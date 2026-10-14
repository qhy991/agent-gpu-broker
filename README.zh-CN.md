# Agent GPU Broker

0.7 新增可选 `gpuq run --allowed-gpus 0,1` 作业范围，编号来自站点允许的物理
设备池。daemon 拒绝无效或无法满足的范围，独占与共享打包都不会分配范围外设备；
准确范围绑定到启动身份和 admission 回执。不传此参数的旧请求保持受管理设备池
默认行为。候选的 CPU 测试不能把历史 0.6 设备资格结论延伸到 0.7。
带范围请求在同一 FIFO 提交路径使用 `op=run-scoped`，使 0.6 server 在准入前拒绝，
避免旧 JSON 解析器忽略新字段后错误地启动到范围外设备。

简体中文 | [English](README.md)

## CUDA 缓存生命周期

broker 为每个开始执行的作业分配唯一 `CUDA_CACHE_PATH`，替换继承或请求中的
缓存路径；实际环境摘要绑定到 admission 回执，环境值不写入回执。排队作业不创建
缓存。作业内部继续使用自己的缓存与 warmup，不与其他作业共享已预热缓存。
成功、失败、执行超时或取消后，broker 回收进程、释放 GPU，再删除自己的临时驱动
缓存；GPU 执行时长不包含清理时间。候选、日志、profiler 报告和回执不属于清理目标。
清理失败在终态 `cuda_cache_cleanup` 中明确记录，不改变任务结果或滞留分配。
容器启动器应使用容器内临时缓存，或转发挂载的 broker 缓存，避免写入持久任务 HOME。

`agent-gpu-broker` 是一个面向编程 Agent 的单机全局 GPU 队列。Agent 只需保持
一条普通的 `gpu-run` 命令运行，broker 就会在同一次调用中持续返回队列位置、
预计等待时间、GPU 分配、命令输出和最终结果。

每个任务可以申请一张或多张 GPU，并明确选择两种资源模式之一：

- `shared`：允许与其他正确性检查等非延迟敏感任务共享 GPU；
- `exclusive`：要求干净、独占的 GPU，适合性能测试和 NCU profiling。

其他本机调度器可以与 broker 使用同一个逐卡锁目录，从而避免主动混用同一张卡。

v0.6.0 在 A800/B200 上的精确 admission-receipt 验收见
[qualification report](docs/v0.6.0-admission-receipt-qualification-2026-08-25.md)。

## 架构

```text
Agent A ─┐
Agent B ─┼─ gpu-run ─ Unix Socket ─ gpuq broker ─ GPU 原子分配
Agent C ─┘                         │                 ├─ shared（每卡限流）
                                  │                 └─ exclusive（干净独占）
                                  └─ 状态、日志、结果
```

daemon 是调度状态的唯一所有者。共享状态目录只保存以下只读投影：

- `status.json`：GPU 状态、运行任务、排队顺序和 ETA；
- `events.jsonl`：任务接受、启动和结束事件；
- `jobs/<job-id>/`：请求信息、标准输出、标准错误和结果 JSON。

每个获准任务还会生成 mode-0600 的 `jobs/<job-id>/admission.json`，由 broker
拥有并投影真实 launch identity。

## 安装

共享机器上无需安装，仓库中的启动脚本会直接使用系统 Python：

```bash
bin/gpuq --help
bin/gpu-run --help
```

也可以安装到虚拟环境：

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

## 启动 broker

```bash
bin/gpuq serve \
  --socket /tmp/agent-gpu-broker.sock \
  --state-dir ~/.local/share/agent-gpu-broker \
  --lock-dir /tmp/agent-gpu-locks \
  --shared-capacity 2
```

默认自动扫描机器上的全部 NVIDIA GPU，并跳过存在外部计算进程或外部锁的卡。
可以通过 `--gpus 1,2` 将 broker 限制在指定的物理 GPU 上。

如果由单个 Unix 账号使用，并将仓库放在 `~/agent-gpu-broker`，可以安装仓库内的
systemd 用户服务：

```bash
install -Dm644 deploy/gpu-agent-broker.service \
  ~/.config/systemd/user/gpu-agent-broker.service
systemctl --user daemon-reload
systemctl --user enable --now gpu-agent-broker.service
```

如果由 root 为可信团队统一部署，应使用无特权的专用服务账号和系统级 service，
具体步骤见 [root 管理部署指南](docs/root-deployment.zh-CN.md)。可直接复制到项目中的
通用 Agent 规则见 [docs/AGENTS.example.md](docs/AGENTS.example.md)。

## Agent 使用方式

查看全局队列：

```bash
bin/gpuq status
```

默认状态输出会显示 broker 版本和进程实例，Agent 可以据此确认实际响应的 endpoint；
自动化程序可以从 `--json` 读取相同字段。运行任务的 `wait_seconds` 在开始时冻结，
实际执行时长单独通过持续增长的 `run_seconds` 表示。

正确性检查允许共享一张 GPU：

```bash
bin/gpu-run \
  --label attention-correctness \
  --mode shared \
  --gpu-count 1 \
  --estimate 2m \
  --queue-timeout 2h \
  --run-timeout 10m \
  -- python test.py
```

NCU profiling 要求一张干净独占的 GPU：

```bash
bin/gpu-run \
  --label ncu-attention \
  --mode exclusive \
  --gpu-count 1 \
  --estimate 10m \
  --queue-timeout 2h \
  --run-timeout 20m \
  -- ncu --set full python profile.py
```

原子申请两张 GPU：

```bash
bin/gpu-run \
  --label distributed-correctness \
  --mode shared \
  --gpu-count 2 \
  --run-timeout 10m \
  -- torchrun --nproc-per-node=2 test.py
```

`--label`、`--mode` 和 `--gpu-count` 均为必填项。多卡任务只有在全部资源同时
可用时才会启动，否则整体留在队列中。队列采用严格 FIFO：队首任务资源不足时，
后续小任务不会越过它。

`--queue-timeout` 和 `--run-timeout` 分别限制排队时间和实际运行时间，排队不会
消耗运行预算。`--timeout` 是 `--run-timeout` 的兼容别名，时间支持 `s`、`m`、
`h` 后缀。

任务进入 FIFO 前，daemon 会用自身的无特权身份检查工作目录和可执行命令。被拒绝
的请求返回 127，且不会占用 GPU。Ctrl-C 会显式发送取消请求；服务端断连检测仍作为
兜底路径。

等待期间，Agent 会持续收到位置变化和心跳消息：

```text
[gpu-run] accepted job gpuq-4a2e9b6f3c1d label=ncu-attention mode=exclusive gpus=1
[gpu-run] queued position=3/5 eta=9m30s
[gpu-run] queued position=2/4 eta=4m10s
[gpu-run] running on physical GPUs 1 (run limit 20m)
```

ETA 是根据任务声明的 `--estimate`、GPU 数量、共享槽位和正在运行的 broker 任务
推算出的参考值。外部任务的结束时间未知，因此相关 ETA 也可能显示为 `unknown`。

结束时间无法预知的常驻服务应使用 `--estimate unknown`。服务自身的开始 ETA 仍可
计算，但依赖该服务结束的后续任务会显示 `eta=unknown`，而不是伪造一个数月后的
时间。有限任务超过其声明 estimate 后，依赖它的 ETA 也会转为 `unknown`；逾期进程
不会被假定为立即结束。

## GPU 探测健康状态

每次 `nvidia-smi` 有 10 秒限时；超时或取消会终止并回收该探测子进程。探测失败
不释放已有作业的 GPU 租约；调度器仍处理队列超时，并在下一轮正常轮询重试。
单次调度异常会被报告，不会让整个循环静默退出。

`gpuq status` 输出最后成功探测时间 `gpu_observed_at`，以及基于 monotonic
时钟的 `gpu_observation_age_seconds`。`updated_at` 仅代表状态响应时间。
超过探测限时加两个轮询周期仍无成功观察时，`probe_error` 报告陈旧，ETA 为
unknown。即使旧 GPU 行仍写 idle，也不能把未知或陈旧观察解释为空闲。

队列同时展示 `allowed_gpu_ids`，其中 `null` 表示受管理设备池。限定卡队首
可能在其他卡空闲时挡住后面的不限卡工作；ETA 也遵循这条 FIFO 顺序。本轮不增加
backfill，也不改变分配策略。

## Broker admission receipt

长驻 evaluator 可以让 `gpu-run` 在进程启动后原子保存 broker 签发的 receipt：

```bash
gpu-run \
  --label fibserve-campaign \
  --mode exclusive \
  --gpu-count 1 \
  --estimate unknown \
  --run-timeout 2h \
  --receipt-out /path/to/fibserve-admission.json \
  --env SERVICE_PORT=10000 \
  -- /path/to/start-fibserve.sh
```

任务存活期间，独立控制面可通过 broker socket 重新查询同一 receipt：

```bash
gpuq receipt gpuq-<job-id> --out /path/to/live-admission.json
```

`gpuq.admission-receipt.v1` 绑定 canonical launch spec、argv、显式 environment
override、cwd、owner、label、资源模式、超时、解析后的 executable 路径与文件
SHA-256、broker instance、GPU allocation，以及包含 broker-owned
`CUDA_VISIBLE_DEVICES` 的完整有效环境 SHA-256。receipt 只公开 environment key 与
digest，不公开 value。

broker 会在 admission 时 fingerprint executable，并在 spawn 前再次检查；内容或
路径漂移会在命令启动前失败。任务结束后 live receipt 查询关闭，私有 job 目录和
terminal result 继续保留证据 digest。

## 共享模式的边界

`shared` 是可信任务之间的协作式共享，不提供显存配额、性能隔离或故障隔离。
daemon 通过 `--shared-capacity` 限制每张 GPU 的最大共享任务数，默认值为 2。
性能测试、NCU 和任何延迟敏感测量都必须使用 `exclusive`。

## 信任边界

daemon 会以自身 Unix 用户身份执行提交的命令，因此只适用于相互信任的 Agent
或用户。对于不可信用户，应在调度器前增加认证、权限控制和按用户隔离的容器执行器，
或者直接使用 Slurm 等集群调度系统。

## 测试

测试使用模拟 GPU 清单和普通 CPU 子进程，不需要本机具备 GPU：

```bash
python -m unittest discover -s tests -v
```
