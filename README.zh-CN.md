# Agent GPU Broker

简体中文 | [English](README.md)

`agent-gpu-broker` 是一个面向编程 Agent 的单机全局 GPU 队列。Agent 只需保持
一条普通的 `gpu-run` 命令运行，broker 就会在同一次调用中持续返回队列位置、
预计等待时间、GPU 分配、命令输出和最终结果。

每个任务可以申请一张或多张 GPU，并明确选择两种资源模式之一：

- `shared`：允许与其他正确性检查等非延迟敏感任务共享 GPU；
- `exclusive`：要求干净、独占的 GPU，适合性能测试和 NCU profiling。

broker 与 KDA 默认共用 `/tmp/kda-gpu-locks` 锁域，因此 KDA 任务不会主动与
broker 管理的任务混用同一张卡。

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
  --lock-dir /tmp/kda-gpu-locks \
  --shared-capacity 2
```

默认自动扫描机器上的全部 NVIDIA GPU，并跳过存在外部计算进程或外部锁的卡。
可以通过 `--gpus 1,2` 将 broker 限制在指定的物理 GPU 上。

在 `verda-b200x4` 上可以安装仓库内的 systemd 用户服务：

```bash
install -Dm644 deploy/gpu-agent-broker.service \
  ~/.config/systemd/user/gpu-agent-broker.service
systemctl --user daemon-reload
systemctl --user enable --now gpu-agent-broker.service
```

## Agent 使用方式

查看全局队列：

```bash
bin/gpuq status
```

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

等待期间，Agent 会持续收到位置变化和心跳消息：

```text
[gpu-run] accepted job gpuq-4a2e9b6f3c1d label=ncu-attention mode=exclusive gpus=1
[gpu-run] queued position=3/5 eta=9m30s
[gpu-run] queued position=2/4 eta=4m10s
[gpu-run] running on physical GPUs 1 (run limit 20m)
```

ETA 是根据任务声明的 `--estimate`、GPU 数量、共享槽位和正在运行的 broker 任务
推算出的参考值。外部任务的结束时间未知，因此相关 ETA 也可能显示为 `unknown`。

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
