# 更新日志

## [Unreleased]

### Added

- 新增可选 `--allowed-gpus` / `allowed_gpu_ids` 作业范围。broker 校验受管理的物理 GPU 编号，在独占分配、共享打包和 ETA 中使用相同范围，并将其绑定到启动身份与 admission 回执；没有范围的旧请求继续使用受管理设备池默认值。此实现提供 KerSor NCU 要求的 broker 0.7 合同（`broker.py`、`server.py`、`cli.py`、GPU 范围测试）。

- 输出最后成功探测时间 `gpu_observed_at` 与基于 monotonic 时钟的
  `gpu_observation_age_seconds`；读取 status 不刷新它们。缺少、失败或陈旧探测
  设置 `probe_error` 并让 ETA 为 unknown；新鲜度上限是探测限时加两个轮询周期。
  status 还展示已有的 `allowed_gpu_ids`，使 Agent 能解释限定卡 FIFO 等待。

### Fixed

- 将主线 scheduler/连接清理修复与已核实的 `807aea5` GPU scope、admission
  源码整合。单次循环异常不再结束队列；wakeup/close 竞争保留取消语义，回收等待任务。
- 每次 `nvidia-smi` 最多等待 10 秒；超时或取消时终止并回收其子进程。
  探测失败禁止新分配，保留已有租约，继续执行队列超时与后续恢复
  （`gpu.py`、`broker.py`）。
- ETA 遵循限定卡 FIFO：可用设备不满足范围时返回 unknown，后续作业不能显示
  早于前序作业的开始时间；CLI 拒绝非有限时长（`broker.py`、`cli.py`）。

这是尚未发布的 `0.8.0.dev0` 后继版本。B300-M3 已部署源码已逐文件核对为
`807aea5`，本轮修复尚未部署。CPU/模拟设备测试不证明设备正确性或性能。
