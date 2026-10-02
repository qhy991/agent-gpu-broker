# 更新日志

## [Unreleased]

### Added

- 新增可选 `--allowed-gpus` / `allowed_gpu_ids` 作业范围。broker 校验受管理的物理 GPU 编号，在独占分配、共享打包和 ETA 中使用相同范围，并将其绑定到启动身份与 admission 回执；没有范围的旧请求继续使用受管理设备池默认值。此实现提供 KerSor NCU 要求的 broker 0.7 合同（`broker.py`、`server.py`、`cli.py`、GPU 范围测试）。

此候选尚未部署到 B300-M3。CPU 测试使用模拟 GPU 清单，不能作为设备执行资格证明。
