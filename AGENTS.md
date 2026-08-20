# Agent GPU Broker

## GPU usage contract

- Never launch a GPU command directly. Use `gpu-run` with an explicit job name,
  sharing mode, and GPU count.
- A `queued` message is normal progress, not a stuck command. Keep the command
  running; do not cancel and resubmit it to obtain a different queue position.
- Set queue and execution limits independently. Queue time does not consume the
  execution time budget.
- `--label`, `--mode shared|exclusive`, and `--gpu-count` are mandatory.
- Use `shared` only for correctness or latency-insensitive work. Use `exclusive`
  for benchmarks, NCU profiling, and any clean-card measurement.
- Do not set `CUDA_VISIBLE_DEVICES`; the broker atomically assigns the requested
  number of physical GPUs and exposes them as logical devices starting at zero.
- Use `gpuq status` to inspect the machine-wide queue. Treat ETA as advisory.
- NCU and other profiling commands use the same `gpu-run` path; never bypass the
  broker for profiling.

Examples:

```bash
gpuq status
gpu-run --label ncu-attention --mode exclusive --gpu-count 1 \
  --queue-timeout 2h --run-timeout 20m -- \
  ncu --set full python profile.py
gpu-run --label correctness-v7 --mode shared --gpu-count 1 \
  --estimate 2m --timeout 10m -- python test.py
```

## Repository rules

- Keep the daemon as the sole owner of queue state and scheduling policy.
- Shared files are projections, logs, and results; clients must not edit them.
- Preserve one FIFO path. Express shared/exclusive and GPU count as resource
  request data, not separate command paths.
- Use only the Python standard library unless a dependency removes more
  complexity than it adds.
- Run `python -m unittest discover -s tests -v` before committing.
