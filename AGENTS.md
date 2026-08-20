# Agent GPU Broker

## GPU usage contract

- Never launch a GPU command directly. Use `gpu-run -- <command>`.
- A `queued` message is normal progress, not a stuck command. Keep the command
  running; do not cancel and resubmit it to obtain a different queue position.
- Set queue and execution limits independently. Queue time does not consume the
  execution time budget.
- Give each job a short `--label` that lets other agents understand the queue
  without exposing the full command.
- Do not set `CUDA_VISIBLE_DEVICES`; the broker assigns exactly one clean,
  exclusively locked physical GPU and exposes it as the command's only device.
- Use `gpuq status` to inspect the machine-wide queue. Treat ETA as advisory.
- NCU and other profiling commands use the same `gpu-run` path; never bypass the
  broker for profiling.

Examples:

```bash
gpuq status
gpu-run --label ncu-attention --queue-timeout 2h --run-timeout 20m -- \
  ncu --set full python profile.py
gpu-run --label benchmark-v7 --estimate 8m --timeout 30m -- python bench.py
```

## Repository rules

- Keep the daemon as the sole owner of queue state and scheduling policy.
- Shared files are projections, logs, and results; clients must not edit them.
- Preserve one FIFO path and one exclusive-clean GPU mode until a real second
  scheduling policy is required.
- Use only the Python standard library unless a dependency removes more
  complexity than it adds.
- Run `python -m unittest discover -s tests -v` before committing.
