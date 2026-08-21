## Shared GPU usage

- Never launch CUDA, benchmark, or profiling commands directly. Submit every GPU
  command through `gpu-run`.
- Run `gpuq status` before submitting work when the current machine state matters.
- Every job must provide a short `--label`, `--mode shared|exclusive`, and a
  positive `--gpu-count`.
- Use `shared` only for correctness checks and latency-insensitive work. Use
  `exclusive` for benchmarks, NCU profiling, and clean-card measurements.
- Set independent `--queue-timeout` and `--run-timeout` limits and a realistic
  `--estimate`. Queue waiting does not consume the run-time budget.
- Treat `[gpu-run] queued` as normal progress. Keep the command running; do not
  cancel and resubmit merely to change its position.
- Do not set `CUDA_VISIBLE_DEVICES`; the broker assigns physical GPUs and exposes
  them as logical devices starting at zero.
- When multiple agents share one Unix identity, pass `--owner TEAM_MEMBER` so the
  queue identifies the submitter. Replace `TEAM_MEMBER` with a stable name.
- CPU-only compilation may run directly. If the broker is unavailable, report
  the failure instead of bypassing it or starting a competing scheduler.

```bash
gpuq status

gpu-run --owner TEAM_MEMBER --label correctness-rmsnorm \
  --mode shared --gpu-count 1 --estimate 2m \
  --queue-timeout 2h --run-timeout 10m -- python test.py

gpu-run --owner TEAM_MEMBER --label ncu-rmsnorm \
  --mode exclusive --gpu-count 1 --estimate 10m \
  --queue-timeout 2h --run-timeout 20m -- \
  ncu --set full python profile.py
```
