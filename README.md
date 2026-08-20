# Agent GPU Broker

`agent-gpu-broker` is a small, machine-wide GPU queue for coding agents. Agents
keep a normal `gpu-run` command open while the broker streams queue position,
advisory ETA, allocation, output, and completion back through the same call.

The broker is deliberately single-host. Each job requests `shared` correctness
capacity or `exclusive` clean-card capacity and one or more GPUs. It shares
KDA's default `/tmp/kda-gpu-locks` lock domain, so KDA work never intentionally
co-tenants a broker-managed card.

## Architecture

```text
Agent A ─┐
Agent B ─┼─ gpu-run ─ Unix socket ─ gpuq broker ─ atomic GPU allocation
Agent C ─┘                         │                 ├─ shared (bounded/card)
                                  │                 └─ exclusive (clean)
                                  └─ status, logs, results
```

The daemon owns all mutable scheduling state. The shared state directory is a
read-only projection containing:

- `status.json`: current GPU states, running jobs, queue order, and ETA;
- `events.jsonl`: accepted, started, and terminal lifecycle events;
- `jobs/<job-id>/`: request metadata, stdout/stderr logs, and result JSON.

## Install

No installation is required on a shared host. The repository launchers use the
system Python directly:

```bash
bin/gpuq --help
bin/gpu-run --help
```

An editable virtual-environment install is optional:

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

## Start the broker

```bash
bin/gpuq serve \
  --socket /tmp/agent-gpu-broker.sock \
  --state-dir ~/.local/share/agent-gpu-broker \
  --lock-dir /tmp/kda-gpu-locks \
  --shared-capacity 2
```

Use `--gpus 1,2` to constrain an experiment to selected physical cards. Without
it, the broker discovers every NVIDIA GPU and still skips cards that are occupied
or locked by another process.

On `verda-b200x4`, install the checked-in user service with:

```bash
install -Dm644 deploy/gpu-agent-broker.service \
  ~/.config/systemd/user/gpu-agent-broker.service
systemctl --user daemon-reload
systemctl --user enable --now gpu-agent-broker.service
```

The account currently has systemd user services but not login lingering, so the
service starts with the user's login session rather than before login.

## Agent commands

```bash
bin/gpuq status

bin/gpu-run \
  --label ncu-attention \
  --mode exclusive \
  --gpu-count 1 \
  --queue-timeout 2h \
  --run-timeout 20m \
  --estimate 10m \
  -- ncu --set full python profile.py

bin/gpu-run \
  --label distributed-correctness \
  --mode shared \
  --gpu-count 2 \
  --run-timeout 10m \
  -- torchrun --nproc-per-node=2 test.py
```

`--timeout` is a compatibility alias for `--run-timeout`. Durations accept `s`,
`m`, or `h`. `--label`, `--mode`, and `--gpu-count` are required. Multi-GPU
allocation is atomic. Queue position is exact FIFO order; a blocked head job is
not bypassed. ETA is advisory and accounts for declared estimates, requested GPU
counts, shared slots, and running broker jobs; externally occupied GPUs have an
unknown release time.

Status messages are emitted when position/ETA changes and as a periodic heartbeat:

```text
[gpu-run] accepted job gpuq-4a2e9b6f3c1d label=ncu-attention mode=exclusive gpus=1
[gpu-run] queued position=3/5 eta~9m30s
[gpu-run] queued position=2/4 eta~4m10s
[gpu-run] running on physical GPUs 1 (run limit 20m)
```

`shared` means cooperative best-effort sharing; it does not impose memory quotas
or isolate failures. Use `exclusive` for benchmarks, NCU, and latency-sensitive
measurements.

## Trust boundary

The daemon executes submitted commands as its own Unix user. Use it only among
trusted agents/users. For mutually untrusted users, place an authenticated API
and per-user container executor in front of the scheduling core, or use a cluster
scheduler such as Slurm.

## Tests

The test suite uses fake GPU inventory with ordinary CPU subprocesses; no GPU is
required:

```bash
python -m unittest discover -s tests -v
```
