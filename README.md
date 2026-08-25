# Agent GPU Broker

[简体中文](README.zh-CN.md) | English

`agent-gpu-broker` is a small, machine-wide GPU queue for coding agents. Agents
keep a normal `gpu-run` command open while the broker streams queue position,
advisory ETA, allocation, output, and completion back through the same call.

The broker is deliberately single-host. Each job requests `shared` correctness
capacity or `exclusive` clean-card capacity and one or more GPUs. Cooperating
local schedulers can use its per-card lock directory to avoid co-tenancy.

The exact v0.6.0 A800/B200 admission-receipt qualification is recorded in
[the qualification report](docs/v0.6.0-admission-receipt-qualification-2026-08-25.md).

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

Each admitted job also has `jobs/<job-id>/admission.json`, a mode-0600
broker-issued projection of the exact launch identity.

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
  --lock-dir /tmp/agent-gpu-locks \
  --shared-capacity 2
```

Use `--gpus 1,2` to constrain an experiment to selected physical cards. Without
it, the broker discovers every NVIDIA GPU and still skips cards that are occupied
or locked by another process.

For a single Unix account with the repository at `~/agent-gpu-broker`, install
the checked-in user service with:

```bash
install -Dm644 deploy/gpu-agent-broker.service \
  ~/.config/systemd/user/gpu-agent-broker.service
systemctl --user daemon-reload
systemctl --user enable --now gpu-agent-broker.service
```

User services normally follow that user's login lifecycle. A root administrator
deploying this for a trusted team should instead follow the
[root-managed deployment guide](docs/root-deployment.md), which uses a dedicated
unprivileged service account and a system service. A ready-to-copy generic Agent
policy is available in [docs/AGENTS.example.md](docs/AGENTS.example.md).

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

The human-readable status header includes the broker version and process
instance so an agent can confirm which endpoint answered. `--json` exposes the
same identity fields for automation. Running jobs expose a frozen queue
`wait_seconds` and a separately increasing `run_seconds`.

`--timeout` is a compatibility alias for `--run-timeout`. Durations accept `s`,
`m`, or `h`. `--label`, `--mode`, and `--gpu-count` are required. Multi-GPU
allocation is atomic. Queue position is exact FIFO order; a blocked head job is
not bypassed. ETA is advisory and accounts for declared estimates, requested GPU
counts, shared slots, and running broker jobs; externally occupied GPUs have an
unknown release time.

Long-lived services whose stop time is not known should use
`--estimate unknown`. Their own start ETA can still be known, while jobs whose
start depends on that service report `eta=unknown` instead of a fabricated
multi-month duration. A bounded job that runs past its declared estimate also
makes dependent ETAs unknown; an overdue process is not treated as finishing
immediately.

## Broker-issued admission receipts

A long-running evaluator can ask `gpu-run` to atomically save the receipt that
the broker emits after process start:

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

While the job is active, an independent controller can re-query the same
receipt through the broker socket:

```bash
gpuq receipt gpuq-<job-id> --out /path/to/live-admission.json
```

Schema `gpuq.admission-receipt.v1` binds:

- canonical launch-spec, argv, and explicit environment-override SHA-256;
- cwd, owner, label, mode, GPU count, and timeouts;
- resolved executable path plus file SHA-256;
- broker version/instance and submit/start timestamps;
- allocated physical GPU IDs;
- SHA-256 of the complete effective child environment, including the
  broker-owned `CUDA_VISIBLE_DEVICES`.

The receipt exposes environment keys and digests, never environment values.
The executable is fingerprinted at admission, rechecked immediately before
spawn, and executed through the resolved path. Content/path drift fails before
the command starts. Active receipt lookup closes when a job becomes terminal;
the private job directory and terminal result retain the durable digests.

Before a request enters the FIFO, the daemon checks that its own unprivileged
identity can enter the working directory and execute the command. Rejected
requests return exit 127 without reserving a GPU. Ctrl-C sends an explicit
cancellation request; disconnect detection remains the server-side fallback.

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
