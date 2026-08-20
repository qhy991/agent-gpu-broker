# Agent GPU Broker

`agent-gpu-broker` is a small, machine-wide GPU queue for coding agents. Agents
keep a normal `gpu-run` command open while the broker streams queue position,
advisory ETA, allocation, output, and completion back through the same call.

The broker is deliberately single-host and exclusive-only. It shares KDA's
default `/tmp/kda-gpu-locks` lock domain, so KDA judge jobs, KDA agent jobs, and
`gpu-run` jobs from this project do not intentionally co-tenant a card.

## Architecture

```text
Agent A ─┐
Agent B ─┼─ gpu-run ─ Unix socket ─ gpuq broker ─ one clean locked GPU/job
Agent C ─┘                         │
                                  └─ shared status, logs, results
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
  --lock-dir /tmp/kda-gpu-locks
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
  --queue-timeout 2h \
  --run-timeout 20m \
  --estimate 10m \
  -- ncu --set full python profile.py
```

`--timeout` is a compatibility alias for `--run-timeout`. Durations accept `s`,
`m`, or `h`. Queue position is exact FIFO order. ETA is advisory and is computed
from each job's declared estimate and the estimated remaining duration of running
broker jobs; it is unknown while every usable GPU is occupied externally.

Status messages are emitted when position/ETA changes and as a periodic heartbeat:

```text
[gpu-run] accepted job gpuq-4a2e9b6f3c1d
[gpu-run] queued position=3/5 eta~9m30s
[gpu-run] queued position=2/4 eta~4m10s
[gpu-run] running on physical GPU 1 (run limit 20m)
```

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
