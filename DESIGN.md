# Design contract

## Goal

Allow multiple trusted coding agents on one GPU machine to submit commands,
observe one global FIFO queue, receive changing position and ETA messages in the
same long-running tool call, and execute on one clean, exclusively locked GPU.

## Canonical owner

The `gpuq serve` process is the only queue owner. Its Unix socket is the command
and event interface. `status.json`, per-job logs, and result files in the shared
state directory are read-only projections for humans and agents.

## Invariants

1. A GPU is assigned to at most one broker job at a time.
2. The broker holds the same per-card `flock` used by KDA for the whole command.
3. A card with a compute process or an external lock is not assigned.
4. Waiting time and running time use separate limits.
5. Jobs start in FIFO order; all jobs currently request the same resource:
   one exclusive, clean physical GPU.
6. Client disconnect cancels its queued or running job and releases resources.
7. Command exit status is returned unchanged, except broker timeouts use 124 and
   explicit cancellation uses 130.

## Failure semantics

- NVIDIA probe failure: fail closed; queued jobs remain queued and status shows
  the probe error.
- Queue timeout: remove the job without launching a command; exit 124.
- Run timeout: terminate the process group, then release the GPU; exit 124.
- Daemon shutdown or client disconnect: terminate the process group; exit 130.
- Existing foreign activity: leave that card parked until a later clean probe.

## Scope

Version 0.1 targets trusted agents running under one Unix identity, or a trusted
group that accepts commands running as the daemon user. It does not provide
hostile multi-tenant isolation, distributed multi-node scheduling, priorities,
preemption, shared-GPU jobs, or recovery of live commands after daemon restart.
