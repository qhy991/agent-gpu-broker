# Design contract

## Goal

Allow multiple trusted coding agents on one GPU machine to submit named GPU
commands, observe one global FIFO queue, receive changing position and advisory
ETA in the same long-running tool call, and request either shareable correctness
capacity or clean exclusive capacity across one or more GPUs.

## Canonical owner

The `gpuq serve` process is the only queue and allocation owner. Its Unix socket
is the command and event interface. `status.json`, per-job logs, and result files
in the shared state directory are read-only projections for humans and agents.

## Resource request

Every job explicitly supplies exactly three scheduling fields:

- `label`: a non-empty human-readable name;
- `mode`: `shared` for correctness and other latency-insensitive work, or
  `exclusive` for benchmarks, NCU, and work requiring clean cards;
- `gpu_count`: the number of distinct physical GPUs required simultaneously.

`shared` is cooperative best-effort isolation, not memory or fault isolation.
The daemon owns one machine-wide `shared_capacity` limit per GPU (default: 2).

## Invariants

1. A job receives all requested GPUs atomically or remains queued.
2. An exclusive allocation never overlaps another broker job on its GPUs.
3. A shared allocation overlaps only shared broker jobs and never exceeds the
   configured per-GPU shared capacity.
4. The broker holds a per-card `flock` while any broker job uses that card. Other
   cooperating schedulers can use the same lock directory; a card with a foreign
   compute process or lock is not newly assigned.
5. Jobs start in strict FIFO order. A blocked head job intentionally blocks jobs
   behind it.
6. Waiting time and running time use separate limits.
7. Client disconnect cancels its queued or running job and releases resources.
8. `CUDA_VISIBLE_DEVICES` is set to the allocated physical IDs; the command sees
   them as logical devices `0..gpu_count-1`.
9. Command exit status is returned unchanged, except broker timeouts use 124 and
   explicit cancellation uses 130.

## Failure semantics

- NVIDIA probe failure: fail closed; queued jobs remain queued and status shows
  the probe error.
- Insufficient clean GPUs or shared slots: keep the complete job queued.
- Queue timeout: remove the job without launching a command; exit 124.
- Run timeout: terminate the process group, then release every GPU; exit 124.
- Daemon shutdown or client disconnect: terminate the process group; exit 130.
- Existing foreign activity: leave that card parked until a later clean probe.

## Acceptance evidence

- Two `shared` one-GPU jobs can overlap on one card up to `shared_capacity`.
- An `exclusive` job waits for all shared users of its selected card to finish.
- A multi-GPU job starts once with exactly `gpu_count` distinct cards, or not at
  all.
- Status and streamed events expose label, mode, requested count, allocated GPU
  IDs, queue position, and advisory ETA.

## Scope

Version 0.2 targets trusted agents running under one Unix identity, or a trusted
group that accepts commands running as the daemon user. It does not provide
hostile multi-tenant isolation, distributed multi-node scheduling, priorities,
preemption, memory quotas, or recovery of live commands after daemon restart.
