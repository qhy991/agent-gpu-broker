# Changelog

## [Unreleased]

### Added

- Add optional `--allowed-gpus` / `allowed_gpu_ids` job scope. The broker validates managed physical indices, applies the same scope to exclusive allocation, shared packing and ETA, and binds it into the launch identity and admission receipt. Legacy requests without a scope retain the managed-pool default. This supplies the broker 0.7 contract required by KerSor NCU (`broker.py`, `server.py`, `cli.py`, GPU scope tests).

- Expose last successful `gpu_observed_at` and monotonic
  `gpu_observation_age_seconds`; status reads never refresh them. Missing, failed
  or stale observation sets `probe_error` and makes queue ETA unknown. Freshness
  expires after the probe bound plus two polling intervals. Status also exposes
  existing `allowed_gpu_ids` so agents can explain scoped FIFO waits.

### Fixed

- Integrate main's scheduler/connection hardening with the exact `807aea5` GPU
  scope and admission source. One iteration error no longer kills the queue;
  wakeup/close races preserve cancellation and leave no polling tasks.
- Bound each `nvidia-smi` call to 10 seconds and kill/reap its child on timeout
  or cancellation. Failed probes block new allocation, preserve existing leases,
  and keep queue expiry and later recovery working (`gpu.py`, `broker.py`).
- Make ETA respect the scoped FIFO head: an unavailable eligible GPU produces
  unknown ETA, and later jobs cannot claim to start before their predecessor.
  Reject nonfinite CLI durations (`broker.py`, `cli.py`).

This is an unreleased `0.8.0.dev0` successor. The deployed B300-M3 source was
independently matched to `807aea5`; these repairs have not been deployed.
CPU/fake-inventory tests do not establish device correctness or performance.
