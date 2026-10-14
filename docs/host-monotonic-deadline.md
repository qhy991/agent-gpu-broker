# Host monotonic job deadlines

`gpu-run --host-monotonic-deadline SECONDS` accepts an absolute Linux
`CLOCK_MONOTONIC` timestamp. The optional deadline covers queue wait and process
execution together. Existing queue and run timeouts remain independent limits;
the first applicable limit wins. It is job lifetime data, not a measurement
policy, a caller budget, or a guarantee that operating-system cleanup finishes
before the timestamp.

The client sends `run-deadline-v1`, plus `host_deadline` containing
`monotonic_s` and `clock_domain`. The ordinary job fields, including optional
`allowed_gpu_ids`, retain the same meaning. A server that does not support the
operation refuses the request; the client never retries it as `run` or
`run-scoped`. Deadline fields on those legacy operations are also refused.

The clock domain is a SHA256 digest of Linux boot ID, CLOCK_MONOTONIC identity,
and the monotonic time namespace offset. The server verifies both its own domain
and the connecting Unix peer's domain through Linux SO_PEERCRED. Missing clock
identity, unavailable time namespace offsets, unsupported peer verification,
nonfinite or negative timestamps, and mismatched domains cause refusal before
job creation. The digest contains no hostname, token or environment value.
Remote clocks cannot be transported through this option.

An expired queued job never starts a process. A running deadline clamps the
existing wait and terminates the process group with TERM, the configured grace,
then KILL and reap. This deadline path kills remaining group descendants even
if the leader exits on TERM. Completed deadline jobs also clean remaining group
descendants before release, including children with closed output streams.
Existing no-deadline termination remains unchanged.
An observed completion at or after expiry has deadline precedence. The first observed timeout or cancellation keeps its terminal cause during
cleanup. A later cancellation does not relabel deadline expiration. Terminal
publication is single-owner and resource release uses the existing broker path.

Deadline expiration returns `state=phase_deadline`, exit code 124, a distinct
reason and retained raw output. Deadline results include `host_deadline`,
`completed_monotonic_s`, and the actual `process_exit_code` or null when no
process started. Deadline metadata is bound into admission and launch digests;
requests without a deadline preserve their previous serialization. No interrupted
job asserts zero kernel calls or a complete measurement cohort. The caller still
owns phase budget auditing and acceptance, including cleanup that crosses its
hard limit. Descendants that escape the owned process group are outside this
primitive's containment contract.
