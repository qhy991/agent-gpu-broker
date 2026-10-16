# mx-smi fixtures

These are command-output excerpts from C500 discovery on 2026-10-16, using
mx-smi 2.3.1. Surrounding shell output was removed and trailing blank lines normalized. Discovery UUIDs were replaced
with distinct synthetic UUIDs; their values are irrelevant to parser behavior.

- `mx-smi-list.txt` is the `--list` output from `c500-management-interface-001.log`.
- `mx-smi-empty.txt` is the `--show-all-process` output from `c500-process-discovery-release-001.log`.
- `mx-smi-holders.txt` is the `--show-all-process` table from
  `c500-process-discovery-holder-001.log`.
  PID 73380 held all four devices with 0 MiB after a native device-count query.

These fixtures establish the observed text grammar, not general process coverage
or runtime isolation.
