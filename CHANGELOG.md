# Changelog

## [Unreleased]

### Added

- Add optional `--allowed-gpus` / `allowed_gpu_ids` job scope. The broker validates managed physical indices, applies the same scope to exclusive allocation, shared packing and ETA, and binds it into the launch identity and admission receipt. Legacy requests without a scope retain the managed-pool default. This supplies the broker 0.7 contract required by KerSor NCU (`broker.py`, `server.py`, `cli.py`, GPU scope tests).

This candidate has not been deployed to B300-M3. CPU tests use fake GPU inventory; they do not qualify device execution.
