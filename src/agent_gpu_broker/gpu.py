"""NVIDIA GPU discovery, occupancy probes, and KDA-compatible card locks."""

from __future__ import annotations

import asyncio
import fcntl
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class GpuInventory(Protocol):
    async def gpu_ids(self) -> list[int]: ...

    async def compute_pids(self) -> dict[int, list[int]]: ...


async def _nvidia_smi(*args: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        "nvidia-smi",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        detail = stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"nvidia-smi failed ({proc.returncode}): {detail}")
    return stdout.decode("utf-8", "replace")


class NvidiaSmiInventory:
    """Probe physical GPU indices and compute PIDs without Python dependencies."""

    def __init__(self) -> None:
        self._uuid_to_index: dict[str, int] = {}

    async def gpu_ids(self) -> list[int]:
        output = await _nvidia_smi(
            "--query-gpu=index,uuid", "--format=csv,noheader,nounits"
        )
        mapping: dict[str, int] = {}
        for line in output.splitlines():
            if not line.strip():
                continue
            index_text, uuid = (part.strip() for part in line.split(",", 1))
            mapping[uuid] = int(index_text)
        self._uuid_to_index = mapping
        return sorted(mapping.values())

    async def compute_pids(self) -> dict[int, list[int]]:
        if not self._uuid_to_index:
            await self.gpu_ids()
        output = await _nvidia_smi(
            "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"
        )
        result = {index: [] for index in self._uuid_to_index.values()}
        for line in output.splitlines():
            if not line.strip():
                continue
            uuid, pid_text = (part.strip() for part in line.split(",", 1))
            index = self._uuid_to_index.get(uuid)
            if index is not None:
                result[index].append(int(pid_text))
        return result


@dataclass
class CardLock:
    gpu_id: int
    fd: int

    def release(self) -> None:
        if self.fd < 0:
            return
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
        finally:
            os.close(self.fd)
            self.fd = -1


def try_card_lock(gpu_id: int, lock_dir: Path) -> CardLock | None:
    """Take the same advisory per-card lock used by KDA."""
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / f"gpu-{gpu_id}.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return CardLock(gpu_id=gpu_id, fd=fd)
