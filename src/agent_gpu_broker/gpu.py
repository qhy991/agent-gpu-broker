from __future__ import annotations

import asyncio
import fcntl
import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping, Protocol, Sequence


GPU_PROBE_TIMEOUT_S = 10.0


class BackendId(str, Enum):
    NVIDIA = "nvidia"
    METAX = "metax"


class GpuInventory(Protocol):
    """Own physical discovery, complete system occupancy, and runtime selectors."""

    @property
    def backend(self) -> BackendId: ...

    async def gpu_ids(self) -> list[int]: ...

    async def compute_pids(self) -> dict[int, list[int]]: ...

    def child_environment(
        self, base: Mapping[str, str], *, physical_ids: tuple[int, ...]
    ) -> dict[str, str]: ...


async def _smi(command: str, *args: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        command,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=GPU_PROBE_TIMEOUT_S
        )
    except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        await proc.communicate()
        if isinstance(exc, asyncio.CancelledError):
            raise
        raise RuntimeError(f"{command} timed out after {GPU_PROBE_TIMEOUT_S:g}s") from exc
    if proc.returncode != 0:
        detail = stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"{command} failed ({proc.returncode}): {detail}")
    return stdout.decode("utf-8", "replace")


class NvidiaSmiInventory:
    """Probe physical GPU indices and compute PIDs without Python dependencies."""

    def __init__(self) -> None:
        self._uuid_to_index: dict[str, int] = {}

    @property
    def backend(self) -> BackendId:
        return BackendId.NVIDIA

    def child_environment(
        self, base: Mapping[str, str], *, physical_ids: tuple[int, ...]
    ) -> dict[str, str]:
        return {**base, "CUDA_VISIBLE_DEVICES": ",".join(map(str, physical_ids))}

    async def gpu_ids(self) -> list[int]:
        output = await _smi(
            "nvidia-smi",
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
        output = await _smi(
            "nvidia-smi",
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


def _metax_device_ids(output: str) -> list[int]:
    lines = output.strip().splitlines()
    if not lines or not re.fullmatch(r"mx-smi  version: [0-9]+\.[0-9]+\.[0-9]+", lines[0]):
        raise RuntimeError("mx-smi discovery: missing version banner")
    ids: set[int] = set()
    pci_addresses: set[str] = set()
    uuids: set[str] = set()
    for line in lines[1:]:
        row = re.fullmatch(
            r"GPU#([0-9]+) +\S+ +([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7])"
            r" +Available \(UUID: (GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
            r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\)",
            line,
        )
        if row is None:
            raise RuntimeError("mx-smi discovery: malformed or unavailable device row")
        device_id = int(row[1])
        pci, uuid = row[2].lower(), row[3].lower()
        if device_id in ids or pci in pci_addresses or uuid in uuids:
            raise RuntimeError("mx-smi discovery: duplicate device identity")
        ids.add(device_id)
        pci_addresses.add(pci)
        uuids.add(uuid)
    if not ids:
        raise RuntimeError("mx-smi discovery: no devices")
    return sorted(ids)


def _metax_compute_pids(
    output: str, *, discovered_ids: Sequence[int]
) -> dict[int, list[int]]:
    lines = [line for line in output.splitlines() if line.strip()]
    border = "+" + "-" * 81 + "+"
    if (
        len(lines) < 12
        or not re.fullmatch(r"mx-smi  version: [0-9]+\.[0-9]+\.[0-9]+", lines[0])
        or lines[1] != "=================== MetaX System Management Interface Log ==================="
        or not re.fullmatch(
            r"Timestamp +: (Mon|Tue|Wed|Thu|Fri|Sat|Sun) "
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) +[0-9]{1,2} "
            r"[0-9]{2}:[0-9]{2}:[0-9]{2} [0-9]{4}", lines[2]
        )
        or lines[4] != border
        or lines[5] != "| Process:" + " " * 72 + "|"
        or lines[6] != "|  GPU                    PID         Process Name                 GPU Memory     |"
        or lines[7] != "|                                                                  Usage(MiB)     |"
        or lines[8] != "|" + "=" * 81 + "|"
        or lines[-2] != border
        or lines[-1] != "End of Log"
    ):
        raise RuntimeError("mx-smi occupancy: incomplete or unrecognized process table")
    count = re.fullmatch(r"Attached GPUs +: ([0-9]+)", lines[3])
    if count is None or int(count[1]) != len(discovered_ids):
        raise RuntimeError("mx-smi occupancy: attached count differs from discovery")
    result: dict[int, list[int]] = {device_id: [] for device_id in discovered_ids}
    rows = lines[9:-2]
    if rows == ["|  no process found" + " " * 63 + "|"]:
        return result
    for line in rows:
        row = re.fullmatch(
            r"\| +([0-9]+) +([1-9][0-9]*) {2,}([^\s|](?:[^|]*?[^\s|])?) {2,}([0-9]+) +\|",
            line,
        )
        if row is None or len(line) != 83:
            raise RuntimeError("mx-smi occupancy: malformed process row")
        device_id, pid = int(row[1]), int(row[2])
        if device_id not in result or pid in result[device_id]:
            raise RuntimeError("mx-smi occupancy: unknown device or duplicate process row")
        result[device_id].append(pid)
    return result


class MetaXSmiInventory:
    def __init__(self) -> None:
        self._device_ids: tuple[int, ...] | None = None

    @property
    def backend(self) -> BackendId:
        return BackendId.METAX

    async def gpu_ids(self) -> list[int]:
        if self._device_ids is None:
            self._device_ids = tuple(_metax_device_ids(await _smi("mx-smi", "--list")))
        return list(self._device_ids)

    async def compute_pids(self) -> dict[int, list[int]]:
        discovered_ids = await self.gpu_ids()
        return _metax_compute_pids(
            await _smi("mx-smi", "--show-all-process"), discovered_ids=discovered_ids
        )

    def child_environment(
        self, base: Mapping[str, str], *, physical_ids: tuple[int, ...]
    ) -> dict[str, str]:
        visible = ",".join(map(str, physical_ids))
        return {**base, "MACA_VISIBLE_DEVICES": visible, "CUDA_VISIBLE_DEVICES": visible}


def inventory_for(backend: BackendId) -> GpuInventory:
    if backend is BackendId.NVIDIA:
        return NvidiaSmiInventory()
    if backend is BackendId.METAX:
        return MetaXSmiInventory()
    raise ValueError(f"unsupported GPU backend: {backend!r}")


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
    """Take an advisory per-card lock shared with cooperating schedulers."""
    lock_dir.mkdir(parents=True, exist_ok=True)
    path = lock_dir / f"gpu-{gpu_id}.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return CardLock(gpu_id=gpu_id, fd=fd)
