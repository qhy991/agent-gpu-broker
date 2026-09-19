"""Device mechanisms only; allocation and queue policy remain in GpuBroker.

An external probe is a read-only argv returning gpuq.device-probe.v1 JSON:
{"schema": "gpuq.device-probe.v1", "devices": [{"id": 0, "compute_pids": []}]}
A missing/null PID list is unknown, never an empty observation.
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from .gpu import NvidiaSmiInventory

VISIBILITY_KEYS = (
    "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL", "GPUQ_BACKEND", "GPUQ_DEVICE_IDS", "GPUQ_OCCUPANCY_SCOPE",
)


async def read_text(argv: list[str], timeout: float = 10.0):
    process = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        start_new_session=True
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except BaseException:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await process.wait()
        raise
    if process.returncode:
        raise RuntimeError(f"device probe failed ({process.returncode}): {stderr.decode(errors='replace')[-1000:]}")
    return stdout.decode("utf-8")


async def read_json(argv: list[str], timeout: float = 10.0):
    return json.loads(await read_text(argv, timeout))


class DeviceBackend:
    def __init__(self, name, inventory, occupancy_scope="system"):
        self.name = name
        self.inventory = inventory
        self.occupancy_scope = occupancy_scope

    def environment(self, inherited, gpu_ids):
        result = {key: value for key, value in inherited.items() if key not in VISIBILITY_KEYS}
        selected = ",".join(map(str, gpu_ids))
        result.update(GPUQ_BACKEND=self.name, GPUQ_DEVICE_IDS=selected,
                      GPUQ_OCCUPANCY_SCOPE=self.occupancy_scope)
        if self.name == "nvidia":
            result["CUDA_VISIBLE_DEVICES"] = selected
        elif self.name == "hygon":
            # HIP runtime ordinal namespace, supplied by the configured probe.
            result["HIP_VISIBLE_DEVICES"] = selected
        elif gpu_ids != (0,) and gpu_ids != [0]:
            raise RuntimeError("Metal backend supports only the single Apple GPU")
        return result


class CommandInventory:
    """Site-specific DTK probe; reject incomplete or changing device inventories."""
    def __init__(self, command):
        self.command = command
        self.ids = None

    async def _probe(self):
        value = await read_json(self.command)
        if not isinstance(value, dict) or value.get("schema") != "gpuq.device-probe.v1":
            raise RuntimeError("invalid device probe schema")
        devices = value.get("devices")
        if not isinstance(devices, list) or not devices:
            raise RuntimeError("device probe requires nonempty devices")
        result = {}
        for item in devices:
            if not isinstance(item, dict):
                raise RuntimeError("invalid device observation")
            index = item.get("id")
            pids = item.get("compute_pids")
            if type(index) is not int or index < 0 or index in result:
                raise RuntimeError("invalid or duplicate runtime device ordinal")
            if not isinstance(pids, list) or any(type(pid) is not int or pid <= 0 for pid in pids):
                raise RuntimeError("device process occupancy unknown or malformed")
            result[index] = pids
        if self.ids is not None and sorted(result) != self.ids:
            raise RuntimeError("device inventory changed; restart and requalify broker")
        return result

    async def gpu_ids(self):
        self.ids = sorted(await self._probe())
        return self.ids

    async def compute_pids(self):
        return await self._probe()


class MetalInventory:
    async def gpu_ids(self):
        if sys.platform != "darwin":
            raise RuntimeError("Metal backend requires macOS")
        value = await read_json(["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"])
        devices = value.get("SPDisplaysDataType", [])
        if len(devices) != 1 or not devices[0].get("sppci_model", "").startswith("Apple "):
            raise RuntimeError("Metal backend requires exactly one Apple GPU")
        if not any("metal" in str(v).lower() for k, v in devices[0].items() if "mtl" in k or "metal" in k):
            raise RuntimeError("Metal support not reported")
        return [0]

    async def compute_pids(self):
        # Explicit cooperative scope: no claim about external/system occupancy.
        return {0: []}


def make_backend(name, *, occupancy_scope="system", probe_command=None, hygon_library="/usr/local/hyhal/lib/librocm_smi64.so"):
    if name == "nvidia":
        if occupancy_scope != "system" or probe_command:
            raise ValueError("NVIDIA uses its built-in system probe")
        return DeviceBackend(name, NvidiaSmiInventory())
    if name == "metal":
        if occupancy_scope != "cooperative" or probe_command:
            raise ValueError("Metal requires --occupancy-scope cooperative; external occupancy is unknown")
        return DeviceBackend(name, MetalInventory(), occupancy_scope)
    if name == "hygon":
        if occupancy_scope != "system":
            raise ValueError("Hygon requires system occupancy observations")
        command = probe_command or [sys.executable, str(Path(__file__).with_name("hygon_probe.py")), hygon_library]
        inventory = CommandInventory(command)
        return DeviceBackend(name, inventory)
    raise ValueError(f"unsupported GPU backend: {name}")
