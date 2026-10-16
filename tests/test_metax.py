from __future__ import annotations

import asyncio
import contextlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import AsyncMock, patch

from agent_gpu_broker import gpu
from agent_gpu_broker.broker import GpuBroker, _validate_occupancy
from agent_gpu_broker.cli import _parser
from test_broker import await_started, spec, terminal_events
from test_probe_reliability import until


FIXTURES = Path(__file__).parent / "fixtures"
DISCOVERY = (FIXTURES / "mx-smi-list.txt").read_text()
EMPTY = (FIXTURES / "mx-smi-empty.txt").read_text()
HOLDERS = (FIXTURES / "mx-smi-holders.txt").read_text()


class MetaXParserTests(unittest.TestCase):
    def test_captured_discovery_empty_and_zero_memory_holders(self):
        ids = gpu._metax_device_ids(DISCOVERY)
        self.assertEqual(ids, [0, 1, 2, 3])
        self.assertEqual(gpu._metax_compute_pids(EMPTY, discovered_ids=ids),
                         {0: [], 1: [], 2: [], 3: []})
        self.assertEqual(gpu._metax_compute_pids(HOLDERS, discovered_ids=ids),
                         {0: [73380], 1: [73380], 2: [73380], 3: [73380]})

    def test_discovery_rejects_unknown_or_duplicate_identity(self):
        for output in (
            "", DISCOVERY.splitlines()[0], DISCOVERY.replace("Available", "Unavailable", 1),
            DISCOVERY.replace("GPU#1", "GPU#0"),
            DISCOVERY.replace("0000:0e:00.0", "0000:09:00.0"),
            DISCOVERY.replace("00000000-0000-0000-0000-000000000002", "00000000-0000-0000-0000-000000000001"),
            DISCOVERY.replace("GPU#1", "GPU#-1"), DISCOVERY + "warning\n",
            DISCOVERY.replace("0000:09:00.0", "unknown"),
        ):
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                gpu._metax_device_ids(output)

    def test_process_report_rejects_missing_or_ambiguous_evidence(self):
        row = next(line for line in HOLDERS.splitlines() if "73380" in line)
        for output in (
            "", '{"attached_gpus":4}', EMPTY.replace("End of Log", ""),
            EMPTY.replace("no process found", "permission denied"),
            EMPTY.replace("| Process:", "| Compute:"),
            EMPTY.replace(" : 4", " : 3"),
            EMPTY.replace("|  no process found" + " " * 63 + "|\n", ""),
            EMPTY + "unexpected trailing output\n",
            HOLDERS.replace("|  0 ", "|  9 ", 1),
            HOLDERS.replace("73380", "00000", 1),
            HOLDERS.replace("73380", "-3380", 1),
            HOLDERS.replace("python3", "python|", 1),
            HOLDERS.replace("python3", "python\n3", 1),
            HOLDERS.replace("python3", "python3 extra", 1),
            HOLDERS.replace(row, row + "\n" + row, 1),
            HOLDERS.replace(row, "|  no process found" + " " * 63 + "|", 1),
            HOLDERS.replace("0              |", "-1             |", 1),
        ):
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                gpu._metax_compute_pids(output, discovered_ids=[0, 1, 2, 3])

    def test_pid_columns_do_not_consume_digits_in_process_name(self):
        output = HOLDERS.replace("python3                      ", "worker 3  17                 ")
        self.assertEqual(gpu._metax_compute_pids(output, discovered_ids=[0, 1, 2, 3]),
                         {0: [73380], 1: [73380], 2: [73380], 3: [73380]})

    def test_inventory_boundary_rejects_incomplete_and_invalid_maps(self):
        for occupancy in (None, {}, {0: []}, {0: [], 2: [0]}, {0: [], 2: [-1]},
                          {0: [], 2: [True]}, {0: [], 2: ""}, {0: [], 2: None},
                          {False: [], 2: []}, {0: [], 2: [], -1: []}):
            with self.subTest(occupancy=occupancy), self.assertRaises(RuntimeError):
                _validate_occupancy(occupancy, required_ids=[0, 2])
        _validate_occupancy({0: [], 1: [73380], 2: []}, required_ids=[2])

    def test_explicit_backend_selection_and_default(self):
        parser = _parser()
        self.assertEqual(parser.parse_args(["serve"]).backend, "nvidia")
        self.assertEqual(parser.parse_args(["serve", "--backend", "metax"]).backend, "metax")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["serve", "--backend", "other"])
        self.assertIsInstance(gpu.inventory_for(gpu.BackendId.NVIDIA), gpu.NvidiaSmiInventory)
        self.assertIsInstance(gpu.inventory_for(gpu.BackendId.METAX), gpu.MetaXSmiInventory)
        with self.assertRaises(ValueError):
            gpu.inventory_for("other")

    def test_backend_is_read_only_and_environment_is_copied(self):
        inventory = gpu.MetaXSmiInventory()
        with self.assertRaises(AttributeError):
            inventory.backend = gpu.BackendId.NVIDIA
        base = {"CUDA_VISIBLE_DEVICES": "3", "MACA_VISIBLE_DEVICES": "1", "KEPT": "yes"}
        self.assertEqual(inventory.child_environment(base, physical_ids=(2, 0)),
                         {"CUDA_VISIBLE_DEVICES": "2,0", "MACA_VISIBLE_DEVICES": "2,0", "KEPT": "yes"})
        self.assertEqual(base["MACA_VISIBLE_DEVICES"], "1")


class MetaXInventoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_commands_discover_once_and_check_full_coverage(self):
        inventory = gpu.MetaXSmiInventory()
        with patch.object(gpu, "_smi", AsyncMock(side_effect=[DISCOVERY, EMPTY, HOLDERS, EMPTY.replace(" : 4", " : 3")])) as command:
            self.assertEqual(await inventory.gpu_ids(), [0, 1, 2, 3])
            ids = await inventory.gpu_ids()
            ids.clear()
            self.assertEqual(await inventory.compute_pids(), {0: [], 1: [], 2: [], 3: []})
            self.assertEqual((await inventory.compute_pids())[2], [73380])
            with self.assertRaises(RuntimeError):
                await inventory.compute_pids()
        self.assertEqual([call.args for call in command.await_args_list],
                         [("mx-smi", "--list")] + [("mx-smi", "--show-all-process")] * 3)

    async def test_missing_configured_device_refuses_start(self):
        with tempfile.TemporaryDirectory() as root, patch.object(gpu, "_smi", AsyncMock(return_value=DISCOVERY)):
            broker = GpuBroker(state_dir=Path(root) / "state", lock_dir=Path(root) / "locks",
                               inventory=gpu.MetaXSmiInventory(), gpu_ids=[4])
            with self.assertRaisesRegex(RuntimeError, "configured GPUs not discovered"):
                await broker.start()
            await broker.close()


class FakeMetaXInventory(gpu.MetaXSmiInventory):
    def __init__(self):
        super().__init__()
        self.occupancy = {0: [], 1: [], 2: [], 3: []}

    async def gpu_ids(self):
        return [0, 1, 2, 3]

    async def compute_pids(self):
        return self.occupancy


class MetaXBrokerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inventory = FakeMetaXInventory()
        self.broker = GpuBroker(state_dir=self.root / "state", lock_dir=self.root / "locks",
                                inventory=self.inventory, gpu_ids=[2, 0],
                                poll_interval_s=0.01, terminate_grace_s=0.05)
        await self.broker.start()
        self.addAsyncCleanup(self.broker.close)
        await until(lambda: self.broker.snapshot()["gpu_observed_at"] is not None)

    async def test_child_identity_selectors_and_cache_follow_actual_allocation(self):
        keys = ("MACA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "GPUQ_BACKEND",
                "GPUQ_DEVICE_IDS", "GPUQ_JOB_ID", "GPUQ_MODE", "GPUQ_OCCUPANCY_SCOPE", "CUDA_CACHE_PATH")
        request = replace(spec(f"import os,json; print(json.dumps({{k:os.environ[k] for k in {keys!r}}}))",
                               label="visibility", gpu_count=2), env={key: "caller" for key in keys})
        with patch.dict(os.environ, {key: "inherited" for key in keys}):
            job = self.broker.submit(request)
            events = await terminal_events(job)
        output = "".join(event["data"] for event in events
                         if event["type"] == "output" and event["stream"] == "stdout")
        self.assertEqual(json.loads(output), {
            "MACA_VISIBLE_DEVICES": "2,0", "CUDA_VISIBLE_DEVICES": "2,0", "GPUQ_BACKEND": "metax",
            "GPUQ_DEVICE_IDS": "2,0", "GPUQ_JOB_ID": job.job_id, "GPUQ_MODE": "exclusive",
            "GPUQ_OCCUPANCY_SCOPE": "system", "CUDA_CACHE_PATH": "caller",
        })
        self.assertEqual(events[-1]["state"], "completed")
        self.assertNotIn("cuda_cache_cleanup", events[-1])
        self.assertEqual(self.broker.snapshot()["backend"], "metax")
        self.assertFalse(self.broker._gpu_locks)

    async def test_zero_memory_holder_blocks_exclusive_allocation(self):
        self.inventory.occupancy = gpu._metax_compute_pids(HOLDERS, discovered_ids=[0, 1, 2, 3])
        self.broker._wakeup.set()
        await until(lambda: all(row["state"] == "foreign" for row in self.broker.snapshot()["gpus"]))
        job = self.broker.submit(spec("print('must not run')", label="held", queue_timeout_s=0.05))
        events = await terminal_events(job)
        self.assertEqual(events[-1]["state"], "queue_timeout")
        self.assertFalse(any(event["type"] == "started" for event in events))

    async def test_missing_occupancy_preserves_running_job_and_recovers(self):
        holder = self.broker.submit(spec("import time; time.sleep(30)", label="holder", run_timeout_s=30))
        await await_started(holder)
        self.inventory.occupancy = {2: []}
        self.broker._wakeup.set()
        await until(lambda: self.broker.snapshot()["probe_error"] is not None)
        snapshot = self.broker.snapshot()
        self.assertEqual(snapshot["external_occupancy"], "unknown")
        self.assertEqual([job["job_id"] for job in snapshot["running"]], [holder.job_id])
        self.assertTrue(self.broker._gpu_locks)
        queued = self.broker.submit(spec("print('must not run')", label="missing", queue_timeout_s=0.05))
        events = await terminal_events(queued)
        self.assertEqual(events[-1]["state"], "queue_timeout")
        self.assertFalse(any(event["type"] == "started" for event in events))
        await self.broker.cancel(holder.job_id)
        self.inventory.occupancy = {0: [], 1: [], 2: [], 3: []}
        self.broker._wakeup.set()
        recovered = self.broker.submit(spec("print('ok')", label="recovered"))
        self.assertEqual((await terminal_events(recovered))[-1]["state"], "completed")

    async def test_environment_failure_releases_allocation_without_spawn(self):
        with patch.object(self.inventory, "child_environment", side_effect=RuntimeError("invalid mapping")):
            job = self.broker.submit(spec("print('must not run')", label="environment"))
            events = await terminal_events(job)
        self.assertEqual(events[-1]["state"], "failed")
        self.assertFalse(any(event["type"] == "started" for event in events))
        self.assertFalse(self.broker._gpu_locks)
        recovered = self.broker.submit(spec("print('ok')", label="after-environment-error"))
        self.assertEqual((await terminal_events(recovered))[-1]["state"], "completed")
