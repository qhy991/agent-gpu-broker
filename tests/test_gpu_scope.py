from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path

from agent_gpu_broker.broker import GpuBroker, admission_receipt_value, launch_spec_value
from agent_gpu_broker.cli import parse_gpu_ids
from agent_gpu_broker.server import BrokerServer
from test_broker import FakeInventory, spec, terminal_events


class GpuScopeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inventory = FakeInventory(gpu_ids=(0, 1, 7))
        self.broker = GpuBroker(
            state_dir=self.root / "state", lock_dir=self.root / "locks",
            inventory=self.inventory, poll_interval_s=0.01,
            heartbeat_s=0.03, terminate_grace_s=0.05,
        )
        self.socket = self.root / "broker.sock"
        self.server = BrokerServer(self.broker, self.socket)
        await self.server.start()

    async def asyncTearDown(self):
        await self.server.close()
        self.temp.cleanup()

    def request(self, allowed, **kwargs):
        return dataclasses.replace(
            spec("import os; print(os.environ['CUDA_VISIBLE_DEVICES'])", label="scope", **kwargs),
            allowed_gpu_ids=allowed,
        )

    async def test_exclusive_worker_and_receipt_use_only_allowed_physical_gpu(self):
        job = self.broker.submit(self.request((7, 1)))
        events = await terminal_events(job)
        self.assertEqual(events[-1]["state"], "completed")
        self.assertEqual(job.gpu_ids, (1,))
        receipt = admission_receipt_value(job)
        self.assertEqual(receipt["allowed_gpu_ids"], [7, 1])
        self.assertEqual(receipt["broker_version"], "0.7.0")
        self.assertIn("1", "".join(e.get("data", "") for e in events))
        self.assertNotEqual(launch_spec_value(self.request((1,))), launch_spec_value(self.request((7,))))

    async def test_busy_allowed_gpu_does_not_spill_to_idle_disallowed_gpu(self):
        self.inventory.occupancy = {1: [999999]}
        await asyncio.sleep(0.03)
        job = self.broker.submit(self.request((1,), queue_timeout_s=0.05))
        events = await terminal_events(job)
        self.assertEqual(events[-1]["state"], "queue_timeout")
        self.assertFalse(any(e["type"] == "started" for e in events))
        self.assertEqual(job.gpu_ids, ())

    async def test_shared_packing_respects_scope(self):
        first = self.broker.submit(dataclasses.replace(
            spec("import time; time.sleep(.2)", label="holder", mode="shared"),
            allowed_gpu_ids=(0,),
        ))
        while (await asyncio.wait_for(first.events.get(), timeout=3))["type"] != "started":
            pass
        second = self.broker.submit(self.request((1,), mode="shared"))
        await terminal_events(second)
        self.assertEqual(first.gpu_ids, (0,))
        self.assertEqual(second.gpu_ids, (1,))
        await terminal_events(first)

    async def test_queue_eta_uses_only_the_allowed_gpu(self):
        first = self.broker.submit(dataclasses.replace(
            spec("import time; time.sleep(.3)", label="eta-holder", estimate_s=0.3),
            allowed_gpu_ids=(1,),
        ))
        while (await asyncio.wait_for(first.events.get(), timeout=3))["type"] != "started":
            pass
        second = self.broker.submit(self.request((1,)))
        queued = next(row for row in self.broker.snapshot()["queue"] if row["job_id"] == second.job_id)
        self.assertGreater(queued["eta_seconds"], 0)
        await asyncio.gather(terminal_events(first), terminal_events(second))

    async def test_invalid_or_impossible_scopes_reject_before_admission(self):
        for allowed in ((), (2,), (1, 1), (True,), (-1,), ("1",)):
            with self.subTest(allowed=allowed), self.assertRaises(ValueError):
                self.broker.submit(self.request(allowed))
        with self.assertRaisesRegex(ValueError, "gpu_count exceeds"):
            self.broker.submit(self.request((1,), gpu_count=2))

    async def test_real_cli_rpc_preserves_scope_in_broker_receipt(self):
        receipt_path = self.root / "receipt.json"
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "agent_gpu_broker.cli", "run",
            "--socket", str(self.socket), "--label", "scope-cli",
            "--mode", "exclusive", "--gpu-count", "1", "--allowed-gpus", "7",
            "--receipt-out", str(receipt_path), "--", sys.executable, "-c",
            "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=5)
        self.assertEqual(proc.returncode, 0, err.decode())
        self.assertEqual(out.strip(), b"7")
        receipt = json.loads(receipt_path.read_text())
        self.assertEqual(receipt["allowed_gpu_ids"], [7])
        self.assertEqual(receipt["gpu_ids"], [7])

    async def test_legacy_rpc_without_scope_still_runs(self):
        value = launch_spec_value(self.request(None))
        value.pop("allowed_gpu_ids")
        value["op"] = "run"
        reader, writer = await asyncio.open_unix_connection(self.socket)
        writer.write((json.dumps(value) + "\n").encode())
        await writer.drain()
        while True:
            event = json.loads(await asyncio.wait_for(reader.readline(), timeout=3))
            self.assertNotEqual(event["type"], "error", event)
            if event["type"] == "finished":
                self.assertEqual(event["state"], "completed")
                break
        writer.close()
        await writer.wait_closed()


class GpuScopeParsingTests(unittest.TestCase):
    def test_cli_rejects_empty_duplicate_or_negative_indices(self):
        self.assertEqual(parse_gpu_ids("0,1,7"), [0, 1, 7])
        for value in ("", "1,", "-1", "1,1", "1.0", "GPU-uuid"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_gpu_ids(value)

    def test_rpc_requires_an_array(self):
        value = launch_spec_value(dataclasses.replace(spec("pass", label="parse"), allowed_gpu_ids=(1,)))
        self.assertEqual(BrokerServer._parse_spec(value).allowed_gpu_ids, (1,))
        value["allowed_gpu_ids"] = "1"
        with self.assertRaisesRegex(ValueError, "must be a list"):
            BrokerServer._parse_spec(value)
