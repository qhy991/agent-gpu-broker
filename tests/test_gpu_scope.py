from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import agent_gpu_broker.broker as broker_module

from agent_gpu_broker import __version__
from agent_gpu_broker.broker import GpuBroker, admission_receipt_value, launch_spec_value
from agent_gpu_broker.cli import parse_gpu_ids
from agent_gpu_broker.server import BrokerServer
from test_broker import FakeInventory, spec, terminal_events


class GpuScopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_disjoint_fifo_eta_is_independent_of_scoped_head(self):
        first = self.broker.submit(dataclasses.replace(
            spec("import time; time.sleep(.3)", label="holder", estimate_s=0.3),
            allowed_gpu_ids=(1,),
        ))
        while (await asyncio.wait_for(first.events.get(), timeout=3))["type"] != "started":
            pass
        head = self.broker.submit(self.request((1,)))
        later = self.broker.submit(self.request((7,)))
        rows = {row["job_id"]: row for row in self.broker.snapshot()["queue"]}
        self.assertEqual(rows[head.job_id]["allowed_gpu_ids"], [1])
        self.assertGreater(rows[head.job_id]["eta_seconds"], 0)
        self.assertEqual(rows[later.job_id]["eta_seconds"], 0.0)
        await asyncio.gather(*(terminal_events(job) for job in (first, head, later)))

    async def test_foreign_scoped_head_leaves_disjoint_eta_known(self):
        self.inventory.occupancy = {1: [999999]}
        await asyncio.sleep(0.03)
        head = self.broker.submit(self.request((1,), queue_timeout_s=0.05))
        later = self.broker.submit(self.request((7,)))
        rows = self.broker.snapshot()["queue"]
        self.assertEqual(len(rows), 2)
        self.assertIsNone(rows[0]["eta_seconds"])
        self.assertEqual(rows[1]["eta_seconds"], 0.0)
        self.assertEqual((await terminal_events(head))[-1]["state"], "queue_timeout")
        self.assertEqual((await terminal_events(later))[-1]["state"], "completed")

    async def test_foreign_head_does_not_hold_a_disjoint_worker(self):
        self.inventory.occupancy = {0: [999999]}
        await asyncio.sleep(0.03)
        first = self.broker.submit(self.request((0,), queue_timeout_s=0.3))
        second = self.broker.submit(self.request((1,)))
        second_events = await terminal_events(second)
        self.assertEqual(second_events[-1]["state"], "completed")
        self.assertEqual(second.gpu_ids, (1,))
        self.assertEqual(first.state, "queued")
        self.assertEqual((await terminal_events(first))[-1]["state"], "queue_timeout")

    async def test_overlap_propagates_through_an_older_broad_waiter(self):
        self.inventory.occupancy = {0: [999999]}
        await asyncio.sleep(0.03)
        head = self.broker.submit(self.request((0,), queue_timeout_s=0.3))
        broad = self.broker.submit(self.request((0, 1), queue_timeout_s=0.3))
        later = self.broker.submit(self.request((1,), queue_timeout_s=0.3))
        await asyncio.sleep(0.05)
        self.assertEqual([j.state for j in (head, broad, later)], ["queued"] * 3)
        self.assertEqual(self.broker.snapshot()["running"], [])
        self.assertTrue(all(row["eta_seconds"] is None for row in self.broker.snapshot()["queue"]))
        for job in (head, broad, later):
            self.assertEqual((await terminal_events(job))[-1]["state"], "queue_timeout")

    async def test_unscoped_multi_gpu_request_retains_priority(self):
        self.inventory.occupancy = {0: [999999]}
        await asyncio.sleep(0.03)
        head = self.broker.submit(self.request(None, gpu_count=3, queue_timeout_s=0.2))
        later = self.broker.submit(self.request((1,), queue_timeout_s=0.2))
        await asyncio.sleep(0.05)
        self.assertEqual(head.state, "queued")
        self.assertEqual(later.state, "queued")
        self.assertTrue(all(row["eta_seconds"] is None for row in self.broker.snapshot()["queue"]))
        for job in (head, later):
            self.assertEqual((await terminal_events(job))[-1]["state"], "queue_timeout")

    async def test_cancelled_head_unblocks_overlapping_request(self):
        self.inventory.occupancy = {0: [999999]}
        await asyncio.sleep(0.03)
        head = self.broker.submit(self.request((0, 1), gpu_count=2))
        later = self.broker.submit(self.request((1,)))
        await asyncio.sleep(0.03)
        self.assertEqual(later.state, "queued")
        self.assertTrue(await self.broker.cancel(head.job_id, reason="owned test cancellation"))
        self.assertEqual((await terminal_events(head))[-1]["state"], "cancelled")
        self.assertEqual((await terminal_events(later))[-1]["state"], "completed")
        self.assertEqual(later.gpu_ids, (1,))

    async def test_older_exclusive_waiter_blocks_new_shared_packing_on_its_scope(self):
        holder = self.broker.submit(dataclasses.replace(
            spec("import time; time.sleep(.3)", label="holder", mode="shared"),
            allowed_gpu_ids=(1,)))
        while (await asyncio.wait_for(holder.events.get(), timeout=3))["type"] != "started":
            pass
        exclusive = self.broker.submit(self.request((1,), queue_timeout_s=0.05))
        shared = self.broker.submit(self.request((1,), mode="shared"))
        await asyncio.sleep(0.02)
        self.assertEqual(exclusive.state, "queued")
        self.assertEqual(shared.state, "queued")
        self.assertEqual((await terminal_events(exclusive))[-1]["state"], "queue_timeout")
        self.assertEqual((await terminal_events(shared))[-1]["state"], "completed")
        self.assertEqual((await terminal_events(holder))[-1]["state"], "completed")

    async def test_later_allocation_error_does_not_replay_an_already_started_worker(self):
        marker = self.root / "worker-executions"
        code = "from pathlib import Path; import time; p=Path(%r); f=p.open('a'); f.write('once\\n'); f.close(); time.sleep(.12)" % str(marker)
        original = broker_module.try_card_lock
        failed = False
        def transient(gpu_id, lock_dir):
            nonlocal failed
            if gpu_id == 1 and not failed:
                failed = True
                raise OSError("transient lock-file error")
            return original(gpu_id, lock_dir)
        with patch.object(broker_module, "try_card_lock", transient):
            first = self.broker.submit(dataclasses.replace(spec(code, label="once"), allowed_gpu_ids=(0,)))
            second = self.broker.submit(self.request((1,)))
            for job in (first, second):
                self.assertEqual((await terminal_events(job))[-1]["state"], "completed")
            await asyncio.sleep(0.15)
        self.assertTrue(failed)
        self.assertEqual(marker.read_text().splitlines(), ["once"])
        self.assertEqual(self.broker.snapshot()["queue"], [])
        self.assertEqual(self.broker.snapshot()["running"], [])

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
        self.assertEqual(receipt["broker_version"], __version__)
        # Frozen B300-M3 NCU Hosts enforce this receipt-version predicate.
        # Change the line only with an explicit consumer migration.
        self.assertTrue(receipt["broker_version"].startswith("0.7."))
        self.assertIn("1", "".join(e.get("data", "") for e in events))
        self.assertNotEqual(launch_spec_value(self.request((1,))), launch_spec_value(self.request((7,))))

    async def test_scoped_worker_retains_broker_owned_cake_allocation_identity(self):
        keys = ("GPUQ_JOB_ID", "GPUQ_MODE", "GPUQ_BACKEND", "GPUQ_DEVICE_IDS",
                "GPUQ_OCCUPANCY_SCOPE", "CUDA_VISIBLE_DEVICES")
        request = dataclasses.replace(
            self.request((7,)),
            argv=(sys.executable, "-c",
                  "import json, os; print(json.dumps({key: os.environ[key] for key in "
                  + repr(keys) + "}))"),
            env={key: "forged" for key in keys},
        )
        job = self.broker.submit(request)
        events = await terminal_events(job)
        self.assertEqual(events[-1]["state"], "completed")
        output = "".join(event["data"] for event in events if event["type"] == "output")
        self.assertEqual(json.loads(output), {
            "GPUQ_JOB_ID": job.job_id, "GPUQ_MODE": "exclusive",
            "GPUQ_BACKEND": "nvidia", "GPUQ_DEVICE_IDS": "7",
            "GPUQ_OCCUPANCY_SCOPE": "system", "CUDA_VISIBLE_DEVICES": "7",
        })
        receipt = admission_receipt_value(job)
        self.assertEqual(receipt["allowed_gpu_ids"], [7])
        self.assertEqual(receipt["gpu_ids"], [7])

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

    async def test_legacy_operation_cannot_silently_drop_a_scope(self):
        value = launch_spec_value(self.request((7,)))
        value["op"] = "run"
        reader, writer = await asyncio.open_unix_connection(self.socket)
        writer.write((json.dumps(value) + "\n").encode())
        await writer.drain()
        response = json.loads(await asyncio.wait_for(reader.readline(), timeout=3))
        self.assertEqual(response["type"], "error")
        self.assertIn("run-scoped", response["message"])
        self.assertEqual(self.broker.snapshot()["running"], [])
        writer.close()
        await writer.wait_closed()

    async def test_new_client_fails_on_legacy_server_before_submitting_a_job(self):
        requests = []

        async def legacy_handler(reader, writer):
            request = json.loads(await reader.readline())
            requests.append(request)
            # v0.6 rejects unknown operations before constructing a JobSpec;
            # its ordinary run parser would ignore an unknown scope field.
            writer.write((json.dumps({"type": "error", "message": "unknown operation"}) + "\n").encode())
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        old_socket = self.root / "old.sock"
        server = await asyncio.start_unix_server(legacy_handler, path=old_socket)
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "agent_gpu_broker.cli", "run",
                "--socket", str(old_socket), "--label", "downgrade",
                "--mode", "exclusive", "--gpu-count", "1", "--allowed-gpus", "7",
                "--", sys.executable, "-c", "print('must-not-run')",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            out, err = await asyncio.wait_for(proc.communicate(), timeout=5)
            self.assertEqual(proc.returncode, 1)
            self.assertEqual(out, b"")
            self.assertIn(b"unknown operation", err)
            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0]["op"], "run-scoped")
        finally:
            server.close()
            await server.wait_closed()


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
