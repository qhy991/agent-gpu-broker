from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

from agent_gpu_broker.broker import GpuBroker
from agent_gpu_broker.server import BrokerServer


class FakeInventory:
    async def gpu_ids(self):
        return [0]

    async def compute_pids(self):
        return {0: []}


async def send(writer: asyncio.StreamWriter, value: dict) -> None:
    writer.write((json.dumps(value) + "\n").encode())
    await writer.drain()


async def receive(reader: asyncio.StreamReader) -> dict:
    return json.loads(await asyncio.wait_for(reader.readline(), timeout=3))


class ServerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        broker = GpuBroker(
            state_dir=root / "state",
            lock_dir=root / "locks",
            inventory=FakeInventory(),
            poll_interval_s=0.01,
            heartbeat_s=0.03,
        )
        self.broker = broker
        self.socket_path = root / "gpuq.sock"
        self.server = BrokerServer(broker, self.socket_path)
        await self.server.start()

    async def asyncTearDown(self):
        await self.server.close()
        self.temp.cleanup()

    async def test_status_observes_job_submitted_by_another_connection(self):
        first_reader, first_writer = await asyncio.open_unix_connection(
            self.socket_path
        )
        await send(
            first_writer,
            {
                "op": "run",
                "argv": [sys.executable, "-c", "import time; time.sleep(.15)"],
                "cwd": str(Path.cwd()),
                "owner": "agent-a",
                "label": "holder",
                "mode": "exclusive",
                "gpu_count": 1,
                "estimate_s": 0.15,
                "run_timeout_s": 2,
            },
        )
        while (await receive(first_reader))["type"] != "started":
            pass

        second_reader, second_writer = await asyncio.open_unix_connection(
            self.socket_path
        )
        await send(
            second_writer,
            {
                "op": "run",
                "argv": [sys.executable, "-c", "print('queued job ran')"],
                "cwd": str(Path.cwd()),
                "owner": "agent-b",
                "label": "waiting",
                "mode": "exclusive",
                "gpu_count": 1,
                "estimate_s": 0.1,
                "run_timeout_s": 2,
            },
        )
        while True:
            event = await receive(second_reader)
            if event["type"] == "queued":
                self.assertEqual(event["position"], 1)
                break

        status_reader, status_writer = await asyncio.open_unix_connection(
            self.socket_path
        )
        await send(status_writer, {"op": "status"})
        snapshot = (await receive(status_reader))["snapshot"]
        self.assertEqual(snapshot["running"][0]["owner"], "agent-a")
        self.assertEqual(snapshot["queue"][0]["owner"], "agent-b")
        self.assertEqual(snapshot["queue"][0]["position"], 1)
        status_writer.close()
        await status_writer.wait_closed()

        for reader in (first_reader, second_reader):
            while (await receive(reader))["type"] != "finished":
                pass
        first_writer.close()
        second_writer.close()
        await first_writer.wait_closed()
        await second_writer.wait_closed()

    async def test_resource_fields_are_required(self):
        reader, writer = await asyncio.open_unix_connection(self.socket_path)
        await send(
            writer,
            {
                "op": "run",
                "argv": [sys.executable, "-c", "print(1)"],
                "cwd": str(Path.cwd()),
                "owner": "agent-a",
                "label": "missing-resource-shape",
            },
        )
        event = await receive(reader)
        self.assertEqual(event["type"], "error")
        writer.close()
        await writer.wait_closed()

    async def test_running_disconnect_preserves_cancel_reason_and_releases_gpu(self):
        reader, writer = await asyncio.open_unix_connection(self.socket_path)
        await send(
            writer,
            {
                "op": "run",
                "argv": [sys.executable, "-c", "import time; time.sleep(30)"],
                "cwd": str(Path.cwd()),
                "owner": "agent-a",
                "label": "disconnect-me",
                "mode": "exclusive",
                "gpu_count": 1,
                "estimate_s": 30,
                "run_timeout_s": 60,
            },
        )
        while (await receive(reader))["type"] != "started":
            pass
        writer.close()
        await writer.wait_closed()

        for _ in range(100):
            snapshot = self.broker.snapshot()
            if snapshot["recent"]:
                break
            await asyncio.sleep(0.01)
        else:
            self.fail("disconnected job did not reach a terminal state")

        result = snapshot["recent"][0]
        self.assertEqual(result["state"], "cancelled")
        self.assertEqual(result["reason"], "client disconnected")
        self.assertEqual(snapshot["running"], [])
        self.assertNotIn("jobs", snapshot["gpus"][0])

    async def test_broken_event_stream_cancels_running_job(self):
        reader, writer = await asyncio.open_unix_connection(self.socket_path)
        await send(
            writer,
            {
                "op": "run",
                "argv": [
                    sys.executable,
                    "-c",
                    "import time; time.sleep(.05); print('late', flush=True); time.sleep(30)",
                ],
                "cwd": str(Path.cwd()),
                "owner": "agent-a",
                "label": "broken-stream",
                "mode": "exclusive",
                "gpu_count": 1,
                "estimate_s": 30,
                "run_timeout_s": 60,
            },
        )
        while (await receive(reader))["type"] != "started":
            pass

        async def broken_send(_writer, _value):
            raise BrokenPipeError("simulated broken event stream")

        self.server._send = broken_send
        for _ in range(200):
            snapshot = self.broker.snapshot()
            if snapshot["recent"]:
                break
            await asyncio.sleep(0.01)
        else:
            self.fail("broken event stream did not cancel the running job")

        result = snapshot["recent"][0]
        self.assertEqual(result["state"], "cancelled")
        self.assertEqual(result["reason"], "client disconnected")
        self.assertEqual(snapshot["running"], [])
        writer.close()
        await writer.wait_closed()

    async def test_explicit_cancel_preserves_reason(self):
        reader, writer = await asyncio.open_unix_connection(self.socket_path)
        await send(
            writer,
            {
                "op": "run",
                "argv": [sys.executable, "-c", "import time; time.sleep(30)"],
                "cwd": str(Path.cwd()),
                "owner": "agent-a",
                "label": "cancel-me",
                "mode": "exclusive",
                "gpu_count": 1,
                "estimate_s": 30,
                "run_timeout_s": 60,
            },
        )
        job_id = None
        while True:
            event = await receive(reader)
            if event["type"] == "accepted":
                job_id = event["job_id"]
            if event["type"] == "started":
                break
        self.assertIsNotNone(job_id)

        cancel_reader, cancel_writer = await asyncio.open_unix_connection(
            self.socket_path
        )
        await send(cancel_writer, {"op": "cancel", "job_id": job_id})
        self.assertTrue((await receive(cancel_reader))["ok"])
        cancel_writer.close()
        await cancel_writer.wait_closed()

        while True:
            terminal = await receive(reader)
            if terminal["type"] == "finished":
                break
        self.assertEqual(terminal["state"], "cancelled")
        self.assertEqual(terminal["reason"], "cancel requested")
        writer.close()
        await writer.wait_closed()


if __name__ == "__main__":
    unittest.main()
