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


if __name__ == "__main__":
    unittest.main()
