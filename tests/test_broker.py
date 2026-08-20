from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

from agent_gpu_broker.broker import GpuBroker, JobSpec


class FakeInventory:
    def __init__(self, gpu_ids=(0,), occupancy=None):
        self.ids = list(gpu_ids)
        self.occupancy = occupancy or {}

    async def gpu_ids(self):
        return list(self.ids)

    async def compute_pids(self):
        return {gpu: list(self.occupancy.get(gpu, [])) for gpu in self.ids}


def spec(
    code: str,
    *,
    label: str,
    mode: str = "exclusive",
    gpu_count: int = 1,
    estimate_s: float = 0.2,
    run_timeout_s: float = 2.0,
    queue_timeout_s: float | None = None,
) -> JobSpec:
    return JobSpec(
        argv=(sys.executable, "-c", code),
        cwd=str(Path.cwd()),
        owner="test-agent",
        label=label,
        mode=mode,
        gpu_count=gpu_count,
        estimate_s=estimate_s,
        run_timeout_s=run_timeout_s,
        queue_timeout_s=queue_timeout_s,
    )


async def terminal_events(job):
    events = []
    while True:
        event = await asyncio.wait_for(job.events.get(), timeout=3)
        events.append(event)
        if event["type"] == "finished":
            return events


class BrokerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.broker = GpuBroker(
            state_dir=root / "state",
            lock_dir=root / "locks",
            inventory=FakeInventory(),
            poll_interval_s=0.01,
            heartbeat_s=0.03,
            terminate_grace_s=0.05,
        )
        await self.broker.start()

    async def asyncTearDown(self):
        await self.broker.close()
        self.temp.cleanup()

    async def test_fifo_queue_streams_position_then_runs(self):
        first = self.broker.submit(
            spec("import time; time.sleep(.15); print('first')", label="first")
        )
        await asyncio.sleep(0.03)
        second = self.broker.submit(spec("print('second')", label="second"))

        first_events, second_events = await asyncio.gather(
            terminal_events(first), terminal_events(second)
        )

        self.assertEqual(first_events[-1]["state"], "completed")
        self.assertEqual(second_events[-1]["state"], "completed")
        queued = [event for event in second_events if event["type"] == "queued"]
        self.assertTrue(any(event["position"] == 1 for event in queued))
        first_started = next(
            index
            for index, event in enumerate(first_events)
            if event["type"] == "started"
        )
        first_finished = next(
            index
            for index, event in enumerate(first_events)
            if event["type"] == "finished"
        )
        self.assertLess(first_started, first_finished)
        output = "".join(
            event["data"] for event in second_events if event["type"] == "output"
        )
        self.assertIn("second", output)

    async def test_queue_timeout_does_not_launch_command(self):
        first = self.broker.submit(
            spec("import time; time.sleep(.2)", label="holder", estimate_s=0.2)
        )
        await asyncio.sleep(0.03)
        waiting = self.broker.submit(
            spec(
                "print('must-not-run')",
                label="expires",
                queue_timeout_s=0.04,
            )
        )
        waiting_events = await terminal_events(waiting)
        self.assertEqual(waiting_events[-1]["state"], "queue_timeout")
        self.assertFalse(any(e["type"] == "started" for e in waiting_events))
        await terminal_events(first)

    async def test_run_timeout_terminates_process_group(self):
        job = self.broker.submit(
            spec(
                "import time; time.sleep(5)",
                label="times-out",
                run_timeout_s=0.05,
            )
        )
        events = await terminal_events(job)
        self.assertEqual(events[-1]["state"], "run_timeout")
        self.assertEqual(events[-1]["exit_code"], 124)
        self.assertEqual(self.broker.snapshot()["running"], [])

    async def test_status_projection_contains_global_queue(self):
        first = self.broker.submit(
            spec("import time; time.sleep(.15)", label="running")
        )
        await asyncio.sleep(0.03)
        second = self.broker.submit(spec("print(1)", label="queued"))
        snapshot = self.broker.snapshot()
        self.assertEqual(snapshot["running"][0]["label"], "running")
        self.assertEqual(snapshot["running"][0]["mode"], "exclusive")
        self.assertEqual(snapshot["running"][0]["gpu_ids"], [0])
        self.assertEqual(snapshot["queue"][0]["label"], "queued")
        self.assertEqual(snapshot["queue"][0]["position"], 1)
        self.assertIsNotNone(snapshot["queue"][0]["eta_seconds"])
        await asyncio.gather(terminal_events(first), terminal_events(second))

    async def test_two_shared_jobs_overlap_on_one_gpu(self):
        first = self.broker.submit(
            spec(
                "import time; time.sleep(.15)",
                label="shared-a",
                mode="shared",
            )
        )
        await asyncio.sleep(0.03)
        second = self.broker.submit(
            spec(
                "import time; time.sleep(.15)",
                label="shared-b",
                mode="shared",
            )
        )
        await asyncio.sleep(0.03)

        snapshot = self.broker.snapshot()
        self.assertEqual(len(snapshot["running"]), 2)
        self.assertEqual(snapshot["gpus"][0]["state"], "shared")
        self.assertEqual(snapshot["gpus"][0]["shared_used"], 2)
        self.assertEqual(snapshot["queue"], [])
        events = await asyncio.gather(terminal_events(first), terminal_events(second))
        self.assertTrue(all(items[-1]["state"] == "completed" for items in events))

    async def test_shared_capacity_queues_third_job(self):
        jobs = [
            self.broker.submit(
                spec(
                    "import time; time.sleep(.15)",
                    label=f"shared-{index}",
                    mode="shared",
                )
            )
            for index in range(3)
        ]
        await asyncio.sleep(0.04)

        snapshot = self.broker.snapshot()
        self.assertEqual(len(snapshot["running"]), 2)
        self.assertEqual(snapshot["queue"][0]["label"], "shared-2")
        self.assertEqual(snapshot["queue"][0]["position"], 1)
        events = await asyncio.gather(*(terminal_events(job) for job in jobs))
        self.assertTrue(any(e["type"] == "queued" for e in events[2]))
        self.assertEqual(events[2][-1]["state"], "completed")

    async def test_exclusive_waits_for_shared_users(self):
        shared = self.broker.submit(
            spec(
                "import time; time.sleep(.12)",
                label="correctness",
                mode="shared",
            )
        )
        await asyncio.sleep(0.03)
        exclusive = self.broker.submit(
            spec("print('benchmark')", label="benchmark", mode="exclusive")
        )
        await asyncio.sleep(0.03)

        snapshot = self.broker.snapshot()
        self.assertEqual(snapshot["running"][0]["label"], "correctness")
        self.assertEqual(snapshot["queue"][0]["label"], "benchmark")
        shared_events, exclusive_events = await asyncio.gather(
            terminal_events(shared), terminal_events(exclusive)
        )
        self.assertEqual(shared_events[-1]["state"], "completed")
        self.assertEqual(exclusive_events[-1]["state"], "completed")

    async def test_rejects_request_larger_than_managed_inventory(self):
        with self.assertRaisesRegex(ValueError, "exceeds managed GPUs"):
            self.broker.submit(spec("print(1)", label="too-large", gpu_count=2))


class MultiGpuBrokerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.broker = GpuBroker(
            state_dir=root / "state",
            lock_dir=root / "locks",
            inventory=FakeInventory(gpu_ids=(0, 1)),
            poll_interval_s=0.01,
            heartbeat_s=0.03,
            terminate_grace_s=0.05,
        )
        await self.broker.start()

    async def asyncTearDown(self):
        await self.broker.close()
        self.temp.cleanup()

    async def test_multi_gpu_allocation_is_atomic_and_visible(self):
        job = self.broker.submit(
            spec(
                "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])",
                label="two-card",
                gpu_count=2,
            )
        )
        events = await terminal_events(job)

        started = next(event for event in events if event["type"] == "started")
        self.assertEqual(started["gpu_ids"], [0, 1])
        self.assertEqual(started["gpu_count"], 2)
        output = "".join(event["data"] for event in events if event["type"] == "output")
        self.assertEqual(output.strip(), "0,1")

    async def test_fifo_head_blocks_smaller_job(self):
        holder = self.broker.submit(
            spec("import time; time.sleep(.15)", label="holder")
        )
        await asyncio.sleep(0.03)
        two_card = self.broker.submit(
            spec("print('two')", label="two-card", gpu_count=2)
        )
        one_card = self.broker.submit(spec("print('one')", label="one-card"))
        await asyncio.sleep(0.03)

        snapshot = self.broker.snapshot()
        self.assertEqual(
            [item["label"] for item in snapshot["queue"]],
            ["two-card", "one-card"],
        )
        events = await asyncio.gather(
            terminal_events(holder),
            terminal_events(two_card),
            terminal_events(one_card),
        )
        self.assertTrue(all(items[-1]["state"] == "completed" for items in events))


if __name__ == "__main__":
    unittest.main()
