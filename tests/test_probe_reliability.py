"""Fault injection with CPU subprocesses and fake GPU inventory."""

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent_gpu_broker import gpu
from agent_gpu_broker.broker import GpuBroker
from test_broker import FakeInventory, spec, terminal_events


async def until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.005)


class NvidiaProbeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.pidfile = self.root / "pid"
        shim = self.root / "nvidia-smi"
        shim.write_text(
            f"#!{sys.executable}\nimport os,time\nfrom pathlib import Path\n"
            f"Path({str(self.pidfile)!r}).write_text(str(os.getpid()))\n"
            "time.sleep(30)\n"
        )
        shim.chmod(0o755)
        self.path = mock.patch.dict(os.environ, {"PATH": str(self.root)})
        self.path.start()
        self.addCleanup(self.path.stop)

    def assert_reaped(self):
        pid = int(self.pidfile.read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    async def test_timeout_kills_and_reaps_probe(self):
        processes = []
        create = asyncio.create_subprocess_exec
        async def spawn(*args, **kwargs):
            process = await create(*args, **kwargs)
            processes.append(process)
            return process
        with mock.patch.object(gpu, "GPU_PROBE_TIMEOUT_S", 0.2), mock.patch.object(
            asyncio, "create_subprocess_exec", side_effect=spawn
        ):
            with self.assertRaisesRegex(RuntimeError, "nvidia-smi timed out"):
                await asyncio.wait_for(gpu._nvidia_smi("--query-gpu=index,uuid"), 3)
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].returncode)
        with self.assertRaises(ProcessLookupError):
            os.kill(processes[0].pid, 0)

    async def test_cancel_kills_and_reaps_probe(self):
        task = asyncio.create_task(gpu._nvidia_smi())
        await until(self.pidfile.exists)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        self.assert_reaped()


class FailingInventory(FakeInventory):
    fail = False

    async def compute_pids(self):
        if self.fail:
            raise RuntimeError("injected GPU probe timeout")
        return await super().compute_pids()


class SchedulerReliabilityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        self.inventory = FailingInventory()
        self.broker = GpuBroker(
            state_dir=self.root / "state", lock_dir=self.root / "locks",
            inventory=self.inventory, poll_interval_s=0.01, terminate_grace_s=0.05,
        )
        await self.broker.start()
        self.addAsyncCleanup(self.broker.close)
        await until(lambda: self.broker.snapshot()["gpu_observed_at"] is not None)

    async def test_probe_failure_blocks_admission_expires_queue_then_recovers(self):
        holder = self.broker.submit(spec("import time; time.sleep(30)", label="holder"))
        while (await asyncio.wait_for(holder.events.get(), 3))["type"] != "started":
            pass
        observed_at = self.broker.snapshot()["gpu_observed_at"]
        self.inventory.fail = True
        self.broker._wakeup.set()
        await until(lambda: self.broker.snapshot()["probe_error"] is not None)
        job = self.broker.submit(spec("print('must not run')", label="blocked", queue_timeout_s=0.05))
        before = self.broker.snapshot()
        self.assertEqual(before["gpu_observed_at"], observed_at)
        self.assertTrue(all(g["state"] == "unavailable" for g in before["gpus"]))
        self.assertEqual([row["job_id"] for row in before["running"]], [holder.job_id])
        self.assertTrue(self.broker._gpu_locks)
        self.assertIsNone(before["queue"][0]["eta_seconds"])
        events = await terminal_events(job)
        self.assertEqual(events[-1]["state"], "queue_timeout")
        self.assertFalse(any(e["type"] == "started" for e in events))
        self.inventory.fail = False
        await self.broker.cancel(holder.job_id, reason="test cleanup")
        self.broker._wakeup.set()
        after = self.broker.submit(spec("print('recovered')", label="after-probe-error"))
        self.assertEqual((await terminal_events(after))[-1]["state"], "completed")
        self.assertIsNone(self.broker.snapshot()["probe_error"])

    async def test_one_scheduler_exception_does_not_kill_loop(self):
        original = self.broker._observe_gpus
        calls = 0
        def observe(occupancy):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("injected observation failure")
            return original(occupancy)
        with mock.patch.object(self.broker, "_observe_gpus", side_effect=observe):
            self.broker._wakeup.set()
            await until(lambda: calls >= 2)
            job = self.broker.submit(spec("print('ok')", label="after-loop-error"))
            self.assertEqual((await terminal_events(job))[-1]["state"], "completed")
        self.assertFalse(self.broker._scheduler.done())
        self.broker._wakeup.set()
        await asyncio.wait_for(self.broker.close(), 3)
        self.assertTrue(self.broker._scheduler.done())
        self.assertFalse(any(t.get_name().startswith("gpuq-scheduler-")
                             for t in asyncio.all_tasks() if not t.done()))

    async def test_status_does_not_refresh_gpu_observation(self):
        first = self.broker.snapshot()
        # Age only the observation, not task/queue clocks or the running loop.
        self.broker._gpu_observed_mono -= 20
        second = self.broker.snapshot()
        self.assertEqual(first["gpu_observed_at"], second["gpu_observed_at"])
        self.assertGreater(second["gpu_observation_age_seconds"], 20)
        self.assertIn("stale", second["probe_error"])


if __name__ == "__main__":
    unittest.main()
