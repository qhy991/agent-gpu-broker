"""Tests for the 2026-08 hardening: admission preflight, swallowed-failure
warning, and status identity fields. All use fake GPU inventory and ordinary
CPU subprocesses; no GPU is required."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import unittest
from pathlib import Path

from agent_gpu_broker import __version__
from agent_gpu_broker.broker import GpuBroker, JobSpec


class FakeInventory:
    def __init__(self, gpu_ids=(0,)):
        self.ids = list(gpu_ids)

    async def gpu_ids(self):
        return list(self.ids)

    async def compute_pids(self):
        return {gpu: [] for gpu in self.ids}


def spec(
    argv: tuple[str, ...],
    *,
    label: str,
    cwd: str | None = None,
    run_timeout_s: float = 5.0,
    env: dict[str, str] | None = None,
) -> JobSpec:
    return JobSpec(
        argv=argv,
        cwd=cwd or str(Path.cwd()),
        owner="test-agent",
        label=label,
        mode="exclusive",
        gpu_count=1,
        estimate_s=0.2,
        run_timeout_s=run_timeout_s,
        queue_timeout_s=None,
        env=env or {},
    )


async def terminal_event(job):
    while True:
        event = await asyncio.wait_for(job.events.get(), timeout=5)
        if event["type"] == "finished":
            return event


class HardeningTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_preflight_rejects_inaccessible_executable(self):
        missing = spec(("/nonexistent/binary", "--flag"), label="missing-exe")
        event = await terminal_event(self.broker.submit(missing))
        self.assertEqual(event["state"], "failed")
        self.assertEqual(event["exit_code"], 127)
        self.assertIn("preflight", event["reason"])
        snapshot = self.broker.snapshot()
        self.assertEqual(snapshot["running"], [])
        self.assertEqual(snapshot["queue"], [])
        self.assertNotIn("jobs", snapshot["gpus"][0])

        valid = spec(
            (sys.executable, "-c", "print('runs after rejection')"),
            label="after-rejection",
        )
        followup = await terminal_event(self.broker.submit(valid))
        self.assertEqual(followup["state"], "completed")
        self.assertGreaterEqual(followup["wait_seconds"], 0)
        self.assertEqual(followup["broker_version"], __version__)
        self.assertEqual(
            followup["broker_instance_id"], self.broker.snapshot()["instance_id"]
        )

    async def test_preflight_rejects_before_waiting_for_gpu(self):
        holder = self.broker.submit(
            spec(
                (sys.executable, "-c", "import time; time.sleep(5)"),
                label="holder",
            )
        )
        while True:
            event = await asyncio.wait_for(holder.events.get(), timeout=1)
            if event["type"] == "started":
                break

        missing = self.broker.submit(
            spec(("/nonexistent/binary",), label="reject-with-busy-gpu")
        )
        rejected = await asyncio.wait_for(terminal_event(missing), timeout=0.2)
        self.assertEqual(rejected["exit_code"], 127)
        self.assertEqual(self.broker.snapshot()["queue"], [])
        self.assertTrue(await self.broker.cancel(holder.job_id, reason="test cleanup"))

    async def test_preflight_uses_job_path_and_cwd(self):
        work = Path(self.temp.name) / "work"
        tools = work / "tools"
        tools.mkdir(parents=True)
        executable = tools / "probe"
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)

        on_path = spec(
            ("probe",),
            label="job-path",
            cwd=str(work),
            env={"PATH": "tools"},
        )
        self.assertEqual((await terminal_event(self.broker.submit(on_path)))["state"], "completed")

        relative = spec(("./tools/probe",), label="relative-exe", cwd=str(work))
        self.assertEqual((await terminal_event(self.broker.submit(relative)))["state"], "completed")

    async def test_preflight_rejects_missing_cwd(self):
        missing_cwd = Path(self.temp.name) / "does-not-exist"
        job = spec(
            (sys.executable, "-c", "print('never runs')"),
            label="missing-cwd",
            cwd=str(missing_cwd),
        )
        event = await terminal_event(self.broker.submit(job))
        self.assertEqual(event["exit_code"], 127)
        self.assertIn("cwd", event["reason"])

    async def test_cancel_before_execute_starts_releases_allocation(self):
        job = self.broker.submit(
            spec(
                (sys.executable, "-c", "import time; time.sleep(30)"),
                label="cancel-before-coroutine-start",
            )
        )
        self.assertIs(self.broker._queue.popleft(), job)
        gpu_ids = self.broker._try_allocate(job, {0: []})
        self.assertEqual(gpu_ids, (0,))
        self.broker._start_job(job, gpu_ids)

        self.assertTrue(
            await self.broker.cancel(job.job_id, reason="cancel before start")
        )
        snapshot = self.broker.snapshot()
        self.assertEqual(snapshot["running"], [])
        self.assertEqual(snapshot["queue"], [])
        self.assertNotIn("jobs", snapshot["gpus"][0])
        self.assertEqual(snapshot["recent"][0]["reason"], "cancel before start")

    async def test_preflight_rejects_inaccessible_cwd(self):
        secret = Path(self.temp.name) / "secret"
        secret.mkdir(mode=0o700)
        try:
            if os.access(secret, os.R_OK | os.X_OK):
                self.skipTest("running as root: permission check cannot fail")
            job = spec(
                (sys.executable, "-c", "print('never runs')"),
                label="bad-cwd",
                cwd=str(secret),
            )
            event = await terminal_event(self.broker.submit(job))
            self.assertEqual(event["state"], "failed")
            self.assertEqual(event["exit_code"], 127)
            self.assertIn("preflight", event["reason"])
            self.assertIn("cwd", event["reason"])
        finally:
            secret.chmod(0o755)

    async def test_swallowed_failure_warning_on_fast_permission_error(self):
        secret = Path(self.temp.name) / "locked"
        secret.write_text("x")
        secret.chmod(0o000)
        try:
            if os.access(secret, os.R_OK):
                self.skipTest("running as root: permission check cannot fail")
            code = (
                "import sys\n"
                "try:\n"
                f"    open({str(secret)!r})\n"
                "except PermissionError as exc:\n"
                "    print(f'PermissionError: {exc}', file=sys.stderr)\n"
                "sys.exit(0)\n"
            )
            job = spec((sys.executable, "-c", code), label="swallowed")
            event = await terminal_event(self.broker.submit(job))
            self.assertEqual(event["state"], "completed")
            self.assertEqual(event["exit_code"], 0)
            self.assertIn("warning", event)
            self.assertIn("possible_swallowed_failure", event["warning"])
        finally:
            secret.chmod(0o644)

    async def test_normal_completion_has_no_warning(self):
        job = spec((sys.executable, "-c", "print('fine')"), label="clean")
        event = await terminal_event(self.broker.submit(job))
        self.assertEqual(event["state"], "completed")
        self.assertNotIn("warning", event)

    async def test_status_exposes_identity(self):
        snapshot = self.broker.snapshot()
        self.assertIn("broker_version", snapshot)
        self.assertIn("instance_id", snapshot)
        self.assertTrue(snapshot["instance_id"])
        self.assertEqual(snapshot["broker_version"], __version__)


if __name__ == "__main__":
    unittest.main()
