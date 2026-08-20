"""FIFO scheduling core and shared read-only state projection."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import tempfile
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gpu import CardLock, GpuInventory, NvidiaSmiInventory, try_card_lock


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class JobSpec:
    argv: tuple[str, ...]
    cwd: str
    owner: str
    label: str
    estimate_s: float
    run_timeout_s: float
    queue_timeout_s: float | None
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class Job:
    job_id: str
    spec: JobSpec
    events: asyncio.Queue[dict[str, Any]]
    submitted_at: str
    submitted_mono: float
    state: str = "queued"
    gpu_id: int | None = None
    started_at: str | None = None
    started_mono: float | None = None
    process: asyncio.subprocess.Process | None = None
    task: asyncio.Task[None] | None = None
    card_lock: CardLock | None = None
    last_queue_key: tuple[Any, ...] | None = None
    last_notice_mono: float = 0.0


class StateStore:
    """Write durable artifacts; never act as a competing scheduler owner."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.jobs_dir = root / "jobs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(exist_ok=True)

    def _atomic_json(self, path: Path, value: Any, *, mode: int = 0o644) -> None:
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.chmod(temp_name, mode)
            os.replace(temp_name, path)
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

    def record_request(self, job: Job) -> None:
        directory = self.jobs_dir / job.job_id
        directory.mkdir(mode=0o700)
        self._atomic_json(
            directory / "request.json",
            {
                "job_id": job.job_id,
                "submitted_at": job.submitted_at,
                "owner": job.spec.owner,
                "label": job.spec.label,
                "cwd": job.spec.cwd,
                "argv": list(job.spec.argv),
                "env_keys": sorted(job.spec.env),
                "estimate_s": job.spec.estimate_s,
                "run_timeout_s": job.spec.run_timeout_s,
                "queue_timeout_s": job.spec.queue_timeout_s,
            },
            mode=0o600,
        )

    def append_output(self, job_id: str, stream: str, data: bytes) -> None:
        path = self.jobs_dir / job_id / f"{stream}.log"
        with path.open("ab") as handle:
            handle.write(data)

    def record_result(self, job_id: str, value: dict[str, Any]) -> None:
        self._atomic_json(self.jobs_dir / job_id / "result.json", value, mode=0o600)

    def append_lifecycle(self, value: dict[str, Any]) -> None:
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")

    def write_status(self, value: dict[str, Any]) -> None:
        self._atomic_json(self.root / "status.json", value)


class GpuBroker:
    def __init__(
        self,
        *,
        state_dir: Path,
        lock_dir: Path,
        inventory: GpuInventory | None = None,
        gpu_ids: list[int] | None = None,
        poll_interval_s: float = 2.0,
        heartbeat_s: float = 30.0,
        terminate_grace_s: float = 5.0,
    ) -> None:
        self._store = StateStore(state_dir)
        self._lock_dir = lock_dir
        self._inventory = inventory or NvidiaSmiInventory()
        self._configured_gpu_ids = list(gpu_ids) if gpu_ids is not None else None
        self._gpu_ids: list[int] = []
        self._poll_interval_s = poll_interval_s
        self._heartbeat_s = heartbeat_s
        self._terminate_grace_s = terminate_grace_s
        self._jobs: dict[str, Job] = {}
        self._queue: deque[Job] = deque()
        self._running: dict[int, Job] = {}
        self._recent: deque[dict[str, Any]] = deque(maxlen=20)
        self._gpu_observation: dict[int, dict[str, Any]] = {}
        self._probe_error: str | None = None
        self._wakeup = asyncio.Event()
        self._scheduler: asyncio.Task[None] | None = None
        self._closed = False

    async def start(self) -> None:
        discovered = await self._inventory.gpu_ids()
        if self._configured_gpu_ids is None:
            self._gpu_ids = discovered
        else:
            missing = sorted(set(self._configured_gpu_ids) - set(discovered))
            if missing:
                raise RuntimeError(f"configured GPUs not discovered: {missing}")
            self._gpu_ids = list(self._configured_gpu_ids)
        if not self._gpu_ids:
            raise RuntimeError("no NVIDIA GPUs discovered or configured")
        self._scheduler = asyncio.create_task(
            self._scheduler_loop(), name="gpuq-scheduler"
        )
        self._wakeup.set()
        await asyncio.sleep(0)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for job_id in list(self._jobs):
            await self.cancel(job_id, reason="broker shutdown")
        if self._scheduler is not None:
            self._scheduler.cancel()
            await asyncio.gather(self._scheduler, return_exceptions=True)
        self._write_status()

    def submit(self, spec: JobSpec) -> Job:
        if self._closed:
            raise RuntimeError("broker is closed")
        if not spec.argv:
            raise ValueError("command is empty")
        if not Path(spec.cwd).is_absolute():
            raise ValueError("cwd must be an absolute path")
        job = Job(
            job_id=f"gpuq-{uuid.uuid4().hex[:12]}",
            spec=spec,
            events=asyncio.Queue(),
            submitted_at=_utc_now(),
            submitted_mono=time.monotonic(),
        )
        self._jobs[job.job_id] = job
        self._queue.append(job)
        self._store.record_request(job)
        self._emit(job, {"type": "accepted", "job_id": job.job_id})
        self._record_lifecycle(job, "accepted")
        self._publish_queue(force=True)
        self._write_status()
        self._wakeup.set()
        return job

    async def cancel(self, job_id: str, *, reason: str = "cancelled") -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job.state == "queued":
            self._queue = deque(item for item in self._queue if item is not job)
            self._finish(job, state="cancelled", exit_code=130, reason=reason)
            self._publish_queue(force=True)
            self._wakeup.set()
            return True
        if job.task is not None:
            job.task.cancel()
            await asyncio.gather(job.task, return_exceptions=True)
            return True
        return False

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        eta = self._queue_eta(now)
        return {
            "version": 1,
            "updated_at": _utc_now(),
            "probe_error": self._probe_error,
            "gpus": [
                {
                    "gpu_id": gpu_id,
                    **self._gpu_observation.get(gpu_id, {"state": "unknown"}),
                }
                for gpu_id in self._gpu_ids
            ],
            "running": [self._public_job(job, now) for job in self._running.values()],
            "queue": [
                {
                    **self._public_job(job, now),
                    "position": position,
                    "eta_seconds": eta.get(job.job_id),
                }
                for position, job in enumerate(self._queue, 1)
            ],
            "recent": list(self._recent),
        }

    async def _scheduler_loop(self) -> None:
        while True:
            self._wakeup.clear()
            self._expire_queued_jobs()
            await self._schedule_clean_cards()
            self._publish_queue(force=False)
            self._write_status()
            try:
                await asyncio.wait_for(
                    self._wakeup.wait(), timeout=self._poll_interval_s
                )
            except asyncio.TimeoutError:
                pass

    def _expire_queued_jobs(self) -> None:
        now = time.monotonic()
        retained: deque[Job] = deque()
        for job in self._queue:
            limit = job.spec.queue_timeout_s
            if limit is not None and now - job.submitted_mono >= limit:
                self._finish(
                    job,
                    state="queue_timeout",
                    exit_code=124,
                    reason=f"queue wait exceeded {limit:g}s",
                )
            else:
                retained.append(job)
        self._queue = retained

    async def _schedule_clean_cards(self) -> None:
        try:
            occupancy = await self._inventory.compute_pids()
        except Exception as exc:  # fail closed when cleanliness is unknown
            self._probe_error = str(exc)
            self._gpu_observation = {
                gpu_id: {"state": "unavailable", "reason": self._probe_error}
                for gpu_id in self._gpu_ids
            }
            return

        self._probe_error = None
        observation: dict[int, dict[str, Any]] = {}
        for gpu_id in self._gpu_ids:
            running = self._running.get(gpu_id)
            if running is not None:
                observation[gpu_id] = {
                    "state": "running",
                    "job_id": running.job_id,
                    "owner": running.spec.owner,
                    "label": running.spec.label,
                }
                continue

            pids = occupancy.get(gpu_id, [])
            if pids:
                observation[gpu_id] = {"state": "foreign", "pids": pids}
                continue

            card_lock = try_card_lock(gpu_id, self._lock_dir)
            if card_lock is None:
                observation[gpu_id] = {"state": "locked"}
                continue

            if not self._queue:
                card_lock.release()
                observation[gpu_id] = {"state": "idle"}
                continue

            job = self._queue.popleft()
            job.state = "running"
            job.gpu_id = gpu_id
            job.started_at = _utc_now()
            job.started_mono = time.monotonic()
            job.card_lock = card_lock
            self._running[gpu_id] = job
            observation[gpu_id] = {
                "state": "running",
                "job_id": job.job_id,
                "owner": job.spec.owner,
                "label": job.spec.label,
            }
            job.task = asyncio.create_task(
                self._execute(job), name=f"gpuq-run-{job.job_id}"
            )

        self._gpu_observation = observation

    async def _execute(self, job: Job) -> None:
        assert job.gpu_id is not None
        output_tasks: list[asyncio.Task[None]] = []
        state = "failed"
        exit_code = 1
        reason: str | None = None
        try:
            environment = {
                **os.environ,
                **job.spec.env,
                "CUDA_VISIBLE_DEVICES": str(job.gpu_id),
            }
            job.process = await asyncio.create_subprocess_exec(
                *job.spec.argv,
                cwd=job.spec.cwd,
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            self._emit(
                job,
                {
                    "type": "started",
                    "job_id": job.job_id,
                    "gpu_id": job.gpu_id,
                    "run_timeout_s": job.spec.run_timeout_s,
                },
            )
            self._record_lifecycle(job, "started")
            assert job.process.stdout is not None and job.process.stderr is not None
            output_tasks = [
                asyncio.create_task(self._pump(job, "stdout", job.process.stdout)),
                asyncio.create_task(self._pump(job, "stderr", job.process.stderr)),
            ]
            try:
                exit_code = await asyncio.wait_for(
                    job.process.wait(), timeout=job.spec.run_timeout_s
                )
                state = "completed" if exit_code == 0 else "failed"
            except asyncio.TimeoutError:
                reason = f"run exceeded {job.spec.run_timeout_s:g}s"
                await self._terminate(job.process)
                state = "run_timeout"
                exit_code = 124
            await asyncio.gather(*output_tasks, return_exceptions=True)
        except asyncio.CancelledError:
            reason = "client disconnected or job cancelled"
            if job.process is not None:
                await self._terminate(job.process)
            await asyncio.gather(*output_tasks, return_exceptions=True)
            state = "cancelled"
            exit_code = 130
        except Exception as exc:  # command launch and broker-side failures
            reason = str(exc)
            if job.process is not None and job.process.returncode is None:
                await self._terminate(job.process)
            await asyncio.gather(*output_tasks, return_exceptions=True)
        finally:
            if job.card_lock is not None:
                job.card_lock.release()
                job.card_lock = None
            self._running.pop(job.gpu_id, None)
            self._finish(job, state=state, exit_code=exit_code, reason=reason)
            self._wakeup.set()

    async def _pump(
        self,
        job: Job,
        stream_name: str,
        stream: asyncio.StreamReader,
    ) -> None:
        while True:
            data = await stream.read(65536)
            if not data:
                return
            self._store.append_output(job.job_id, stream_name, data)
            self._emit(
                job,
                {
                    "type": "output",
                    "stream": stream_name,
                    "data": data.decode("utf-8", "replace"),
                },
            )

    async def _terminate(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=self._terminate_grace_s)
            return
        except asyncio.TimeoutError:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        await process.wait()

    def _finish(
        self, job: Job, *, state: str, exit_code: int, reason: str | None
    ) -> None:
        if job.job_id not in self._jobs:
            return
        completed_at = _utc_now()
        duration_s = (
            max(0.0, time.monotonic() - job.started_mono)
            if job.started_mono is not None
            else None
        )
        result = {
            "job_id": job.job_id,
            "state": state,
            "exit_code": exit_code,
            "reason": reason,
            "gpu_id": job.gpu_id,
            "owner": job.spec.owner,
            "label": job.spec.label,
            "submitted_at": job.submitted_at,
            "started_at": job.started_at,
            "completed_at": completed_at,
            "duration_seconds": duration_s,
        }
        job.state = state
        self._store.record_result(job.job_id, result)
        self._emit(job, {"type": "finished", **result})
        self._record_lifecycle(job, state, result=result)
        self._recent.appendleft(result)
        self._jobs.pop(job.job_id, None)
        self._write_status()

    def _queue_eta(self, now: float) -> dict[str, float | None]:
        slots: list[float] = []
        for gpu_id in self._gpu_ids:
            running = self._running.get(gpu_id)
            if running is not None and running.started_mono is not None:
                elapsed = now - running.started_mono
                slots.append(max(0.0, running.spec.estimate_s - elapsed))
            elif self._gpu_observation.get(gpu_id, {}).get("state") == "idle":
                slots.append(0.0)
        if not slots:
            return {job.job_id: None for job in self._queue}
        result: dict[str, float | None] = {}
        for job in self._queue:
            slot_index = min(range(len(slots)), key=slots.__getitem__)
            result[job.job_id] = slots[slot_index]
            slots[slot_index] += job.spec.estimate_s
        return result

    def _publish_queue(self, *, force: bool) -> None:
        now = time.monotonic()
        eta = self._queue_eta(now)
        total = len(self._queue)
        for position, job in enumerate(self._queue, 1):
            eta_s = eta.get(job.job_id)
            eta_bucket = None if eta_s is None else int(eta_s // 10)
            key = (position, total, eta_bucket, self._probe_error)
            heartbeat_due = now - job.last_notice_mono >= self._heartbeat_s
            if force or key != job.last_queue_key or heartbeat_due:
                self._emit(
                    job,
                    {
                        "type": "queued",
                        "job_id": job.job_id,
                        "position": position,
                        "queue_length": total,
                        "eta_seconds": eta_s,
                        "probe_error": self._probe_error,
                        "heartbeat": key == job.last_queue_key,
                    },
                )
                job.last_queue_key = key
                job.last_notice_mono = now

    def _public_job(self, job: Job, now: float) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "state": job.state,
            "owner": job.spec.owner,
            "label": job.spec.label,
            "gpu_id": job.gpu_id,
            "submitted_at": job.submitted_at,
            "started_at": job.started_at,
            "wait_seconds": max(0.0, now - job.submitted_mono),
            "estimate_seconds": job.spec.estimate_s,
        }

    def _emit(self, job: Job, event: dict[str, Any]) -> None:
        job.events.put_nowait(event)

    def _record_lifecycle(
        self, job: Job, event: str, *, result: dict[str, Any] | None = None
    ) -> None:
        value: dict[str, Any] = {
            "ts": _utc_now(),
            "event": event,
            "job_id": job.job_id,
            "owner": job.spec.owner,
            "label": job.spec.label,
            "gpu_id": job.gpu_id,
        }
        if result is not None:
            value.update(
                {key: result[key] for key in ("exit_code", "reason") if key in result}
            )
        self._store.append_lifecycle(value)

    def _write_status(self) -> None:
        self._store.write_status(self.snapshot())
