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

MODES = frozenset({"shared", "exclusive"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class JobSpec:
    argv: tuple[str, ...]
    cwd: str
    owner: str
    label: str
    mode: str
    gpu_count: int
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
    gpu_ids: tuple[int, ...] = ()
    started_at: str | None = None
    started_mono: float | None = None
    process: asyncio.subprocess.Process | None = None
    task: asyncio.Task[None] | None = None
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
                "mode": job.spec.mode,
                "gpu_count": job.spec.gpu_count,
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
        shared_capacity: int = 2,
        poll_interval_s: float = 2.0,
        heartbeat_s: float = 30.0,
        terminate_grace_s: float = 5.0,
    ) -> None:
        if shared_capacity < 1:
            raise ValueError("shared_capacity must be positive")
        self._store = StateStore(state_dir)
        self._lock_dir = lock_dir
        self._inventory = inventory or NvidiaSmiInventory()
        self._configured_gpu_ids = list(gpu_ids) if gpu_ids is not None else None
        self._gpu_ids: list[int] = []
        self._shared_capacity = shared_capacity
        self._poll_interval_s = poll_interval_s
        self._heartbeat_s = heartbeat_s
        self._terminate_grace_s = terminate_grace_s
        self._jobs: dict[str, Job] = {}
        self._queue: deque[Job] = deque()
        self._running_jobs: dict[str, Job] = {}
        self._gpu_jobs: dict[int, list[Job]] = {}
        self._gpu_locks: dict[int, CardLock] = {}
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
        if len(set(self._gpu_ids)) != len(self._gpu_ids):
            raise RuntimeError("configured GPU indices must be unique")
        self._scheduler = asyncio.create_task(
            self._scheduler_loop(), name="gpuq-scheduler"
        )
        self._wakeup.set()
        await asyncio.sleep(0)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._scheduler is not None:
            self._scheduler.cancel()
            await asyncio.gather(self._scheduler, return_exceptions=True)
        for job_id in list(self._jobs):
            await self.cancel(job_id, reason="broker shutdown")
        for card_lock in self._gpu_locks.values():
            card_lock.release()
        self._gpu_locks.clear()
        self._write_status()

    def submit(self, spec: JobSpec) -> Job:
        if self._closed:
            raise RuntimeError("broker is closed")
        if not spec.argv:
            raise ValueError("command is empty")
        if not Path(spec.cwd).is_absolute():
            raise ValueError("cwd must be an absolute path")
        if not isinstance(spec.label, str) or not spec.label.strip():
            raise ValueError("label is required")
        if spec.mode not in MODES:
            raise ValueError("mode must be 'shared' or 'exclusive'")
        if (
            not isinstance(spec.gpu_count, int)
            or isinstance(spec.gpu_count, bool)
            or spec.gpu_count < 1
        ):
            raise ValueError("gpu_count must be a positive integer")
        if spec.gpu_count > len(self._gpu_ids):
            raise ValueError(
                f"gpu_count={spec.gpu_count} exceeds managed GPUs={len(self._gpu_ids)}"
            )
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
        self._emit(
            job,
            {
                "type": "accepted",
                "job_id": job.job_id,
                "label": job.spec.label,
                "mode": job.spec.mode,
                "gpu_count": job.spec.gpu_count,
            },
        )
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
        running = sorted(
            self._running_jobs.values(), key=lambda job: job.started_mono or 0.0
        )
        return {
            "version": 2,
            "updated_at": _utc_now(),
            "probe_error": self._probe_error,
            "shared_capacity": self._shared_capacity,
            "gpus": [
                {
                    "gpu_id": gpu_id,
                    **self._gpu_observation.get(gpu_id, {"state": "unknown"}),
                }
                for gpu_id in self._gpu_ids
            ],
            "running": [self._public_job(job, now) for job in running],
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
            await self._schedule_jobs()
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

    async def _schedule_jobs(self) -> None:
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
        while self._queue:
            job = self._queue[0]
            gpu_ids = self._try_allocate(job, occupancy)
            if gpu_ids is None:
                break
            self._queue.popleft()
            self._start_job(job, gpu_ids)
        self._gpu_observation = self._observe_gpus(occupancy)

    def _try_allocate(
        self, job: Job, occupancy: dict[int, list[int]]
    ) -> tuple[int, ...] | None:
        selected: list[int] = []
        if job.spec.mode == "shared":
            shared_cards = sorted(
                (
                    gpu_id
                    for gpu_id, jobs in self._gpu_jobs.items()
                    if jobs
                    and all(item.spec.mode == "shared" for item in jobs)
                    and len(jobs) < self._shared_capacity
                ),
                key=lambda gpu_id: (-len(self._gpu_jobs[gpu_id]), gpu_id),
            )
            selected.extend(shared_cards[: job.spec.gpu_count])

        needed = job.spec.gpu_count - len(selected)
        acquired: dict[int, CardLock] = {}
        if needed:
            free_cards = [
                gpu_id
                for gpu_id in self._gpu_ids
                if gpu_id not in self._gpu_jobs
                and gpu_id not in selected
                and not occupancy.get(gpu_id, [])
            ]
            for gpu_id in free_cards:
                card_lock = try_card_lock(gpu_id, self._lock_dir)
                if card_lock is None:
                    continue
                acquired[gpu_id] = card_lock
                selected.append(gpu_id)
                if len(selected) == job.spec.gpu_count:
                    break

        if len(selected) != job.spec.gpu_count:
            for card_lock in acquired.values():
                card_lock.release()
            return None

        self._gpu_locks.update(acquired)
        return tuple(selected)

    def _start_job(self, job: Job, gpu_ids: tuple[int, ...]) -> None:
        job.state = "running"
        job.gpu_ids = gpu_ids
        job.started_at = _utc_now()
        job.started_mono = time.monotonic()
        self._running_jobs[job.job_id] = job
        for gpu_id in gpu_ids:
            self._gpu_jobs.setdefault(gpu_id, []).append(job)
        job.task = asyncio.create_task(
            self._execute(job), name=f"gpuq-run-{job.job_id}"
        )

    def _observe_gpus(
        self, occupancy: dict[int, list[int]]
    ) -> dict[int, dict[str, Any]]:
        observation: dict[int, dict[str, Any]] = {}
        for gpu_id in self._gpu_ids:
            jobs = self._gpu_jobs.get(gpu_id, [])
            if jobs:
                mode = jobs[0].spec.mode
                value: dict[str, Any] = {
                    "state": mode,
                    "jobs": [self._job_brief(job) for job in jobs],
                }
                if mode == "shared":
                    value.update(
                        {
                            "shared_used": len(jobs),
                            "shared_capacity": self._shared_capacity,
                        }
                    )
                observation[gpu_id] = value
                continue

            pids = occupancy.get(gpu_id, [])
            if pids:
                observation[gpu_id] = {"state": "foreign", "pids": pids}
                continue

            card_lock = try_card_lock(gpu_id, self._lock_dir)
            if card_lock is None:
                observation[gpu_id] = {"state": "locked"}
            else:
                card_lock.release()
                observation[gpu_id] = {"state": "idle"}
        return observation

    async def _execute(self, job: Job) -> None:
        assert job.gpu_ids
        output_tasks: list[asyncio.Task[None]] = []
        state = "failed"
        exit_code = 1
        reason: str | None = None
        try:
            environment = {
                **os.environ,
                **job.spec.env,
                "CUDA_VISIBLE_DEVICES": ",".join(map(str, job.gpu_ids)),
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
                    "label": job.spec.label,
                    "mode": job.spec.mode,
                    "gpu_count": job.spec.gpu_count,
                    "gpu_ids": list(job.gpu_ids),
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
            self._release_allocation(job)
            self._finish(job, state=state, exit_code=exit_code, reason=reason)
            self._wakeup.set()

    def _release_allocation(self, job: Job) -> None:
        self._running_jobs.pop(job.job_id, None)
        for gpu_id in job.gpu_ids:
            jobs = self._gpu_jobs.get(gpu_id, [])
            if job in jobs:
                jobs.remove(job)
            if jobs:
                self._gpu_observation[gpu_id] = {
                    "state": "shared",
                    "jobs": [self._job_brief(item) for item in jobs],
                    "shared_used": len(jobs),
                    "shared_capacity": self._shared_capacity,
                }
                continue
            self._gpu_jobs.pop(gpu_id, None)
            card_lock = self._gpu_locks.pop(gpu_id, None)
            if card_lock is not None:
                card_lock.release()
            self._gpu_observation[gpu_id] = {"state": "unknown"}

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
            "gpu_ids": list(job.gpu_ids),
            "gpu_count": job.spec.gpu_count,
            "mode": job.spec.mode,
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
        available_ids = [
            gpu_id
            for gpu_id in self._gpu_ids
            if gpu_id in self._gpu_jobs
            or self._gpu_observation.get(gpu_id, {}).get("state") == "idle"
        ]
        if not available_ids:
            return {job.job_id: None for job in self._queue}

        virtual: dict[int, dict[str, Any]] = {
            gpu_id: {"exclusive_until": 0.0, "shared_ends": []}
            for gpu_id in available_ids
        }
        for job in self._running_jobs.values():
            if job.started_mono is None:
                continue
            remaining = max(0.0, job.spec.estimate_s - (now - job.started_mono))
            for gpu_id in job.gpu_ids:
                if gpu_id not in virtual:
                    continue
                if job.spec.mode == "exclusive":
                    virtual[gpu_id]["exclusive_until"] = max(
                        virtual[gpu_id]["exclusive_until"], remaining
                    )
                else:
                    virtual[gpu_id]["shared_ends"].append(remaining)

        result: dict[str, float | None] = {}
        queue = list(self._queue)
        for index, job in enumerate(queue):
            if job.spec.gpu_count > len(virtual):
                for blocked in queue[index:]:
                    result[blocked.job_id] = None
                break
            if job.spec.mode == "exclusive":
                ready = sorted(
                    (
                        max(
                            state["exclusive_until"],
                            max(state["shared_ends"], default=0.0),
                        ),
                        gpu_id,
                    )
                    for gpu_id, state in virtual.items()
                )
                selected = ready[: job.spec.gpu_count]
                start = max(item[0] for item in selected)
                for _, gpu_id in selected:
                    virtual[gpu_id]["exclusive_until"] = start + job.spec.estimate_s
                    virtual[gpu_id]["shared_ends"] = []
            else:
                ready = sorted(
                    (self._shared_ready_time(state), gpu_id)
                    for gpu_id, state in virtual.items()
                )
                selected = ready[: job.spec.gpu_count]
                start = max(item[0] for item in selected)
                for _, gpu_id in selected:
                    state = virtual[gpu_id]
                    state["shared_ends"] = [
                        end for end in state["shared_ends"] if end > start
                    ]
                    state["shared_ends"].append(start + job.spec.estimate_s)
            result[job.job_id] = start
        return result

    def _shared_ready_time(self, state: dict[str, Any]) -> float:
        start = float(state["exclusive_until"])
        active = sorted(end for end in state["shared_ends"] if end > start)
        if len(active) < self._shared_capacity:
            return start
        return active[len(active) - self._shared_capacity]

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
                        "label": job.spec.label,
                        "mode": job.spec.mode,
                        "gpu_count": job.spec.gpu_count,
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
            "mode": job.spec.mode,
            "gpu_count": job.spec.gpu_count,
            "gpu_ids": list(job.gpu_ids),
            "submitted_at": job.submitted_at,
            "started_at": job.started_at,
            "wait_seconds": max(0.0, now - job.submitted_mono),
            "estimate_seconds": job.spec.estimate_s,
        }

    @staticmethod
    def _job_brief(job: Job) -> dict[str, Any]:
        return {
            "job_id": job.job_id,
            "owner": job.spec.owner,
            "label": job.spec.label,
            "mode": job.spec.mode,
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
            "mode": job.spec.mode,
            "gpu_count": job.spec.gpu_count,
            "gpu_ids": list(job.gpu_ids),
        }
        if result is not None:
            value.update(
                {key: result[key] for key in ("exit_code", "reason") if key in result}
            )
        self._store.append_lifecycle(value)

    def _write_status(self) -> None:
        self._store.write_status(self.snapshot())
