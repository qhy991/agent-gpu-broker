"""Unix-socket JSONL service for the broker."""

from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path
from typing import Any

from .broker import GpuBroker, JobSpec


class BrokerServer:
    def __init__(self, broker: GpuBroker, socket_path: Path) -> None:
        self.broker = broker
        self.socket_path = socket_path
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        await self.broker.start()
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            mode = self.socket_path.stat().st_mode
            if not stat.S_ISSOCK(mode):
                raise RuntimeError(
                    f"refusing to replace non-socket path: {self.socket_path}"
                )
            if await self._socket_is_live():
                raise RuntimeError(f"broker socket already active: {self.socket_path}")
            self.socket_path.unlink()
        self._server = await asyncio.start_unix_server(
            self._handle, path=self.socket_path, limit=8 * 1024 * 1024
        )
        self.socket_path.chmod(0o660)

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        await self.broker.close()
        try:
            if stat.S_ISSOCK(self.socket_path.stat().st_mode):
                self.socket_path.unlink()
        except FileNotFoundError:
            pass

    async def _socket_is_live(self) -> bool:
        try:
            reader, writer = await asyncio.open_unix_connection(self.socket_path)
        except OSError:
            return False
        writer.close()
        await writer.wait_closed()
        del reader
        return True

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        job_id: str | None = None
        try:
            line = await reader.readline()
            if not line:
                return
            request = json.loads(line)
            operation = request.get("op")
            if operation == "status":
                await self._send(
                    writer, {"type": "status", "snapshot": self.broker.snapshot()}
                )
                return
            if operation == "cancel":
                target = str(request.get("job_id", ""))
                cancelled = await self.broker.cancel(target, reason="cancel requested")
                await self._send(writer, {"type": "cancelled", "ok": cancelled})
                return
            if operation != "run":
                raise ValueError(f"unknown operation: {operation!r}")

            spec = self._parse_spec(request)
            job = self.broker.submit(spec)
            job_id = job.job_id
            forward = asyncio.create_task(self._forward_events(job.events, writer))
            disconnected = asyncio.create_task(reader.read())
            done, pending = await asyncio.wait(
                {forward, disconnected}, return_when=asyncio.FIRST_COMPLETED
            )
            if disconnected in done and not forward.done():
                await self.broker.cancel(job.job_id, reason="client disconnected")
                await forward
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            await self._safe_send(writer, {"type": "error", "message": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            if job_id is not None:
                await self.broker.cancel(job_id, reason="client disconnected")
        except Exception as exc:
            await self._safe_send(writer, {"type": "error", "message": str(exc)})
            if job_id is not None:
                await self.broker.cancel(job_id, reason="client handler failed")
        finally:
            writer.close()
            await writer.wait_closed()

    async def _forward_events(
        self, events: asyncio.Queue[dict[str, Any]], writer: asyncio.StreamWriter
    ) -> None:
        while True:
            event = await events.get()
            await self._send(writer, event)
            if event.get("type") == "finished":
                return

    @staticmethod
    def _parse_spec(request: dict[str, Any]) -> JobSpec:
        argv = request["argv"]
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(item, str) for item in argv)
        ):
            raise ValueError("argv must be a non-empty string list")
        env = request.get("env", {})
        if not isinstance(env, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in env.items()
        ):
            raise ValueError("env must map strings to strings")
        label = request["label"]
        if not isinstance(label, str) or not label.strip():
            raise ValueError("label must be a non-empty string")
        mode = request["mode"]
        if mode not in {"shared", "exclusive"}:
            raise ValueError("mode must be 'shared' or 'exclusive'")
        gpu_count = request["gpu_count"]
        if (
            not isinstance(gpu_count, int)
            or isinstance(gpu_count, bool)
            or gpu_count < 1
        ):
            raise ValueError("gpu_count must be a positive integer")
        queue_timeout = request.get("queue_timeout_s")
        return JobSpec(
            argv=tuple(argv),
            cwd=str(request["cwd"]),
            owner=str(request.get("owner") or "unknown"),
            label=label.strip(),
            mode=mode,
            gpu_count=gpu_count,
            estimate_s=max(1.0, float(request.get("estimate_s", 600.0))),
            run_timeout_s=max(1.0, float(request.get("run_timeout_s", 900.0))),
            queue_timeout_s=(
                None if queue_timeout is None else max(0.01, float(queue_timeout))
            ),
            env=env,
        )

    @staticmethod
    async def _send(writer: asyncio.StreamWriter, value: dict[str, Any]) -> None:
        writer.write((json.dumps(value, ensure_ascii=False) + "\n").encode())
        await writer.drain()

    @classmethod
    async def _safe_send(
        cls, writer: asyncio.StreamWriter, value: dict[str, Any]
    ) -> None:
        try:
            await cls._send(writer, value)
        except (BrokenPipeError, ConnectionResetError):
            pass
