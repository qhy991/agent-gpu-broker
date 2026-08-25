"""Command-line client and daemon entry point."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import signal
import socket
import sys
import tempfile
from pathlib import Path
from typing import Any

from .broker import GpuBroker
from .server import BrokerServer

DEFAULT_SOCKET = Path("/tmp/agent-gpu-broker.sock")
DEFAULT_STATE_DIR = Path.home() / ".local/share/agent-gpu-broker"
DEFAULT_LOCK_DIR = Path(os.environ.get("GPUQ_LOCK_DIR", "/tmp/agent-gpu-locks"))


def parse_duration(value: str) -> float:
    suffixes = {"s": 1.0, "m": 60.0, "h": 3600.0}
    text = value.strip().lower()
    multiplier = 1.0
    if text and text[-1] in suffixes:
        multiplier = suffixes[text[-1]]
        text = text[:-1]
    try:
        result = float(text) * multiplier
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid duration: {value}") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("duration must be positive")
    return result


def parse_estimate(value: str) -> float | None:
    if value.strip().lower() == "unknown":
        return None
    return parse_duration(value)


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value}") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpuq")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="run the machine-local broker")
    serve.add_argument("--socket", type=Path, default=DEFAULT_SOCKET)
    serve.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    serve.add_argument("--lock-dir", type=Path, default=DEFAULT_LOCK_DIR)
    serve.add_argument("--gpus", help="comma-separated physical GPU indices")
    serve.add_argument("--shared-capacity", type=positive_int, default=2)
    serve.add_argument("--poll-interval", type=parse_duration, default=2.0)
    serve.add_argument("--heartbeat", type=parse_duration, default=30.0)

    status = subparsers.add_parser("status", help="show the global queue")
    status.add_argument("--socket", type=Path, default=DEFAULT_SOCKET)
    status.add_argument("--json", action="store_true")

    cancel = subparsers.add_parser("cancel", help="cancel a queued/running job")
    cancel.add_argument("job_id")
    cancel.add_argument("--socket", type=Path, default=DEFAULT_SOCKET)

    receipt = subparsers.add_parser(
        "receipt", help="show one active job's broker-issued admission receipt"
    )
    receipt.add_argument("job_id")
    receipt.add_argument("--socket", type=Path, default=DEFAULT_SOCKET)
    receipt.add_argument("--out", type=Path)

    run = subparsers.add_parser("run", help="queue and run one GPU command")
    run.add_argument("--socket", type=Path, default=DEFAULT_SOCKET)
    run.add_argument("--label", required=True, help="human-readable job name")
    run.add_argument("--mode", choices=("shared", "exclusive"), required=True)
    run.add_argument("--gpu-count", type=positive_int, required=True)
    run.add_argument("--owner", default=getpass.getuser())
    run.add_argument("--cwd", type=Path, default=Path.cwd())
    run.add_argument("--estimate", type=parse_estimate, default=600.0)
    run.add_argument("--queue-timeout", type=parse_duration)
    run.add_argument(
        "--run-timeout",
        "--timeout",
        dest="run_timeout",
        type=parse_duration,
        default=900.0,
    )
    run.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    run.add_argument(
        "--receipt-out",
        type=Path,
        help="atomically write the started job's broker-issued admission receipt",
    )
    run.add_argument("argv", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "serve":
        return asyncio.run(_serve(args))
    if args.command == "status":
        return _status(args)
    if args.command == "cancel":
        return _cancel(args)
    if args.command == "receipt":
        return _receipt(args)
    if args.command == "run":
        return _run(args)
    raise AssertionError(args.command)


def gpu_run_main() -> int:
    return main(["run", *sys.argv[1:]])


async def _serve(args: argparse.Namespace) -> int:
    gpu_ids = None
    if args.gpus:
        gpu_ids = [
            int(value.strip()) for value in args.gpus.split(",") if value.strip()
        ]
    broker = GpuBroker(
        state_dir=args.state_dir.expanduser(),
        lock_dir=args.lock_dir,
        gpu_ids=gpu_ids,
        shared_capacity=args.shared_capacity,
        poll_interval_s=args.poll_interval,
        heartbeat_s=args.heartbeat,
    )
    server = BrokerServer(broker, args.socket)
    await server.start()
    print(
        f"gpuq: serving {args.socket} with state in {args.state_dir.expanduser()}",
        flush=True,
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    try:
        await stop.wait()
    finally:
        await server.close()
    return 0


def _request(socket_path: Path, value: dict[str, Any]):
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        client.connect(str(socket_path))
    except OSError as exc:
        client.close()
        raise RuntimeError(f"cannot reach broker at {socket_path}: {exc}") from exc
    connection = client.makefile("rwb")
    connection.write((json.dumps(value) + "\n").encode())
    connection.flush()
    return client, connection


def _status(args: argparse.Namespace) -> int:
    try:
        client, connection = _request(args.socket, {"op": "status"})
        with client, connection:
            response = json.loads(connection.readline())
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"gpuq: {exc}", file=sys.stderr)
        return 1
    snapshot = response["snapshot"]
    if args.json:
        print(json.dumps(snapshot, indent=2, ensure_ascii=False))
        return 0
    print(
        f"BROKER version={snapshot.get('broker_version', 'unknown')} "
        f"instance={snapshot.get('instance_id', 'unknown')}"
    )
    if snapshot.get("probe_error"):
        print(f"probe error: {snapshot['probe_error']}")
    print("GPUS")
    for gpu in snapshot["gpus"]:
        jobs = gpu.get("jobs", [])
        detail = ",".join(job["label"] for job in jobs)
        if gpu.get("state") == "shared":
            detail = (
                f"{gpu['shared_used']}/{gpu['shared_capacity']} " + detail
            ).rstrip()
        if gpu.get("pids"):
            detail = f"pids={','.join(map(str, gpu['pids']))}"
        print(f"  {gpu['gpu_id']}: {gpu['state']} {detail}".rstrip())
    print("RUNNING")
    if not snapshot["running"]:
        print("  -")
    for job in snapshot["running"]:
        print(
            f"  {job['job_id']} gpus={','.join(map(str, job['gpu_ids']))} "
            f"mode={job['mode']} owner={job['owner']} label={job['label']}"
        )
    print("QUEUE")
    if not snapshot["queue"]:
        print("  -")
    for job in snapshot["queue"]:
        eta = _format_eta(job.get("eta_seconds"))
        print(
            f"  {job['position']}. {job['job_id']} owner={job['owner']} "
            f"label={job['label']} mode={job['mode']} gpus={job['gpu_count']} "
            f"eta={eta}"
        )
    return 0


def _cancel(args: argparse.Namespace) -> int:
    try:
        cancelled = _request_cancel(args.socket, args.job_id)
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"gpuq: {exc}", file=sys.stderr)
        return 1
    if not cancelled:
        print(f"gpuq: active job not found: {args.job_id}", file=sys.stderr)
        return 1
    return 0


def _request_cancel(socket_path: Path, job_id: str) -> bool:
    client, connection = _request(
        socket_path, {"op": "cancel", "job_id": job_id}
    )
    with client, connection:
        response = json.loads(connection.readline())
    return bool(response.get("ok"))


def _receipt(args: argparse.Namespace) -> int:
    try:
        client, connection = _request(
            args.socket, {"op": "receipt", "job_id": args.job_id}
        )
        with client, connection:
            response = json.loads(connection.readline())
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"gpuq: {exc}", file=sys.stderr)
        return 1
    receipt = response.get("receipt")
    if not response.get("ok") or not isinstance(receipt, dict):
        print(f"gpuq: active job receipt not found: {args.job_id}", file=sys.stderr)
        return 1
    if args.out is not None:
        _atomic_json(args.out.expanduser().resolve(), receipt)
    else:
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


def _parse_env(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--env requires KEY=VALUE, got: {value}")
        key, item = value.split("=", 1)
        if not key:
            raise ValueError("--env key cannot be empty")
        result[key] = item
    return result


def _run(args: argparse.Namespace) -> int:
    command = list(args.argv)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        print("gpu-run: missing command after --", file=sys.stderr)
        return 2
    try:
        env = _parse_env(args.env)
        client, connection = _request(
            args.socket,
            {
                "op": "run",
                "argv": command,
                "cwd": str(args.cwd.expanduser().resolve()),
                "owner": args.owner,
                "label": args.label,
                "mode": args.mode,
                "gpu_count": args.gpu_count,
                "estimate_s": args.estimate,
                "queue_timeout_s": args.queue_timeout,
                "run_timeout_s": args.run_timeout,
                "env": env,
            },
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"gpu-run: {exc}", file=sys.stderr)
        return 1

    exit_code = 1
    job_id: str | None = None
    try:
        for line in connection:
            event = json.loads(line)
            kind = event.get("type")
            if kind == "accepted":
                job_id = str(event["job_id"])
                admission = event.get("admission_receipt") or {}
                digest = admission.get("launch_spec_sha256")
                suffix = f" admission={str(digest)[:12]}" if digest else ""
                _note(
                    f"accepted job {event['job_id']} label={event['label']} "
                    f"mode={event['mode']} gpus={event['gpu_count']}{suffix}"
                )
            elif kind == "queued":
                message = (
                    f"queued position={event['position']}/{event['queue_length']} "
                    f"eta={_format_eta(event.get('eta_seconds'))}"
                )
                if event.get("probe_error"):
                    message += f" probe_error={event['probe_error']}"
                _note(message)
            elif kind == "started":
                if args.receipt_out is not None:
                    receipt = event.get("admission_receipt")
                    if not isinstance(receipt, dict):
                        raise RuntimeError(
                            "broker did not provide the requested admission receipt"
                        )
                    _atomic_json(args.receipt_out.expanduser().resolve(), receipt)
                _note(
                    "running on physical GPUs "
                    f"{','.join(map(str, event['gpu_ids']))} "
                    f"(run limit {_format_eta(event['run_timeout_s'])})"
                )
            elif kind == "output":
                stream = sys.stderr if event["stream"] == "stderr" else sys.stdout
                stream.write(event["data"])
                stream.flush()
            elif kind == "finished":
                if event.get("warning"):
                    _note(f"warning: {event['warning']}")
                reason = f" reason={event['reason']}" if event.get("reason") else ""
                _note(
                    f"finished state={event['state']} exit={event['exit_code']}{reason}"
                )
                exit_code = int(event["exit_code"])
                break
            elif kind == "error":
                print(f"gpu-run: broker error: {event['message']}", file=sys.stderr)
                exit_code = 1
                break
    except KeyboardInterrupt:
        return _interrupt_and_cancel(args.socket, job_id)
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"gpu-run: connection failed: {exc}", file=sys.stderr)
        return 1
    finally:
        connection.close()
        client.close()
    return exit_code


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _interrupt_and_cancel(socket_path: Path, job_id: str | None) -> int:
    if job_id is None:
        _note("interrupted before broker accepted the job")
        return 130
    _note(f"interrupted; requesting cancellation for {job_id}")
    try:
        if _request_cancel(socket_path, job_id):
            _note(f"cancellation confirmed for {job_id}")
        else:
            _note(f"job {job_id} is already terminal or cancelling")
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        _note(f"explicit cancellation could not be confirmed: {exc}")
    return 130


def _note(message: str) -> None:
    print(f"[gpu-run] {message}", file=sys.stderr, flush=True)


def _format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


if __name__ == "__main__":
    raise SystemExit(main())
