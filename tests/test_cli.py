from __future__ import annotations

import argparse
import contextlib
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_gpu_broker.cli import (
    _run,
    _status,
    parse_duration,
    parse_estimate,
    positive_int,
)


class ContextResource:
    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class InterruptAfterAccepted(ContextResource):
    def __init__(self):
        self.sent = False

    def __iter__(self):
        return self

    def __next__(self):
        if not self.sent:
            self.sent = True
            return b'{"type":"accepted","job_id":"gpuq-test","label":"job","mode":"exclusive","gpu_count":1}\n'
        raise KeyboardInterrupt


class ResponseConnection(ContextResource):
    def __init__(self, response: dict):
        self.response = response

    def readline(self):
        return (json.dumps(self.response) + "\n").encode()


class DurationTests(unittest.TestCase):
    def test_suffixes(self):
        self.assertEqual(parse_duration("30s"), 30)
        self.assertEqual(parse_duration("2m"), 120)
        self.assertEqual(parse_duration("1.5h"), 5400)

    def test_rejects_non_positive(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_duration("0")

    def test_positive_integer(self):
        self.assertEqual(positive_int("2"), 2)
        with self.assertRaises(argparse.ArgumentTypeError):
            positive_int("0")

    def test_unknown_estimate(self):
        self.assertIsNone(parse_estimate("unknown"))
        self.assertEqual(parse_estimate("2m"), 120)


class RunTests(unittest.TestCase):
    def test_interrupt_explicitly_requests_cancellation(self):
        requests = []

        def fake_request(socket_path, value):
            requests.append((socket_path, value))
            if value["op"] == "run":
                return ContextResource(), InterruptAfterAccepted()
            return ContextResource(), ResponseConnection({"ok": True})

        args = argparse.Namespace(
            argv=["--", "true"],
            env=[],
            socket=Path("/tmp/test-gpuq.sock"),
            cwd=Path.cwd(),
            owner="test",
            label="job",
            mode="exclusive",
            gpu_count=1,
            estimate=1.0,
            queue_timeout=None,
            run_timeout=2.0,
        )
        stderr = io.StringIO()
        with patch("agent_gpu_broker.cli._request", side_effect=fake_request):
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(_run(args), 130)

        self.assertEqual(requests[-1][1], {"op": "cancel", "job_id": "gpuq-test"})
        self.assertIn("cancellation confirmed", stderr.getvalue())

    def test_finished_warning_is_visible(self):
        class FinishedConnection(ContextResource):
            def __iter__(self):
                return iter(
                    [
                        b'{"type":"accepted","job_id":"gpuq-test","label":"job","mode":"exclusive","gpu_count":1}\n',
                        b'{"type":"finished","state":"completed","exit_code":0,"reason":null,"warning":"possible_swallowed_failure"}\n',
                    ]
                )

        args = argparse.Namespace(
            argv=["true"],
            env=[],
            socket=Path("/tmp/test-gpuq.sock"),
            cwd=Path.cwd(),
            owner="test",
            label="job",
            mode="exclusive",
            gpu_count=1,
            estimate=1.0,
            queue_timeout=None,
            run_timeout=2.0,
        )
        stderr = io.StringIO()
        with patch(
            "agent_gpu_broker.cli._request",
            return_value=(ContextResource(), FinishedConnection()),
        ):
            with contextlib.redirect_stderr(stderr):
                self.assertEqual(_run(args), 0)
        self.assertIn("possible_swallowed_failure", stderr.getvalue())


class StatusTests(unittest.TestCase):
    def test_human_status_shows_broker_identity(self):
        snapshot = {
            "broker_version": "0.5.1",
            "instance_id": "host-pid1-start1",
            "probe_error": None,
            "gpus": [{"gpu_id": 0, "state": "idle"}],
            "running": [],
            "queue": [],
        }
        args = argparse.Namespace(
            socket=Path("/tmp/test-gpuq.sock"),
            json=False,
        )
        stdout = io.StringIO()
        with patch(
            "agent_gpu_broker.cli._request",
            return_value=(
                ContextResource(),
                ResponseConnection({"snapshot": snapshot}),
            ),
        ):
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(_status(args), 0)
        self.assertIn("version=0.5.1", stdout.getvalue())
        self.assertIn("instance=host-pid1-start1", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
