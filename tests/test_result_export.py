from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_gpu_broker.cli import _parser, _run


class Resource:
    def close(self):
        pass


class Events(Resource):
    def __init__(self, values):
        self.values = values

    def __iter__(self):
        for value in self.values:
            if isinstance(value, BaseException):
                raise value
            yield value if isinstance(value, bytes) else (json.dumps(value) + '\n').encode()


class ResultExportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.output = Path(self.directory.name) / 'result.json'
        self.accepted = dict(type='accepted', job_id='gpuq-test', label='job',
                             mode='exclusive', gpu_count=1)

    def args(self, *options):
        return _parser().parse_args(['run', '--label', 'job', '--mode', 'exclusive',
                                     '--gpu-count', '1', *options, '--', 'true'])

    def run_events(self, values, *options):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch('agent_gpu_broker.cli._request',
                   return_value=(Resource(), Events(values))) as request:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = _run(self.args(*options))
        return result, stdout.getvalue(), stderr.getvalue(), request

    def terminal(self, **changes):
        value = dict(type='finished', job_id='gpuq-test', state='completed',
                     exit_code=0, reason=None, future_field={'nested': [1, None, 'a']})
        value.update(changes)
        return value

    def test_exact_terminal_export_and_unchanged_streams_and_request(self):
        terminal = self.terminal(warning='possible_swallowed_failure')
        events = [self.accepted, dict(type='output', stream='stdout', data='worker output\n'),
                  dict(type='output', stream='stderr', data='worker error\n'), terminal]
        ordinary = self.run_events(events)
        exported = self.run_events(events, '--result-out', str(self.output))
        self.assertEqual(exported[:3], ordinary[:3])
        self.assertEqual(exported[3].call_args, ordinary[3].call_args)
        self.assertEqual(json.loads(self.output.read_text()), terminal)
        self.assertEqual(list(self.output.parent.glob('.result.json.*')), [])

    def test_queued_and_running_deadlines_preserve_exact_event(self):
        for started in (False, True):
            with self.subTest(started=started):
                terminal = self.terminal(state='phase_deadline', exit_code=124,
                    reason='host deadline expired', host_deadline={'monotonic_s': 100},
                    started_at=99 if started else None, duration_seconds=1 if started else None,
                    gpu_ids=[0] if started else [], process_exit_code=-15 if started else None)
                receipt_out = self.output.parent / 'receipt.json'
                events = [self.accepted]
                receipt = dict(job_id='gpuq-test', receipt_sha256='receipt-sha')
                if started:
                    events.append(dict(type='started', gpu_ids=[0], run_timeout_s=2,
                                       admission_receipt=receipt))
                events.append(terminal)
                result = self.run_events(events, '--result-out', str(self.output),
                                         '--receipt-out', str(receipt_out))
                self.assertEqual(result[0], 124)
                self.assertEqual(json.loads(self.output.read_text()), terminal)
                self.assertEqual(receipt_out.exists(), started)
                if started:
                    self.assertEqual(json.loads(receipt_out.read_text()), receipt)

    def test_other_terminal_failures_keep_original_exit_code(self):
        for state, exit_code in [('failed', 7), ('cancelled', 130), ('queue_timeout', 124)]:
            with self.subTest(state=state):
                terminal = self.terminal(state=state, exit_code=exit_code)
                result = self.run_events([self.accepted, terminal], '--result-out', str(self.output))
                self.assertEqual(result[0], exit_code)
                self.assertEqual(json.loads(self.output.read_text()), terminal)

    def test_refusal_eof_malformed_json_and_interrupt_do_not_export(self):
        for events in ([dict(type='error', message='refused')], [self.accepted],
                       [self.accepted, b'not json\n'], [self.accepted, KeyboardInterrupt()]):
            with self.subTest(events=events), patch('agent_gpu_broker.cli._request_cancel', return_value=True):
                result = self.run_events(events, '--result-out', str(self.output))
                self.assertNotEqual(result[0], 0)
                self.assertFalse(self.output.exists())
                self.assertEqual(result[3].call_count, 1)

    def test_missing_or_mismatched_accepted_job_rejects_terminal(self):
        for events in ([self.terminal()], [self.accepted, self.terminal(job_id='other')],
                       [self.accepted, self.terminal(job_id=None)]):
            with self.subTest(events=events):
                result = self.run_events(events, '--result-out', str(self.output))
                self.assertEqual(result[0], 1)
                self.assertIn('result export failed', result[2])
                self.assertFalse(self.output.exists())

    def test_resolved_receipt_collision_rejected_before_submission(self):
        alias = self.output.parent / 'alias'
        alias.symlink_to(self.output.parent, target_is_directory=True)
        result = self.run_events([], '--receipt-out', str(self.output),
                                 '--result-out', str(alias / self.output.name))
        self.assertEqual(result[0], 1)
        result[3].assert_not_called()
        self.assertFalse(self.output.exists())

    def test_atomic_replace_failure_preserves_old_file_and_removes_temporary(self):
        self.output.write_text('old custody')
        with patch('agent_gpu_broker.cli.os.replace', side_effect=OSError('write denied')):
            result = self.run_events([self.accepted, self.terminal()],
                                     '--result-out', str(self.output))
        self.assertEqual(result[0], 1)
        self.assertIn('result export failed: write denied', result[2])
        self.assertEqual(self.output.read_text(), 'old custody')
        self.assertEqual(list(self.output.parent.glob('.result.json.*')), [])
        self.assertEqual(result[3].call_count, 1)

    def test_serialization_failure_is_reported_without_retry(self):
        with patch('agent_gpu_broker.cli._atomic_json', side_effect=TypeError('cannot serialize')):
            result = self.run_events([self.accepted, self.terminal()],
                                     '--result-out', str(self.output))
        self.assertEqual(result[0], 1)
        self.assertIn('result export failed: cannot serialize', result[2])
        self.assertFalse(self.output.exists())
        self.assertEqual(result[3].call_count, 1)


if __name__ == '__main__':
    unittest.main()
