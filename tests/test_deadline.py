from __future__ import annotations

import asyncio
import json
import sys
import time
import unittest
from dataclasses import replace
from pathlib import Path

from agent_gpu_broker.broker import launch_spec_value
from agent_gpu_broker.deadline import HostDeadline, monotonic_clock_domain
from agent_gpu_broker.server import BrokerServer
import test_broker as broker_fixture
from test_broker import spec, terminal_events, await_started


class DeadlineTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = broker_fixture.BrokerTests.asyncSetUp
    asyncTearDown = broker_fixture.BrokerTests.asyncTearDown
    def deadline(self, after):
        return HostDeadline(time.monotonic() + after, monotonic_clock_domain())

    async def test_expired_queue_never_starts_process_and_finishes_once(self):
        marker = Path(self.temp.name) / 'must-not-exist'
        request = replace(spec(f"from pathlib import Path; Path({str(marker)!r}).touch()",
                               label='expired'), deadline=self.deadline(-1))
        job = self.broker.submit(request)
        events = await terminal_events(job)
        self.assertEqual(events[-1]['state'], 'phase_deadline')
        self.assertIsNone(job.process)
        self.assertFalse(marker.exists())
        self.assertEqual(events[-1]['gpu_ids'], [])
        self.assertEqual(sum(e['type'] == 'finished' for e in events), 1)
        self.assertEqual(self.broker.snapshot()['running'], [])

    async def test_waiting_deadline_expires_behind_fifo_holder(self):
        holder = self.broker.submit(spec('import time; time.sleep(.3)', label='holder'))
        await await_started(holder)
        job = self.broker.submit(replace(spec('print("never")', label='waiting'),
                                        deadline=self.deadline(.06)))
        ended = (await terminal_events(job))[-1]
        self.assertEqual(ended['state'], 'phase_deadline')
        self.assertIsNone(job.process)
        self.assertEqual((await terminal_events(holder))[-1]['state'], 'completed')

    async def test_running_deadline_preserves_output_reaps_and_releases(self):
        request = replace(spec('import time; print("partial", flush=True); time.sleep(10)',
                               label='running', run_timeout_s=5), deadline=self.deadline(.5))
        job = self.broker.submit(request)
        events = await terminal_events(job)
        self.assertEqual(events[-1]['state'], 'phase_deadline')
        self.assertEqual(events[-1]['exit_code'], 124)
        self.assertIn('partial', ''.join(e.get('data', '') for e in events))
        self.assertIsNotNone(job.process.returncode)
        self.assertEqual(self.broker.snapshot()['running'], [])
        self.assertEqual(events[-1]['cuda_cache_cleanup'], {'removed': True})
        self.assertEqual((await terminal_events(self.broker.submit(
            spec('pass', label='after-release'))))[-1]['state'], 'completed')

    async def test_ordinary_timeout_remains_distinct_when_earlier(self):
        job = self.broker.submit(replace(spec('import time; time.sleep(10)', label='timeout',
                                              run_timeout_s=.05), deadline=self.deadline(1)))
        self.assertEqual((await terminal_events(job))[-1]['state'], 'run_timeout')

    async def test_completed_scoped_job_before_deadline_preserves_scope(self):
        job = self.broker.submit(replace(spec('print("complete")', label='complete'),
                                        deadline=self.deadline(2), allowed_gpu_ids=(0,)))
        self.assertEqual((await terminal_events(job))[-1]['state'], 'completed')
        self.assertEqual(job.gpu_ids, (0,))

    async def test_cancel_before_expiry_publishes_one_terminal_result(self):
        job = self.broker.submit(replace(spec('import time; time.sleep(10)', label='race'),
                                        deadline=self.deadline(1)))
        await await_started(job)
        await self.broker.cancel(job.job_id)
        events = await terminal_events(job)
        self.assertIn(events[-1]['state'], ('phase_deadline', 'cancelled'))
        self.assertEqual(sum(e['type'] == 'finished' for e in events), 1)
        self.assertIsNotNone(job.process.returncode)
        self.assertFalse(await self.broker.cancel(job.job_id))
        self.assertEqual(self.broker.snapshot()['running'], [])

    async def test_invalid_clock_and_nan_are_refused_without_job(self):
        for value in (HostDeadline(float('nan'), monotonic_clock_domain()),
                      HostDeadline(float('inf'), monotonic_clock_domain()),
                      HostDeadline(True, monotonic_clock_domain()),
                      HostDeadline(1, 'wrong-domain')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.broker.submit(replace(spec('pass', label='invalid'), deadline=value))
        self.assertEqual(self.broker.snapshot()['queue'], [])

    async def test_legacy_digest_has_no_deadline_and_rpc_roundtrips(self):
        legacy = spec('pass', label='legacy', estimate_s=1)
        self.assertNotIn('host_deadline', launch_spec_value(legacy))
        request = replace(legacy, deadline=self.deadline(1))
        self.assertEqual(BrokerServer._parse_spec(launch_spec_value(request)), request)

    async def test_deadline_kills_descendant_that_ignores_term_and_holds_pipe(self):
        code = ('import os,signal,time; child=os.fork(); '
                'signal.signal(signal.SIGTERM, signal.SIG_IGN) if child==0 else None; '
                'print(child,flush=True); time.sleep(10)')
        job = self.broker.submit(replace(spec(code, label='descendant'),
                                        deadline=self.deadline(.5)))
        events = await terminal_events(job)
        self.assertEqual(events[-1]['state'], 'phase_deadline')
        self.assertIsNotNone(job.process.returncode)
        self.assertEqual(self.broker.snapshot()['running'], [])
        self.assertTrue(any(e.get('data') for e in events))


    async def test_cancel_during_deadline_cleanup_preserves_first_cause(self):
        entered = asyncio.Event()
        terminate = self.broker._terminate
        calls = 0
        async def paused_terminate(process, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                await asyncio.sleep(10)
            await terminate(process, **kwargs)
        with patch.object(self.broker, '_terminate', side_effect=paused_terminate):
            job = self.broker.submit(replace(spec('import time; time.sleep(10)', label='cleanup-race'),
                                            deadline=self.deadline(.2)))
            await asyncio.wait_for(entered.wait(), 2)
            await self.broker.cancel(job.job_id, reason='late cancellation')
            events = await terminal_events(job)
        self.assertEqual(events[-1]['state'], 'phase_deadline')
        self.assertIn('phase deadline', events[-1]['reason'])
        self.assertEqual(sum(e['type'] == 'finished' for e in events), 1)
        self.assertIsNotNone(job.process.returncode)
        self.assertEqual(self.broker.snapshot()['running'], [])


    async def test_exited_leader_cannot_leave_descendant_pipe_past_deadline(self):
        code = ("import os,signal,time; child=os.fork(); "
                "os._exit(0) if child else None; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "print('child-output', flush=True); time.sleep(10)")
        job = self.broker.submit(replace(spec(code, label='exited-leader'),
                                        deadline=self.deadline(.5)))
        events = await terminal_events(job)
        self.assertEqual(events[-1]['state'], 'phase_deadline')
        self.assertEqual(events[-1]['process_exit_code'], 0)
        self.assertIn('child-output', ''.join(e.get('data', '') for e in events))
        self.assertEqual(self.broker.snapshot()['running'], [])

    async def test_queue_timeout_earlier_than_deadline_keeps_its_cause(self):
        self.broker._inventory.occupancy = {0: [123]}
        job = self.broker.submit(replace(spec('pass', label='earlier-queue', queue_timeout_s=.02),
                                        deadline=self.deadline(.04)))
        await asyncio.sleep(.07)
        self.broker._expire_queued_jobs()
        self.assertEqual((await terminal_events(job))[-1]['state'], 'queue_timeout')
        self.assertIsNone(job.process)


    async def test_concurrent_cancel_and_shutdown_cannot_interrupt_reap(self):
        entered, release = asyncio.Event(), asyncio.Event()
        terminate = self.broker._terminate
        async def paused_terminate(process, **kwargs):
            entered.set()
            await release.wait()
            await terminate(process, **kwargs)
        with patch.object(self.broker, '_terminate', side_effect=paused_terminate):
            job = self.broker.submit(replace(spec('import time; time.sleep(10)', label='double-cancel'),
                                            deadline=self.deadline(2)))
            await await_started(job)
            first = asyncio.create_task(self.broker.cancel(job.job_id, reason='first-cancel'))
            await asyncio.wait_for(entered.wait(), 2)
            second = asyncio.create_task(self.broker.cancel(job.job_id, reason='second-cancel'))
            shutdown = asyncio.create_task(self.broker.close())
            await asyncio.sleep(0)
            self.assertFalse(job.task.done())
            release.set()
            await asyncio.gather(first, second, shutdown)
            events = await terminal_events(job)
        self.assertEqual(events[-1]['state'], 'cancelled')
        self.assertEqual(events[-1]['reason'], 'first-cancel')
        self.assertEqual(sum(e['type'] == 'finished' for e in events), 1)
        self.assertIsNotNone(job.process.returncode)
        self.assertEqual(self.broker.snapshot()['running'], [])


    async def test_completed_leader_cleans_descendant_with_closed_output(self):
        code = ("import os,signal,time\n"
                "child=os.fork()\n"
                "if child:\n print(child,flush=True); os._exit(0)\n"
                "os.close(1); os.close(2)\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "time.sleep(10)\n")
        job = self.broker.submit(replace(spec(code, label='closed-pipes'),
                                        deadline=self.deadline(2)))
        events = await terminal_events(job)
        self.assertEqual(events[-1]['state'], 'completed')
        child = int(''.join(e.get('data', '') for e in events).strip())
        status = Path(f'/proc/{child}/stat')
        if status.exists():
            self.assertEqual(status.read_text().split()[2], 'Z')
        self.assertEqual(events[-1]['process_exit_code'], 0)
        self.assertEqual(self.broker.snapshot()['running'], [])


import test_server as server_fixture
from unittest.mock import patch
from agent_gpu_broker import cli


class DeadlineProtocolTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = server_fixture.ServerTests.asyncSetUp
    asyncTearDown = server_fixture.ServerTests.asyncTearDown

    def request(self):
        value = launch_spec_value(replace(spec('pass', label='protocol'),
                      deadline=HostDeadline(time.monotonic()+2, monotonic_clock_domain())))
        return {**value, 'op': 'run-deadline-v1'}

    async def rpc(self, request):
        reader, writer = await asyncio.open_unix_connection(self.socket_path)
        await server_fixture.send(writer, request)
        events = []
        while True:
            event = await server_fixture.receive(reader)
            events.append(event)
            if event['type'] in ('error', 'finished'):
                break
        writer.close()
        await writer.wait_closed()
        return events

    async def test_explicit_deadline_operation_with_scope(self):
        request = self.request()
        request['allowed_gpu_ids'] = [0]
        events = await self.rpc(request)
        self.assertEqual(events[-1]['state'], 'completed')
        self.assertEqual(events[-1]['host_deadline'], request['host_deadline'])

    async def test_legacy_operation_cannot_drop_deadline(self):
        request = self.request()
        request['op'] = 'run'
        events = await self.rpc(request)
        self.assertEqual(events[-1]['type'], 'error')
        self.assertIn('run-deadline-v1', events[-1]['message'])
        self.assertEqual(self.broker.snapshot()['queue'], [])

    async def test_peer_clock_mismatch_refuses_before_job(self):
        with patch('agent_gpu_broker.server.monotonic_clock_domain', return_value='different-peer'):
            events = await self.rpc(self.request())
        self.assertEqual(events[-1]['type'], 'error')
        self.assertIn('peer clock', events[-1]['message'])
        self.assertEqual(self.broker.snapshot()['running'], [])

    async def test_new_cli_refuses_legacy_server_without_downgrade(self):
        requests = []
        async def legacy(reader, writer):
            requests.append(json.loads(await reader.readline()))
            await server_fixture.send(writer, {'type': 'error', 'message': 'unknown operation'})
            writer.close()
            await writer.wait_closed()
        old_socket = self.socket_path.parent / 'old.sock'
        server = await asyncio.start_unix_server(legacy, path=old_socket)
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, '-m', 'agent_gpu_broker.cli', 'run', '--socket', str(old_socket),
                '--label', 'old-server', '--mode', 'exclusive', '--gpu-count', '1',
                '--host-monotonic-deadline', str(time.monotonic()+2),
                '--', sys.executable, '-c', 'print("never")',
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            out, err = await asyncio.wait_for(process.communicate(), 3)
            self.assertEqual(process.returncode, 1)
            self.assertEqual(out, b'')
            self.assertIn(b'unknown operation', err)
            self.assertEqual([r['op'] for r in requests], ['run-deadline-v1'])
        finally:
            server.close()
            await server.wait_closed()
