import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from agent_gpu_broker.backends import (CommandInventory, DeviceBackend, MetalInventory, make_backend)
from agent_gpu_broker.broker import GpuBroker
from test_broker import FakeInventory, spec, terminal_events


class BackendTests(unittest.IsolatedAsyncioTestCase):
    def test_backend_environment_cannot_be_overridden(self):
        inherited = {'HIP_VISIBLE_DEVICES': '9', 'CUDA_VISIBLE_DEVICES': '8', 'ROCR_VISIBLE_DEVICES': '7', 'GPUQ_BACKEND': 'wrong', 'PATH': '/bin'}
        for name, key in [('nvidia', 'CUDA_VISIBLE_DEVICES'), ('hygon', 'HIP_VISIBLE_DEVICES')]:
            env = DeviceBackend(name, None).environment(inherited, (2,))
            self.assertEqual(env[key], '2')
            self.assertEqual(env['GPUQ_BACKEND'], name)
            self.assertNotIn('ROCR_VISIBLE_DEVICES', env)
            self.assertEqual(env['PATH'], '/bin')
        env = DeviceBackend('metal', None, 'cooperative').environment(inherited, (0,))
        self.assertNotIn('CUDA_VISIBLE_DEVICES', env)
        self.assertNotIn('HIP_VISIBLE_DEVICES', env)
        with self.assertRaises(RuntimeError):
            DeviceBackend('metal', None).environment({}, (1,))

    def test_metal_requires_explicit_scope(self):
        with self.assertRaises(ValueError):
            make_backend('metal')
        self.assertEqual(make_backend('metal', occupancy_scope='cooperative').occupancy_scope, 'cooperative')

    async def test_command_probe_rejects_unknown_and_drift(self):
        inventory = CommandInventory(['probe'])
        def payload(devices):
            return {'schema': 'gpuq.device-probe.v1', 'devices': devices}
        with patch('agent_gpu_broker.backends.read_json', AsyncMock(return_value=payload([{'id': 0, 'compute_pids': []}]))):
            self.assertEqual(await inventory.gpu_ids(), [0])
        for devices in [[{'id': 0}], [{'id': 0, 'compute_pids': None}], [{'id': 1, 'compute_pids': []}], [{'id': True, 'compute_pids': []}], [{'id': 0, 'compute_pids': [-1]}]]:
            with patch('agent_gpu_broker.backends.read_json', AsyncMock(return_value=payload(devices))):
                with self.assertRaises(RuntimeError):
                    await inventory.compute_pids()

    async def test_hygon_child_environment_and_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            broker = GpuBroker(state_dir=root/'state', lock_dir=root/'locks', backend=DeviceBackend('hygon', FakeInventory()), poll_interval_s=.01)
            await broker.start()
            try:
                job = broker.submit(spec("import os; assert os.environ['HIP_VISIBLE_DEVICES']=='0'; assert 'CUDA_VISIBLE_DEVICES' not in os.environ", label='hygon-env'))
                events = await terminal_events(job)
                self.assertEqual(events[-1]['state'], 'completed')
                self.assertEqual(broker.snapshot()['backend'], 'hygon')
                self.assertEqual(next(e for e in events if e['type']=='started')['backend'], 'hygon')
            finally:
                await broker.close()
