import ctypes as c
import unittest
from unittest.mock import patch
from agent_gpu_broker.hygon_probe import ProcessInfo, probe


class Function:
    def __init__(self, callback):
        self.callback = callback
    def __call__(self, *args):
        return self.callback(*args)


class Library:
    def __init__(self, mapping_error=False):
        self.closed = False
        self.mapping_error = mapping_error
        self.rsmi_init = Function(lambda flags: 0)
        self.rsmi_shut_down = Function(self.close)
        self.rsmi_num_monitor_devices = Function(self.count)
        self.rsmi_compute_process_info_get = Function(self.processes)
        self.rsmi_compute_process_gpus_get = Function(self.mapping)
    def close(self):
        self.closed = True
        return 0
    def count(self, count):
        c.cast(count, c.POINTER(c.c_uint32))[0] = 2
        return 0
    def processes(self, array, count):
        c.cast(count, c.POINTER(c.c_uint32))[0] = 1
        if array is not None:
            array[0].process_id = 123
        return 0
    def mapping(self, pid, array, count):
        if self.mapping_error:
            return 7
        array[0] = 1
        c.cast(count, c.POINTER(c.c_uint32))[0] = 1
        return 0


class HygonProbeTests(unittest.TestCase):
    def test_busy_pid_maps_to_only_its_device(self):
        library = Library()
        with patch('agent_gpu_broker.hygon_probe.c.CDLL', return_value=library):
            value = probe('test')
        self.assertEqual(value['devices'], [{'id':0,'compute_pids':[]},{'id':1,'compute_pids':[123]}])
        self.assertTrue(library.closed)

    def test_mapping_race_is_unknown_and_library_closes(self):
        library = Library(mapping_error=True)
        with patch('agent_gpu_broker.hygon_probe.c.CDLL', return_value=library):
            with self.assertRaisesRegex(RuntimeError, 'status 7'):
                probe('test')
        self.assertTrue(library.closed)
