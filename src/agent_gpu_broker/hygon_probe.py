"""Read-only BW1101 ROCm SMI v2 ABI probe, isolated in a bounded subprocess.

Compatible with Python 3.7 for standalone host inventory diagnostics. The
broker itself still requires Python 3.10+. Any API error, including a process
race or insufficient buffer, fails the whole observation closed.
"""
import ctypes as c
import json
import sys


class ProcessInfo(c.Structure):
    _fields_ = [('process_id', c.c_uint32), ('pasid', c.c_uint32),
                ('vram_usage', c.c_uint64), ('sdma_usage', c.c_uint64),
                ('cu_occupancy', c.c_uint32)]


def probe(library):
    lib = c.CDLL(library)
    signatures = {
        'rsmi_init': [c.c_uint64],
        'rsmi_shut_down': [],
        'rsmi_num_monitor_devices': [c.POINTER(c.c_uint32)],
        'rsmi_compute_process_info_get': [c.POINTER(ProcessInfo), c.POINTER(c.c_uint32)],
        'rsmi_compute_process_gpus_get': [c.c_uint32, c.POINTER(c.c_uint32), c.POINTER(c.c_uint32)],
    }
    for name, arguments in signatures.items():
        function = getattr(lib, name)
        function.argtypes = arguments
        function.restype = c.c_int

    def call(name, *args):
        status = getattr(lib, name)(*args)
        if status != 0:
            raise RuntimeError('{} failed with status {}'.format(name, status))

    call('rsmi_init', 0)
    try:
        count = c.c_uint32()
        call('rsmi_num_monitor_devices', c.byref(count))
        if not 0 < count.value <= 1024:
            raise RuntimeError('invalid device count')
        devices = [{'id': index, 'compute_pids': []} for index in range(count.value)]
        n = c.c_uint32()
        call('rsmi_compute_process_info_get', None, c.byref(n))
        if n.value > 1048576:
            raise RuntimeError('invalid process count')
        # Query again even at zero to detect a process appearing between calls.
        capacity = max(1, n.value)
        processes = (ProcessInfo * capacity)()
        n.value = capacity
        call('rsmi_compute_process_info_get', processes, c.byref(n))
        if n.value > capacity:
            raise RuntimeError('process inventory grew')
        for process in processes[:n.value]:
            if process.process_id <= 0:
                raise RuntimeError('invalid process id')
            indices = (c.c_uint32 * count.value)()
            used = c.c_uint32(count.value)
            call('rsmi_compute_process_gpus_get', process.process_id, indices, c.byref(used))
            if used.value > count.value:
                raise RuntimeError('device mapping grew')
            for index in indices[:used.value]:
                if index >= count.value:
                    raise RuntimeError('unknown device mapping')
                devices[index]['compute_pids'].append(process.process_id)
        return {'schema': 'gpuq.device-probe.v1', 'devices': devices}
    finally:
        call('rsmi_shut_down')


if __name__ == '__main__':
    try:
        print(json.dumps(probe(sys.argv[1] if len(sys.argv) > 1 else '/usr/local/hyhal/lib/librocm_smi64.so')))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
