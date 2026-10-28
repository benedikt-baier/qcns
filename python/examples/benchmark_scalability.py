"""Scaling evidence, not brittle runtime unit tests. Run with --help for options."""
import argparse
import cProfile
import ctypes
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import platform
import pstats
import statistics
import subprocess
import sys
import time
import tracemalloc

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from qcns.python.examples.scaling_workload import LoadConfig, build_workload
from qcns.python.components.qubit.qec import StabilizerPauliCode


class ProcessMemory(ctypes.Structure):
    _fields_ = [('cb', ctypes.c_uint32), ('faults', ctypes.c_uint32)] + [
        (name, ctypes.c_size_t) for name in ('peak_working', 'working', 'peak_paged', 'paged',
                                           'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]


def process_peak_mib() -> float | None:
    if sys.platform != 'win32':
        return None
    kernel = ctypes.WinDLL('kernel32')
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL('psapi')
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessMemory), ctypes.c_uint32]
    record = ProcessMemory()
    record.cb = ctypes.sizeof(record)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(record), record.cb):
        return None
    return record.peak_working / 1024**2


def sample(config: LoadConfig, profiler: cProfile.Profile | None = None) -> dict:
    started = time.perf_counter()
    scenario = build_workload(config)
    setup = time.perf_counter() - started
    started = time.perf_counter()
    if profiler is not None:
        profiler.enable()
    try:
        scenario.simulation.run()
    finally:
        if profiler is not None:
            profiler.disable()
    simulation = time.perf_counter() - started
    started = time.perf_counter()
    results = scenario.validate()
    validation = time.perf_counter() - started
    results.update(setup_seconds=setup, simulation_seconds=simulation,
                   output_validation_seconds=validation, run_and_validation_seconds=simulation + validation)
    results['events_by_type'] = dict(scenario.simulation.scheduled)
    scenario.release_outputs()
    return results


def profile_summary(profiler: cProfile.Profile) -> dict:
    profile = pstats.Stats(profiler)
    stats = profile.stats
    done_calls = sum(values[1] for (_, _, name), values in stats.items() if "'done' of '_asyncio.Task'" in name)
    top = sorted(stats.items(), key=lambda item: item[1][3], reverse=True)[:18]
    self_top = sorted(stats.items(), key=lambda item: item[1][2], reverse=True)[:15]
    task_self = sum(values[2] for (_, _, name), values in stats.items()
                    if any(f"'{method}' of '_asyncio.Task'" in name for method in ('done', 'cancelled', 'exception')))
    scheduler_self = sum(values[2] for (filename, _, name), values in stats.items()
                         if filename.endswith('simulation.py') and name == '_run_events')
    return {'task_done_calls': done_calls, 'task_check_self_seconds': task_self,
            'scheduler_self_seconds': scheduler_self, 'profile_total_seconds': profile.total_tt,
            'top_self': [{'function': f'{Path(location[0]).name}:{location[1]}:{location[2]}',
                         'calls': values[1], 'self_seconds': values[2]} for location, values in self_top],
            'top_cumulative': [
        {'function': f'{Path(location[0]).name}:{location[1]}:{location[2]}',
         'calls': values[1], 'self_seconds': values[2], 'cumulative_seconds': values[3]}
        for location, values in top]}


def worker(config: LoadConfig, repeats: int) -> dict:
    sample(config)  # Warm caches, and validate before collecting timing samples.
    runs = [sample(config) for _ in range(repeats)]
    unprofiled_peak = process_peak_mib()
    gc.collect()
    tracemalloc.start()
    sample(config)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    profiler = cProfile.Profile()
    sample(config, profiler)
    seconds = [run['run_and_validation_seconds'] for run in runs]
    return {'config': asdict(config), 'runs': runs, 'median_seconds': statistics.median(seconds),
            'min_seconds': min(seconds), 'max_seconds': max(seconds),
            'peak_traced_scenario_mib': peak / 1024**2,
            'worker_peak_process_mib': unprofiled_peak,
            'instrumented_worker_peak_process_mib': process_peak_mib(), 'profile': profile_summary(profiler)}


def qec_scaling() -> list[dict]:
    """Synthetic repetition codes isolate exact-state growth, not code quality."""
    rows = []
    for length in (3, 5, 7, 9, 11, 13):
        checks = ['I'*i + 'ZZ' + 'I'*(length-i-2) for i in range(length-1)]
        started = time.perf_counter()
        code = StabilizerPauliCode(f'repetition_{length}', checks, 'X'*length, 'Z'+'I'*(length-1))
        construction = time.perf_counter() - started
        probabilities = np.tile([.99, .01/3, .01/3, .01/3], (length, 1))
        started = time.perf_counter()
        output = code.logical_channel(probabilities)
        elapsed = time.perf_counter() - started
        if not np.isclose(output.sum(), 1.):
            raise AssertionError('QEC channel is not normalized')
        rows.append({'physical_qubits': length, 'syndrome_frame_states': len(code._indices),
                     'precomputed_array_bytes': sum(value.nbytes for value in vars(code).values() if isinstance(value, np.ndarray)),
                     'construct_seconds': construction, 'channel_seconds': elapsed})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', default='20,50,100,200,400')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--supply', choices=['eager', 'prepared', 'ondemand'], default='eager')
    parser.add_argument('--topology', choices=['ba', 'ring'], default='ba')
    parser.add_argument('--noise-time', type=float, default=1.)
    parser.add_argument('--attachment', type=int, default=2)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if args.worker is not None:
        config = LoadConfig(nodes=args.worker, supply=args.supply, topology=args.topology,
                            noise_time=args.noise_time, attachment=args.attachment)
        print(json.dumps(worker(config, args.repeats)))
        return
    results = []
    for nodes in map(int, args.sizes.split(',')):
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(nodes),
                   '--repeats', str(args.repeats), '--supply', args.supply, '--topology', args.topology,
                   '--noise-time', str(args.noise_time), '--attachment', str(args.attachment)]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode:
            results.append({'nodes': nodes, 'status': 'failed', 'error': completed.stderr})
            print(f'{nodes} nodes: FAILED (recorded in report)', flush=True)
            continue
        result = json.loads(completed.stdout)
        results.append(result)
        print(f'{nodes} nodes, {nodes*4//5} requests: {result["median_seconds"]:.4f} s; '
              f'{result["profile"]["task_done_calls"]} Task.done calls', flush=True)
    report = {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
              'processor': platform.processor(), 'network_cases': results, 'qec_scaling': qec_scaling(),
              'notes': 'Runtime excludes setup/imports, includes output validation and retained-half memory noise. '
                       'Memory/profile use separate runs; workers isolate each size. Perfect generation/gates, '
                       'finite link delays, existing DepolarizationError and DephasingError on each memory. '
                       'Despite its name, the existing DepolarizationError implements amplitude damping. '
                       'Profiles have instrumentation overhead; profile percentages are not unprofiled runtime fractions.'}
    component_root = Path(__file__).resolve().parents[1] / 'components'
    report['source_sha256'] = {str(path.relative_to(component_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in component_root.rglob('*.py')}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf8')
    else:
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
