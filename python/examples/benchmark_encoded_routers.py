"""Benchmark the actual six-node example: python python/examples/benchmark_encoded_routers.py.

Wall-clock times are separate from simulated network time. Memory tracing runs
separately so instrumentation does not inflate the reported runtime samples.
"""
import contextlib
import io
import json
import platform
from pathlib import Path
import runpy
import statistics
import subprocess
import sys
import time
import tracemalloc
from unittest.mock import patch


def summary(samples: list[float]) -> dict[str, float]:
    return {'minimum_seconds': min(samples), 'median_seconds': statistics.median(samples),
            'maximum_seconds': max(samples)}


def main() -> None:
    example = Path(__file__).with_name('encoded_routers.py')
    started = time.perf_counter()
    namespace = runpy.run_path(str(example))
    import_seconds = time.perf_counter() - started
    simulation = namespace['Simulation']
    original_run = simulation.run
    simulation_times: list[float] = []
    scenario_times: list[float] = []

    def timed_run(instance: object, end_time: float = 0) -> None:
        started = time.perf_counter()
        original_run(instance, end_time)
        simulation_times.append(time.perf_counter() - started)

    output = io.StringIO()
    with patch.object(simulation, 'run', timed_run), contextlib.redirect_stdout(output):
        for _ in range(21):
            started = time.perf_counter()
            namespace['main']()
            scenario_times.append(time.perf_counter() - started)

    with contextlib.redirect_stdout(io.StringIO()):
        tracemalloc.start()
        namespace['main']()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    process_times: list[float] = []
    for _ in range(5):
        started = time.perf_counter()
        subprocess.run([sys.executable, str(example)], check=True, capture_output=True, text=True)
        process_times.append(time.perf_counter() - started)

    print(json.dumps({
        'python': sys.version,
        'numpy': namespace['np'].__version__,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'workload': 'two clients, four encoded routers, five_qubit, one output, seed 7',
        'example_import_seconds': import_seconds,
        'first_simulation_seconds': simulation_times[0],
        'first_scenario_seconds': scenario_times[0],
        'warm_simulation_20_runs': summary(simulation_times[1:]),
        'warm_scenario_20_runs': summary(scenario_times[1:]),
        'fresh_process_5_runs': summary(process_times),
        'separate_run_peak_traced_allocations_mib': peak / 1024**2,
        'validation': output.getvalue().splitlines()[:2],
    }, indent=2))


if __name__ == '__main__':
    main()
