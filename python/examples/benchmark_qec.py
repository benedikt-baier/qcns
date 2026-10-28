"""QEC kernel benchmark; setup/imports and traced memory are timed separately."""
import argparse
import hashlib
import importlib.util
import json
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qcns.python.components.qubit.qec import StabilizerPauliCode


def repetition_decoder(bits: tuple[int, ...]) -> str:
    errors = [0]
    for bit in bits:
        errors.append(errors[-1] ^ bit)
    if sum(errors) > len(errors) // 2:
        errors = [1 - value for value in errors]
    return ''.join('X' if value else 'I' for value in errors)


def construct(factory: Callable, name: str, length: int, method: str, shots: int) -> StabilizerPauliCode:
    if name != 'repetition':
        return factory(name, method=method, shots=shots, seed=7)
    checks = tuple('I'*i + 'ZZ' + 'I'*(length-i-2) for i in range(length-1))
    return factory(f'repetition_{length}', checks, 'X'*length, 'Z'+'I'*(length-1),
                   decoder=repetition_decoder, method=method, shots=shots, seed=7)


def measure(factory: Callable, name: str, length: int, method: str, shots: int, repeats: int) -> dict:
    started = time.perf_counter()
    code = construct(factory, name, length, method, shots)
    construction = time.perf_counter() - started
    noise = np.tile([.97, .01, .01, .01], (length, 1))
    code.logical_channel(noise, rng=np.random.default_rng(7))
    loops = 500 if method == 'exact' else 1
    times = []
    for _ in range(repeats):
        rng = np.random.default_rng(7)
        started = time.perf_counter()
        for _ in range(loops):
            output = code.logical_channel(noise, rng=rng)
        times.append((time.perf_counter() - started) / loops)
    tracemalloc.start()
    traced = construct(factory, name, length, method, shots)
    traced.logical_channel(noise, rng=np.random.default_rng(7))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {'construction_seconds': construction, 'median_seconds': statistics.median(times),
            'seconds': times, 'peak_traced_bytes_construct_and_evaluate': peak,
            'precomputed_array_bytes': sum(value.nbytes for value in vars(code).values() if isinstance(value, np.ndarray)),
            'output': output.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, help='Optional saved pre-change qec.py for paired comparison')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    factories = {'current': StabilizerPauliCode}
    if args.reference is not None:
        spec = importlib.util.spec_from_file_location('qec_reference', args.reference)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        factories['reference'] = module.StabilizerPauliCode
    cases = [('five_qubit', 5, 'exact', 20000), ('steane', 7, 'exact', 20000),
             ('five_qubit', 5, 'sample', 20000), ('steane', 7, 'sample', 20000),
             ('repetition', 65, 'sample', 10000), ('repetition', 257, 'sample', 2000)]
    rows = []
    for name, length, method, shots in cases:
        row = {'code': name, 'length': length, 'method': method, 'shots': shots}
        for label, factory in factories.items():
            row[label] = measure(factory, name, length, method, shots, args.repeats)
        if 'reference' in row:
            np.testing.assert_allclose(row['current']['output'], row['reference']['output'], atol=1e-12, rtol=0)
            row['speedup'] = row['reference']['median_seconds'] / row['current']['median_seconds']
        rows.append(row)
        print(f'{name} n={length} {method}: {row["current"]["median_seconds"]:.6f} s', flush=True)
    source = Path(__file__).resolve().parents[1] / 'components' / 'qubit' / 'qec.py'
    report = {'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
              'processor': platform.processor(), 'seed': 7, 'cases': rows,
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest() if args.reference else None,
              'notes': 'Independent I/X/Y/Z noise [0.97,0.01,0.01,0.01], ideal syndrome/correction. '
                       'Kernel timing excludes construction; separate traced memory includes construction. '
                       'Traced memory is not whole-process RSS. Repetition codes demonstrate scaling, not general Pauli protection.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
