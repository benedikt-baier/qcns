"""Joint Pauli trajectories for tensor products of the [[5,1,3]] code.

Run from the repository root. Prints reproducible kernel timing and memory
measurements; no network generation, gates or communication are scheduled.
"""
from collections.abc import Callable
import json
import platform
from pathlib import Path
import statistics
import sys
import time
import tracemalloc

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from qcns.python.components.qubit.qec import StabilizerPauliCode


def product_code(k: int, shots: int = 5000) -> StabilizerPauliCode:
    """Define [[5k,k,3]] with independent five-qubit minimum-weight decoders."""
    generators = ('XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ')
    corrections = {(0, 0, 0, 0): 'IIIII'}
    for position in range(5):
        for pauli in 'XYZ':
            bits = tuple(int(row[position] not in ('I', pauli)) for row in generators)
            corrections[bits] = 'I'*position + pauli + 'I'*(4-position)
    decoder: Callable[[tuple[int, ...]], str] = lambda bits: ''.join(
        corrections[bits[4*i:4*i+4]] for i in range(k))
    checks = tuple('I'*(5*i) + row + 'I'*(5*(k-i-1)) for i in range(k) for row in generators)
    xs = tuple('I'*(5*i) + 'XXXXX' + 'I'*(5*(k-i-1)) for i in range(k))
    zs = tuple('I'*(5*i) + 'ZZZZZ' + 'I'*(5*(k-i-1)) for i in range(k))
    return StabilizerPauliCode(f'five_qubit_product_{k}', checks, xs, zs, distance=3,
                               decoder=decoder, method='sample', shots=shots)


def run(code: StabilizerPauliCode, noise: np.ndarray) -> int:
    """Count blocks with any residual logical error, keeping joint outcomes."""
    return sum(int(np.count_nonzero(np.any(batch != 0, axis=1)))
               for batch in code.iter_logical_errors(noise, rng=np.random.default_rng(7)))


def main() -> None:
    rows = []
    for k in (1, 2, 8, 32):
        start = time.perf_counter()
        code = product_code(k)
        construction = time.perf_counter()-start
        noise = np.tile([.97, .01, .01, .01], (code.length, 1))
        run(code, noise)
        times = []
        for _ in range(3):
            start = time.perf_counter()
            failures = run(code, noise)
            times.append(time.perf_counter()-start)
        tracemalloc.start()
        traced = product_code(k)
        run(traced, noise)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append(dict(n=code.length, k=k, distance=3, shots=code.shots,
                         construction_seconds=construction, median_seconds=statistics.median(times),
                         peak_traced_bytes=peak, blocks_with_logical_errors=failures))
    print(json.dumps(dict(python=sys.version, numpy=np.__version__, platform=platform.platform(),
                          processor=platform.processor(), seed=7, cases=rows,
                          assumptions='Independent Pauli noise [0.97,0.01,0.01,0.01], ideal correction. '
                          'Median of three warmed evaluations; construction excluded from timing. '
                          'Traced peak includes fresh construction/evaluation, not whole-process RSS. '
                          'Failures are residual logical errors, not heralded protocol failures.'), indent=2))


if __name__ == '__main__':
    main()
