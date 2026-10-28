"""One configuration and public API for three routing modes.

Run: python python/examples/unified_experiment.py --mode raw
Also accepts purification or encoded. Times are seconds, lengths kilometers.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from qcns.api import (
    ExperimentConfig, HardwareConfig, ProtocolConfig, TrafficRequest, run_experiment,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('raw', 'purification', 'encoded'), default='raw')
    parser.add_argument('--backend', choices=('python', 'cpp'), default='python')
    parser.add_argument('--backend-threads', type=int, default=1)
    parser.add_argument('--save', type=Path, help='Save the effective configuration for replay')
    arguments = parser.parse_args()
    config = ExperimentConfig(
        seed=7,
        backend=arguments.backend,
        backend_threads=arguments.backend_threads,
        protocol=ProtocolConfig(mode=arguments.mode),
        hardware=HardwareConfig(fidelity=.99, dephasing_time=10.),
        requests=(TrafficRequest(4, 5, fidelity=.97),),
    )
    result = run_experiment(config)
    if arguments.save:
        ExperimentConfig.from_dict(result.configuration).save(arguments.save)
    for request in result.requests:
        print(f'{arguments.mode}: {request.status}, {request.delivered_pairs} delivered pair(s), '
              f'{request.finished_at:.6g} s; estimated fidelity={request.estimated_fidelity}; '
              f'{request.reason}')


if __name__ == '__main__':
    main()
