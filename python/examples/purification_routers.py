"""Run: python python/examples/purification_routers.py.

Two clients and four PRs; on-demand Werner pairs, ideal local gates and memory,
finite classical propagation delays. Times are seconds and channel lengths km.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from qcns.python.components.hardware.connection import L3C_Model, PChannel_Model
from qcns.python.components.hardware.memory import PQM_Model
from qcns.python.components.network.router import PurificationRouter
from qcns.python.components.network.client import PurificationClient
from qcns.python.components.simulation.simulation import Simulation


def main() -> None:
    # The existing quantum backend uses NumPy's process-global random generator.
    simulation = Simulation(seed=7)
    routes = [{j: i - 1 if j < i else i + 1 for j in range(6) if j != i}
              for i in range(6)]
    source = PurificationClient(0, simulation, routes[0], destination=5, fidelity=.97)
    routers = [PurificationRouter(i, simulation, routes[i]) for i in range(1, 5)]
    destination = PurificationClient(5, simulation, routes[5])
    nodes = [source, *routers, destination]
    for left, right in zip(nodes, nodes[1:]):
        connection = L3C_Model(fidelity=.99, duration=1e-4,
                               pchannel=PChannel_Model(length=1))
        memory = PQM_Model(size=16, extraction_mode='fifo')
        left.set_eqs_connection(right, connection, connection, memory, memory)
    simulation.run()
    result = source.result
    if result.status != 'complete':
        raise RuntimeError(result.reason)
    first = source.pr_retrieve_output(result.outputs[0])
    last = destination.pr_retrieve_output(result.outputs[0])
    assert first.qsystem is last.qsystem
    phi = np.array([1, 0, 0, 1]) / np.sqrt(2)
    # Ground-truth state inspection is analysis only, never a routing decision.
    actual_fidelity = float((phi @ first.state @ phi).real)
    assert actual_fidelity >= .97
    entries = [node.pr_requests[result.key] for node in nodes]
    print(f'{result.status}: conditional Phi+ fidelity={actual_fidelity:.6f}, '
          f'estimate={result.estimated_fidelity:.6f}, time={result.finished_at:.6g} s')
    print(f'Raw pairs={sum(entry.raw_pairs for entry in entries)}, '
          f'purification attempts={sum(entry.purification_attempts for entry in entries)}, '
          f'rejected attempts={sum(entry.failed_attempts for entry in entries)}')
    for node in nodes:
        node.pr_archive(result.key)


if __name__ == '__main__':
    main()
