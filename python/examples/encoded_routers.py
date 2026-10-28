"""Run: python python/examples/encoded_routers.py.

Ideal effective graph interfaces; independent Werner links, perfect storage.
Times are seconds, classical link lengths km. No preloaded Bell pairs.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from qcns.python.components.hardware.connection import L3C_Model, PChannel_Model
from qcns.python.components.hardware.memory import PQM_Model
from qcns.python.components.network.client import EncodedClient
from qcns.python.components.network.router import EncodedRouter
from qcns.python.components.simulation.simulation import Simulation


def main() -> None:
    np.random.seed(7)  # Existing quantum backend RNG; not independent streams.
    simulation = Simulation(seed=7)
    routes = [{j: i-1 if j < i else i+1 for j in range(6) if j != i} for i in range(6)]
    source = EncodedClient(0, simulation, routes[0], destination=5, code='five_qubit', fidelity=.97)
    nodes = [source, *[EncodedRouter(i, simulation, routes[i]) for i in range(1, 5)],
             EncodedClient(5, simulation, routes[5])]
    for left, right in zip(nodes, nodes[1:]):
        model = L3C_Model(fidelity=.99, duration=1e-4, pchannel=PChannel_Model(length=1))
        memory = PQM_Model(size=16, extraction_mode='fifo')
        left.set_eqs_connection(right, model, model, memory, memory)
    simulation.run()
    result = source.result
    if result.status != 'complete':
        raise RuntimeError(result.reason)
    first = source.encoded_retrieve_output(result.outputs[0])
    last = nodes[-1].encoded_retrieve_output(result.outputs[0])
    assert first.qsystem is last.qsystem
    phi = np.array([1, 0, 0, 1]) / np.sqrt(2)
    actual = float((phi @ first.state @ phi).real)
    assert actual >= .97
    entries = [node.pr_requests[result.key] for node in nodes]
    print(f'{result.status}: conditional Phi+ fidelity={actual:.6f}, time={result.finished_at:.6g} s')
    print(f'Raw pairs={sum(entry.raw_pairs for entry in entries)}, '
          f'encoded blocks={sum(entry.encoded_blocks for entry in entries)}, '
          f'graph resource qubits prepared={sum(entry.graph_resource_qubits for entry in entries)}')
    for node in nodes:
        node.pr_archive(result.key)


if __name__ == '__main__':
    main()
