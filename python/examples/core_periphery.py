"""Two clients and four PRs, assembled through the standardized topology API.

Run: python python/examples/core_periphery.py
On-demand effective Werner-pair generation, ideal local gates, 10-second
dephasing memory, finite source/classical delays. Times are seconds; link lengths km.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from qcns.python.components.network.utils import (
    Network, NodeContext, LinkConfig, create_topology, build_network, summarize_topology,
)
from qcns.python.components.network.router import PurificationRouter
from qcns.python.components.network.client import PurificationClient
from qcns.python.components.simulation.simulation import Simulation
from qcns.python.components.hardware.connection import L3C_Model, PChannel_Model
from qcns.python.components.hardware.memory import PQM_Model
from qcns.python.components.qubit.error import DephasingError


def router_factory(context: NodeContext) -> PurificationRouter:
    return PurificationRouter(context.node_id, context.simulation, context.routing_table, **context.options)


def client_factory(context: NodeContext) -> PurificationClient:
    return PurificationClient(context.node_id, context.simulation, context.routing_table, **context.options)


def build_example(seed: int = 7) -> Network:
    topology = create_topology(core='path_graph', num_routers=4, num_clients=2,
                               attachment='explicit', client_attachments={4: 0, 5: 3}, seed=seed)
    connection = L3C_Model(fidelity=.99, duration=1e-4,
                           pchannel=PChannel_Model(length=1, data_rate=1e8))
    memory = PQM_Model(size=32, extraction_mode='fifo', memory_errors=[DephasingError(10.)])
    links = LinkConfig(connection, connection, memory, memory)
    return build_network(simulation=Simulation(seed=seed), topology=topology,
                         router_factory=router_factory, client_factory=client_factory,
                         core_links=links, access_links=links,
                         node_options={4: {'destination': 5, 'fidelity': .97}})


def main() -> None:
    network = build_example()
    print(summarize_topology(network.topology, include_paths=True))
    network.simulation.run()
    source, destination = network.clients[4], network.clients[5]
    result = source.result
    if result.status != 'complete':
        raise RuntimeError(result.reason)
    first = source.pr_retrieve_output(result.outputs[0])
    second = destination.pr_retrieve_output(result.outputs[0])
    if first.qsystem is not second.qsystem:
        raise RuntimeError('Output halves do not share a state')
    phi = np.array([1, 0, 0, 1]) / np.sqrt(2)
    fidelity = float((phi @ first.state @ phi).real)
    print(f'{result.status}: conditional Phi+ fidelity={fidelity:.6f}; estimate={result.estimated_fidelity:.6f}; '
          f'finished_at={result.finished_at:.6g} s')
    for node in network.nodes.values():
        node.pr_archive(result.key)


if __name__ == '__main__':
    main()
