"""Reproducible raw-swapping traffic for tests and profiling, without timing asserts."""
from collections import Counter
from dataclasses import dataclass

import networkx as nx
import numpy as np

from qcns.python.components.hardware.connection import L3C_Model, PChannel_Model
from qcns.python.components.hardware.memory import PQM_Model
from qcns.python.components.network.node import Node, SEND, RECEIVE
from qcns.python.components.network.qprotocol import L1_EGP, L2_NEC, L3_QProtocol, QProtocolStack
from qcns.python.components.network.router import Router
from qcns.python.components.packet.packet import Packet
from qcns.python.components.qubit.qubit import Qubit, remove_qubits
from qcns.python.components.qubit.error import DepolarizationError, DephasingError
from qcns.python.components.simulation.event import Event
from qcns.python.components.simulation.simulation import Simulation


@dataclass(frozen=True)
class LoadConfig:
    nodes: int = 100
    seed: int = 7
    topology: str = 'ba'
    supply: str = 'eager'
    timeout: float = 1.
    noise_time: float | None = 1.
    attachment: int = 2


class CountedSimulation(Simulation):
    def __init__(self, seed: int = 7) -> None:
        super().__init__(seed=seed)
        self.scheduled: Counter[str] = Counter()
        self.queue_peak = 0

    def schedule_event(self, event: Event) -> None:
        super().schedule_event(event)
        self.scheduled[type(event).__name__] += 1
        self.queue_peak = max(self.queue_peak, len(self._event_queue))


class ClientHeralds(L3_QProtocol):
    required_layers = ()

    def on_qubits_ready(self, neighbor: int, store: int, count: int,
                        request_id: int | None = None) -> None:
        if store == SEND:
            self.host.ready_pairs += count


class LoadClient(Node):
    def __init__(self, node_id: int, sim: Simulation, access: int, destination: int,
                 expected: int, config: LoadConfig) -> None:
        stack = QProtocolStack(l1_qprotocol=L1_EGP(reattempt=False), l2_qprotocol=L2_NEC(),
                               l3_qprotocol=ClientHeralds({}))
        super().__init__(node_id, sim, qprotocols=stack)
        self.access, self.destination, self.expected = access, destination, expected
        self.config = config
        self.ready_pairs = int(config.supply == 'prepared')
        self.sent = False
        self.retained: Qubit | None = None
        self.received: dict[int, Qubit] = {}
        self.delivery_times: dict[int, float] = {}
        self.token: tuple[int, int] | None = None

    def _send_request(self) -> None:
        packet = Packet(self.id, self.access, l3_src=self.id, l3_dst=self.destination,
                        l3_requested=1, l3_needed=1, l3_protocol=1)
        self.token = (self.id, packet.request_id)
        if not self.pr_capture(self.access, SEND, self.token, self.token):
            raise RuntimeError('Source memory extraction failed')
        self.send_packet(packet)
        self.sent = True

    def _accept(self, packet: Packet) -> None:
        if packet.has_l1:
            self.handle_packet(packet)
            return
        if packet.rejected or packet.l3_dst != self.id:
            raise RuntimeError('Raw request rejected or misrouted')
        qubit = self.retrieve_ready_pair(packet.src, RECEIVE)
        if qubit is None or packet.l3_src in self.received:
            raise RuntimeError('Missing output half or duplicate delivery')
        if packet.l3_x_count[0]:
            self.apply_gate('X', qubit)
        if packet.l3_z_count[0]:
            self.apply_gate('Z', qubit)
        self.received[packet.l3_src] = qubit
        self.delivery_times[packet.l3_src] = self._time

    async def run(self) -> None:
        if self.config.supply != 'prepared':
            self.attempt_bell_pairs(self.access, 1)
        while not self.sent or len(self.received) < self.expected:
            if not self.sent and self.ready_pairs:
                self._send_request()
            if self.sent and len(self.received) == self.expected:
                return
            packet = await self.receive_packet(time_out=max(0., self.config.timeout - self._time))
            if packet is None:
                raise RuntimeError(f'Client {self.id} timed out: {len(self.received)}/{self.expected} deliveries')
            self._accept(packet)


class LoadRouter(Router):
    def __init__(self, node_id: int, sim: Simulation, routes: dict[int, int],
                 demand: dict[int, int], supply: str) -> None:
        super().__init__(node_id, sim, routes)
        self.demand, self.supply = demand, supply

    async def run(self) -> None:
        if self.supply == 'eager':
            for neighbor, count in self.demand.items():
                self.attempt_bell_pairs(neighbor, count)
        await super().run()


@dataclass
class LoadScenario:
    simulation: CountedSimulation
    nodes: dict[int, Node]
    clients: list[LoadClient]
    required_pairs: int
    mean_hops: float

    def validate(self) -> dict[str, int | float]:
        phi = np.array([1, 0, 0, 1]) / np.sqrt(2)
        minimum_fidelity = 1.
        for source in self.clients:
            destination = self.nodes[source.destination]
            # Analysis evaluates the retained half when its peer output was ready.
            # It remains in physical memory throughout the actual protocol run.
            if source.retained is None:
                source._time = max(source._time, destination.delivery_times.get(source.id, source._time))
                source.retained = source.pr_retrieve_output(source.token)
            first, last = source.retained, destination.received.get(source.id)
            if first is None or last is None or first.qsystem is not last.qsystem:
                raise AssertionError(f'Wrong or missing entanglement for source {source.id}')
            minimum_fidelity = min(minimum_fidelity, float((phi @ first.state @ phi).real))
            if not np.isclose(np.trace(first.state), 1.) or np.linalg.eigvalsh(first.state).min() < -1e-9:
                raise AssertionError('Delivered an invalid density matrix')
        if self.clients[0].config.noise_time is None and minimum_fidelity < 1 - 1e-9:
            raise AssertionError(f'Incorrect ideal output state: fidelity={minimum_fidelity}')
        leftovers = sum(len(memory) for node in self.nodes.values() for directions in node._memory.values()
                        for memory in directions.values())
        return {'delivered': len(self.clients), 'minimum_fidelity': minimum_fidelity,
                'required_raw_pairs': self.required_pairs, 'remaining_memory_halves': leftovers,
                'mean_path_edges': self.mean_hops,
                'scheduled_events': sum(self.simulation.scheduled.values()),
                'event_queue_peak': self.simulation.queue_peak}

    def release_outputs(self) -> None:
        outputs = [node.retained for node in self.clients if node.retained is not None]
        outputs += [qubit for node in self.clients for qubit in node.received.values()]
        remove_qubits(outputs)


def build_workload(config: LoadConfig) -> LoadScenario:
    if config.nodes < 15 or config.nodes % 5:
        raise ValueError('Use a node count >=15 divisible by five (80% clients)')
    if config.supply not in ('prepared', 'eager', 'ondemand'):
        raise ValueError('supply must be prepared, eager or ondemand')
    if config.topology not in ('ba', 'ring'):
        raise ValueError('topology must be ba or ring')
    np.random.seed(config.seed)  # Existing backend uses the global stream.
    rng = np.random.default_rng(config.seed)
    routers = config.nodes // 5
    graph = nx.barabasi_albert_graph(routers, min(config.attachment, routers-1), seed=config.seed)
    if config.topology == 'ring':
        graph = nx.cycle_graph(routers)
    client_ids = list(range(routers, config.nodes))
    access = {client: int(rng.integers(routers)) for client in client_ids}
    graph.add_edges_from(access.items())
    destinations = {client: int(rng.choice([other for other in client_ids if other != client]))
                    for client in client_ids}
    expected = Counter(destinations.values())
    routing = {}
    for router in range(routers):
        shortest = nx.single_source_shortest_path(graph, router)
        routing[router] = {client: shortest[client][1] for client in client_ids}
    paths = {}
    for client, destination in destinations.items():
        path = [client, access[client]]
        while path[-1] != destination:
            path.append(routing[path[-1]][destination])
        paths[client] = path
    demand = Counter(edge for path in paths.values() for edge in zip(path, path[1:]))
    sim = CountedSimulation(config.seed)
    nodes: dict[int, Node] = {}
    for router in range(routers):
        routes = routing[router]
        needs = {peer: count for (owner, peer), count in demand.items() if owner == router}
        nodes[router] = LoadRouter(router, sim, routes, needs, config.supply)
    clients = [LoadClient(client, sim, access[client], destinations[client], expected[client], config)
               for client in client_ids]
    nodes.update({client.id: client for client in clients})
    for left, right in graph.edges:
        capacity = max(16, demand[left, right], demand[right, left])
        # On-demand is a diagnostic: expose surplus generation without capacity crashes.
        errors = [] if config.noise_time is None else [DepolarizationError(config.noise_time), DephasingError(config.noise_time)]
        memory = PQM_Model(size=-1 if config.supply == 'ondemand' else capacity, extraction_mode='fifo', memory_errors=errors)
        length = .1 if left in access or right in access else 1.
        model = L3C_Model(fidelity=1., duration=1e-4, pchannel=PChannel_Model(length=length, data_rate=1e8))
        nodes[left].set_eqs_connection(nodes[right], model, model, memory, memory)
    if config.supply == 'prepared':
        for (left, right), count in demand.items():
            for _ in range(count):
                nodes[left]._connections['eqs'][right].success_creation(0.)
    return LoadScenario(sim, nodes, clients, sum(demand.values()),
                        float(np.mean([len(path)-1 for path in paths.values()])))
