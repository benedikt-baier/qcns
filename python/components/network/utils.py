"""Core/periphery topology, hardware assembly and independent traffic generation.

Graph node labels are translated to stable integer simulator IDs in insertion
order. Clients are single-connected endpoints, never transit routers. Durations
are seconds; physical link units are defined by the supplied QCNS models.
"""
from collections import Counter
from collections.abc import Callable, Hashable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral, Real
from typing import Any

import networkx as nx
import numpy as np

from qcns.python.components.network.node import Node
from qcns.python.components.simulation.simulation import Simulation
from qcns.python.components.simulation.timing import EventTimings
from qcns.python.components.hardware.connection import EQC_Model, L3C_Model
from qcns.python.components.hardware.memory import PQM_Model, LQM_Model

__all__ = [
    'Topology', 'TopologySummary', 'NodeContext', 'LinkConfig', 'Network', 'TrafficRequest',
    'create_core', 'create_topology', 'build_network', 'generate_requests',
    'validate_topology', 'summarize_topology', 'create_graph', 'calculate_routing_table',
    'get_access_routers', 'create_single_traffic_matrix', 'create_multi_traffic_matrix',
    'calculate_single_packet_count', 'calculate_multi_packet_count',
    'add_attributes_to_node', 'add_attributes_to_nodes', 'add_attributes_to_link',
    'add_attributes_to_links', 'set_connection', 'set_l3_connection',
]

CORE_GRAPH_FUNCTIONS = {
    name: getattr(nx, name) for name in (
        'complete_graph', 'cycle_graph', 'circulant_graph', 'path_graph', 'star_graph',
        'wheel_graph', 'grid_2d_graph', 'fast_gnp_random_graph', 'newman_watts_strogatz_graph',
        'barabasi_albert_graph', 'dual_barabasi_albert_graph', 'extended_barabasi_albert_graph')
}
_RANDOM_CORES = {'fast_gnp_random_graph', 'newman_watts_strogatz_graph', 'barabasi_albert_graph',
                 'dual_barabasi_albert_graph', 'extended_barabasi_albert_graph'}
Weight = str | Callable[[int, int, Mapping[str, Any]], float] | None
MemoryModel = PQM_Model | LQM_Model


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return float(value)


def _random(seed: int | None, rng: np.random.Generator | None = None) -> np.random.Generator:
    if seed is not None and rng is not None:
        raise ValueError('Supply seed or rng, not both')
    return np.random.default_rng(seed) if rng is None else rng


def _simple_graph(graph: nx.Graph) -> None:
    if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
        raise ValueError('Supply a simple undirected NetworkX graph')
    if nx.number_of_selfloops(graph):
        raise ValueError('Self-loops are not supported')


@dataclass
class Topology:
    graph: nx.Graph
    routers: tuple[int, ...]
    clients: tuple[int, ...]
    access_routers: dict[int, int]
    original_to_id: dict[Hashable, int]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def core_graph(self) -> nx.Graph:
        """Independent copy of the router-only graph."""
        return deepcopy(self.graph.subgraph(self.routers).copy())


@dataclass(frozen=True)
class TopologySummary:
    routers: int
    clients: int
    core_links: int
    access_links: int
    degree_distribution: dict[int, int]
    clients_per_router: dict[int, int]
    mean_client_hops: float | None
    max_client_hops: int | None


def create_core(kind: str, num_routers: int, *, seed: int | None = None,
                shape: tuple[int, int] | None = None, **parameters: Any) -> nx.Graph:
    """Generate exactly num_routers. Grid requires shape=(rows, columns).

    Other keyword parameters follow the selected NetworkX generator. Random
    presets take a local seed; disconnected draws are not silently repaired.
    """
    count = _integer(num_routers, 'num_routers', 1)
    if kind not in CORE_GRAPH_FUNCTIONS:
        raise ValueError(f'Unknown core {kind!r}; choose from {tuple(CORE_GRAPH_FUNCTIONS)}')
    if kind == 'grid_2d_graph':
        if shape is None or len(shape) != 2 or any(_integer(v, 'grid dimension', 1) < 1 for v in shape):
            raise ValueError('grid_2d_graph requires shape=(rows, columns)')
        if shape[0] * shape[1] != count:
            raise ValueError('Grid shape must contain exactly num_routers nodes')
        return nx.grid_2d_graph(*shape, **parameters)
    if shape is not None:
        raise ValueError('shape is only supported for grid_2d_graph')
    if kind in _RANDOM_CORES:
        parameters['seed'] = int(np.random.SeedSequence(seed).entropy)
    graph = CORE_GRAPH_FUNCTIONS[kind](count-1 if kind == 'star_graph' else count, **parameters)
    if len(graph) != count:
        raise ValueError('Core generator did not create num_routers nodes')
    return graph


def create_topology(*, core: nx.Graph | str | Callable[..., nx.Graph], num_clients: int = 0,
                    num_routers: int | None = None, core_parameters: Mapping[str, Any] | None = None,
                    attachment: str = 'uniform', client_attachments: Mapping[int, Hashable] | None = None,
                    seed: int | None = None) -> Topology:
    """Describe a core with leaf clients; no simulator nodes or traffic created.

    A custom core factory receives seed=... plus core_parameters. Explicit
    attachments map new client IDs to ORIGINAL core labels. IDs are assigned
    0..R-1 to routers in core insertion order and R..R+C-1 to clients.
    """
    count = _integer(num_clients, 'num_clients')
    root = np.random.SeedSequence(seed)
    core_stream, access_stream = root.spawn(2)
    core_seed = int(core_stream.generate_state(1)[0])
    parameters = dict(core_parameters or {})
    if isinstance(core, str):
        graph = create_core(core, num_routers, seed=core_seed, **parameters)
    elif callable(core):
        graph = core(seed=core_seed, **parameters)
    else:
        if parameters:
            raise ValueError('core_parameters are only for generators')
        graph = core
    _simple_graph(graph)
    if not len(graph) or not nx.is_connected(graph):
        raise ValueError('The router core must be nonempty and connected')
    if num_routers is not None and len(graph) != _integer(num_routers, 'num_routers', 1):
        raise ValueError('num_routers does not match the supplied core')
    mapping = {label: index for index, label in enumerate(graph)}
    full = nx.relabel_nodes(deepcopy(graph), mapping, copy=True)
    routers = tuple(range(len(graph)))
    clients = tuple(range(len(graph), len(graph)+count))
    rng = np.random.default_rng(access_stream)
    if attachment not in ('uniform', 'balanced', 'explicit'):
        raise ValueError('attachment must be uniform, balanced or explicit')
    if attachment == 'explicit':
        if client_attachments is None or set(client_attachments) != set(clients):
            raise ValueError('Explicit attachments must specify every generated client ID exactly once')
        if any(label not in mapping for label in client_attachments.values()):
            raise ValueError('Attachments must reference original core labels')
        access = {client: mapping[client_attachments[client]] for client in clients}
    else:
        if client_attachments is not None:
            raise ValueError('client_attachments requires attachment="explicit"')
        assignments = rng.integers(len(routers), size=count) if attachment == 'uniform' else np.resize(rng.permutation(routers), count)
        access = {client: int(router) for client, router in zip(clients, assignments)}
    for label, identifier in mapping.items():
        full.nodes[identifier].update(role='router', original_label=label)
    for client in clients:
        full.add_node(client, role='client')
        full.add_edge(client, access[client], kind='access')
    for left, right in graph.edges:
        full.edges[mapping[left], mapping[right]]['kind'] = 'core'
    result = Topology(full, routers, clients, access, mapping,
                      {'seed': root.entropy, 'core_seed': core_seed, 'attachment': attachment,
                       'core': core if isinstance(core, str) else 'custom', 'core_parameters': deepcopy(parameters),
                       'networkx_version': nx.__version__, 'numpy_version': np.__version__})
    validate_topology(result)
    return result


def validate_topology(topology: Topology) -> None:
    graph = topology.graph
    _simple_graph(graph)
    routers, clients = set(topology.routers), set(topology.clients)
    if len(routers) != len(topology.routers) or len(clients) != len(topology.clients):
        raise ValueError('Router and client IDs must be unique')
    if not routers or routers & clients or routers | clients != set(graph):
        raise ValueError('Router/client roles must partition the graph with a nonempty core')
    for identifier in graph:
        _integer(identifier, 'node ID')
        expected = 'router' if identifier in routers else 'client'
        if graph.nodes[identifier].get('role') != expected:
            raise ValueError(f'Node {identifier} has an inconsistent role')
    if not nx.is_connected(graph.subgraph(routers)):
        raise ValueError('Router core must be connected')
    if set(topology.original_to_id.values()) != routers or len(topology.original_to_id) != len(routers):
        raise ValueError('Original core labels must map one-to-one to routers')
    if get_access_routers(graph, topology.clients) != topology.access_routers:
        raise ValueError('access_routers disagrees with the graph')


def summarize_topology(topology: Topology, *, include_paths: bool = False) -> TopologySummary:
    """Path statistics are opt-in: O(C*(V+E)), over unordered client pairs."""
    validate_topology(topology)
    total_hops, pair_count, maximum = 0, 0, 0
    if include_paths:
        for index, client in enumerate(topology.clients):
            distances = nx.single_source_shortest_path_length(topology.graph, client)
            for other in topology.clients[index+1:]:
                hops = distances[other]
                total_hops += hops
                pair_count += 1
                maximum = max(maximum, hops)
    load = Counter(topology.access_routers.values())
    return TopologySummary(len(topology.routers), len(topology.clients), topology.core_graph.number_of_edges(),
                           len(topology.clients), dict(Counter(dict(topology.graph.degree()).values())),
                           {router: load[router] for router in topology.routers},
                           total_hops / pair_count if pair_count else None, maximum if pair_count else None)


def calculate_routing_table(graph: nx.Graph, routers: Sequence[int], clients: Sequence[int],
                            weight: Weight = None) -> dict[int, dict[int, int]]:
    """Next hops from every supplied source to every supplied destination.

    One shortest-path tree per destination, avoiding inconsistent tie choices.
    Supports undirected graphs and nonnegative finite static weights. Missing
    named weights default to one, following NetworkX. No silent unreachable routes.
    """
    _simple_graph(graph)
    sources, destinations = tuple(routers), tuple(clients)
    if not set(sources + destinations) <= set(graph):
        raise ValueError('Routing sources and destinations must exist in the graph')
    weighted = graph
    if weight is not None:
        weighted = graph.copy()
        for left, right, attributes in graph.edges(data=True):
            cost = weight(left, right, attributes) if callable(weight) else attributes.get(weight, 1)
            weighted.edges[left, right]['_qcns_cost'] = _nonnegative(cost, 'routing weight')
    routes = {source: {} for source in sources}
    for destination in destinations:
        paths = (nx.single_source_shortest_path(weighted, destination) if weight is None else
                 nx.single_source_dijkstra_path(weighted, destination, weight='_qcns_cost'))
        for source in sources:
            if source not in paths:
                raise ValueError(f'No route from {source} to {destination}')
            if source != destination:
                routes[source][destination] = paths[source][-2]
    return routes


@dataclass(frozen=True)
class NodeContext:
    node_id: int
    simulation: Simulation
    role: str
    routing_table: dict[int, int]
    access_router: int | None
    attributes: dict[str, Any]
    options: dict[str, Any]


@dataclass
class LinkConfig:
    """QCNS models for both generation directions; no inferred link physics.

    Forward is lower-ID to higher-ID. Memory models follow set_eqs_connection's
    sending-direction convention. Defaults are L3 effective generation with
    default PQM memory; supply explicit models for a physical experiment.
    """
    forward: EQC_Model = field(default_factory=L3C_Model)
    reverse: EQC_Model = field(default_factory=L3C_Model)
    forward_memory: MemoryModel = field(default_factory=PQM_Model)
    reverse_memory: MemoryModel = field(default_factory=PQM_Model)

    def validate(self) -> None:
        if not isinstance(self.forward, EQC_Model) or not isinstance(self.reverse, EQC_Model):
            raise TypeError('Link directions require EQC_Model configurations')
        if not isinstance(self.forward_memory, (PQM_Model, LQM_Model)) or not isinstance(self.reverse_memory, (PQM_Model, LQM_Model)):
            raise TypeError('Link memory requires PQM_Model or LQM_Model configurations')

    def connect(self, left: Node, right: Node) -> None:
        self.validate()
        models = deepcopy((self.forward, self.reverse, self.forward_memory, self.reverse_memory))
        left.set_eqs_connection(right, *models)


@dataclass
class Network:
    simulation: Simulation
    topology: Topology
    nodes: dict[int, Node]
    routing_tables: dict[int, dict[int, int]]

    @property
    def routers(self) -> dict[int, Node]:
        return {identifier: self.nodes[identifier] for identifier in self.topology.routers}

    @property
    def clients(self) -> dict[int, Node]:
        return {identifier: self.nodes[identifier] for identifier in self.topology.clients}


def build_network(*, simulation: Simulation, topology: Topology,
                  router_factory: Callable[[NodeContext], Node], client_factory: Callable[[NodeContext], Node],
                  core_links: LinkConfig | None = None, access_links: LinkConfig | None = None,
                  link_overrides: Mapping[tuple[int, int], LinkConfig] | None = None,
                  node_options: Mapping[int, Mapping[str, Any]] | None = None,
                  node_timings: Mapping[int, EventTimings] | None = None,
                  routing_metric: Weight = None) -> Network:
    """Create nodes via factories, then wire models through Node's public API.

    Factories receive context and must return a Node registered in this simulation
    with the requested ID. They own protocol selection and interpret options.
    Topology attributes are metadata, never blindly forwarded as constructor args.
    """
    topology = deepcopy(topology)
    validate_topology(topology)
    ids = set(topology.graph)
    if ids & set(simulation.hosts):
        raise ValueError('Topology node IDs collide with hosts already in the simulation')
    for values in (node_options or {}, node_timings or {}):
        if not set(values) <= ids:
            raise ValueError('Node overrides reference unknown IDs')
    if any(not isinstance(value, EventTimings) for value in (node_timings or {}).values()):
        raise TypeError('node_timings values must be EventTimings')
    overrides = {}
    for edge, config in (link_overrides or {}).items():
        if len(edge) != 2 or not topology.graph.has_edge(*edge):
            raise ValueError(f'Link override {edge} does not exist')
        key = tuple(sorted(edge))
        if key in overrides:
            raise ValueError(f'Duplicate undirected override for {key}')
        overrides[key] = config
    core_links = LinkConfig() if core_links is None else core_links
    access_links = LinkConfig() if access_links is None else access_links
    configs = {tuple(sorted((left, right))): overrides.get(tuple(sorted((left, right))),
               core_links if left in topology.routers and right in topology.routers else access_links)
               for left, right in topology.graph.edges}
    if any(not isinstance(config, LinkConfig) for config in configs.values()):
        raise TypeError('All link configurations must be LinkConfig instances')
    for config in configs.values():
        config.validate()
    routes = calculate_routing_table(topology.graph, tuple(topology.graph), tuple(topology.graph), routing_metric)
    nodes = {}
    for identifier in topology.routers + topology.clients:
        role = topology.graph.nodes[identifier]['role']
        context = NodeContext(identifier, simulation, role, dict(routes[identifier]),
                              topology.access_routers.get(identifier), deepcopy(dict(topology.graph.nodes[identifier])),
                              deepcopy(dict((node_options or {}).get(identifier, {}))))
        factory = router_factory if role == 'router' else client_factory
        node = factory(context)
        if not isinstance(node, Node) or node.id != identifier or simulation.hosts.get(identifier) is not node:
            raise ValueError('Factory must return a Node registered with the requested simulation and ID')
        if identifier in (node_timings or {}):
            node.event_timings = node_timings[identifier]
        nodes[identifier] = node
    for (left, right), config in configs.items():
        config.connect(nodes[left], nodes[right])
    return Network(simulation, topology, nodes, routes)


def get_access_routers(graph: nx.Graph, clients: Sequence[int]) -> dict[int, int]:
    client_set = set(clients)
    result = {}
    for client in clients:
        if client not in graph or graph.degree(client) != 1:
            raise ValueError(f'Client {client} must have exactly one access router')
        router = next(iter(graph.neighbors(client)))
        if router in client_set:
            raise ValueError('Client-to-client links are not supported')
        result[client] = router
    return result


def _ids(values: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(_integer(value, name) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f'{name} must contain unique IDs')
    return result


@dataclass(frozen=True)
class TrafficRequest:
    source: int
    destination: int
    qubits: int = 1
    arrival_time: float = 0.
    fidelity: float = .5

    def __post_init__(self) -> None:
        _integer(self.source, 'source')
        _integer(self.destination, 'destination')
        _integer(self.qubits, 'qubits', 1)
        _nonnegative(self.arrival_time, 'arrival_time')
        if self.source == self.destination:
            raise ValueError('Source and destination must differ')
        if not .5 <= _nonnegative(self.fidelity, 'fidelity') <= 1:
            raise ValueError('fidelity must be in [0.5, 1]')


def generate_requests(sources: Sequence[int], destinations: Sequence[int], *, requests_per_source: int = 1,
                      qubits: int = 1, fidelity: float = .5, arrival_interval: float = 0.,
                      arrival_rate: float | None = None, seed: int | None = None,
                      rng: np.random.Generator | None = None) -> tuple[TrafficRequest, ...]:
    """Uniform destinations excluding self; periodic or Poisson arrivals per source.

    Times are absolute seconds from zero. Periodic traffic begins at zero;
    Poisson traffic begins after its first exponential interarrival. Returned
    requests are sorted by arrival time, stably resolving ties by source order.
    This describes traffic only; client factories decide how to execute it.
    """
    sources, destinations = _ids(sources, 'sources'), _ids(destinations, 'destinations')
    count = _integer(requests_per_source, 'requests_per_source')
    _integer(qubits, 'qubits', 1)
    if not .5 <= _nonnegative(fidelity, 'fidelity') <= 1:
        raise ValueError('fidelity must be in [0.5, 1]')
    interval = _nonnegative(arrival_interval, 'arrival_interval')
    if arrival_rate is not None and (_nonnegative(arrival_rate, 'arrival_rate') == 0 or interval != 0):
        raise ValueError('A positive arrival_rate requires arrival_interval=0')
    random = _random(seed, rng)
    result = []
    for source in sources:
        choices = [target for target in destinations if target != source]
        if count and not choices:
            raise ValueError(f'Source {source} has no eligible destination')
        time = 0.
        for index in range(count):
            time = index * interval if arrival_rate is None else time + float(random.exponential(1/arrival_rate))
            result.append(TrafficRequest(source, int(random.choice(choices)), qubits, time, fidelity))
    return tuple(sorted(result, key=lambda request: request.arrival_time))


def create_single_traffic_matrix(src_nodes: Sequence[int], dst_nodes: Sequence[int], max_qubits: int, *,
                                 seed: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
    """Legacy rows [destination ID, count], source order; max_qubits is exclusive."""
    upper = _integer(max_qubits, 'max_qubits', 2)
    random = _random(seed, rng)
    requests = generate_requests(src_nodes, dst_nodes, rng=random)
    return np.array([(request.destination, int(random.integers(1, upper))) for request in requests], dtype=np.int64).reshape(-1, 2)


def create_multi_traffic_matrix(src_nodes: Sequence[int], dst_nodes: Sequence[int], max_qubits: int | Sequence[int], *,
                                seed: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
    """Rows/columns follow supplied IDs; exclusive per-source upper bounds."""
    sources, destinations = _ids(src_nodes, 'sources'), _ids(dst_nodes, 'destinations')
    bounds = [max_qubits]*len(sources) if isinstance(max_qubits, Integral) else list(max_qubits)
    if len(bounds) != len(sources):
        raise ValueError('max_qubits must supply one bound per source')
    bounds = [_integer(value, 'max_qubits', 1) for value in bounds]
    random = _random(seed, rng)
    matrix = np.zeros((len(sources), len(destinations)), dtype=np.int64)
    for index, source in enumerate(sources):
        matrix[index] = random.integers(bounds[index], size=len(destinations))
        matrix[index, np.array(destinations) == source] = 0
    return matrix


def calculate_single_packet_count(traffic_matrix: np.ndarray, dst_nodes: Sequence[int] | None = None) -> np.ndarray:
    """Counts in destination order; legacy omission assumes IDs 0..row_count-1."""
    matrix = np.asarray(traffic_matrix)
    if matrix.ndim != 2 or matrix.shape[1] != 2:
        raise ValueError('Expected rows [destination, count]')
    destinations = _ids(range(len(matrix)) if dst_nodes is None else dst_nodes, 'destinations')
    lookup = {identifier: index for index, identifier in enumerate(destinations)}
    counts = np.zeros(len(destinations), dtype=np.int64)
    for target, qubits in matrix:
        if not np.isfinite(target) or target != int(target) or int(target) not in lookup:
            raise ValueError('Destination is unknown; pass dst_nodes explicitly for arbitrary IDs')
        if not np.isfinite(qubits) or qubits < 0 or qubits != int(qubits):
            raise ValueError('Qubit counts must be nonnegative integers')
        counts[lookup[int(target)]] += qubits > 0
    return counts


def calculate_multi_packet_count(traffic_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(traffic_matrix)
    if matrix.ndim != 2 or np.any(~np.isfinite(matrix)) or np.any(matrix < 0) or np.any(matrix != np.floor(matrix)):
        raise ValueError('Expected a two-dimensional nonnegative integer traffic matrix')
    return np.sum(matrix > 0, axis=0)


def create_graph(graph_name: str, num_routers: int, *args: Any, num_clients: int = 0, **kwargs: Any) -> nx.Graph:
    """Legacy NetworkX-shaped arguments; new code should use create_topology.

    Unlike create_core, star/grid arguments retain NetworkX's original meaning.
    Clients start after the ACTUAL core node count. All nodes receive integer IDs.
    """
    if graph_name not in CORE_GRAPH_FUNCTIONS:
        raise ValueError(f'Unknown core {graph_name!r}')
    seed = kwargs.get('seed') if graph_name in _RANDOM_CORES else kwargs.pop('seed', None)
    if graph_name in _RANDOM_CORES:
        seed = int(np.random.SeedSequence(seed).entropy)
        kwargs['seed'] = seed
    graph = CORE_GRAPH_FUNCTIONS[graph_name](num_routers, *args, **kwargs)
    return create_topology(core=graph, num_clients=num_clients, seed=seed).graph


def add_attributes_to_node(graph: nx.Graph, node: int, attributes: Mapping[str, Any]) -> None:
    graph.nodes[node].update(deepcopy(dict(attributes)))


def add_attributes_to_nodes(graph: nx.Graph, attributes: Mapping[int, Mapping[str, Any]] | None) -> None:
    for node, values in (attributes or {}).items():
        add_attributes_to_node(graph, node, values)


def add_attributes_to_link(graph: nx.Graph, source: int, target: int, attributes: Mapping[str, Any]) -> None:
    graph.edges[source, target].update(deepcopy(dict(attributes)))


def add_attributes_to_links(graph: nx.Graph, attributes: Mapping[tuple[int, int], Mapping[str, Any]] | None) -> None:
    for (source, target), values in (attributes or {}).items():
        add_attributes_to_link(graph, source, target, values)


def set_connection(graph: nx.Graph, nodes: Sequence[Node] | Mapping[int, Node]) -> None:
    """Legacy wiring helper; graph edges may carry a LinkConfig in 'config'."""
    lookup = dict(nodes) if isinstance(nodes, Mapping) else {node.id: node for node in nodes}
    _simple_graph(graph)
    if not set(graph) <= set(lookup):
        raise ValueError('Every graph node needs a simulator node')
    for left, right, data in graph.edges(data=True):
        config = data.get('config', LinkConfig())
        if not isinstance(config, LinkConfig):
            raise TypeError('Edge config must be LinkConfig')
        low, high = sorted((left, right))
        config.connect(lookup[low], lookup[high])


def set_l3_connection(graph: nx.Graph, nodes: Sequence[Node] | Mapping[int, Node]) -> None:
    """Legacy effective-link helper using the current set_eqs_connection API."""
    set_connection(graph, nodes)


CONNECTION_DICT = {'l1': set_connection, 'l3': set_l3_connection}
