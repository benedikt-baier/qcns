"""Declarative experiments composed from the existing network and simulation APIs.

The built-in hardware preset uses effective Werner-pair generation, symmetric
links, ideal gates and optional Markovian memory channels. It is not a model of
optical hardware preparation. Advanced models remain available via build_network.
"""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any, Literal

import networkx as nx
import numpy as np

from qcns.python.components.hardware.connection import L3C_Model, PChannel_Model
from qcns.python.components.hardware.memory import PQM_Model
from qcns.python.components.network.node import Node, PRResult
from qcns.python.components.network.qprotocol import (
    L1_EGP, L2_NEC, L3_PR, L3_Encoded, L3_RawEndpoint, QProtocolStack,
)
from qcns.python.components.network.router import Router, PurificationRouter, EncodedRouter
from qcns.python.components.network.utils import (
    LinkConfig, Network, NodeContext, Topology, TrafficRequest, create_topology, build_network,
)
from qcns.python.components.qubit.error import DepolarizingError, DephasingError
from qcns.python.components.qubit.qec import get_qec_code
from qcns.python.components.simulation.simulation import Simulation
from qcns.python.components.simulation.timing import EventTimings
from qcns.python.components.utils import save_config, load_config, experiment_metadata
from qcns.python.components.qubit.backend import CppBackend, PythonBackend

__all__ = ['TopologyConfig', 'HardwareConfig', 'ProtocolConfig', 'TimingConfig',
           'ExperimentConfig', 'RequestResult', 'ExperimentResult', 'Experiment', 'run_experiment']


def _number(value: float, name: str, *, minimum: float = 0., positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum or (positive and value == minimum):
        raise ValueError(f'{name} must be finite and {">" if positive else ">="} {minimum}')


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


@dataclass(frozen=True)
class TopologyConfig:
    """Router core plus leaf clients; IDs follow create_topology's conventions."""
    core: str | nx.Graph = 'path_graph'
    num_routers: int = 4
    num_clients: int = 2
    core_parameters: dict[str, Any] = field(default_factory=dict)
    attachment: str = 'balanced'
    client_attachments: dict[int, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _integer(self.num_routers, 'num_routers', 1)
        _integer(self.num_clients, 'num_clients')
        if not isinstance(self.core, (str, nx.Graph)):
            raise TypeError('core must be a graph generator name or a NetworkX graph')
        if self.attachment not in ('uniform', 'balanced', 'explicit'):
            raise ValueError('attachment must be uniform, balanced or explicit')

    def build(self, seed: int) -> Topology:
        return create_topology(core=self.core, num_routers=self.num_routers,
                               num_clients=self.num_clients, core_parameters=self.core_parameters,
                               attachment=self.attachment, client_attachments=self.client_attachments or None,
                               seed=seed)


@dataclass(frozen=True)
class HardwareConfig:
    """Symmetric effective links. Seconds, kilometers and bits/second are explicit.

    None for a noise time disables that channel; -1 memory slots means unbounded.
    Fidelity is Phi+ overlap of the initial Werner state. Gates are ideal.
    """
    fidelity: float = .99
    generation_duration: float = 1e-4
    generation_success: float = 1.
    classical_length_km: float = 1.
    classical_data_rate: float = 1e8
    memory_slots: int = 128
    depolarization_time: float | None = None
    dephasing_time: float | None = None

    def __post_init__(self) -> None:
        _number(self.fidelity, 'fidelity', minimum=.5)
        _number(self.generation_success, 'generation_success', minimum=1e-6, positive=True)
        if self.fidelity > 1 or self.generation_success > 1:
            raise ValueError('fidelity and generation_success must be <= 1')
        _number(self.generation_duration, 'generation_duration')
        _number(self.classical_length_km, 'classical_length_km')
        _number(self.classical_data_rate, 'classical_data_rate', positive=True)
        if isinstance(self.memory_slots, bool) or not isinstance(self.memory_slots, int):
            raise ValueError('memory_slots must be -1 or a positive integer')
        if self.memory_slots != -1:
            _integer(self.memory_slots, 'memory_slots', 1)
        for name in ('depolarization_time', 'dephasing_time'):
            value = getattr(self, name)
            if value is not None:
                _number(value, name, positive=True)

    def build(self) -> LinkConfig:
        errors = []
        if self.depolarization_time is not None:
            errors.append(DepolarizingError(self.depolarization_time))
        if self.dephasing_time is not None:
            errors.append(DephasingError(self.dephasing_time))
        memory = PQM_Model(size=self.memory_slots, extraction_mode='fifo', memory_errors=errors)
        model = L3C_Model(fidelity=self.fidelity, duration=self.generation_duration,
                          success_prob=self.generation_success,
                          pchannel=PChannel_Model(length=self.classical_length_km,
                                                   data_rate=self.classical_data_rate))
        return LinkConfig(model, model, memory, memory)


@dataclass(frozen=True)
class ProtocolConfig:
    """Use existing FRP, hierarchical BBPSSW PR, or exact encoded graph interfaces."""
    mode: Literal['raw', 'purification', 'encoded'] = 'raw'
    timeout: float = 1.
    max_attempts: int = 64
    max_rounds: int = 8
    max_raw_pairs: int = 4096
    max_requests: int = 256
    code: str = 'five_qubit'
    max_retries: int = 3
    purification_rounds: int = 0
    preparation_duration: float = 1e-4
    max_graph_qubits: int = 4096

    def __post_init__(self) -> None:
        if self.mode not in ('raw', 'purification', 'encoded'):
            raise ValueError('mode must be raw, purification or encoded')
        _number(self.timeout, 'timeout', positive=True)
        for name in ('max_attempts', 'max_rounds', 'max_raw_pairs', 'max_requests', 'max_graph_qubits'):
            _integer(getattr(self, name), name, 1)
        _integer(self.max_retries, 'max_retries')
        _integer(self.purification_rounds, 'purification_rounds')
        if self.purification_rounds > 8:
            raise ValueError('purification_rounds must be <= 8')
        _number(self.preparation_duration, 'preparation_duration')
        if self.mode == 'encoded':
            backend = get_qec_code(self.code)
            if backend.method != 'exact' or backend.num_logical != 1:
                raise ValueError('Encoded experiments require an exact [[n,1]] code')
            if backend.length + 1 > self.max_graph_qubits:
                raise ValueError('Code interface exceeds max_graph_qubits')

    def options(self) -> dict[str, Any]:
        if self.mode == 'raw':
            return {'timeout': self.timeout, 'max_attempts': self.max_attempts}
        options = {name: getattr(self, name) for name in
                   ('timeout', 'max_attempts', 'max_rounds', 'max_raw_pairs', 'max_requests')}
        if self.mode == 'encoded':
            options.update({name: getattr(self, name) for name in
                            ('code', 'max_retries', 'purification_rounds', 'preparation_duration', 'max_graph_qubits')})
        return options


@dataclass(frozen=True)
class TimingConfig:
    default_duration: float = 0.
    durations: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.build()

    def build(self) -> EventTimings:
        return EventTimings(self.default_duration, self.durations)


@dataclass(frozen=True)
class ExperimentConfig:
    """Saveable inputs. Each source initiates its requests sequentially in arrival order.

    Clients continue serving other sources after their own requests finish.
    end_time=None stops when all requests finish; finite end_time also permits
    background-only runs. Every request has a finite protocol timeout.
    """
    topology: TopologyConfig = field(default_factory=lambda: TopologyConfig(
        attachment='explicit', client_attachments={4: 0, 5: 3}))
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    access_hardware: HardwareConfig | None = None
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    requests: tuple[TrafficRequest, ...] = field(default_factory=lambda: (TrafficRequest(4, 5, fidelity=.97),))
    seed: int | None = 7
    end_time: float | None = None
    timings: TimingConfig = field(default_factory=TimingConfig)
    node_timings: dict[int, TimingConfig] = field(default_factory=dict)
    routing_metric: str | None = None
    backend: str = 'python'
    backend_threads: int = 1
    backend_library: str | None = None

    def __post_init__(self) -> None:
        if self.backend not in ('python', 'cpp'):
            raise ValueError('backend must be python or cpp')
        _integer(self.backend_threads, 'backend_threads', minimum=1)
        if self.backend_threads > 1024:
            raise ValueError('backend_threads must be at most 1024')
        if self.backend_library is not None and not isinstance(self.backend_library, str):
            raise TypeError('backend_library must be a path string or None')
        for name, kind in (('topology', TopologyConfig), ('hardware', HardwareConfig),
                           ('protocol', ProtocolConfig), ('timings', TimingConfig)):
            if not isinstance(getattr(self, name), kind):
                raise TypeError(f'{name} must be {kind.__name__}')
        if self.access_hardware is not None and not isinstance(self.access_hardware, HardwareConfig):
            raise TypeError('access_hardware must be HardwareConfig or None')
        if self.seed is not None:
            _integer(self.seed, 'seed')
        if self.end_time is not None:
            _number(self.end_time, 'end_time')
        if not self.requests and self.end_time is None:
            raise ValueError('An empty workload requires a finite end_time')
        object.__setattr__(self, 'requests', tuple(self.requests))
        if any(not isinstance(request, TrafficRequest) for request in self.requests):
            raise TypeError('requests must contain TrafficRequest values')
        if self.protocol.mode == 'encoded' and len({request.source for request in self.requests}) > 1:
            raise ValueError('Encoded experiments currently support one initiating source per run; concurrent encoded initiators are not validated')
        if self.protocol.mode != 'raw' and len(self.requests) > self.protocol.max_requests:
            raise ValueError('Workload exceeds max_requests; increase the request table limit or split the experiment')
        for node, timing in self.node_timings.items():
            _integer(node, 'node_timings node ID')
            if not isinstance(timing, TimingConfig):
                raise TypeError('node_timings values must be TimingConfig')
        if self.routing_metric is not None and not isinstance(self.routing_metric, str):
            raise TypeError('routing_metric must be an edge attribute name or None')

    def to_dict(self) -> dict[str, Any]:
        return {'experiment_version': 1, **asdict(self)}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ExperimentConfig:
        data = deepcopy(dict(values))
        if data.pop('experiment_version', 1) != 1:
            raise ValueError('Unsupported experiment_version')
        for name, kind in (('topology', TopologyConfig), ('hardware', HardwareConfig),
                           ('access_hardware', HardwareConfig), ('protocol', ProtocolConfig), ('timings', TimingConfig)):
            if name in data and data[name] is not None:
                data[name] = kind(**data[name])
        if 'requests' in data:
            data['requests'] = tuple(TrafficRequest(**item) for item in data['requests'])
        if 'node_timings' in data:
            data['node_timings'] = {int(key): TimingConfig(**value) for key, value in data['node_timings'].items()}
        if 'topology' in data:
            topology = data['topology']
            data['topology'] = TopologyConfig(**{**asdict(topology), 'client_attachments':
                                                {int(key): value for key, value in topology.client_attachments.items()}})
        return cls(**data)

    def save(self, path: str | Path) -> None:
        save_config(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> ExperimentConfig:
        data = load_config(path)
        if not isinstance(data, dict):
            raise ValueError('Expected an experiment mapping, not a legacy network tuple')
        return cls.from_dict(data)


@dataclass(frozen=True)
class RequestResult:
    index: int
    request: TrafficRequest
    status: str
    finished_at: float
    protocol_result: PRResult | None = None
    reason: str = ''
    estimated_fidelity: float | None = None

    @property
    def delivered_pairs(self) -> int:
        return len(self.protocol_result.outputs) if self.status == 'complete' and self.protocol_result else 0


@dataclass(frozen=True)
class ExperimentResult:
    requests: tuple[RequestResult, ...]
    finished_at: float
    configuration: dict[str, Any]
    metadata: dict[str, Any]

    @property
    def delivered_pairs(self) -> int:
        return sum(item.delivered_pairs for item in self.requests)


class _ExperimentClient(Node):
    def __init__(self, context: NodeContext, protocol: ProtocolConfig,
                 requests: list[tuple[int, TrafficRequest]],
                 report: Callable[[RequestResult], None]) -> None:
        operation = {'raw': L3_RawEndpoint, 'purification': L3_PR, 'encoded': L3_Encoded}[protocol.mode]
        stack = QProtocolStack(l1_qprotocol=L1_EGP(reattempt=False), l2_qprotocol=L2_NEC(),
                               l3_qprotocol=operation(context.routing_table, **protocol.options()))
        super().__init__(context.node_id, context.simulation, stop=False, qprotocols=stack)
        self._requests, self._report = requests, report
        self._mode = protocol.mode

    async def run(self) -> None:
        for index, request in self._requests:
            await self.l3_qprotocol.serve_until(request.arrival_time)
            outcome = await self.l3_qprotocol.establish(request.destination, requested=request.qubits,
                                                       fidelity=request.fidelity)
            self._report(RequestResult(index, request, outcome.status, outcome.finished_at, outcome, outcome.reason,
                                       outcome.estimated_fidelity if self._mode != 'raw' and outcome.status == 'complete' else None))
        await self.l3_qprotocol.serve()


class _CompletionHost(Node):
    def __init__(self, identifier: int, simulation: Simulation, completed: asyncio.Event,
                 finished_at: Callable[[], float]) -> None:
        super().__init__(identifier, simulation)
        self._completed = completed
        self._finished_at = finished_at

    async def run(self) -> None:
        await self._completed.wait()
        await self.wait(max(0., self._finished_at() - self._time))


class Experiment:
    """One isolated run; use a new instance for replay, or run_experiment for automatic cleanup.

    build() exposes the ordinary Network for advanced inspection. run() retains
    output resources until close(); context-manager exit releases all memories.
    """
    def __init__(self, config: ExperimentConfig) -> None:
        if not isinstance(config, ExperimentConfig):
            raise TypeError('config must be ExperimentConfig')
        self.config = deepcopy(config)
        self.network: Network | None = None
        self._results: dict[int, RequestResult] = {}
        self._requests_by_source: dict[int, list[tuple[int, TrafficRequest]]] = {}
        for index, request in enumerate(self.config.requests):
            self._requests_by_source.setdefault(request.source, []).append((index, request))
        for requests in self._requests_by_source.values():
            requests.sort(key=lambda item: item[1].arrival_time)
        self._completed = asyncio.Event()
        self._ran = False
        self._closed = False

    def _report(self, result: RequestResult) -> None:
        self._results[result.index] = result
        if len(self._results) == len(self.config.requests):
            self._completed.set()

    def _router(self, context: NodeContext) -> Node:
        protocol = self.config.protocol
        if protocol.mode == 'raw':
            return Router(context.node_id, context.simulation, context.routing_table, l1_reattempt=False)
        kind = PurificationRouter if protocol.mode == 'purification' else EncodedRouter
        return kind(context.node_id, context.simulation, context.routing_table, **protocol.options())

    def _client(self, context: NodeContext) -> Node:
        requests = self._requests_by_source.get(context.node_id, [])
        return _ExperimentClient(context, self.config.protocol, requests, self._report)

    def build(self) -> Network:
        if self._closed:
            raise RuntimeError('Experiment is closed; construct a new instance')
        if self.network is not None:
            return self.network
        backend = (CppBackend(self.config.backend_library, threads=self.config.backend_threads)
                   if self.config.backend == 'cpp' else PythonBackend())
        simulation = Simulation(seed=self.config.seed, end_time=self.config.end_time, backend=backend)
        topology = self.config.topology.build(int(simulation.seed))
        for request in self.config.requests:
            if request.source not in topology.clients or request.destination not in topology.clients:
                raise ValueError('Traffic endpoints must be client IDs in this topology')
            if self.config.protocol.mode == 'raw' and request.qubits > 256:
                raise ValueError('Raw request headers support at most 256 output pairs')
        timings = {node: self.config.timings.build() for node in topology.graph}
        timings.update({node: timing.build() for node, timing in self.config.node_timings.items()})
        self.network = build_network(simulation=simulation, topology=topology,
                                     router_factory=self._router, client_factory=self._client,
                                     core_links=self.config.hardware.build(),
                                     access_links=(self.config.access_hardware or self.config.hardware).build(),
                                     node_timings=timings, routing_metric=self.config.routing_metric)
        if self.config.requests:
            _CompletionHost(max(topology.graph) + 1, simulation, self._completed,
                            lambda: max(row.finished_at for row in self._results.values()))
        return self.network

    def run(self) -> ExperimentResult:
        if self._ran:
            raise RuntimeError('Experiments are single-use; construct a new Experiment to repeat a run')
        network = self.build()
        self._ran = True
        try:
            network.simulation.run()
        except BaseException:
            self.close()
            raise
        results = tuple(self._results.get(index, RequestResult(index, request, 'incomplete',
                                                              network.simulation.time, reason='end_time reached'))
                        for index, request in enumerate(self.config.requests))
        effective = self.config.to_dict()
        effective['seed'] = int(network.simulation.seed)
        metadata = experiment_metadata(seed=int(network.simulation.seed))
        metadata['quantum_backend'] = network.simulation.backend.name
        metadata['native_operations'] = network.simulation.backend.native_operations
        if isinstance(network.simulation.backend, CppBackend):
            metadata['backend_library'] = str(network.simulation.backend.library_path)
            metadata['backend_threads'] = network.simulation.backend.threads
        metadata['model'] = 'effective Werner generation; ideal gates; configured Markovian memory noise'
        metadata['backend'] = ('exact_graph_interface; independent Bell-diagonal inputs; ideal graph resources and decoding'
                               if self.config.protocol.mode == 'encoded' else 'density_matrix')
        metadata['fidelity'] = 'protocol estimates only; raw forwarding does not enforce target fidelity'
        return ExperimentResult(results, network.simulation.time, effective, metadata)

    def close(self) -> None:
        if not self._closed and self.network is not None:
            self.network.simulation.reset()
        self._closed = True

    def __enter__(self) -> Experiment:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run and release resources. Use Experiment directly to retain quantum outputs."""
    with Experiment(config) as experiment:
        return experiment.run()
