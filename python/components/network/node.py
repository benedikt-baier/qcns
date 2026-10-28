from __future__ import annotations

import logging
import traceback
import numpy as np
import asyncio as asc
from functools import partial
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Set, Any

from qcns.python.components.simulation.simulation import Simulation
from qcns.python.components.simulation.event import SendEvent, ReceiveEvent, TimeoutEvent, CompletionEvent
from qcns.python.components.simulation.timing import _advance_local_time, _duration, EventTimings, node_event, gate_event, untimed_events
from qcns.python.components.qubit.qubit import Qubit, combine_state, remove_qubits
from qcns.python.components.qubit.qec import get_qec_code
from qcns.python.components.qubit.graph_state import MeasurementBasedRepeaterModel
from qcns.python.components.hardware.connection import *
from qcns.python.components.hardware.memory import PhysicalQuantumMemory, LogicalQuantumMemory, PQM_Model, LQM_Model
from qcns.python.components.connection.channel import QChannel, PChannel
from qcns.python.components.connection.connection import PChannel_Model, SingleQubitConnection, SenderReceiverConnection, TwoPhotonSourceConnection, BellStateMeasurementConnection, FockStateConnection, L3Connection
from qcns.python.components.packet import Packet
from qcns.python.components.network.qprotocol import QProtocol, QProtocolStack

__all__ = ['Node', 'EventTimings', 'node_event', 'gate_event',
           'PRRequest', 'PRSession', 'PRPair', 'PRResult', 'EncodedResult',
           'PRFailure', 'PairExtractionFailure']

# Local L3 protocol state and outcomes. These records are owned by nodes and
# applications; they are not packet fields and do not affect wire header lengths.
@dataclass
class PRRequest:
    """Node-local request progress and resource accounting; not a packet header."""

    key: tuple[int, int]
    destination: int
    count: int
    fidelity_ir: int
    deadline: float
    path: tuple[int, ...] = ()
    hop_index: int = 0
    hierarchy_index: int = 0
    status: str = 'syn'
    owner_ready: bool = False
    peer_ready: bool = False
    resets: int = 0
    raw_pairs: int = 0
    purification_attempts: int = 0
    failed_attempts: int = 0
    swaps: int = 0
    encoded_blocks: int = 0
    encoded_retries: int = 0
    graph_resource_qubits: int = 0
    outputs: list[tuple[int, int]] = field(default_factory=list)
    code: str | None = None


@dataclass
class PRSession:
    """Node-local FIFO requests and retained pairs shared by a purification session."""

    owner: int
    peer: int
    fidelity_ir: int
    memory_store: int
    requests: deque[tuple[int, int]] = field(default_factory=deque)
    # Successful intermediate pairs remain available across failed attempts.
    cache: dict[int, list[PRPair]] = field(default_factory=dict)
    generation: int = 0
    status: str = 'open'


@dataclass(frozen=True)
class PRPair:
    """Local resource token with an operational fidelity estimate and timestamp."""

    token: tuple[int, int]
    fidelity: float
    created: float


@dataclass(frozen=True)
class PRResult:
    """Application-facing establishment outcome; not serialized as a header."""

    key: tuple[int, int]
    status: str
    outputs: tuple[tuple[int, int], ...] = ()
    estimated_fidelity: float = 0.0
    finished_at: float = 0.0
    reason: str = ''


@dataclass(frozen=True)
class EncodedResult(PRResult):
    """Encoded establishment outcome with the backend and modeling assumptions."""

    code: str = 'five_qubit'
    backend: str = 'exact_graph_interface'
    assumptions: str = 'independent Bell-diagonal inputs; ideal graph resources and decoding'


class PRFailure(Exception):
    """A bounded, expected request failure; never masks programming errors."""


class PairExtractionFailure(PRFailure):
    """Heralded loss while preparing a new pair; other reservations survive."""


@dataclass
class L2Batch:
    """Node-owned local results and staged physical qubits for one L2 batch."""
    request_id: int
    results: np.ndarray
    qubits: List[Qubit]
    needed: int
    code: str | None = None

@dataclass
class L4Batch(L2Batch):
    """End-to-end batch addressed by the remote endpoint, not the last hop."""
    next_hop: int = 0

class QuantumError:
    pass

_GATE_DURATION = {'X': 5e-6, 'Y': 5e-6, 'Z': 3e-6, 'H': 6e-6, 'T': 1e-6, 'bsm': 1e-5, 'prob_bsm': 1e-5, 'CNOT': 12e-5, 'measure': 1e-6}

_GATE_PARAMETERS = {'prob_bsm_p': 0.57, 'prob_bsm_a': 0.57}

SEND = 0
RECEIVE = 1
GATE = 2

# Packet-queue purpose: delivered messages not yet claimed by receive_packet.
# SEND/RECEIVE queues contain protocol-owned transactions and reservations.
PENDING_RECEIVE = 2

L0 = 0
L1 = 1
L2 = 2
L3 = 3
L4 = 4
L7 = 5

POSITIVE = 1
NEGATIVE = -1
NEUTRAL = 0

class Node:
    
    """
    Represents a Node
    
    Attr:
        _node_id (int): ID of node
        _sim (Simulation): simulation object
        protocol_stack (QProtocolStack): protocols owned by this node
        _pulse_duration (float): duration to send one bit of a packet
        _gates (dict): gates to apply to a qubit 
        _gate_duration (dict): duration of gates
        _gate_parameters (dict): parameters for quantum gates
        _connections (dict): collection of single, entangled, packet and memory connections to other hosts
        _neighbors (set): collection of other host this host is connected with
        _layer_results (dict): saved results of packet layer
        _l1_packets (list): L1 packet puffer
        _l2_packets (list): L2 packet puffer
        _l3_packets (list): L3 packet puffer
        _resume (asc.Event): event to resume execution
        stop (bool): stop flag for infinitly running hosts  
    """
    
    def __init__(self, node_id: int, sim: Simulation, stop: bool=True, qprotocols: QProtocolStack=None, gate_duration: Dict[str, float]=None, 
                 gate_parameters: Dict[str, float]=None, *, event_timings: EventTimings | None = None) -> None:
        
        """Create a node with an independent copy of protocol configuration.

        ``stop=True`` identifies a finite node; ``stop=False`` identifies a
        background service. Gate configuration dictionaries are copied.
        """
        
        self.event_timings = EventTimings() if event_timings is None else event_timings
        self._event_timing_suspended = False
        self._node_id: int = node_id
        self._sim: Simulation = sim
        self._rng = sim.rng('node', node_id)
        
        self.protocol_stack = (qprotocols or QProtocolStack()).clone()
        self.protocol_stack.bind(self)
        self._qprotocols = self.protocol_stack._qprotocols
        
        self._gates: Dict[str, Qubit] = {k: v for k, v in Qubit.__dict__.items() if not k.startswith(('__', 'f'))}
        self._gate_duration = dict(_GATE_DURATION if gate_duration is None else gate_duration)
        self._gate_parameters = dict(_GATE_PARAMETERS if gate_parameters is None else gate_parameters)
        
        self._channels: Dict[str, Dict[str, Any]] = {'qc': {}, 'pc': {}}
        self._connections: Dict[str, Dict[str, Any]] = {'sqs': {}, 'eqs': {}}
        self._memory: Dict[str, Any] = {}
        self._link_fidelities: dict[int, dict[int, float]] = {}
        self._neighbors: Set[int] = set()
        
        self._layer_results: Dict[int, Dict[int, Dict[int, List[np.ndarray | Tuple[int, np.ndarray] | L2Batch]]]] = {}
        self._packets: dict[int, dict[int, dict[int, list[Packet | tuple[Packet, int | None] | tuple[int, Packet]]]]] = {}
        self.event_dict = {0: self.__S, 1: self.__F, 2: self.__N}
        
        self._time: float = 0.
        
        self._resume: asc.Event = asc.Event()
        self._packet_receive_sequence = 0
        self._scheduled_dispatch = False
        self._pending_swaps: list[tuple[QProtocol, Packet, int | None]] = []
        self._l4_ready = {}
        self.pr_requests: dict[tuple[int, int], PRRequest] = {}
        self.pr_sessions: dict[tuple[int, int, int], PRSession] = {}
        self._pr_halves: dict[tuple[int, int], tuple[Qubit, int, int, tuple[int, int]]] = {}
        self.stop: bool = stop
    
        self.run = partial(self.log_exceptions, self.run)
        self._sim.add_host(self)
    
    async def run(self):
        
        """
        Run function of Node, should be overwritten
        
        Args:
            /
            
        Returns:
            /
        """
        
        pass
    
    async def log_exceptions(self, func) -> None:
        
        """
        Wrapper to log exceptions in the run function
        
        Args:
            func (Function): function to wrap around
            
        Returns:
            /
        """
        
        try:
            await func()
        except Exception as e:
            logging.error(f'Node {self.id} has Exception')
            logging.error(traceback.format_exc())
            self._sim.stop_simulation()
            raise
    
    @property
    def event_timings(self) -> EventTimings:
        """Per-node processing overhead in simulation seconds."""
        return self._event_timings

    @event_timings.setter
    def event_timings(self, value: EventTimings) -> None:
        if not isinstance(value, EventTimings):
            raise TypeError('event_timings must be an EventTimings configuration')
        self._event_timings = value
    
    @untimed_events
    def reset(self) -> None:
        
        for token in list(self._pr_halves):
            self.pr_discard(token)
        self.pr_requests.clear()
        self.pr_sessions.clear()
        self._pr_halves.clear()
        self._time = 0.
        self._resume = asc.Event()
        self._packet_receive_sequence = 0
        self._pending_swaps.clear()
        self._scheduled_dispatch = False
        remove_qubits(self._l4_owned_qubits())
        self._l4_ready.clear()
        self._layer_results = {peer: stores for peer, stores in self._layer_results.items()
                               if peer in self._channels['pc']}
        self._packets = {peer: stores for peer, stores in self._packets.items()
                         if peer in self._channels['pc']}
        for stores in self._memory.values():
            for memory in stores.values():
                memory.discard_qubits()
        for channels in self._channels.values():
            for stores in channels.values():
                for channel in stores.values():
                    channel._channel.clear()
        for neighbor in self._channels['pc']:
            self._layer_results[neighbor] = {store: {layer: [] for layer in (L1, L2, L3)} for store in (SEND, RECEIVE)}
            self._packets[neighbor] = {store: {layer: [] for layer in (L1, L2, L3)} for store in (SEND, RECEIVE)}
        for protocol in self._qprotocols.values():
            protocol.reset()
    
    def set_sqs_connection(self, host: Node, sender_config: SQC_Model=SQC_Model(), receiver_config: SQC_Model=SQC_Model()) -> None:
        
        """
        Sets up a single qubit source connection
        
        Args:
            host (Node): other host to establish connection with
            sender_source (str): name of sender sided source to use
            receiver_source (str): name of receiver sided source to use
            sender_num_sources (int): number of sources at sender side
            receiver_num_sources (int): number of sources at receiver side
            sender_length (float): length from sender to receiver
            sender_attenuation (float): sender attenuation coefficient
            sender_in_coupling_prob (float): probability of coupling qubit into fiber
            sender_out_coupling_prob (float): probability pf coupling qubit out of fiber
            sender_lose_qubits (bool): whether to lose qubits of sender channel
            sender_channel_errors (list): list of errors on sender channel
            receiver_length (float): length from receiver to sender
            receiver_attenuation (float): receiver attenuation coefficient
            receiver_in_coupling_prob (float): probability of coupling qubit into fiber
            receiver_out_coupling_prob (float): probability of coupling qubit out of fiber
            receiver_lose_qubits (bool): whether to lose qubits of receiver channel
            receiver_channel_errors (list): list of errors on receiver channel
            
        Returns:
            /
        """
        
        self._neighbors.add(host.id)
        host._neighbors.add(self.id)
        
        connection_s_r = SingleQubitConnection(self, host.id, self._sim, sender_config)
        connection_r_s = SingleQubitConnection(host, self.id, self._sim, receiver_config)
        
        self._connections['sqs'][host.id] = connection_s_r
        host._connections['sqs'][self.id] = connection_r_s
        
        channel_s_r = QChannel(sender_config._qchannel_model, sender_config._channel_errors, rng=self._sim.rng('channel', self.id, host.id))
        channel_r_s = QChannel(receiver_config._qchannel_model, receiver_config._channel_errors, rng=self._sim.rng('channel', host.id, self.id))
        
        self._channels['qc'][host.id] = {SEND: channel_s_r, RECEIVE: channel_r_s}
        host._channels['qc'][self.id] = {SEND: channel_r_s, RECEIVE: channel_s_r}
    
    def set_eqs_connection(self, host: Node, sender_connection_config: EQC_Model=L3C_Model(), receiver_connection_config: EQC_Model=L3C_Model(), sender_memory_config: PQM_Model | LQM_Model=LQM_Model(), receiver_memory_config: PQM_Model | LQM_Model=LQM_Model()) -> None:
        
        """
        Sets up a heralded entangled qubit connection between this host and another host
        
        Args:
            host (Node): other host to establish connection with
            sender_type (str): type of entanglement connection from sender to receiver
            sender_model (str): model of photon source at sender
            receiver_type (str): type of entanglement connection from receiver to sender
            receiver_model (str): model of photon source at receiver
            sender_source (str): model of sender sided photon source 
            receiver_source (str): model of receiver sided photon source
            sender_detector (str): model of sender sided photon detector
            receiver_detector (str): model of receiver sided photon detector
            sender_num_sources (int): number of sources at sender side
            receiver_num_sources (int): number of sources at receiver side
            sender_length (float): length from sender to receiver
            sender_attenuation (float): sender attenuation coefficient
            sender_in_coupling_prob (float): probability of coupling qubit into fiber
            sender_out_coupling_prob (float): probability of coupling qubit out of fiber
            sender_lose_qubits (bool): whether to lose qubits of sender channel
            receiver_length (float): length from receiver to sender
            receiver_attenuation (float): receiver attenuation coefficient
            receiver_in_coupling_prob (float): probability of coupling qubit into fiber
            receiver_out_coupling_prob (float): probability of coupling qubit out of fiber
            receiver_lose_qubits (bool): whether to lose qubits of receiver channel
            sender_memory_mode (str): mode how to extract qubits at the sender, fifo or lifo
            sender_memory_size (int): size of sender memory
            sender_efficiency (float): efficiency of sender memory
            sender_memory_errors (list): list of memory errors at sender
            receiver_memory_mode (str): mode how to extract qubits at the receiver, fifo or lifo
            receiver_memory_size (int): size of receiver memory
            receiver_efficiency (float): efficiency of receiver memory
            receiver_memory_errors (list): list of memory errors at receiver
            
        Returns:
            /
        """
        
        if sender_connection_config._connection_type not in ['sr', 'tps', 'bsm', 'fs', 'l3c']:
            raise ValueError('Connection should a well known entanglement generation method')
        if receiver_connection_config._connection_type not in ['sr', 'tps', 'bsm', 'fs', 'l3c']:
            raise ValueError('Connection should a well known entanglement generation method')
        if not (isinstance(sender_memory_config, PQM_Model) or isinstance(sender_memory_config, LQM_Model)):
            raise ValueError('Memory should be either physical or logical')
        if not (isinstance(receiver_memory_config, PQM_Model) or isinstance(receiver_memory_config, LQM_Model)):
            raise ValueError('Memory should be either physical or logical')
            
        self._neighbors.add(host.id)
        host._neighbors.add(self.id)
        
        connections = {'sr': SenderReceiverConnection, 'tps': TwoPhotonSourceConnection, 'bsm': BellStateMeasurementConnection, 'fs': FockStateConnection, 'l3c': L3Connection}
        memories = {'pqm': PhysicalQuantumMemory, 'lqm': LogicalQuantumMemory}
        
        sender_memory_send = memories[sender_memory_config._memory_type](sender_memory_config)
        sender_memory_receive = memories[receiver_memory_config._memory_type](receiver_memory_config)
        receiver_memory_send = memories[receiver_memory_config._memory_type](receiver_memory_config)
        receiver_memory_receive = memories[sender_memory_config._memory_type](sender_memory_config)
        
        # sender_memory_receive._size = -1
        # receiver_memory_receive._size = -1
        
        connection_s_r = connections[sender_connection_config._connection_type](self, host.id, self._sim, sender_memory_send, receiver_memory_receive, sender_connection_config)
        connection_r_s = connections[receiver_connection_config._connection_type](host, self.id, self._sim, receiver_memory_send, sender_memory_receive, receiver_connection_config)
        
        self._connections['eqs'][host.id] = connection_s_r
        host._connections['eqs'][self.id] = connection_r_s
        outgoing = connection_s_r.estimate_fidelity()
        incoming = connection_r_s.estimate_fidelity()
        self._link_fidelities[host.id] = {SEND: outgoing, RECEIVE: incoming}
        host._link_fidelities[self.id] = {SEND: incoming, RECEIVE: outgoing}
        
        self._memory[host.id] = {SEND: sender_memory_send, RECEIVE: sender_memory_receive}
        host._memory[self.id] = {SEND: receiver_memory_send, RECEIVE: receiver_memory_receive}
        for owner, peer in ((self, host), (host, self)):
            for direction, memory in owner._memory[peer.id].items():
                memory.bind_rng(self._sim.rng('memory', owner.id, peer.id, direction))
        
        if sender_connection_config._connection_type in ['sr', 'l3c']:
            sender_pchannel = sender_connection_config._pchannel 
        if sender_connection_config._connection_type in ['tps', 'bsm', 'fs']:
            sender_pchannel = PChannel_Model(sender_connection_config._sender_pchannel._length + sender_connection_config._receiver_pchannel._length, 0.5 * (sender_connection_config._sender_pchannel._data_rate + sender_connection_config._receiver_pchannel._data_rate))
        
        if receiver_connection_config._connection_type in ['sr', 'l3c']:
            receiver_pchannel = receiver_connection_config._pchannel
        if receiver_connection_config._connection_type in ['tps', 'bsm', 'fs']:
            receiver_pchannel = PChannel_Model(receiver_connection_config._sender_pchannel._length + receiver_connection_config._receiver_pchannel._length, 0.5 * (receiver_connection_config._sender_pchannel._data_rate + receiver_connection_config._receiver_pchannel._data_rate))
        
        self.set_packet_connection(host, sender_pchannel, receiver_pchannel)

    def set_packet_connection(self, host: Node, sender_config: PChannel_Model=PChannel_Model(), receiver_config: PChannel_Model=PChannel_Model()) -> None:
        
        """
        Sets up a packet connection between this host an another
        
        Args:
            host (Node): host to set up connection with
            length (float): length of connection
            
        Returns:
            /
        """
        
        self._neighbors.add(host.id)
        host._neighbors.add(self.id)
        
        channel_s_r = PChannel(sender_config)
        channel_r_s = PChannel(receiver_config)
        
        self._channels['pc'][host.id] = {SEND: channel_s_r, RECEIVE: channel_r_s}
        host._channels['pc'][self.id] = {SEND: channel_r_s, RECEIVE: channel_s_r}
    
        self._layer_results[host.id] = {SEND: {L1: [], L2: [], L3: []}, RECEIVE: {L1: [], L2: [], L3: []}}
        host._layer_results[self.id] = {SEND: {L1: [], L2: [], L3: []}, RECEIVE: {L1: [], L2: [], L3: []}}
        
        self._packets[host.id] = {SEND: {L1: [], L2: [], L3: []}, RECEIVE: {L1: [], L2: [], L3: []}}
        host._packets[self.id] = {SEND: {L1: [], L2: [], L3: []}, RECEIVE: {L1: [], L2: [], L3: []}}
    
    @node_event
    def attempt_qubit(self, receiver: int, num_requested: int=1, estimate: bool=True) -> None:
        
        """
        Attempts to create the number of requested qubits, can estimate the number of needed qubits
        
        Args:
            receiver (int): receiver to which photon source points to
            num_requested (int): number of requested qubits
            estimate (bool): whether to estimate the number of needed qubits or not
            
        Returns:
            /
        """
        
        if not isinstance(num_requested, (int, np.integer)) or num_requested < 1:
            raise ValueError("Requested count must be a positive integer")
        self._sim.schedule_event(SendEvent(self._time, self.id))
        
        _num_needed = num_requested
        if estimate:
            probability = self._connections['sqs'][receiver]._success_prob
            if probability <= 0:
                raise ValueError("Cannot estimate a zero-success connection")
            _num_needed = int(np.ceil(_num_needed / probability))
        
        self._connections['sqs'][receiver].attempt_qubit(_num_needed)

    @node_event
    def create_qubit(self, receiver: int, num_requested: int=1) -> None:
        
        """
        Creates the number of requested qubits, no matter how long it takes
        
        Args:
            receiver (int): receiver to which photon source points to
            num_requested (int): number of requested qubits
            
        Returns:
            /
        """
        
        if not isinstance(num_requested, (int, np.integer)) or num_requested < 1:
            raise ValueError("Requested count must be a positive integer")
        self._sim.schedule_event(SendEvent(self._time, self.id))
        
        self._connections['sqs'][receiver].create_qubit(num_requested)
    
    @node_event
    def attempt_bell_pairs(self, receiver: int, num_requested: int=1, estimate: bool=False, *, request: Packet=None) -> None:
        
        """
        Attempts to create the number of requested bell pairs, can estimate the number of needed qubits based on success probability
        
        Args:
            receiver (int): receiver of bell pairs
            num_requested (int): number of requested bell pairs
            estimate (bool): whether to estimate the number of needed qubits or not
            
        Returns:
            /
        """
        
        if not isinstance(num_requested, (int, np.integer)) or num_requested < 1:
            raise ValueError("Requested count must be a positive integer")
        if num_requested < 1:
            raise ValueError('Requesting 0 qubits')
        
        self._sim.schedule_event(SendEvent(self._time, self.id))
        
        _num_needed = num_requested
        if estimate:
            probability = self._connections['eqs'][receiver]._success_prob
            if probability <= 0:
                raise ValueError("Cannot estimate a zero-success connection")
            _num_needed = int(np.ceil(_num_needed / probability))
        
        if not self.has_space(receiver, 0, _num_needed):
            _num_needed = self.remaining_space(receiver, 0)
        
        if not self._sim._hosts[receiver].has_space(self.id, 1, _num_needed):
            _num_needed = self._sim._hosts[receiver].remaining_space(self.id, 1)
        
        if not _num_needed:
            return
        
        if _num_needed < num_requested:
            num_requested = _num_needed
        
        self._connections['eqs'][receiver].attempt_bell_pairs(num_requested, _num_needed, request=request)
        return num_requested
    
    @node_event
    def create_bell_pairs(self, receiver: int, num_requested: int=1, *, request: Packet=None) -> None:
        
        """
        Creates the number of requested qubits, no matter how long it takes
        
        Args:
            receiver (int): receiver of bell pairs
            num_requested (int): number of requested bell pairs
            
        Returns:
            /
        """
        
        if not isinstance(num_requested, (int, np.integer)) or num_requested < 1:
            raise ValueError("Requested count must be a positive integer")
        if num_requested < 1:
            raise ValueError('Attempting 0 qubits')
        
        self._sim.schedule_event(SendEvent(self._time, self.id))
        
        if not self.has_space(receiver, 0, num_requested):
            num_requested = self.remaining_space(receiver, 0)
        
        if not self._sim._hosts[receiver].has_space(self.id, 1, num_requested):
            num_requested = self._sim._hosts[receiver].remaining_space(self.id, 1)
        
        if not num_requested:
            return
        
        self._connections['eqs'][receiver].create_bell_pairs(num_requested, request=request)
        return num_requested
    
    @gate_event
    async def execute_gate(self, gate: str, *args: Any, success_prob: float = 1.,
                           false_prob: float = 0., combine: bool = True, remove: bool = True) -> int | None:
        """Apply a gate at completion, yielding to other hosts during its duration.

        The caller owns/reserves input qubits until this await completes. Noise
        during the gate is supplied by the gate model, not memory extraction.
        Synchronous apply_gate remains available for offline calculations.
        """
        return self.apply_gate(gate, *args, success_prob=success_prob,
                               false_prob=false_prob, combine=combine, remove=remove, _scheduled=True)

    @gate_event
    def apply_gate(self, gate: str, *args: List[Any], apply: bool=True, success_prob: float=1., false_prob: float=0., combine: bool=True, remove: bool=True, _scheduled: bool=False) -> int | None:
        
        """
        Applys a gate to qubits
        
        Args:
            gate (str): gate to apply
            *args (list): variable length argument list
            combine (bool): to combine qubits that are in different qsystems
            remove (bool): to remove qubits
            
        Returns:
            res (int/np.ndarray/None): result of the gate
        """
        
        if not apply:
            return self._gates[gate](*args, apply=False)
        
        if combine and gate in ['CNOT', 'CX', 'CY', 'CZ', 'CPHASE', 'CU', 'SWAP', 'iSWAP', 'bell_state', 'bsm']:
            combine_state(args[:2])
        if combine and gate in ['QAND', 'QOR', 'QXOR', 'QNAND', 'QNOR', 'QXNOR', 'CCU', 'CSWAP']:
            combine_state(args[:3])
        
        prob = self._rng.random()
        if prob < success_prob:
            event = 0
        elif prob < success_prob + false_prob:
            event = 1
        else:
            event = 2
        
        return self.event_dict[event](gate, remove, args)
    
    @node_event
    def send_qubit(self, receiver: int, qubit: Qubit) -> None:
        
        """
        Sends a qubit to the specified receiver
        
        Args:
            receiver (int): receiver to send qubit to
            qubit (Qubit): qubit to send
            
        Returns:
            /
        """
        
        self._sim.schedule_event(SendEvent(self._time, self.id))
        recv_event = ReceiveEvent(self._time + self._channels['qc'][receiver][SEND]._propagation_time, receiver)
        self._sim.schedule_event(recv_event)
        
        self._channels['qc'][receiver][SEND].put(qubit, recv_event._end_time)
    
    @node_event
    async def receive_qubit(self, sender=None, time_out=None):
        """Receive an arrived qubit; timeout is measured in simulation seconds."""
        return await self._receive('qc', sender, time_out)
         
    @node_event
    def send_packet(self, _packet: Packet) -> None:
        
        """
        Sends a packet to the specified receiver in the packet
        
        Args:
            _packet (Packet): packet to send
            
        Returns:
            /
        """
        
        _packet = _packet.copy()
        _packet.set_hop(self.id, _packet.dst)
        self._time += self._channels['pc'][_packet.dst][SEND]._sending_time(len(_packet))
        self._sim.schedule_event(SendEvent(self._time, self.id))
        _recv_event = ReceiveEvent(self._time + self._channels['pc'][_packet.dst][SEND]._propagation_time, _packet.dst)
        self._sim.schedule_event(_recv_event)
        
        self._channels['pc'][_packet.dst][SEND].put(_packet, _recv_event._end_time, _recv_event._event_counter)
        
    @node_event
    async def receive_packet(self, sender=None, time_out=None, *, layer=None, end_sender=None):
        """Receive an arrived packet, optionally selecting a stack layer.

        sender selects a neighboring last hop; end_sender selects the L3 source.
        A filtered application receive dispatches installed L1/L2/L4 control
        protocols; unrelated application messages remain in the inbox.
        Timeout is measured in simulation seconds.
        """
        return await self._receive('pc', sender, time_out, layer, end_sender=end_sender)
    
    async def wait(self, duration: float) -> None:
        """Suspend this coroutine for simulation seconds, without blocking peers.

        The scheduler advances the node clock at completion. A zero duration
        still yields through a completion event; cancellation invalidates it.
        This primitive is deliberately not decorated with node_event.
        """
        duration = _duration(duration)
        future = asc.get_running_loop().create_future()

        def complete() -> None:
            if not future.done():
                future.set_result(None)

        event = CompletionEvent(self._time + duration, self.id, complete)
        self._sim.schedule_event(event)
        try:
            await future
        finally:
            event.active = False
    
    @property
    def id(self) -> int:
        
        """
        Returns the node id of host
        
        Args:
            /
            
        Returns:
            _id (int): id of host
        """
        
        return self._node_id
    
    @property
    def neighbors(self) -> Set[int]:
        
        """
        Retrieves the neighbors of the host
        
        Args:
            /
            
        Returns:
            _neighbors (set): neighbors
        """
        
        return self._neighbors
    
    @property
    def l1_qprotocol(self) -> QProtocol:
        
        """
        Returns the L1 QProtocol
        
        Args:
            /
            
        Returns:
            L1_QProgram (QProgram): L1 program to return
        """
        
        return self._qprotocols[L1]
    
    @property
    def l2_qprotocol(self) -> QProtocol:
        
        """
        Returns the L2 QProtocol
        
        Args:
            /
            
        Returns:
            L2_QProgram (QProgram): L2 program to return
        """
        
        return self._qprotocols[L2]
    
    @property
    def l3_qprotocol(self) -> QProtocol:
        
        """
        Returns the L3 QProgram
        
        Args:
            /
            
        Returns:
            L3_QProgram (QProgram): L3 program to return
        """
        
        return self._qprotocols[L3]
    
    @property
    def l4_qprotocol(self) -> QProtocol:
        
        """
        Returns the L4 QProgram
        
        Args:
            /
            
        Returns:
            L4_QProgram (QProgram): L4 program to return
        """
        
        return self._qprotocols[L4]
    
    @property
    def l7_qprotocol(self) -> QProtocol:
        
        """
        Returns the L7 QProgram
        
        Args:
            /
            
        Returns:
            L7_QProgram (QProgram): L7 program to return
        """
        
        return self._qprotocols[L7]
    
    @node_event
    def has_space(self, host: int, store: int, num_qubits: int=1) -> bool:
        
        """
        Checks whether the memory has space for the number of qubits
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            num_qubits (int): number of qubits to check if they fit into memory
            
        Returns:
            has_space (bool): whether number of qubits fit into memory
        """
        
        return self._memory[host][store].has_space(num_qubits)
    
    @node_event
    def remaining_space(self, host: int, store: int) -> int:
        
        """
        Returns the remaining size of the memory
        
        Args:
            host (int): the host the memory points to 
            store (int): SEND or RECEIVE store
            
        Returns:
            remaining_space (int): remaining size of the memory
        """
        
        return self._memory[host][store].remaining_space()
    
    @node_event
    def memory_size(self, host: int, store: int) -> int:
        
        """
        Returns the size of a given memory
        
        Args:
            host (int): host the memory points to
            store (int): SEND or RECEIVE store
            
        Returns:
            memory_size (int): size of the memory
        """
        
        return self._memory[host][store].size
    
    @node_event
    def change_memory_size(self, host: int, store: int, size: int) -> None:
        
        """
        Changes the size of the specified memory
        
        Args:
            host (int): the host the memory points to 
            store (int): SEND or RECEIVE store
            size (int): new size of the memory
            
        Returns:
            /
        """
        
        self._memory[host][store].change_size(size)
    
    @node_event
    def l0_num_qubits(self, host: int, store: int) -> int:
        
        """
        Returns the number of qubits from the L0 store
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            
        Returns:
            l0_num_qubits (int): number of qubits in the L0 store
        """
        
        return self._memory[host][store].num_qubits(L0)
    
    @node_event
    def l1_num_qubits(self, host: int, store: int) -> int:
        
        """
        Returns the number of qubits from the L1 store
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            
        Returns:
            l1_num_qubits (int): number of qubits in the L1 store
        """
        
        return self._memory[host][store].num_qubits(L1)
    
    @node_event
    def l2_num_qubits(self, host: int, store: int) -> int:
        
        """
        Returns the number of qubits from the L2 store
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            
        Returns:
            l2_num_qubits (int): number of qubits in the L2 store
        """
        
        return self._memory[host][store].num_qubits(L2)
    
    @node_event
    def l3_num_qubits(self, host: int, store: int) -> int:
        """Compatibility alias for ready_pair_count; not a physical stage count."""
        return self.ready_pair_count(host, store)

    @node_event
    def memory_stage_count(self, neighbor: int, store: int, layer: int) -> int:
        """Count physical slots in exactly one of the four lifecycle stages."""
        if layer not in (L0, L1, L2, L3):
            raise ValueError('Memory layer must be L0, L1, L2 or L3')
        return self._memory[neighbor][store].num_qubits(layer)

    @node_event
    def ready_pair_count(self, host: int, store: int) -> int:
        """Confirmed inputs usable by forwarding (raw L1 or improved L3)."""
        if L2 in self._qprotocols and self.l2_qprotocol.fip_mode != 'necp':
            return self._memory[host][store].available_qubits(L3)
        return self._memory[host][store].ready_count()
    
    @node_event
    def l0_store_qubit(self, qubit: Qubit, host: int, store: int, index: int=-1) -> None:
        
        """
        Stores a qubit in the L0 memory
        
        Args:
            qubit (Qubit): list of qubits
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to qubits at
            
        Returns:
            /
        """
        
        self._memory[host][store].store_qubit(L0, qubit, index, self._time)
        
    @node_event
    def l1_store_qubit(self, qubit: Qubit, host: int, store: int, index: int=-1) -> None:
        
        """
        Stores a qubit in the L1 memory
        
        Args:
            qubit (Qubit): list of qubits
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to qubits at
            
        Returns:
            /
        """
        
        self._memory[host][store].store_qubit(L1, qubit, index, self._time)
        
    @node_event
    def l2_store_qubit(self, qubit: Qubit, host: int, store: int, index: int=-1) -> None:
        
        """
        Stores qubits in the L2 memory
        
        Args:
            qubit (Qubit): list of qubits
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to qubits at
            
        Returns:
            /
        """
        
        self._memory[host][store].store_qubit(L2, qubit, index, self._time)
        
    @node_event
    def l3_store_qubit(self, qubit: Qubit, host: int, store: int, index: int=-1) -> None:
        
        """
        Stores qubits in the L3 memory
        
        Args:
            qubit (Qubit): list of qubits
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to qubits at
            
        Returns:
            /
        """
        
        self._memory[host][store].store_qubit(L3, qubit, index, self._time)
     
    @node_event
    def l0_retrieve_qubit(self, host: int, store: int, index: int=None, _mode: bool=False) -> Qubit | None:
        
        """
        Retrieves a qubit from the L0 store
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to retrieve from
            
        Returns:
            qubit (Qubit/None): retrieved qubit
        """
        
        return self._memory[host][store].retrieve_qubit(L0, index, self._time, _mode)
    
    @node_event
    def l1_retrieve_qubit(self, host: int, store: int, index: int=None, _mode: bool=False) -> Qubit | None:
        
        """
        Retrieves a qubit from the L1 store
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to retrieve from
            
        Returns:
            qubit (Qubit/None): retrieved qubit
        """
        
        return self._memory[host][store].retrieve_qubit(L1, index, self._time, _mode)
    
    @node_event
    def l2_retrieve_qubit(self, host: int, store: int, index: int=None, _mode: bool=False) -> Qubit | None:
        
        """
        Retrieves a qubit from the L2 store
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to retrieve from
            
        Returns:
            qubit (Qubit/None): retrieved qubit
        """
        
        return self._memory[host][store].retrieve_qubit(L2, index, self._time, _mode)
    
    @node_event
    def l3_retrieve_qubit(self, host, store, index=None, offset_index=None):
        """Compatibility alias for retrieve_ready_pair."""
        return self.retrieve_ready_pair(host, store, index, offset_index)

    @node_event
    def retrieve_ready_pair(self, host: int, store: int, index: int | None = None,
                            offset_index: int | None = None) -> Qubit | None:
        """Consume a confirmed forwarding input without changing its stage."""
        if L2 in self._qprotocols and self.l2_qprotocol.fip_mode != 'necp':
            return self._memory[host][store].retrieve_qubit(L3, index, self._time, False, offset_index)
        return self._memory[host][store].retrieve_ready(self._time, index, offset_index)
    
    @node_event
    def l0_move_qubits_l1(self, host: int, store: int, indices: List[int]) -> None:
        
        """
        Moves qubits given the indices from L0 memory to L1 memory
        
        Args:
            host (int): the host the memory points to
            store (int): entanglement store Send or Receive
            indices (list): indices to remove
            
        Returns:
            /
        """
        
        self._memory[host][store].move_qubits(L0, L1, indices)

    @node_event
    def confirm_generation(self, neighbor: int, store: int, request_id: int, success: np.ndarray) -> bool | None:
        """Match a herald to concrete L0 inputs rather than a moving prefix."""
        if neighbor not in self._memory:
            return False
        return self._memory[neighbor][store].confirm_generation(request_id, success)
        
    @node_event
    def l1_move_qubits_l2(self, host: int, store: int, indices: List[int]) -> None:
        
        """
        Moves qubits given the indices from L1 memory to L2 memory
        
        Args:
            host (int): the host the memory points to
            store (int): entanglement store Send or Receive
            indices (list): indices to remove
            
        Returns:
            /
        """
        
        self._memory[host][store].move_qubits(L1, L2, indices)
    
    @node_event
    def l1_move_qubits_l3(self, host: int, store: int, indices: List[int]) -> None:
        
        """
        Moves qubits from the L1 store to the L3 store
        
        Args:
            host (int): the host the memory points to
            store (int): entanglement store Send or Receive
            indices (list): indices to remove
            
        Returns:
            /
        """
        
        self._memory[host][store].move_qubits(L1, L3, indices)
    
    @node_event
    def l2_move_qubits_l3(self, host: int, store: int, indices: List[int]) -> None:
        
        """
        Moves qubits given the indices from L2 memory to L3 memory
        
        Args:
            host (int): the host the memory points to
            store (int): entanglement store Send or Receive
            indices (list): indices to remove
            
        Returns:
            /
        """
        
        self._memory[host][store].move_qubits(L2, L3, indices)

    @node_event
    def l3_move_qubits_l1(self, host: int, store: int, indices: List[int]) -> None:
        
        """
        Moves qubits given the indices from L3 memory to L1 memory
        
        Args:
            host (int): the host the memory points to
            store (int): entanglement store Send or Receive
            indices (list): indices to remove
            
        Returns:
            /
        """
        
        self._memory[host][store].move_qubits(L3, L1, indices)
    
    @node_event
    def l3_remove_qubits(self, host: int, store: int, indices: List[int]) -> None:
        
        """
        Removes qubits in the L3 memory based on the indices
        
        Args:
            host (int): the host the memory points to
            store (int): entanglement store Send or Receive
            indices (list): indices to remove
            
        Returns:
            /
        """
        
        self._memory[host][store].move_qubits(L3, L3, indices)
      
    @node_event
    def l0_discard_qubits(self, host: int, store: int) -> None:
        
        """
        Discards all qubits in L0 store
        
        Args:
            host (int): the host the memory points to
            store (int) SEND or RECEIVE store
            
        Returns:
            /
        """
        
        self._memory[host][store].discard_qubits(L0)
        
    @node_event
    def l1_discard_qubits(self, host: int, store: int) -> None:
        
        """
        Discards all qubits in L1 store
        
        Args:
            host (int): the host the memory points to
            store (int) SEND or RECEIVE store
            
        Returns:
            /
        """
        
        self._memory[host][store].discard_qubits(L1)
        
    @node_event
    def l2_discard_qubits(self, host: int, store: int) -> None:
        
        """
        Discards all qubits in L2 store
        
        Args:
            host (int): the host the memory points to
            store (int) SEND or RECEIVE store
            
        Returns:
            /
        """
        
        self._memory[host][store].discard_qubits(L2)
        
    @node_event
    def l3_discard_qubits(self, host: int, store: int) -> None:
        
        """
        Discards all qubits in L0 store
        
        Args:
            host (int): the host the memory points to
            store (int) SEND or RECEIVE store
            
        Returns:
            /
        """
        
        self._memory[host][store].discard_qubits(L3)
    
    @node_event
    def discard_all_qubits(self, host: int, store: int) -> None:
        
        """
        Discards all qubits in all the stores
        
        Args:
            host (int): the host the memory points to
            store (int) SEND or RECEIVE store
            
        Returns:
            /
        """
        
        self._memory[host][store].discard_qubits()
    
    @node_event
    def l2_num_purification(self, host: int, store: int) -> int:
        
        """
        Returns the number of purifications in the store
        
        Args:
            host (int): the host the memory points to
            store (int) SEND or RECEIVE store
            
        Returns:
            l2_num_purification (int): number of available purifications
        """
        
        return int(np.floor(self.l1_num_qubits(host, store) / 2))
    
    @node_event
    def l0_estimate_fidelity(self, host: int, store: int, index: int=None) -> float:
        
        """
        Estimates the fidelity of a qubit in L0 memory
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE memory
            index (int): index of qubit
            
        Returns:
            fidelity (float): estimated fidelity
        """
        
        return self._memory[host][store].estimate_fidelity(L0, index, self._time)
    
    @node_event
    def l1_estimate_fidelity(self, host: int, store: int, index: int=None) -> float:
        
        """
        Estimates the fidelity of a qubit in L1 memory
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE memory
            index (int): index of qubit
            
        Returns:
            fidelity (float): estimated fidelity
        """
        
        return self._memory[host][store].estimate_fidelity(L1, index, self._time)
    
    @node_event
    def l2_estimate_fidelity(self, host: int, store: int, index: int=None) -> float:
        
        """
        Estimates the fidelity of a qubit in L2 memory
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE memory
            index (int): index of qubit
            
        Returns:
            fidelity (float): estimated fidelity
        """
        
        return self._memory[host][store].estimate_fidelity(L2, index, self._time)
    
    @node_event
    def l3_estimate_fidelity(self, host: int, store: int, index: int=None) -> float:
        
        """
        Estimates the fidelity of a qubit in L3 memory
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE memory
            index (int): index of qubit
            
        Returns:
            fidelity (float): estimated fidelity
        """
        
        return self._memory[host][store].estimate_fidelity(L3, index, self._time)
    
    @node_event
    def l1_check_results(self, host: int, store: int, *, request_id: int=None) -> bool:
        """Check the L1 buffer, optionally for a particular request."""
        return self._l1_result_index(host, store, request_id) is not None
    
    @node_event
    def l2_check_results(self, host, store, *, request_id=None):
        return self._l2_result_index(host, store, request_id) is not None
    
    @node_event
    def l1_num_results(self, host: int, store: int) -> int:
        
        """
        Returns the number of results in L1 store
        
        Args:
            host (int): connected host
            store (int): SEND or RECEIVE store
            
        Returns:
            l1_num_results (int): number of results in L1 store
        """
        
        return len(self._layer_results[host][store][L1])
    
    @node_event
    def l2_num_results(self, host: int, store: int) -> int:
        
        """
        Returns the number of results in L2 store
        
        Args:
            host (int): connected host
            store (int): SEND or RECEIVE store
            
        Returns:
            l2_num_results (int): number of results in L2 store
        """
        
        return len(self._layer_results[host][store][L2])

    @node_event
    def l1_store_result(self, store: int, packet: Packet) -> None:
        """Keep a local result and its request ID in the node's L1 buffer."""
        self._layer_results[packet.src][store][L1].append(
            (packet.request_id, packet.l1_success.copy()))
        
    @node_event
    def l2_store_result(self, store, packet, qubits=None):
        """Keep an outgoing batch in the existing per-neighbor L2 result buffer."""
        batch = L2Batch(packet.request_id, packet.l2_success.copy(),
                        list(qubits or []), packet.l2_needed, packet.l2_code)
        self._layer_results[packet.dst][store][L2].append(batch)
        
    @node_event
    def l1_retrieve_result(self, host: int, store: int, *, request_id: int=None) -> np.ndarray:
        """Consume a matching L1 result, or the oldest result when ID is omitted."""
        index = self._l1_result_index(host, store, request_id)
        if index is None:
            raise KeyError(f'No L1 result for node {host}, store {store}, request {request_id}')
        _, result = self._layer_results[host][store][L1].pop(index)
        return result

    def _l1_result_index(self, host, store, request_id):
        results = self._layer_results[host][store][L1]
        return next((index for index, (identifier, _) in enumerate(results)
                     if request_id is None or identifier == request_id), None)
    
    @node_event
    def l2_retrieve_result(self, host, store, *, request_id=None):
        """Consume a matching result; the default preserves FIFO array retrieval."""
        entry = self._l2_result_entry(host, store, request_id)
        index = self._l2_result_index(host, store, request_id)
        self._layer_results[host][store][L2].pop(index)
        return getattr(entry, 'results', entry)
    
    @node_event
    def l1_compare_results(self, packet: Packet) -> np.ndarray:
        
        """
        Compares the L1 results of the packet with the result in storage
        
        Args:
            packet (Packet): packet to compare results of
            
        Returns:
            res (np.ndarray): result of the comparison
        """
        
        store = int(not packet.l1_ack)
        index = self._l1_result_index(packet.src, store, packet.request_id)
        if index is None:
            raise KeyError(f'No local L1 result for request {packet.request_id}')
        _, local = self._layer_results[packet.src][store][L1][index]
        if local.shape != packet.l1_success.shape:
            raise ValueError('L1 result counts do not match')
        stor_res = self.l1_retrieve_result(packet.src, store, request_id=packet.request_id)
        return np.logical_and(packet.l1_success, stor_res)
    
    @node_event
    def l2_compare_results(self, packet, *, consume=True):
        store = int(packet.l2_ack)
        entry = self._l2_result_entry(packet.src, store, packet.request_id)
        local = getattr(entry, 'results', entry)
        if local.shape != packet.l2_success.shape:
            raise ValueError('Purification result counts do not match')
        if consume:
            self.l2_retrieve_result(packet.src, store, request_id=packet.request_id)
        return local == packet.l2_success
    
    @node_event
    def l1_packets(self, host: int, store: int) -> List[Packet]:
        
        """
        Returns all packets in L1 packet store
        
        Args:
            /
            
        Returns:
            l1_packets (list): packets in L1 store
        """
        
        return self._packets[host][store][L1]
    
    @node_event
    def l2_packets(self, host: int, store: int) -> List[Packet]:
        
        """
        Returns all packets in L2 packet store
        
        Args:
            /
            
        Returns:
            l2_packets (list): packets in L2 store
        """
        
        return self._packets[host][store][L2]
    
    @node_event
    def l3_packets(self, host: int, store: int) -> List[Packet]:
        
        """
        Returns all packets in L3 packet store
        
        Args:
            /
            
        Returns:
            l3_packets (list): packets in L3 store
        """
        
        return self._packets[host][store][L3]
    
    @node_event
    def l1_check_packets(self, host: int, store: int, *, request_id: int=None) -> bool:
        """Check queued L1 packets, optionally selecting a request."""
        return any(request_id is None or packet.request_id == request_id
                   for packet in self._packets[host][store][L1])
    
    @node_event
    def l2_check_packets(self, host, store, *, request_id=None):
        return any(request_id is None or packet.request_id == request_id
                   for packet in self._packets[host][store][L2])
    
    @node_event
    def l3_check_packets(self, host: int, store: int) -> bool:
        
        """
        Checks whether packets are in L3 store
        
        Args:
            /
            
        Returns:
            l3_check_packets (bool): check if packets in L3 store
        """
        
        if self._packets[host][store][L3]:
            return True
        return False
    
    @node_event
    def l1_num_packets(self, host: int, store: int) -> int:
        
        """
        Returns the number of packets in L1 store
        
        Args:
            /
            
        Returns:
            l1_num_packets (int): number of packets in L1 store
        """
        
        return len(self._packets[host][store][L1])
    
    @node_event
    def l2_num_packets(self, host: int, store: int) -> int:
        
        """
        Returns the number of packets in L2 store
        
        Args:
            /
            
        Returns:
            l2_num_packets (int): number of packets in L2 store
        """
        
        return len(self._packets[host][store][L2])
    
    @node_event
    def l3_num_packets(self, host: int, store: int) -> int:
        
        """
        Returns the number of packets in L3 store
        
        Args:
            /
            
        Returns:
            l3_num_packets (int): number of packets in L3 store
        """
        
        return len(self._packets[host][store][L3])
    
    @node_event
    def l1_store_packet(self, host: int, store: int, packet: Packet) -> None:
        
        """
        Stores a packet in the L1 store
        
        Args:
            packet (Packet): packet to store
            
        Returns:
            /
        """
        
        self._packets[host][store][L1].append(packet)
        
    @node_event
    def l2_store_packet(self, host: int, store: int, packet: Packet) -> None:
        
        """
        Stores a packet in the L2 store
        
        Args:
            packet (Packet): packet to store
            
        Returns:
            /
        """
        
        self._packets[host][store][L2].append(packet)
        
    @node_event
    def l3_store_packet(self, host: int, store: int, packet: Packet, offset_index: int=None) -> None:
        
        """
        Stores a packet in the L3 store
        
        Args:
            packet (Packet): packet to store
            
        Returns:
            /
        """
        
        self._packets[host][store][L3].append((packet, offset_index))
        
    @node_event
    def l1_retrieve_packet(self, host: int, store: int, index: int=0, *, request_id: int=None) -> Packet:
        """Consume by queue index, or by request ID when provided."""
        packets = self._packets[host][store][L1]
        if request_id is not None:
            index = next((i for i, packet in enumerate(packets)
                          if packet.request_id == request_id), None)
        if index is None:
            raise KeyError(f'No queued L1 packet for request {request_id}')
        return packets.pop(index)
    
    @node_event
    def l2_retrieve_packet(self, host, store, index=0, *, request_id=None):
        packets = self._packets[host][store][L2]
        if request_id is not None:
            index = next((i for i, packet in enumerate(packets)
                          if packet.request_id == request_id), None)
        if index is None:
            raise KeyError(f'No L2 packet for request {request_id}')
        return packets.pop(index)
        
    @node_event
    def l3_retrieve_packet(self, host: int, store: int, index: int=0) -> Packet:
        
        return self._packets[host][store][L3].pop(index)
    
    @node_event
    def l3_add_offset(self, host: int, store: int, offset: int) -> int:
        
        """
        Adds a L3 offset to the memory
        
        Args:
            host (int): host the memory points to
            store (int): SEND or RECEIVE store
            offset (int): offset to add
            
        Returns:
            /
        """
        
        raw = L2 not in self._qprotocols or self.l2_qprotocol.fip_mode == 'necp'
        return self._memory[host][store].add_offset(offset, ready=raw)
    
    @node_event
    def l3_remove_offset(self, host: int, store: int, index: int=0) -> None:
        
        """
        Removes a L3 offset memory
        
        Args:
            host (int): host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to remove
            
        Returns:
            /
        """
        
        return self._memory[host][store].remove_offset(index)

    @node_event
    def l0_retrieve_time_stamp(self, host: int, store: int, index: int=None) -> float | None:
        
        """
        Retrieves the time stamp of a qubit in L0 memory
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to retrieve from
            
        Returns:
            time_stamp (float/None): time stamp of qubit
        """
        
        return self._memory[host][store].retrieve_time_stamp(L0, index)
    
    @node_event
    def l1_retrieve_time_stamp(self, host: int, store: int, index: int=None) -> float | None:
        
        """
        Retrieves the time stamp of a qubit in L1 memory
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to retrieve from
            
        Returns:
            time_stamp (float/None): time stamp of qubit
        """
        
        return self._memory[host][store].retrieve_time_stamp(L1, index)
    
    @node_event
    def l2_retrieve_time_stamp(self, host: int, store: int, index: int=None) -> float | None:
        
        """
        Retrieves the time stamp of a qubit in L2 memory
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to retrieve from
            
        Returns:
            time_stamp (float/None): time stamp of qubit
        """
        
        return self._memory[host][store].retrieve_time_stamp(L2, index)
    
    @node_event
    def l3_retrieve_time_stamp(self, host: int, store: int, index: int=None) -> float | None:
        
        """
        Retrieves the time stamp of a qubit in L3 memory
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to retrieve from
            
        Returns:
            time_stamp (float/None): time stamp of qubit
        """
        
        return self._memory[host][store].retrieve_time_stamp(L3, index)
    
    @node_event
    def estimate_link_fidelity(self, neighbor: int, store: int) -> float:
        """Estimate conditional Phi+ fidelity for a fresh adjacent Bell pair.

        ``store`` is SEND (0) or RECEIVE (1). Estimates come from the directional
        connection models installed by set_eqs_connection, without consulting
        remote outcomes or stored states. Later memory noise is excluded.
        Override this method to supply a calibrated/custom link estimator.
        """
        if store not in (SEND, RECEIVE):
            raise ValueError('store must be SEND (0) or RECEIVE (1)')
        if neighbor not in self._link_fidelities:
            raise ValueError(f'No entanglement link configured for neighbor {neighbor}')
        return self._link_fidelities[neighbor][store]

    @node_event
    def pr_owns(self, token: tuple[int, int], key: tuple[int, int]) -> bool:
        entry = self._pr_halves.get(token)
        return entry is not None and entry[3] == key

    @node_event
    def pr_archive(self, key: tuple[int, int]) -> None:
        """Release completed control metadata after all outputs were retrieved."""
        entry = self.pr_requests[key]
        if entry.status not in ('complete', 'failed') or any(value[3] == key for value in self._pr_halves.values()):
            raise ValueError('Cannot archive active requests or outstanding quantum outputs')
        del self.pr_requests[key]
        for identity, session in list(self.pr_sessions.items()):
            if not session.requests and not any(session.cache.values()):
                del self.pr_sessions[identity]

    @node_event
    def pr_has(self, token: tuple[int, int]) -> bool:
        """Whether this node still owns a reserved PR half."""
        return token in self._pr_halves

    @node_event
    def pr_capture(self, neighbor: int, store: int, token: tuple[int, int], key: tuple[int, int]) -> bool:
        """Reserve a confirmed raw input; session ownership is metadata."""
        if token in self._pr_halves:
            raise ValueError('Duplicate PR pair token')
        memory = self._memory[neighbor][store]
        layer = memory.ready_layer()
        qubit = memory.retrieve_ready(self._time)
        if qubit is None:
            return False
        self._pr_put(token, qubit, neighbor, store, key, layer=layer)
        return True

    def _pr_put(self, token: tuple[int, int], qubit: Qubit, neighbor: int, store: int, key: tuple[int, int], *, layer: int = L2) -> None:
        self._memory[neighbor][store].store_qubit(layer, qubit, -1, self._time)
        self._memory[neighbor][store].reserve_session(qubit, layer)
        self._pr_halves[token] = (qubit, neighbor, store, key)

    @node_event
    def pr_confirm(self, token: tuple[int, int]) -> None:
        """Promote a survivor only after classical acceptance."""
        qubit, neighbor, store, _ = self._pr_halves[token]
        self._memory[neighbor][store].confirm_session(qubit)

    @node_event
    def pr_stage_inputs(self, tokens: tuple[tuple[int, int], ...]) -> None:
        """Reserve fidelity-improvement inputs in L2 before preparation begins."""
        for token in tokens:
            qubit, neighbor, store, _ = self._pr_halves[token]
            self._memory[neighbor][store].stage_session(qubit, L2)

    def _pr_take(self, token: tuple[int, int]) -> tuple[Qubit | None, int, int, tuple[int, int]]:
        qubit, neighbor, store, key = self._pr_halves.pop(token)
        memory = self._memory[neighbor][store]
        qubit = memory.retrieve_session(qubit, self._time)
        return qubit, neighbor, store, key

    @node_event
    def pr_purify(self, control: tuple[int, int], target: tuple[int, int]) -> int | None:
        """Local BBPSSW CNOT and Z measurement; acceptance needs the peer bit."""
        first, neighbor, store, key = self._pr_take(control)
        second, _, _, _ = self._pr_take(target)
        if first is None or second is None:
            remove_qubits([q for q in (first, second) if q is not None])
            return None
        self.apply_gate('CNOT', first, second)
        bit = self.apply_gate('measure', second, 'z')
        self._pr_put(control, first, neighbor, store, key)
        return int(bit)

    def _pr_begin_operation(self, token: tuple[int, int], *, layer: int | None = None) -> Qubit | None:
        owned, neighbor, store, _ = self._pr_halves[token]
        layer = self._memory[neighbor][store].session_layer(owned) if layer is None else layer
        qubit, neighbor, store, key = self._pr_take(token)
        if qubit is not None:
            self._pr_put(token, qubit, neighbor, store, key, layer=layer)
        return qubit

    def _pr_finish_operation(self, token: tuple[int, int], layer: int = L2, *, consumed: bool = False) -> None:
        qubit, neighbor, store, _ = self._pr_halves[token]
        self._memory[neighbor][store].finish_session_operation(qubit, self._time, layer, consumed=consumed)
        if consumed:
            del self._pr_halves[token]

    @node_event
    async def execute_pr_purification(self, control: tuple[int, int], target: tuple[int, int]) -> int | None:
        """Keep both physical slots in L2 throughout the local parity circuit."""
        first = self._pr_begin_operation(control, layer=L2)
        second = self._pr_begin_operation(target, layer=L2)
        if first is None or second is None:
            self.pr_discard(control)
            self.pr_discard(target)
            return None
        await self.execute_gate('CNOT', first, second)
        bit = await self.execute_gate('measure', second, 'z')
        self._pr_finish_operation(target, consumed=True)
        self._pr_finish_operation(control)
        return int(bit)

    @node_event
    async def execute_pr_swap(self, incoming: tuple[int, int], outgoing: tuple[int, int]) -> int | None:
        first = self._pr_begin_operation(incoming)
        second = self._pr_begin_operation(outgoing)
        if first is None or second is None:
            self.pr_discard(incoming)
            self.pr_discard(outgoing)
            return None
        outcome = await self.execute_gate('bsm', first, second)
        self._pr_finish_operation(incoming, consumed=True)
        self._pr_finish_operation(outgoing, consumed=True)
        return int(outcome)

    @node_event
    async def execute_pr_correction(self, token: tuple[int, int], outcome: int, new_token: tuple[int, int]) -> bool:
        owned, neighbor, store, _ = self._pr_halves[token]
        layer = self._memory[neighbor][store].session_layer(owned)
        qubit = self._pr_begin_operation(token)
        if qubit is None:
            return False
        if outcome & 1:
            await self.execute_gate('X', qubit)
        if outcome & 2:
            await self.execute_gate('Z', qubit)
        self._pr_finish_operation(token, layer)
        self._pr_halves[new_token] = self._pr_halves.pop(token)
        return True

    @node_event
    async def execute_pr_twirl(self, token: tuple[int, int]) -> bool:
        qubit = self._pr_begin_operation(token)
        if qubit is None:
            return False
        await self.wait(self._gate_duration.get('twirl', 5e-6))
        self._twirl_pair(qubit)
        self._pr_finish_operation(token, L3)
        return True

    @node_event
    def pr_twirl(self, token: tuple[int, int]) -> bool:
        """Ideal bilateral Phi+-preserving twirl, applied once to the joint pair.

        This is a physical ensemble channel, not a fidelity observation exposed
        to protocol logic. Both local operations are coordinated by PR messages.
        No state larger than the surviving two-qubit pair is allocated here.
        """
        qubit, neighbor, store, key = self._pr_take(token)
        if qubit is None:
            return False
        self._twirl_pair(qubit)
        _advance_local_time(self, self._gate_duration.get('twirl', 5e-6))
        self._pr_put(token, qubit, neighbor, store, key, layer=L3)
        return True

    @staticmethod
    def _twirl_pair(qubit: Qubit) -> None:
        if qubit.state.shape != (4, 4):
            raise ValueError('PR twirling requires an independent two-qubit pair')
        bell = np.array([[.5, 0, 0, .5], [0, 0, 0, 0],
                         [0, 0, 0, 0], [.5, 0, 0, .5]], dtype=complex)
        fidelity = float(np.trace(bell @ qubit.state).real)
        qubit.state = fidelity * bell + (1 - fidelity) * (np.eye(4) - bell) / 3

    @node_event
    def pr_swap(self, incoming: tuple[int, int], outgoing: tuple[int, int]) -> int | None:
        first, _, _, _ = self._pr_take(incoming)
        second, _, _, _ = self._pr_take(outgoing)
        if first is None or second is None:
            remove_qubits([q for q in (first, second) if q is not None])
            return None
        return int(self.apply_gate('bsm', first, second))

    @node_event
    def pr_correct(self, token: tuple[int, int], outcome: int, new_token: tuple[int, int]) -> bool:
        owned, neighbor, store, _ = self._pr_halves[token]
        layer = self._memory[neighbor][store].session_layer(owned)
        qubit, neighbor, store, key = self._pr_take(token)
        if qubit is None:
            return False
        if outcome & 1:
            self.apply_gate('X', qubit)
        if outcome & 2:
            self.apply_gate('Z', qubit)
        self._pr_put(new_token, qubit, neighbor, store, key, layer=layer)
        return True

    @node_event
    def pr_discard(self, token: tuple[int, int]) -> None:
        entry = self._pr_halves.pop(token, None)
        if entry is not None:
            qubit, neighbor, store, _ = entry
            self._memory[neighbor][store].discard_session_qubit(qubit)

    @node_event
    def pr_discard_request(self, key: tuple[int, int], keep: tuple[tuple[int, int], ...] = ()) -> None:
        for token, entry in list(self._pr_halves.items()):
            if entry[3] == key and token not in keep:
                self.pr_discard(token)

    @node_event
    def pr_retrieve_output(self, token: tuple[int, int]) -> Qubit | None:
        """Consume an endpoint half, applying its elapsed-time memory noise."""
        return self._pr_take(token)[0]

    @node_event
    def encoded_refresh(self, tokens: tuple[tuple[int, int], ...]) -> bool:
        """Apply local waiting noise/extraction to a reserved physical block."""
        success = True
        for token in tokens:
            qubit, neighbor, store, key = self._pr_take(token)
            success &= qubit is not None
            if qubit is not None:
                self._pr_put(token, qubit, neighbor, store, key)
        return success

    @node_event
    def estimate_encoded_fidelity(self, code: str, input_fidelities: tuple[float, ...]) -> float:
        """Planning estimate for independent Werner inputs and ideal interfaces.

        Uses only advertised/aged input estimates, never stored quantum states.
        """
        probabilities = np.array([[fidelity, *([(1-fidelity)/3]*3)] for fidelity in input_fidelities])
        return float(self._graph_state_model(code).exact_segment_channel(probabilities)[0])

    @node_event
    def encoded_decode(self, tokens: tuple[tuple[int, int], ...], code: str) -> bool:
        """Consume n physical inputs into one effective logical interface pair.

        Both peers must refresh before this operation. The first token is the
        logical representative, not an unencoded surviving physical qubit.
        This ideal graph-interface channel averages Bell records and corrections;
        it does not expose residual errors or fidelity to protocol decisions.
        """
        model = self._graph_state_model(code)
        if len(tokens) != model.code.length or len(set(tokens)) != len(tokens):
            raise ValueError('Encoded decoding requires one complete block of distinct tokens')
        records = [self._pr_take(token) for token in tokens]
        qubits = [record[0] for record in records]
        if any(qubit is None for qubit in qubits):
            remove_qubits([qubit for qubit in qubits if qubit is not None])
            return False
        backend = get_qec_code(code)
        if len({qubit.qsystem_id for qubit in qubits}) != len(qubits):
            raise ValueError('Encoded graph interfaces require independent input pairs')
        probabilities = np.array([backend.bell_probabilities(qubit) for qubit in qubits])
        output = model.exact_segment_channel(probabilities)
        qubits[0].state = backend.bell_state(output)
        remove_qubits(qubits[1:])
        _, neighbor, store, key = records[0]
        self._pr_put(tokens[0], qubits[0], neighbor, store, key)
        return True

    @node_event
    def encoded_fuse(self, incoming: tuple[int, int], outgoing: tuple[int, int]) -> int | None:
        """Logical BSM on decoded graph-interface representatives, consuming both.

        This is the ideal logical fusion in the effective representation, not
        a physical BSM substituted for n uncorrected input pairs.
        """
        return self.pr_swap(incoming, outgoing)

    @node_event
    async def execute_encoded_fusion(self, incoming: tuple[int, int], outgoing: tuple[int, int]) -> int | None:
        """Await the ideal effective logical BSM, consuming both interfaces."""
        return await self.execute_pr_swap(incoming, outgoing)

    @node_event
    def encoded_retrieve_output(self, token: tuple[int, int]) -> Qubit | None:
        """Consume a decoded logical output half, applying its storage noise."""
        return self.pr_retrieve_output(token)

    @node_event
    def l2_purification(self, host: int, store: int, direction: bool=0, gate: str='CNOT', basis: str='Z', index_src: int=None, index_dst: int=None, rounds: int=1) -> List[int]:
    
        if not isinstance(rounds, int) or rounds < 1:
            raise ValueError('Purification rounds must be positive')
        if self.l1_num_qubits(host, store) < 2 * rounds:
            raise ValueError('Not enough L1 qubits for purification')
        results = np.zeros(rounds, dtype=np.bool_)
        for i in range(rounds):
            qubit_src = self.l1_retrieve_qubit(host, store, index_src)
            qubit_dst = self.l1_retrieve_qubit(host, store, index_dst)
            if qubit_src is None or qubit_dst is None:
                raise RuntimeError('Purification lost a qubit during extraction')
            if direction:
                self.apply_gate(gate, qubit_dst, qubit_src)
            else:
                self.apply_gate(gate, qubit_src, qubit_dst)
            results[i] = self.apply_gate('measure', qubit_dst, basis.lower(), remove=True)
            self.l2_store_qubit(qubit_src, host, store)
        return results
                
    def __S(self, gate: str, remove: bool, args: List[Any]):
            
            options = {'rng': self._rng} if gate in ('measure', 'bsm') else {}
            res = self._gates[gate](*args, **options)
            
            if remove and gate == 'measure':
                remove_qubits(args[:1])
            if remove and gate == 'bsm':
                remove_qubits(args[:2])
                
            return res
        
    def __F(self, gate: str, remove: bool, args: List[Any]):
        
        if gate in ['X', 'Y', 'Z', 'H', 'SX', 'SY', 'SZ', 'T', 'K', 'iSX', 'iSY', 'iSZ', 'iK', 'iT', 'Rx', 'Ry', 'Rz', 'PHASE', 'general_rotation', 'exp_pauli', 'custom_single_gate', 'measure']:
            remove_qubits(args[:1])
        if gate in ['CNOT', 'CX', 'CY', 'CZ', 'CPHASE', 'CU', 'SWAP', 'iSWAP', 'bell_state', 'bsm']:
            remove_qubits(args[:2])
        if gate in ['QAND', 'QOR', 'QXOR', 'QNAND', 'QNOR', 'QXNOR', 'CCU', 'CSWAP']:
            remove_qubits(args[:3])
    
    def __N(self, gate: str, remove: bool, args: List[Any]):
        
        if gate not in ['measure', 'bsm']:
            return None
        
        if remove and gate == 'measure':
            remove_qubits(args[:1])
            return int(self._rng.integers(0, 2))
        if remove and gate == 'bsm':
            remove_qubits(args[:2])
            return int(self._rng.integers(0, 4))

    @node_event
    def handle_packet(self, packet):
        """Dispatch a control or application packet through the owned stack."""
        return self.protocol_stack.handle_packet(packet)

    @node_event
    async def handle_packet_scheduled(self, packet: Packet) -> None:
        """Dispatch raw-router decisions, then await each reserved operation."""
        self._scheduled_dispatch = True
        try:
            self.handle_packet(packet)
            while self._pending_swaps:
                protocol, request, reservation = self._pending_swaps.pop(0)
                source, destination = request.src, request.dst
                await protocol.swap_entanglement_scheduled(request, reservation)
                if reservation is not None:
                    self.l3_remove_offset(source, RECEIVE, reservation)
                protocol.handle_stored_packets(destination)
        finally:
            self._scheduled_dispatch = False

    @node_event
    async def swap_pairs_scheduled(self, packet: Packet, reservation: int | None, *, correct: bool) -> None:
        """Execute raw BSMs with completion-time measurements and corrections."""
        if not self.l3_can_swap(packet, reservation):
            raise ValueError('Not enough reserved or outgoing Bell pairs for swapping')
        for index in range(packet.l3_needed):
            source = self.l3_retrieve_qubit(packet.src, RECEIVE, offset_index=reservation)
            target = self.l3_retrieve_qubit(packet.dst, SEND)
            if source is None or target is None:
                remove_qubits([qubit for qubit in (source, target) if qubit is not None])
                raise RuntimeError('Swapping lost a qubit during extraction')
            await self._swap_pair_scheduled(packet, index, source, target, correct=correct)

    async def _swap_pair_scheduled(self, packet: Packet, index: int, source: Qubit, target: Qubit, *, correct: bool) -> None:
        if correct and packet.l3_x_count[index]:
            await self.execute_gate('X', source)
        if correct and packet.l3_z_count[index]:
            await self.execute_gate('Z', source)
        if correct:
            packet.l3_reset_es(index)
        packet.l3_update_es(await self.execute_gate('bsm', source, target), index)

    @node_event
    def l2_purify(self, host, store, **kwargs):
        """Compatibility helper for one purification round."""
        return int(self.l2_purification(host, store, **kwargs)[0])

    @staticmethod
    def _packet_matches(packet: Packet, sender: int | None, layer: int | None, end_sender: int | None) -> bool:
        return ((sender is None or packet.src == sender) and
                (layer is None or packet.layer == layer) and
                (end_sender is None or (packet.has_l3 and packet.l3_src == end_sender)))

    def _buffer_received_packet(self, packet: Packet) -> None:
        """Keep an unmatched delivery in the shared neighbor/layer packet table."""
        layers = self._packets.setdefault(packet.src, {}).setdefault(PENDING_RECEIVE, {})
        layers.setdefault(packet.layer, []).append((self._packet_receive_sequence, packet))
        self._packet_receive_sequence += 1

    def _retrieve_buffered_packet(self, sender: int | None, layer: int | None,
                                  end_sender: int | None) -> Packet | None:
        """Return the oldest matching delivery without consuming protocol queues."""
        candidates = (
            (sequence, queue, index)
            for neighbor, stores in self._packets.items()
            if sender is None or neighbor == sender
            for queue in stores.get(PENDING_RECEIVE, {}).values()
            for index, (sequence, packet) in enumerate(queue)
            if self._packet_matches(packet, sender, layer, end_sender)
        )
        selected = min(candidates, key=lambda entry: entry[0], default=None)
        if selected is None:
            return None
        _, queue, index = selected
        return queue.pop(index)[1]

    async def _receive(self, kind, sender, time_out, layer=None, *, end_sender=None):
        channels = self._channels[kind]
        if sender is not None and sender not in channels:
            raise ValueError(f'No {kind} channel from node {sender}')
        if time_out is not None and (not np.isfinite(time_out) or time_out < 0):
            raise ValueError('Timeout must be finite and nonnegative')
        deadline = None if time_out is None else self._time + time_out
        timeout_event = None
        if time_out is not None and time_out > 0:
            timeout_event = TimeoutEvent(deadline, self.id)
            self._sim.schedule_event(timeout_event)
        try:
            while True:
                if kind == 'pc':
                    packet = self._retrieve_buffered_packet(sender, layer, end_sender)
                    if packet is not None:
                        return packet
                candidates = [(channel[RECEIVE]._channel[0][0], neighbor, channel[RECEIVE])
                              for neighbor, channel in channels.items()
                              if (sender is None or sender == neighbor)
                              and channel[RECEIVE].ready(self._time)]
                if candidates:
                    _, _, channel = min(candidates, key=lambda item: (item[0], item[1]))
                    value = channel.get()
                    if kind == 'qc' or self._packet_matches(value, sender, layer, end_sender):
                        return value
                    if (value.layer in (L1, L2, L4) and value.layer in self._qprotocols) or (
                            value.layer == L3 and L4 in self._qprotocols and value.l3_dst == self.id):
                        self.handle_packet(value)
                    else:
                        self._buffer_received_packet(value)
                    continue
                if deadline is not None and self._time >= deadline:
                    return None
                self._resume.clear()
                await self._resume.wait()
        finally:
            if timeout_event is not None:
                timeout_event.active = False

    def _l2_result_index(self, host, store, request_id):
        entries = self._layer_results[host][store][L2]
        return next((index for index, entry in enumerate(entries)
                     if request_id is None or getattr(entry, 'request_id', None) in (None, request_id)), None)

    def _l2_result_entry(self, host, store, request_id):
        index = self._l2_result_index(host, store, request_id)
        if index is None:
            raise KeyError(f'No L2 result for request {request_id}')
        return self._layer_results[host][store][L2][index]

    @node_event
    def l2_peek_batch(self, host, store):
        """Oldest local batch awaiting peer confirmation, without consuming it."""
        entries = self._layer_results[host][store][L2]
        return entries[0] if entries else None

    @node_event
    def l7_num_pairs(self, remote, neighbor, store):
        """Count teleportation resources from the highest installed pair layer."""
        if L4 in self._qprotocols:
            return self.l4_num_qubits(remote, store)
        return self.l3_num_qubits(neighbor, store)

    def _l7_retrieve_pair(self, remote, neighbor, store):
        if L4 in self._qprotocols:
            return self.l4_retrieve_qubit(remote, store)
        return self.l3_retrieve_qubit(neighbor, store)

    def _l7_peek_pairs(self, remote, neighbor, store, count):
        if L4 in self._qprotocols:
            return self._l4_ready.get((remote, store), [])[:count]
        entries = self._memory[neighbor][store]._ready_entries()
        return [qubit for qubit, _ in entries[:count]]

    @node_event
    def apply_teleportation_send(self, qubits, remote, neighbor):
        """Consume data and local shared halves through Bell measurements."""
        self._l7_validate_data(qubits)
        if self.l7_num_pairs(remote, neighbor, SEND) < len(qubits):
            raise ValueError('Not enough Bell pairs for teleportation')
        shared = self._l7_peek_pairs(remote, neighbor, SEND, len(qubits))
        if {id(qubit) for qubit in qubits} & {id(qubit) for qubit in shared}:
            raise ValueError('Data qubits cannot also be selected Bell-pair resources')
        return [self._l7_measure(qubit, remote, neighbor) for qubit in qubits]

    @staticmethod
    def _l7_validate_data(qubits):
        if not qubits or not all(isinstance(qubit, Qubit) for qubit in qubits):
            raise ValueError('At least one data qubit is required')
        if len({id(qubit) for qubit in qubits}) != len(qubits):
            raise ValueError('Data qubits must be distinct')
        if any(not any(item is qubit for item in qubit.qsystem._qubits) for qubit in qubits):
            raise ValueError('Data qubits have already been consumed')

    def _l7_measure(self, qubit, remote, neighbor):
        shared = self._l7_retrieve_pair(remote, neighbor, SEND)
        if shared is None:
            raise RuntimeError('Teleportation lost a shared qubit during extraction')
        return self.apply_gate('bsm', qubit, shared)

    @node_event
    def apply_teleportation_receive(self, packet):
        """Consume receiving halves and combine BSM and swapping corrections."""
        self._l7_validate_outcomes(packet)
        if self.l7_num_pairs(packet.l3_src, packet.src, RECEIVE) < packet.l7_requested:
            raise ValueError('Not enough receiving Bell pairs for teleportation')
        return [self._l7_correct_pair(packet, index, outcome)
                for index, outcome in enumerate(packet.payload)]

    @staticmethod
    def _l7_validate_outcomes(packet):
        if packet.l7_requested < 1 or len(packet.payload) != packet.l7_requested or packet.l3_needed != packet.l7_requested:
            raise ValueError('Teleportation outcome and correction counts do not match')
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
               or not 0 <= value <= 3 for value in packet.payload):
            raise ValueError('A BSM outcome must be an integer between 0 and 3')

    def _l7_correct_pair(self, packet, index, outcome):
        qubit = self._l7_retrieve_pair(packet.l3_src, packet.src, RECEIVE)
        if qubit is None:
            raise RuntimeError('Teleportation lost a receiving qubit during extraction')
        if bool(outcome & 1) ^ bool(packet.l3_x_count[index]):
            self.apply_gate('X', qubit)
        if bool(outcome & 2) ^ bool(packet.l3_z_count[index]):
            self.apply_gate('Z', qubit)
        return qubit

    def _l4_buffers(self, remote, store):
        results = self._layer_results.setdefault(remote, {}).setdefault(store, {}).setdefault(L4, [])
        packets = self._packets.setdefault(remote, {}).setdefault(store, {}).setdefault(L4, [])
        return results, packets

    def _l4_owned_qubits(self):
        batches = [batch for stores in self._layer_results.values() for layers in stores.values()
                   for batch in layers.get(L4, [])]
        return [qubit for batch in batches for qubit in batch.qubits] + [
            qubit for qubits in self._l4_ready.values() for qubit in qubits]

    @node_event
    def l4_prepare_inputs(self, neighbor, store, packet, mode, *, corrections=None):
        """Take end-to-end halves from L3; correct their swapping frame first."""
        if self.l3_num_qubits(neighbor, store) < packet.l4_needed:
            raise ValueError('Not enough end-to-end input pairs for L4')
        if corrections is not None and (len(corrections) != 2 or
                                        any(len(bits) != packet.l4_needed for bits in corrections)):
            raise ValueError('Swapping correction and L4 input counts differ')
        qubits = [self.l3_retrieve_qubit(neighbor, store) for _ in range(packet.l4_needed)]
        if any(qubit is None for qubit in qubits):
            raise RuntimeError('L4 lost a qubit during extraction')
        if corrections is not None:
            self._l4_apply_frames(qubits, corrections)
        if mode == 'epp':
            return self._l4_purify_inputs(qubits)
        return np.ones(packet.l4_requested, dtype=bool), qubits

    def _l4_apply_frames(self, qubits, corrections):
        for qubit, x, z in zip(qubits, *corrections):
            self._l4_apply_frame(qubit, x, z)

    def _l4_apply_frame(self, qubit, x, z):
        if x:
            self.apply_gate('X', qubit)
        if z:
            self.apply_gate('Z', qubit)

    def _l4_purify_inputs(self, qubits):
        pairs = [self._l4_purify_pair(control, target)
                 for control, target in zip(qubits[::2], qubits[1::2])]
        return np.array([bit for bit, _ in pairs], dtype=bool), [qubit for _, qubit in pairs]

    def _l4_purify_pair(self, control, target):
        self.apply_gate('CNOT', control, target)
        return self.apply_gate('measure', target, 'z'), control

    @node_event
    def l4_store_batch(self, remote, store, packet, qubits):
        results, _ = self._l4_buffers(remote, store)
        if any(batch.request_id == packet.request_id for batch in results):
            raise ValueError('Duplicate L4 batch request ID')
        results.append(L4Batch(packet.request_id, packet.l4_success.copy(),
                               qubits, packet.l4_needed, packet.l4_code, packet.dst))

    @node_event
    def l4_store_packet(self, remote, store, packet):
        _, packets = self._l4_buffers(remote, store)
        packets.append(packet.copy())

    @node_event
    def l4_check_results(self, remote, store, request_id):
        results, _ = self._l4_buffers(remote, store)
        return any(batch.request_id == request_id for batch in results)

    @node_event
    def l4_peek_batch(self, remote, store):
        results, _ = self._l4_buffers(remote, store)
        return results[0] if results else None

    @node_event
    def l4_peer_packet(self, remote, store, request_id):
        _, packets = self._l4_buffers(remote, store)
        return next((packet for packet in packets if packet.request_id == request_id), None)

    @node_event
    def l4_apply_qec(self, remote, store, request_id):
        results, _ = self._l4_buffers(remote, store)
        batch = next(batch for batch in results if batch.request_id == request_id)
        return self.apply_qec(batch.qubits, batch.code)

    @node_event
    def l4_confirm_batch(self, remote, store, batch, success):
        success = np.asarray(success, dtype=bool)
        size = 1 if batch.code is None else self._l4_code_size(batch.code)
        if success.shape != batch.results.shape or len(batch.qubits) != len(success)*size:
            raise ValueError('L4 survivor and result counts differ')
        outputs = [qubit for qubit, keep in zip(batch.qubits[::size], success) if keep]
        retained = {id(qubit) for qubit in outputs}
        remove_qubits([qubit for qubit in batch.qubits if id(qubit) not in retained])
        self._l4_ready.setdefault((remote, store), []).extend(outputs)
        results, packets = self._l4_buffers(remote, store)
        results.remove(batch)
        packets[:] = [packet for packet in packets if packet.request_id != batch.request_id]
        return len(outputs)

    @staticmethod
    def _l4_code_size(code):
        return get_qec_code(code).length

    @node_event
    def l4_store_ready(self, remote, store, qubits):
        self._l4_ready.setdefault((remote, store), []).extend(qubits)

    @node_event
    def l4_num_qubits(self, remote, store):
        return len(self._l4_ready.get((remote, store), []))

    @node_event
    def l4_retrieve_qubit(self, remote, store):
        qubits = self._l4_ready.get((remote, store), [])
        return qubits.pop(0) if qubits else None

    @node_event
    def l3_can_swap(self, packet, offset_index=None):
        """Check both memory directions without consuming their resources."""
        incoming = self._memory[packet.src][RECEIVE]
        return (incoming.ready_count(offset_index) >= packet.l3_needed and
                self.l3_num_qubits(packet.dst, SEND) >= packet.l3_needed)

    @node_event
    def apply_entanglement_swapping_crp(self, packet, offset_index=None):
        """Correct the incoming Pauli frame locally before Bell measurements."""
        self._l3_swap_pairs(packet, offset_index, correct=True)

    @node_event
    def apply_entanglement_swapping_frp(self, packet, offset_index=None):
        """Perform Bell measurements and accumulate the outgoing Pauli frame."""
        self._l3_swap_pairs(packet, offset_index, correct=False)

    def prepare_graph_state_resource(self, code):
        """Construct packed stabilizer generators for a 2n-qubit resource."""
        return self._graph_state_model(code).prepare_resource()

    def simulate_graph_state_segment(self, code, **parameters):
        """Sample joint logical Pauli frames without a block density matrix."""
        return self._graph_state_model(code).simulate_segment(**parameters)

    def simulate_graph_state_chain(self, code, segments, **parameters):
        """Compose independent segment trajectories, retaining logical correlations."""
        return self._graph_state_model(code).simulate_chain(segments, **parameters)

    @node_event
    def apply_entanglement_swapping_graph_state(self, packet, offset_index=None, *,
                                               code=None, first_record=None, second_record=None):
        """Compute recovery frames from two encoded-interface Bell records.

        Encoded resources and measurement records are supplied explicitly.
        Ordinary Qubit memory is not an encoded-resource representation.
        """
        if not packet.has_l3:
            raise ValueError('Graph-state swapping requires an L3 header')
        if offset_index is not None:
            raise NotImplementedError('Ordinary Bell-pair reservations cannot represent encoded graph resources')
        if code is None or first_record is None or second_record is None:
            raise ValueError("Supply a code and both endpoints' Bell measurement records")
        model = self._graph_state_model(code)
        if packet.l3_needed != model.code.num_logical:
            raise ValueError('L3 correction count must equal the number of logical outputs')
        recovery = model.recover(first_record, second_record)
        for index in range(model.code.num_logical):
            outcome = int(recovery.bit_flips[index]) | (int(recovery.phase_flips[index]) << 1)
            packet.l3_update_es(outcome, index)
        return recovery

    @staticmethod
    def _graph_state_model(code):
        if isinstance(code, MeasurementBasedRepeaterModel):
            return code
        if isinstance(code, str):
            code = get_qec_code(code)
        return MeasurementBasedRepeaterModel(code)

    def _l3_swap_pairs(self, packet, offset_index, *, correct):
        if not self.l3_can_swap(packet, offset_index):
            raise ValueError('Not enough reserved or outgoing Bell pairs for swapping')
        for index in range(packet.l3_needed):
            self._l3_swap_pair(packet, index, offset_index, correct=correct)

    def _l3_swap_pair(self, packet, index, offset_index, *, correct):
        source = self.l3_retrieve_qubit(packet.src, RECEIVE, offset_index=offset_index)
        target = self.l3_retrieve_qubit(packet.dst, SEND)
        if source is None or target is None:
            raise RuntimeError('Swapping lost a qubit during extraction')
        if correct:
            self._l3_apply_frame(packet, index, source)
        packet.l3_update_es(self.apply_gate('bsm', source, target), index)

    def _l3_apply_frame(self, packet, index, qubit):
        if packet.l3_x_count[index]:
            self.apply_gate('X', qubit)
        if packet.l3_z_count[index]:
            self.apply_gate('Z', qubit)
        packet.l3_reset_es(index)

    @node_event
    def l2_purify_inputs(self, host, store, rounds):
        """Run local bilateral-parity operations and stage the survivors in L2."""
        if self.l1_num_qubits(host, store) < 2 * rounds:
            raise ValueError('Purification requires two input pairs per round')
        pairs = [self._l2_purify_pair(host, store) for _ in range(rounds)]
        return (np.array([result for result, _ in pairs], dtype=bool),
                [qubit for _, qubit in pairs])

    def _l2_purify_pair(self, host, store):
        control = self.l1_retrieve_qubit(host, store)
        target = self.l1_retrieve_qubit(host, store)
        if control is None or target is None:
            raise RuntimeError('Purification lost a qubit during extraction')
        self.apply_gate('CNOT', control, target)
        result = self.apply_gate('measure', target, 'z')
        self.l2_store_qubit(control, host, store)
        return result, control

    @node_event
    def l2_stage_qec_inputs(self, host, store, needed):
        """Hold physical inputs until both endpoints are ready for decoding."""
        if self.l1_num_qubits(host, store) < needed:
            raise ValueError('Not enough raw input pairs for QEC')
        qubits = [self.l1_retrieve_qubit(host, store) for _ in range(needed)]
        if any(qubit is None for qubit in qubits):
            raise RuntimeError('QEC lost a qubit during extraction; erasure decoding is unsupported')
        for qubit in qubits:
            self.l2_store_qubit(qubit, host, store)
        return qubits

    @node_event
    def apply_qec(self, qubits, code):
        """Apply the code's ideal effective channel without combining pair states.

        This operation must run once after both peers have staged their halves.
        The caller owns disposal of the physical pairs not retained as outputs.
        """
        model = get_qec_code(code)
        if model.num_logical != 1:
            raise NotImplementedError('Network QEC currently retains one pair per block; use the joint QEC backend for k>1')
        if not qubits or len(qubits) % model.length:
            raise ValueError('QEC inputs must contain complete code blocks')
        if len({qubit.qsystem_id for qubit in qubits}) != len(qubits):
            raise ValueError('Correlated input pairs cannot use the independent-noise model')
        noise = np.array([model.bell_probabilities(qubit) for qubit in qubits])
        blocks = noise.reshape(-1, model.length, 4)
        states = [model.bell_state(model.logical_channel(block, rng=self._sim.rng('qec', self.id))) for block in blocks]
        for survivor, state in zip(qubits[::model.length], states):
            survivor.state = state
        return len(states)

    @node_event
    def apply_qec_multiple_outputs(self, qubits, code):
        """Placeholder for QEC producing multiple logical output pairs."""
        raise NotImplementedError('QEC with multiple logical outputs is not modeled yet')

    @node_event
    def apply_qec_non_stabilizer(self, qubits, code):
        """Placeholder for applying a non-stabilizer QEC code."""
        raise NotImplementedError('Non-stabilizer QEC is not modeled yet')

    @node_event
    def apply_qec_coherent_noise(self, qubits, code):
        """Placeholder for QEC on inputs with coherent noise."""
        raise NotImplementedError('QEC with coherent noise is not modeled yet')

    @node_event
    def l2_apply_qec(self, host, store, request_id):
        """Decode a node-owned batch once the peer has confirmed readiness."""
        batch = self._l2_result_entry(host, store, request_id)
        return self.apply_qec(batch.qubits, batch.code)

    @node_event
    def l2_confirm_batch(self, host, store, request_id, success):
        """Promote the right survivors and consume this node-owned batch."""
        batch = self._l2_result_entry(host, store, request_id)
        block_size = 1 if batch.code is None else get_qec_code(batch.code).length
        success = np.asarray(success, dtype=bool)
        if success.shape != batch.results.shape or len(batch.qubits) != len(success) * block_size:
            raise ValueError('L2 survivor and result counts differ')
        mask = np.repeat(success, block_size) & (np.arange(len(batch.qubits)) % block_size == 0)
        self._memory[host][store].move_selected_qubits(L2, L3, batch.qubits, mask)
        self.l2_retrieve_result(host, store, request_id=request_id)

# def _IF(IF: int) -> float:
    
#     """
#     Conversion function for an Integer Fidelity (IF) for use in the fidelity formulas for entanglement swapping and fidelity improvement
    
#     Args:
#         IF (int): integer representation of a fidelity
        
#     Returns:
#         _IF (float): converted integer fidelity
#     """
    
#     return (2 ** (8 * IF / 255)).astype(np.float32)

# def IF_entanglement_swapping(IF_1: int, IF_2: int) -> int:
    
#     """
#     Computes the integer fidelity for entanglement swapping based on two input integer fidelities
    
#     Args:
#         IF_1 (int): first integer fidelity
#         IF_2 (int): second integer fidelity
        
#     Returns:
#         _IF (int): output integer fidelity
#     """
    
#     IF_1 = _IF(IF_1)
#     IF_2 = _IF(IF_2)
    
#     return np.ceil(31.875 * np.log2(IF_1 + IF_2 - 1 - 2 / 765 * (IF_1 - 1) * (IF_2 - 1))).astype(np.uint8)

# def IF_fidelity_improvement(IF_g: int, IF_b: int) -> int:
    
#     """
#     Computes the integer fidelity of the fidelity improvement for a GHZ and BP
    
#     Args:
#         IF_g (int): integer fidelity of the GHZ
#         IF_b (int): integer fidelity of the BP
        
#     Returns:
#         _IF (int): output integer fidelity
#     """
    
#     IF_g = _IF(IF_g)
#     IF_b = _IF(IF_b)
    
#     return np.ceil(31.875 * np.log2((327679 + 584456 * IF_g + 454151 * IF_b - 761 * IF_g * IF_b) / (1368844 - 1534 * IF_g - 1789 * IF_b + 4 * IF_g * IF_b))).astype(np.uint8)

# def FF_entanglement_swapping(fid_1: float, fid_2: float) -> float:
    
#     """
#     Computes the float fidelity for the entanglement swapping
    
#     Args:
#         fid_1 (float): first float fidelity
#         fid_2 (float): second float fidelity
        
#     Returns:
#         _FF (float): output float fidelity
#     """
    
#     return (4 * fid_1 * fid_2 - fid_1 - fid_2 + 1) / 3

# def FF_fidelity_improvement(fid_g: float, fid_b: float) -> float:
    
#     """
#     Computes the float fidelity of the fidelity improvement with a GHZ and BP
    
#     Args:
#         fid_g (float): float fidelity of GHZ
#         fid_b (float): float fidelity of BP
        
#     Returns:
#         _FF (float): output fidelity
#     """
    
#     return (22 * fid_g * fid_b - fid_g - fid_b + 1) / (16 * fid_g * fid_b - 4 * fid_g - 2 * fid_b + 11)
