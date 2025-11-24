import logging
import traceback
import numpy as np
import asyncio as asc
from functools import partial
from typing import List, Dict, Set, Any

from qcns.python.components.simulation.simulation import Simulation
from qcns.python.components.simulation.event import StopEvent, SendEvent, ReceiveEvent, GateEvent, WaitEvent
from qcns.python.components.qubit.qubit import Qubit, combine_state, remove_qubits
from qcns.python.components.hardware.connection import *
from qcns.python.components.connection.channel import QChannel, PChannel
from qcns.python.components.packet import Packet
from qcns.python.components.hardware.memory import PhysicalQuantumMemory, LogicalQuantumMemory, PQM_Model, LQM_Model
from qcns.python.components.connection.connection import PChannel_Model, SingleQubitConnection, SenderReceiverConnection, TwoPhotonSourceConnection, BellStateMeasurementConnection, FockStateConnection, L3Connection
from qcns.python.components.network.qprogram import QProgram, QProgram_Model

__all__ = ['Node', 'IF_entanglement_swapping', 'IF_fidelity_improvement', 'FF_entanglement_swapping', 'FF_fidelity_improvement']

class QuantumError:
    pass

_GATE_DURATION = {'X': 5e-6, 'Y': 5e-6, 'Z': 3e-6, 'H': 6e-6, 'T': 1e-6, 'bsm': 1e-5, 'prob_bsm': 1e-5, 'CNOT': 12e-5, 'measure': 1e-6}

_GATE_PARAMETERS = {'prob_bsm_p': 0.57, 'prob_bsm_a': 0.57}

SEND = 0
RECEIVE = 1
GATE = 2

L0 = 0
L1 = 1
L2 = 2
L3 = 3
L4 = 4
L7 = 5

class Node:
    
    pass

class Node:
    
    """
    Represents a Node
    
    Attr:
        _node_id (int): ID of node
        _sim (Simulation): simulation object
        _qprograms (dict): dictonary holding all quantum programs on the host
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
    
    def __init__(self, node_id: int, sim: Simulation, stop: bool=True, qprograms: QProgram_Model=QProgram_Model(), gate_duration: Dict[str, float]=_GATE_DURATION, 
                 gate_parameters: Dict[str, float]=_GATE_PARAMETERS) -> None:
        
        """
        Initializes a Node
        
        Args:
            node_id (int): ID of Node
            sim (Simulation): Simulation
            stop (bool): whether the Node has a finite number of events
            l1_qprogram (QProgram): Program handling L1 packets
            l2_qprogram (QProgram): Program handling L2 packets
            l3_qprogram (QProgram): Program handling L3 packets
            l4_qprogram (QProgram): Program handling L4 packets
            l7_qprogram (QProgram): Program handling L7 packets
            gate_duration (dict): durations of gates
            gate_parameters (dict): parameters for gates
            pulse_duration (float): duration of pulses for sending packets
            
        Returns:
            /
        """
        
        self._node_id: int = node_id
        self._sim: Simulation = sim
        
        self._qprograms: Dict[int, QProgram] = qprograms._qprograms
        
        self._gates: Dict[str, Qubit] = {k: v for k, v in Qubit.__dict__.items() if not k.startswith(('__', 'f'))}
        self._gate_duration: Dict[str, float] = gate_duration
        self._gate_parameters: Dict[str, float] = gate_parameters
        
        self._channels: Dict[str, Dict[str, Any]] = {'qc': {}, 'pc': {}}
        self._connections: Dict[str, Dict[str, Any]] = {'sqs': {}, 'eqs': {}}
        self._memory: Dict[str, Any] = {}
        self._neighbors: Set[int] = set()
        
        self._layer_results: Dict[int, Dict[int, Dict[int, List[np.ndarray]]]] = {}
        self._packets: Dict[int, Dict[int, Dict[int, List[Packet]]]] = {}
        
        self._time: float = 0.
        
        self._resume: asc.Event = asc.Event()
        self._receive_tasks: Dict[int, asc.Task] = {}
        self._receive_queue: asc.Queue = asc.Queue()
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
        
        if not self.stop:
            self._sim.schedule_event(StopEvent(self.id))
        
        try:
            await func()
            if self.stop:
                self._sim.schedule_event(StopEvent(self.id))
        except Exception as e:
            logging.error(f'Node {self.id} has Exception')
            logging.error(traceback.format_exc())
            self._sim.stop_simulation()
    
    def reset(self) -> None:
        
        """
        Resets a host
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._time = 0.
        
        for neighbor in self.neighbors:
            self._resume[neighbor].clear()
            self._memory[neighbor][SEND].discard_qubits()
            self._memory[neighbor][RECEIVE].discard_qubits()
            self._layer_results[neighbor] = {SEND: {L1: [], L2: [], L3: []}, RECEIVE: {L1: [], L2: [], L3: []}}
            self._packets[neighbor] = {SEND: {L1: [], L2: [], L3: []}, RECEIVE: {L1: [], L2: [], L3: []}}
    
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
        
        channel_s_r = QChannel(sender_config._qchannel_model, sender_config._channel_errors)
        channel_r_s = QChannel(receiver_config._qchannel_model, receiver_config._channel_errors)
        
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
        sender_memory_receive = memories[sender_memory_config._memory_type](receiver_memory_config)
        receiver_memory_send = memories[receiver_memory_config._memory_type](receiver_memory_config)
        receiver_memory_receive = memories[receiver_memory_config._memory_type](sender_memory_config)
        
        # sender_memory_receive._size = -1
        # receiver_memory_receive._size = -1
        
        connection_s_r = connections[sender_connection_config._connection_type](self, host.id, self._sim, sender_memory_send, receiver_memory_receive, sender_connection_config)
        connection_r_s = connections[receiver_connection_config._connection_type](host, self.id, self._sim, receiver_memory_send, sender_memory_receive, receiver_connection_config)
        
        self._connections['eqs'][host.id] = connection_s_r
        host._connections['eqs'][self.id] = connection_r_s
        
        self._memory[host.id] = {SEND: sender_memory_send, RECEIVE: sender_memory_receive}
        host._memory[self.id] = {SEND: receiver_memory_send, RECEIVE: receiver_memory_receive}
        
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
        
        self._sim.schedule_event(SendEvent(self._time, self.id))
        
        _num_needed = num_requested
        if estimate:
            _num_needed = int(np.ceil(_num_needed / self._connections['sqs'][receiver][SEND]._success_prob))
        
        self._connections['sqs'][receiver][SEND].attempt_qubit(_num_needed)

    def create_qubit(self, receiver: int, num_requested: int=1) -> None:
        
        """
        Creates the number of requested qubits, no matter how long it takes
        
        Args:
            receiver (int): receiver to which photon source points to
            num_requested (int): number of requested qubits
            
        Returns:
            /
        """
        
        self._sim.schedule_event(SendEvent(self._time, self.id))
        
        self._connections['sqs'][receiver][SEND].create_qubit(num_requested)
    
    def attempt_bell_pairs(self, receiver: int, num_requested: int=1, estimate: bool=False) -> None:
        
        """
        Attempts to create the number of requested bell pairs, can estimate the number of needed qubits based on success probability
        
        Args:
            receiver (int): receiver of bell pairs
            num_requested (int): number of requested bell pairs
            estimate (bool): whether to estimate the number of needed qubits or not
            
        Returns:
            /
        """
        
        if num_requested < 1:
            raise ValueError('Requesting 0 qubits')
        
        self._sim.schedule_event(SendEvent(self._time, self.id))
        
        _num_needed = num_requested
        if estimate:
            _num_needed = int(np.ceil(_num_needed / self._connections['eqs'][receiver]._success_prob))
        
        if not self.has_space(receiver, 0, _num_needed):
            _num_needed = self.remaining_space(receiver, 0)
        
        if not self._sim._hosts[receiver].has_space(self.id, 1, _num_needed):
            _num_needed = self._sim._hosts[receiver].remaining_space(self.id, 1)
        
        if not _num_needed:
            return
        
        if _num_needed < num_requested:
            num_requested = _num_needed
        
        self._connections['eqs'][receiver].attempt_bell_pairs(num_requested, _num_needed)
    
    def create_bell_pairs(self, receiver: int, num_requested: int=1) -> None:
        
        """
        Creates the number of requested qubits, no matter how long it takes
        
        Args:
            receiver (int): receiver of bell pairs
            num_requested (int): number of requested bell pairs
            
        Returns:
            /
        """
        
        if num_requested < 1:
            raise ValueError('Attempting 0 qubits')
        
        self._sim.schedule_event(SendEvent(self._time, self.id))
        
        if not self.has_space(receiver, 0, num_requested):
            num_requested = self.remaining_space(receiver, 0)
        
        if not self._sim._hosts[receiver].has_space(self.id, 1, num_requested):
            num_requested = self._sim._hosts[receiver].remaining_space(self.id, 1)
        
        if not num_requested:
            return
        
        self._connections['eqs'][receiver].create_bell_pairs(num_requested)
    
    def apply_gate(self, gate: str, *args: List[Any], apply: bool=True, success_prob: float=1., false_prob: float=0., combine: bool=True, remove: bool=True) -> int | None:
        
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
        
        self._time += self._gate_duration.get(gate, 5e-6)
        self._sim.schedule_event(GateEvent(self._time, self.id))
        
        if combine and gate in ['CNOT', 'CX', 'CY', 'CZ', 'CPHASE', 'CU', 'SWAP', 'iSWAP', 'bell_state', 'bsm']:
            combine_state(args[:2])
        if combine and gate in ['QAND', 'QOR', 'QXOR', 'QNAND', 'QNOR', 'QXNOR', 'CCU', 'CSWAP']:
            combine_state(args[:3])
        
        prob = np.random.uniform(0, 1)
        if prob < success_prob:
            event = 0
        elif prob < success_prob + false_prob:
            event = 1
        else:
            event = 2
           
        def __S(self, gate: str, remove: bool, args: List[Any]):
            
            res = self._gates[gate](*args)
            
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
                return np.random.randint(0, 2)
            if remove and gate == 'bsm':
                remove_qubits(args[:2])
                return np.random.randint(0, 4)

        event_dict = {0: __S, 1: __F, 2: __N}
        
        return event_dict[event](self, gate, remove, args)
    
    def l2_purify(self, host: int, store: int, direction: bool=0, gate: str='CNOT', basis: str='Z', index_src: int=None, index_dst: int=None) -> int:
        
        """
        Purifies two qubits in the store given the indices
        
        Args:
            host (int): the host the memory points to
            store (int): send or receive entanglement store
            direction (int): whether to apply src->dst or dst->src
            gate (str): gate to apply
            basis (str): in which basis to measure the target qubit
            index_src (int): index of source qubit
            index_dst (int): index of dest qubit
            
        Returns:
            res (int): measurement result
        """
        
        self._time += self._gate_duration.get(gate, 5e-6)
        self._sim.schedule_event(GateEvent(self._time, self.id))
        
        _qubit_src, _qubit_dst = self._memory[host][store].purify(index_src, index_dst, self._time)
        
        combine_state([_qubit_src, _qubit_dst])
        
        _res = self._gates['purification'](_qubit_src, _qubit_dst, direction, gate, basis)
        
        remove_qubits([_qubit_dst])
        
        self._memory[host][store].store_qubit(L2, _qubit_src, -1, self._time)
        
        return _res
    
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
    
    async def receive_qubit(self, sender: int=None, time_out: float=None) -> Qubit | None:
        
        """
        Waits until a qubit is received
        
        Args:
            sender (int): sender to receive qubit from
            
        Returns:
            _qubit (Qubit): received qubit
        """
        
        try:
            await asc.wait_for(self._resume.wait(), timeout=time_out)
            self._resume.clear()
        except asc.TimeoutError:
            return None
        
        if sender is not None:
            return self._channels['qc'][sender][RECEIVE].get()
        
        for neighbor in self._neighbors:
            if not self._channels['qc'][neighbor][RECEIVE].empty():
                return self._channels['qc'][neighbor][RECEIVE].get()
         
    def send_packet(self, _packet: Packet) -> None:
        
        """
        Sends a packet to the specified receiver in the packet
        
        Args:
            _packet (Packet): packet to send
            
        Returns:
            /
        """
        
        self._time += self._channels['pc'][_packet.l2_dst][SEND]._sending_time(len(_packet))
        self._sim.schedule_event(SendEvent(self._time, self.id))
        _recv_event = ReceiveEvent(self._time + self._channels['pc'][_packet.l2_dst][SEND]._propagation_time, _packet.l2_dst)
        self._sim.schedule_event(_recv_event)
        
        self._channels['pc'][_packet.l2_dst][SEND].put(_packet, _recv_event._end_time, _recv_event._event_counter)
        
    async def receive_packet(self, sender: int=None, time_out: float=None) -> Packet | None:
        
        """
        Receives a packet
        
        Args:
            sender (int): channel to listen on
            
        Returns:
            _packet (Packet): received packet
        """
            
        try:
            await asc.wait_for(self._resume.wait(), timeout=time_out)
            self._resume.clear()
        except asc.TimeoutError:
            return None
        
        if sender is not None:
            return self._channels['pc'][sender][RECEIVE].get()
        
        for neighbor in self._neighbors:
            if not self._channels['pc'][neighbor][RECEIVE].empty():
                return self._channels['pc'][neighbor][RECEIVE].get()
    
    def wait(self, duration: float) -> None:
        
        """
        Waits the defined seconds
        
        Args:
            duration (float): time to wait
            
        Returns:
            /
        """
        
        self._time += duration
        self._sim.schedule_event(WaitEvent(self._time, self.id))

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
    def l1_qprogram(self) -> QProgram:
        
        """
        Returns the L1 QProgram
        
        Args:
            /
            
        Returns:
            L1_QProgram (QProgram): L1 program to return
        """
        
        return self._qprograms[L1]
    
    @property
    def l2_qprogram(self) -> QProgram:
        
        """
        Returns the L2 QProgram
        
        Args:
            /
            
        Returns:
            L2_QProgram (QProgram): L2 program to return
        """
        
        return self._qprograms[L2]
    
    @property
    def l3_qprogram(self) -> QProgram:
        
        """
        Returns the L3 QProgram
        
        Args:
            /
            
        Returns:
            L3_QProgram (QProgram): L3 program to return
        """
        
        return self._qprograms[L3]
    
    @property
    def l4_qprogram(self) -> QProgram:
        
        """
        Returns the L4 QProgram
        
        Args:
            /
            
        Returns:
            L4_QProgram (QProgram): L4 program to return
        """
        
        return self._qprograms[L4]
    
    @property
    def l7_qprogram(self) -> QProgram:
        
        """
        Returns the L7 QProgram
        
        Args:
            /
            
        Returns:
            L7_QProgram (QProgram): L7 program to return
        """
        
        return self._qprograms[L7]
    
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
    
    def l3_num_qubits(self, host: int, store: int) -> int:
        
        """
        Returns the number of qubits from the L3 store
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            
        Returns:
            l3_num_qubits (int): number of qubits in the L3 store
        """
        
        return self._memory[host][store].num_qubits(L3)
    
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
        
    def l0_retrieve_qubit(self, host: int, store: int, index: int=None) -> Qubit | None:
        
        """
        Retrieves a qubit from the L0 store
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to retrieve from
            
        Returns:
            qubit (Qubit/None): retrieved qubit
        """
        
        return self._memory[host][store].retrieve_qubit(L0, index, self._time)
    
    def l1_retrieve_qubit(self, host: int, store: int, index: int=None) -> Qubit | None:
        
        """
        Retrieves a qubit from the L1 store
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to retrieve from
            
        Returns:
            qubit (Qubit/None): retrieved qubit
        """
        
        return self._memory[host][store].retrieve_qubit(L1, index, self._time)
    
    def l2_retrieve_qubit(self, host: int, store: int, index: int=None) -> Qubit | None:
        
        """
        Retrieves a qubit from the L2 store
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to retrieve from
            
        Returns:
            qubit (Qubit/None): retrieved qubit
        """
        
        return self._memory[host][store].retrieve_qubit(L2, index, self._time)
    
    def l3_retrieve_qubit(self, host: int, store: int, index: int=None, offset_index: int=None) -> Qubit | None:
        
        """
        Retrieves a qubit from the L3 store
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to retrieve from
            offset_index (int): index of the offset needed if host needs to recirculate packets
            
        Returns:
            qubit (Qubit/None): retrieved qubit
        """
        
        return self._memory[host][store].retrieve_qubit(L3, index, self._time, offset_index)
    
    def l0_peek_qubit(self, host: int, store: int, index: int=None) -> Qubit | None:
        
        """
        Looks at the qubit without retrieving it from the L0 memory
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to retrieve from
            
        Returns:
            _qubit (Qubit/None): peeked at qubit
        """
        
        return self._memory[host][store].peek_qubit(L0, index)
    
    def l1_peek_qubit(self, host: int, store: int, index: int=None) -> Qubit | None:
        
        """
        Looks at the qubit without retrieving it from the L1 memory
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to retrieve from
            
        Returns:
            qubit (Qubit/None): peeked at qubit
        """
        
        return self._memory[host][store].peek_qubit(L1, index)
    
    def l2_peek_qubit(self, host: int, store: int, index: int=None) -> Qubit | None:
        
        """
        Looks at the qubit without retrieving it from the L2 memory
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to retrieve from
            
        Returns:
            qubit (Qubit/None): peeked at qubit
        """
        
        return self._memory[host][store].peek_qubit(L2, index)
    
    def l3_peek_qubit(self, host: int, store: int, index: int=None) -> Qubit | None:
        
        """
        Looks at the qubit without retrieving it from the L3 memory
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to retrieve from
            
        Returns:
            qubit (Qubit/None): peeked at qubit
        """
        
        return self._memory[host][store].peek_qubit(L3, index)
    
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
    
    def l1_check_results(self, host: int, store: int) -> bool:
        
        """
        Checks if there is a L1 result
        
        Args:
            host (int): connected host
            store (int): SEND or RECEIVE store
            
        Returns:
            res (bool): whether results exist or not
        """
        
        if self._layer_results[host][store][L1]:
            return True
        return False
    
    def l2_check_results(self, host: int, store: int) -> bool:
        
        """
        Checks if there is a L2 result
        
        Args:
            host (int): connected host
            store (int): SEND or RECEIVE store
            
        Returns:
            res (bool): whether results exist or not
        """
        
        if self._layer_results[host][store][L2]:
            return True
        return False
    
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

    def l1_store_result(self, store: int, packet: Packet) -> None:
        
        """
        Stores the L1 result of packet
        
        Args:
            store (int): SEND or RECEIVE store
            packet (Packet): packet to store results of
            
            
        Returns:
            /
        """
        
        self._layer_results[packet.l2_src][store][L1].append(packet.l1_success)
        
    def l2_store_result(self, store: int, packet: Packet) -> None:
        
        """
        Stores the L2 result of packet
        
        Args:
            store (int): SEND or RECEIVE store
            packet (Packet): packet to store results of
            
        Returns:
            /
        """
        
        self._layer_results[packet.l2_dst][store][L2].append(packet.l2_success)
        
    def l1_retrieve_result(self, host: int, store: int) -> np.ndarray:
        
        """
        Retrieves the first result in the L1 store
        
        Args:
            host (int): connected host
            store (int): SEND or RECEIVE store
            
        Returns:
            res (np.ndarray): L1 result
        """
        
        return self._layer_results[host][store][L1].pop(0)
    
    def l2_retrieve_result(self, host: int, store: int) -> np.ndarray:
        
        """
        Retrieves the first result in the L2 store
        
        Args:
            host (int): connected host
            store (int): SEND or RECEIVE store
            
        Returns:
            res (np.ndarray): L2 result
        """
        
        return self._layer_results[host][store][L2].pop(0)
    
    def l1_compare_results(self, packet: Packet) -> np.ndarray:
        
        """
        Compares the L1 results of the packet with the result in storage
        
        Args:
            packet (Packet): packet to compare results of
            
        Returns:
            res (np.ndarray): result of the comparison
        """
        
        stor_res = self.l1_retrieve_result(packet.l2_src, not packet.l1_ack)
        return np.logical_and(packet.l1_success, stor_res)
    
    def l2_compare_results(self, packet: Packet) -> np.ndarray:
        
        """
        Compares the L1 results of the packet with the result in storage
        
        Args:
            packet (Packet): packet to compare results of
            
        Returns:
            res (np.ndarray): result of the comparison
        """
        
        stor_res = self.l2_retrieve_result(packet.l2_src, not packet.l2_ack)
        return np.logical_not(np.logical_xor(packet.l2_success, stor_res))
    
    def l1_packets(self, host: int, store: int) -> List[Packet]:
        
        """
        Returns all packets in L1 packet store
        
        Args:
            /
            
        Returns:
            l1_packets (list): packets in L1 store
        """
        
        return self._packets[host][store][L1]
    
    def l2_packets(self, host: int, store: int) -> List[Packet]:
        
        """
        Returns all packets in L2 packet store
        
        Args:
            /
            
        Returns:
            l2_packets (list): packets in L2 store
        """
        
        return self._packets[host][store][L2]
    
    def l3_packets(self, host: int, store: int) -> List[Packet]:
        
        """
        Returns all packets in L3 packet store
        
        Args:
            /
            
        Returns:
            l3_packets (list): packets in L3 store
        """
        
        return self._packets[host][store][L3]
    
    def l1_check_packets(self, host: int, store: int) -> bool:
        
        """
        Checks whether packets are in L1 store
        
        Args:
            store (int): SEND or RECEIVE store
            
        Returns:
            l1_check_packets (bool): check if packets in L1 store
        """
        
        if self._packets[host][store][L1]:
            return True
        return False
    
    def l2_check_packets(self, host: int, store: int) -> bool:
        
        """
        Checks whether packets are in L2 store
        
        Args:
            /
            
        Returns:
            l2_check_packets (bool): check if packets in L2 store
        """
        
        if self._packets[host][store][L2]:
            return True
        return False
    
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
    
    def l1_num_packets(self, host: int, store: int) -> int:
        
        """
        Returns the number of packets in L1 store
        
        Args:
            /
            
        Returns:
            l1_num_packets (int): number of packets in L1 store
        """
        
        return len(self._packets[host][store][L1])
    
    def l2_num_packets(self, host: int, store: int) -> int:
        
        """
        Returns the number of packets in L2 store
        
        Args:
            /
            
        Returns:
            l2_num_packets (int): number of packets in L2 store
        """
        
        return len(self._packets[host][store][L2])
    
    def l3_num_packets(self, host: int, store: int) -> int:
        
        """
        Returns the number of packets in L3 store
        
        Args:
            /
            
        Returns:
            l3_num_packets (int): number of packets in L3 store
        """
        
        return len(self._packets[host][store][L3])
    
    def l1_store_packet(self, host: int, store: int, packet: Packet) -> None:
        
        """
        Stores a packet in the L1 store
        
        Args:
            packet (Packet): packet to store
            
        Returns:
            /
        """
        
        self._packets[host][store][L1].append(packet)
        
    def l2_store_packet(self, host: int, store: int, packet: Packet) -> None:
        
        """
        Stores a packet in the L2 store
        
        Args:
            packet (Packet): packet to store
            
        Returns:
            /
        """
        
        self._packets[host][store][L2].append(packet)
        
    def l3_store_packet(self, host: int, store: int, packet: Packet, offset_index: int=None) -> None:
        
        """
        Stores a packet in the L3 store
        
        Args:
            packet (Packet): packet to store
            
        Returns:
            /
        """
        
        self._packets[host][store][L3].append((packet, offset_index))
        
    def l1_retrieve_packet(self, host: int, store: int, index: int=0) -> Packet:
        
        """
        Retrieves a packet from the L1 store
        
        Args:
            /
            
        Returns:
            packet (Packet): retrieved packet
        """
        
        return self._packets[host][store][L1].pop(index)
    
    def l2_retrieve_packet(self, host: int, store: int, index: int=0) -> Packet:
        
        """
        Retrieves a packet from the L2 store
        
        Args:
            /
            
        Returns:
            packet (Packet): retrieved packet
        """
        
        return self._packets[host][store][L1].pop(index)
        
    def l3_retrieve_packet(self, host: int, store: int, index: int=0) -> Packet:
        
        """
        Retrieves a packet from the L3 store
        
        Args:
            /
            
        Returns:
            packet (Packet): retrieved packet
        """
        
        packet, offset_index = self._packets[host][store][L3].pop(index)
        
        if offset_index is None:
            return packet
        
        return packet, offset_index
    
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
        
        return self._memory[host][store].add_offset(offset)
    
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

def _IF(IF: int) -> float:
    
    """
    Conversion function for an Integer Fidelity (IF) for use in the fidelity formulas for entanglement swapping and fidelity improvement
    
    Args:
        IF (int): integer representation of a fidelity
        
    Returns:
        _IF (float): converted integer fidelity
    """
    
    return (2 ** (8 * IF / 255)).astype(np.float32)

def IF_entanglement_swapping(IF_1: int, IF_2: int) -> int:
    
    """
    Computes the integer fidelity for entanglement swapping based on two input integer fidelities
    
    Args:
        IF_1 (int): first integer fidelity
        IF_2 (int): second integer fidelity
        
    Returns:
        _IF (int): output integer fidelity
    """
    
    IF_1 = _IF(IF_1)
    IF_2 = _IF(IF_2)
    
    return np.ceil(31.875 * np.log2(IF_1 + IF_2 - 1 - 2 / 765 * (IF_1 - 1) * (IF_2 - 1))).astype(np.uint8)

def IF_fidelity_improvement(IF_g: int, IF_b: int) -> int:
    
    """
    Computes the integer fidelity of the fidelity improvement for a GHZ and BP
    
    Args:
        IF_g (int): integer fidelity of the GHZ
        IF_b (int): integer fidelity of the BP
        
    Returns:
        _IF (int): output integer fidelity
    """
    
    IF_g = _IF(IF_g)
    IF_b = _IF(IF_b)
    
    return np.ceil(31.875 * np.log2((327679 + 584456 * IF_g + 454151 * IF_b - 761 * IF_g * IF_b) / (1368844 - 1534 * IF_g - 1789 * IF_b + 4 * IF_g * IF_b))).astype(np.uint8)

def FF_entanglement_swapping(fid_1: float, fid_2: float) -> float:
    
    """
    Computes the float fidelity for the entanglement swapping
    
    Args:
        fid_1 (float): first float fidelity
        fid_2 (float): second float fidelity
        
    Returns:
        _FF (float): output float fidelity
    """
    
    return (4 * fid_1 * fid_2 - fid_1 - fid_2 + 1) / 3

def FF_fidelity_improvement(fid_g: float, fid_b: float) -> float:
    
    """
    Computes the float fidelity of the fidelity improvement with a GHZ and BP
    
    Args:
        fid_g (float): float fidelity of GHZ
        fid_b (float): float fidelity of BP
        
    Returns:
        _FF (float): output fidelity
    """
    
    return (22 * fid_g * fid_b - fid_g - fid_b + 1) / (16 * fid_g * fid_b - 4 * fid_g - 2 * fid_b + 11)