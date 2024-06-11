import traceback
import numpy as np
import asyncio as asc

from copy import deepcopy
from functools import partial
from typing import List, Dict, Tuple, Set, Union, Type, Any

from python.components.simulation import Simulation
from python.components.event import StopEvent, SendEvent, ReceiveEvent, GateEvent 
from python.components.qubit import Qubit, QSystem, combine_state, remove_qubits
from python.components.channel import PChannel
from python.components.packet import Packet
from python.components.memory import QuantumMemory
from python.components.connection import SingleQubitConnection, SenderReceiverConnection, TwoPhotonSourceConnection, BellStateMeasurementConnection

__all__ = ['Host']

class QuantumError:
    pass

_GATE_DURATION = {'X': 5e-6,
                 'Y': 5e-6,
                 'Z': 3e-6,
                 'H': 6e-6,
                 'S': 2e-6,
                 'T': 1e-6,
                 'bsm': 1e-5,
                 'prob_bsm': 1e-5,
                 'CNOT': 12e-5,
                 'measure': 1e-6
                 }

_GATE_PARAMETERS = {
                'prob_bsm_p': 0.57,
                'prob_bsm_a': 0.57 
}

SEND = 0
RECEIVE = 1

L1 = 0
L2 = 1
L3 = 2

class Host:
    
    pass

class PhotonSource:
    
    pass

class Host:
    
    """
    Represents a Host
    
    Attr:
        _node_id (int): ID of node
        _sim (Simulation): simulation object
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
    
    def __init__(self, _node_id: int, _sim: Simulation, _gate_duration: Dict[str, float]=_GATE_DURATION, _gate_parameters: Dict[str, float]=_GATE_PARAMETERS, _pulse_duration: float=10 ** -11) -> None:
        
        """
        Initializes a Host
        
        Args:
            _node_id (int): ID of Host
            _sim (Simulation): Simulation
            _gate_duration (dict): durations of gates
            _gate_parameters (dict): parameters for gates
            _pulse_duration (float): duration of pulses for sending packets
            
        Returns:
            /
        """
        
        self._node_id: int = _node_id
        self._sim: Simulation = _sim
        self._pulse_duration: float = _pulse_duration
        self._gates: Dict[str, Qubit] = {k: v for k, v in Qubit.__dict__.items() if not k.startswith(('__', 'f'))}
        self._gate_duration: Dict[str, float] = _gate_duration
        self._gate_parameters: Dict[str, float] = _gate_parameters
        
        self._connections: Dict[str, Dict[str, Any]] = {'sqs': {}, 'eqs': {}, 'packet': {}, 'memory': {}}
        self._neighbors: Set[int] = set()
        
        self._layer_results: Dict[str, Dict[int, Dict[int, np.array]]] = {}
        self._l1_packets: List[Packet] = []
        self._l2_packets: List[Packet] = []
        self._l3_packets: List[Packet] = []
        
        self._resume: asc.Event = asc.Event()
        self.stop: bool = False
    
        self.run = partial(self.log_exceptions, self.run)
    
    async def run(self):
        
        """
        Run function of Host, should be overwritten
        
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
            self._sim.schedule_event(StopEvent(self._node_id))
        except Exception as e:
            print(traceback.format_exc())
    
    def set_sqs_connection(self, host: Host, sender_source: str='perfect', receiver_source: str='perfect',
                           sender_length: float=0., sender_attenuation: float=-0.016, sender_coupling_prob: float=1., sender_com_errors: List[QuantumError]=None,
                           receiver_length: float=0., receiver_attenuation: float=-0.016, receiver_coupling_prob: float=1., receiver_com_errors: List[QuantumError]=None) -> None:
        
        """
        Sets up a single qubit source connection
        
        Args:
            host (Host): other host to establish connection with
            sender_source (str): name of sender sided source to use
            receiver_source (str): name of receiver sided source to use
            sender_length (float): length from sender to receiver
            sender_attenuation (float): sender attenuation coefficient
            sender_coupling_prob (float): probability of coupling qubit into fiber
            sender_com_errors (list): list of errors on sender channel
            receiver_length (float): length from receiver to sender
            receiver_attenuation (float): receiver attenuation coefficient
            receiver_coupling_prob (float): probability of coupling qubit into fiber
            receiver_com_errors (list): list of errors on receiver channel
            
        Returns:
            /
        """
            
        if sender_com_errors is None:
            sender_com_errors = []
        if receiver_com_errors is None:
            receiver_com_errors = []
            
        for com_error in sender_com_errors:
            com_error.add_signal_time(sender_length + receiver_length)
        
        for com_error in receiver_com_errors:
            com_error.add_signal_time(sender_length + receiver_length)
        
        connection_s_r = SingleQubitConnection(self, host, self._sim, sender_source, sender_length + receiver_length, sender_attenuation, sender_coupling_prob, sender_com_errors)
        connection_r_s = SingleQubitConnection(host, self, self._sim, receiver_source, sender_length + receiver_length, receiver_attenuation, receiver_coupling_prob, sender_com_errors)
        
        self._connections['sqs'][host._node_id] = {SEND: connection_s_r, RECEIVE: connection_r_s._channel}
        host._connections['sqs'][self._node_id] = {SEND: connection_r_s, RECEIVE: connection_s_r._channel}
    
    def set_eqs_connection(self, host: Host, sender_type: str='sr', sender_model: str='perfect', 
                           receiver_type: str='sr', receiver_model: str='perfect',
                           sender_source: str='perfect', receiver_source: str='perfect', 
                           sender_detector: str='perfect', receiver_detector: str='perfect',
                           sender_length: float=0., sender_attenuation: float=-0.016, sender_coupling_prob: float=1., sender_com_errors: List[QuantumError]=None,
                           receiver_length: float=0., receiver_attenuation: float=-0.016, receiver_coupling_prob: float=1., receiver_com_errors: List[QuantumError]=None,
                           sender_mem_size: int=-1, sender_efficiency: float=1., sender_mem_errors: List[QuantumError]=None, 
                           receiver_mem_size: int=-1, receiver_efficiency: float=1., receiver_mem_errors: List[QuantumError]=None) -> None:
        
        """
        Sets up a heralded entangled qubit connection between this host and another host
        
        Args:
            host (Host): other host to establish connection with
            sender_type (str): type of entanglement connection from sender to receiver
            sender_model (str): model of photon source at sender
            receiver_type (str): type of entanglement connection from receiver to sender
            receiver_model (str): model of photon source at receiver
            sender_source (str): model of sender sided photon source 
            receiver_source (str): model of receiver sided photon source
            sender_detector (str): model of sender sided photon detector
            receiver_detector (str): model of receiver sided photon detector
            sender_length (float): length from sender to receiver
            sender_attenuation (float): sender attenuation coefficient
            sender_coupling_prob (float): probability of coupling qubit into fiber
            sender_com_errors (list): list of errors on sender channel
            receiver_length (float): length from receiver to sender
            receiver_attenuation (float): receiver attenuation coefficient
            receiver_coupling_prob (float): probability of coupling qubit into fiber
            receiver_com_errors (list): list of errors on receiver channel
            sender_mem_size (int): size of sender memory
            sender_efficiency (float): efficiency of sender memory
            sender_mem_errors (list): list of memory errors at sender
            receiver_mem_size (int): size of receiver memory
            receiver_efficiency (float): efficiency of receiver memory
            receiver_mem_errors (list): list of memory errors at receiver
            
        Returns:
            /
        """
        
        if sender_com_errors is None:
            sender_com_errors = []
        if receiver_com_errors is None:
            receiver_com_errors = []
        if sender_mem_errors is None:
            sender_mem_errors = []
        if receiver_mem_errors is None:
            receiver_mem_errors = []
        
        sender_memory_send = QuantumMemory(sender_mem_size, sender_efficiency, sender_mem_errors)
        sender_memory_receive = QuantumMemory(sender_mem_size, sender_efficiency, sender_mem_errors)
        receiver_memory_send = QuantumMemory(receiver_mem_size, receiver_efficiency, receiver_mem_errors)
        receiver_memory_receive = QuantumMemory(receiver_mem_size, receiver_efficiency, receiver_mem_errors)
        
        if sender_type == 'sr':
            connection_s_r = SenderReceiverConnection(self, host, self._sim, sender_model, sender_source, receiver_detector, 
                                                      sender_length + receiver_length, sender_attenuation, sender_coupling_prob, sender_com_errors, 
                                                      sender_memory_send, receiver_memory_receive)
            sender_length += receiver_length
        
        if sender_type == 'tps':
            connection_s_r = TwoPhotonSourceConnection(self, host, self._sim, sender_model, sender_source, 
                                                       sender_detector, receiver_detector, 
                                                       sender_length, sender_attenuation, sender_coupling_prob, sender_com_errors, 
                                                       receiver_length, receiver_attenuation, receiver_coupling_prob, receiver_com_errors, 
                                                       sender_memory_send, receiver_memory_receive)
        
        if sender_type == 'bsm':
            connection_s_r = BellStateMeasurementConnection(self, host, self._sim, sender_model, sender_source, receiver_source, 
                                                            sender_detector, receiver_detector, 
                                                            sender_length, sender_attenuation, sender_coupling_prob, sender_com_errors, 
                                                            receiver_length, receiver_attenuation, receiver_coupling_prob, receiver_com_errors, 
                                                            sender_memory_send, receiver_memory_receive)
        
        if receiver_type == 'sr':
            connection_r_s = SenderReceiverConnection(host, self, self._sim, receiver_model, receiver_source, sender_detector, 
                                                      sender_length + receiver_length, receiver_attenuation, receiver_coupling_prob, receiver_com_errors, 
                                                      receiver_memory_send, sender_memory_receive)
            receiver_length += sender_length
            
        if receiver_type == 'tps':
            connection_r_s = TwoPhotonSourceConnection(host, self, self._sim, receiver_model, receiver_source,
                                                       receiver_detector, sender_detector, 
                                                       receiver_length, receiver_attenuation, receiver_coupling_prob, receiver_com_errors, 
                                                       sender_length, sender_attenuation, sender_coupling_prob, sender_com_errors, 
                                                       receiver_memory_send, sender_memory_receive)
        
        if receiver_model == 'bsm':
            connection_r_s = BellStateMeasurementConnection(host, self, self._sim, receiver_model, receiver_source, sender_source, 
                                                            receiver_detector, sender_detector, 
                                                            receiver_length, receiver_attenuation, receiver_coupling_prob, receiver_com_errors, 
                                                            sender_length, sender_attenuation, sender_coupling_prob, sender_com_errors, 
                                                            receiver_memory_send, sender_memory_receive)
        
        for com_error in sender_com_errors:
            com_error.add_signal_time(sender_length)
        
        for com_error in receiver_com_errors:
            com_error.add_signal_time(receiver_length)
        
        self._connections['eqs'][host._node_id] = connection_s_r
        host._connections['eqs'][self._node_id] = connection_r_s
        
        self._connections['memory'][host._node_id] = {SEND: sender_memory_send, RECEIVE: sender_memory_receive}
        host._connections['memory'][self._node_id] = {SEND: receiver_memory_send, RECEIVE: receiver_memory_receive}
    
    def set_pconnection(self, host: Host, length: float=0.) -> None:
        
        """
        Sets up a packet connection between this host an another
        
        Args:
            host (Host): host to set up connection with
            length (float): length of connection
            
        Returns:
            /
        """
        
        channel_s = PChannel(length)
        channel_r = PChannel(length)
        
        self._connections['packet'][host._node_id] = {SEND: channel_s, RECEIVE: channel_r}
        host._connections['packet'][self._node_id] = {SEND: channel_r, RECEIVE: channel_s}
    
        self._layer_results[host._node_id] = {SEND: {L1: [], L2: [], L3: []}, RECEIVE: {L1: [], L2: [], L3: []}}
        host._layer_results[self._node_id] = {SEND: {L1: [], L2: [], L3: []}, RECEIVE: {L1: [], L2: [], L3: []}}
    
    def set_connection(self, host: Host, sender_type: str='sr', sender_model: str='perfect', 
                       receiver_type: str='sr', receiver_model: str='perfect',
                        sp_sender_source: str='perfect', sp_receiver_source: str='perfect',
                        he_sender_source: str='perfect', he_receiver_source: str='perfect', 
                        sender_detector: str='perfect', receiver_detector: str='perfect',
                        sender_length: float=0., sender_attenuation: float=-0.016, sender_coupling_prob: float=1., sender_com_errors: List[QuantumError]=None,
                        receiver_length: float=0., receiver_attenuation: float=-0.016, receiver_coupling_prob: float=1., receiver_com_errors: List[QuantumError]=None,
                        sender_mem_size: int=-1, sender_efficiency: float=1., sender_mem_errors: List[QuantumError]=None, 
                        receiver_mem_size: int=-1, receiver_efficiency: float=1., receiver_mem_errors: List[QuantumError]=None) -> None:
        
        """
        Sets a single photon source connection, a entangled photon source connection and a packet connection
        
        Args:
            host (Host): other host to establish connection with
            sender_type (str): type of entanglement connection from sender to receiver
            sender_model (str): model of photon source at sender
            receiver_type (str): type of entanglement connection from receiver to sender
            receiver_model (str): model of photon source at receiver
            sp_sender_source (str): model for single photon source at sender
            sp_receiver_source (str): model for single photon source at receiver
            he_sender_source (str): model of sender sided heralded photon source 
            he_receiver_source (str): model of receiver sided heralded photon source
            sender_detector (str): model of sender sided photon detector
            receiver_detector (str): model of receiver sided photon detector
            sender_length (float): length from sender to receiver
            sender_attenuation (float): sender attenuation coefficient
            sender_coupling_prob (float): probability of coupling qubit into fiber
            sender_com_errors (list): list of errors on sender channel
            receiver_length (float): length from receiver to sender
            receiver_attenuation (float): receiver attenuation coefficient
            receiver_coupling_prob (float): probability of coupling qubit into fiber
            receiver_com_errors (list): list of errors on receiver channel
            sender_mem_size (int): size of sender memory
            sender_efficiency (float): efficiency of sender memory
            sender_mem_errors (list): list of memory errors at sender
            receiver_mem_size (int): size of receiver memory
            receiver_efficiency (float): efficiency of receiver memory
            receiver_mem_errors (list): list of memory errors at receiver
            
        Returns:
            /
        """
        
        self._neighbors.add(host._node_id)
        
        self.set_sqs_connection(host, sp_sender_source, sp_receiver_source, 
                                sender_length, sender_attenuation, sender_coupling_prob, sender_com_errors, 
                                receiver_length, receiver_attenuation, receiver_coupling_prob, receiver_com_errors)
        self.set_eqs_connection(host, sender_type, sender_model, 
                                receiver_type, receiver_model, 
                                he_sender_source, he_receiver_source, 
                                sender_detector, receiver_detector, 
                                sender_length, sender_attenuation, sender_coupling_prob, sender_com_errors, 
                                receiver_length, receiver_attenuation, receiver_coupling_prob, receiver_com_errors, 
                                sender_mem_size, sender_efficiency, sender_mem_errors, 
                                receiver_mem_size, receiver_efficiency, receiver_mem_errors)
        self.set_pconnection(host, sender_length + receiver_length)
       
    def create_qsystem(self, _num_qubits: int, fidelity: float=1., sparse: bool=False) -> QSystem:
        
        """
        Creates a new qsystem at the host
        
        Args:
            _num_qubits (int): number of qubits in the qsystem
            fidelity (float): fidelity of quantum system
            sparse (float): sparsity of qsystem
            
        Returns:
            qsys (QSystem): new qsystem
        """
        
        return Simulation.create_qsystem(_num_qubits, fidelity, sparse)
    
    def delete_qsystem(self, _qsys: QSystem) -> None:
        
        """
        Deletes a qsystem at the host
        
        Args:
            qsys (QSystem): qsystem to delete
            
        Returns:
            /
        """
        
        Simulation.delete_qsystem(_qsys)
    
    async def create_qubit(self, _receiver: int, num_requested: int=1, estimate: bool=True) -> List[Qubit]:
        
        """
        Creates a number of requested qubits from the source pointed to receiver
        
        Args:
            _receiver (int): receiver to which photon source points to
            _num_requested (int): number of requested qubits
            
        Returns:
            qubits (list): created qubits
        """
        
        self._sim.schedule_event(SendEvent(self._sim._sim_time, self._node_id))
        
        await self._resume.wait()
        self._resume.clear()
        
        _num_needed = num_requested
        if estimate:
            _num_needed = int(np.ceil(_num_needed / (1 - self._connections['sqs'][_receiver][SEND]._success_prob)))
        
        self._connections['sqs'][_receiver][SEND].create_qubit(_num_needed)
    
    async def create_bell_pairs(self, _receiver: int, num_requested: int=1, estimate: bool=False) -> None:
        
        """
        Creates number of requested bell pairs
        
        Args:
            _receiver (int): receiver of bell pairs
            num_requested (int): number of requested bell pairs
            
        Returns:
            /
        """
        
        self._sim.schedule_event(SendEvent(self._sim._sim_time, self._node_id))
        
        await self._resume.wait()
        self._resume.clear()
        
        _num_needed = num_requested
        if estimate:
            _num_needed = int(np.ceil(_num_needed / (1 - self._connections['eqs'][_receiver]._connection._success_prob)))
        
        self._connections['eqs'][_receiver].create_bell_pairs(num_requested, _num_needed)
    
    async def apply_gate(self, _gate: str, *args: str, combine: bool=False, remove: bool=False) -> Union[int, None]:
        
        """
        Applys a gate to qubits
        
        Args:
            _gate (str): gate to apply
            *args (list): variable length argument list
            combine (bool): to combine qubits that are in different qsystems
            remove (bool): to remove qubits
            
        Returns:
            res (int/None): result of the gate
        """
        
        self._sim.schedule_event(GateEvent(self._sim._sim_time + self._gate_duration.get(_gate, 5e-6), self._node_id))
        
        await self._resume.wait()
        self._resume.clear()
        
        if combine and _gate in ['CNOT', 'CY', 'CZ', 'CH', 'CPHASE', 'CU', 'SWAP', 'bell_state', 'bsm', 'prob_bsm', 'purification']:
            combine_state(args[:2])
        if combine and _gate in ['TOFFOLI', 'CCU', 'CSWAP']:
            combine_state(args[:3])
        
        res = self._gates[_gate](*args)
        
        if remove and _gate == 'measure':
            remove_qubits(args[:1])
        if remove and _gate in ['bsm', 'prob_bsm']:
            remove_qubits(args[:2])
        if remove and _gate in ['purification']:
            remove_qubits(args[1:2])
        
        return res
    
    async def send_qubit(self, _receiver: int, _qubit: Qubit) -> None:
        
        """
        Sends a qubit to the specified receiver
        
        Args:
            _receiver (int): receiver to send qubit to
            _qubit (Qubit): qubit to send
            
        Returns:
            /
        """
        
        self._sim.schedule_event(SendEvent(self._sim._sim_time, self._node_id))
        self._sim.schedule_event(ReceiveEvent(self._sim._sim_time + self._connections['sqs'][_receiver][SEND]._duration, _receiver))
        
        await self._resume.wait()
        self._resume.clear()
        
        self._connections['sqs'][_receiver][SEND]._channel.put(_qubit)
    
    async def send_qubit_prob(self, _receiver: int, _qubit: Qubit) -> None:
        
        """
        Sends a qubit to the specified receiver
        
        Args:
            _receiver (int): receiver to send qubit to
            _qubit (Qubit): qubit to send
            
        Returns:
            /
        """
        
        self._sim.schedule_event(SendEvent(self._sim._sim_time, self._node_id))
        self._sim.schedule_event(ReceiveEvent(self._sim._sim_time + self._connections['sqs'][_receiver][SEND]._duration, _receiver))
        
        await self._resume.wait()
        self._resume.clear()
        
        self._connections['sqs'][_receiver][SEND]._channel.put_prob(_qubit)
    
    async def receive_qubit(self, sender: int=None, time_out: float=None) -> Union[Qubit, Tuple[int, Qubit]]:
        
        """
        Waits until a qubit is received
        
        Args:
            sender (str): sender to receive qubit from
            
        Returns:
            _qubit (Qubit): received qubit
        """
        
        try:
            await asc.wait_for(self._resume.wait(), timeout=time_out)
            self._resume.clear()
        except asc.TimeoutError:
            return None
        
        if sender is None:            
            for _sender in self._connections.keys():
                if not self._connections['sqs'][_sender][RECEIVE].empty():
                    return _sender, self._connections['sqs'][_sender][RECEIVE].get()
        return self._connections['sqs'][sender][RECEIVE].get()
    
    async def receive_qubit_prob(self, sender: int=None, time_out: float=None) -> Union[Qubit, Tuple[int, Qubit]]:
        
        """
        Waits until a qubit is received
        
        Args:
            sender (str): sender to receive qubit from
            
        Returns:
            _qubit (Qubit): received qubit
        """
        
        try:
            await asc.wait_for(self._resume.wait(), timeout=time_out)
            self._resume.clear()
        except asc.TimeoutError:
            return None
        
        if sender is None:            
            for _sender in self._connections.keys():
                if not self._connections['sqs'][_sender][RECEIVE].empty():
                    return _sender, self._connections['sqs'][_sender][RECEIVE].get_prob()
        return self._connections['sqs'][sender][RECEIVE].get_prob()
    
    async def send_packet(self, _packet: Packet) -> None:
        
        """
        Sends a packet to the specified receiver in the packet
        
        Args:
            _packet (Packet): packet to send
            
        Returns:
            /
        """
        
        self._sim.schedule_event(SendEvent(self._sim._sim_time, self._node_id))
        self._sim.schedule_event(ReceiveEvent(self._sim._sim_time + len(_packet) * self._pulse_duration + self._connections['packet'][_packet._l2._dst][SEND]._signal_time, _packet._l2._dst))
        
        await self._resume.wait()
        self._resume.clear()
        
        self._connections['packet'][_packet._l2._dst][SEND].put(_packet)
        
    async def receive_packet(self, sender: int=None, time_out: float=None) -> Union[Packet, None]:
        
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
        
        if sender is None:
            for _sender in self.neighbors:
                if not self._connections['packet'][_sender][RECEIVE].empty():
                    return self._connections['packet'][_sender][RECEIVE].get()
            return None

        return self._connections['packet'][sender][RECEIVE].get()
    
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
    
    def l0_num_qubits(self, _host: int, _store: int) -> int:
        
        """
        Returns the number of qubits from the L0 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel of entanglement store
            
        Returns:
            l0_num_qubits (int): number of qubits in the L0 store
        """
        
        return self._connections['memory'][_host][_store].l0_num_qubits()
    
    def l1_num_qubits(self, _host: int, _store: int) -> int:
        
        """
        Returns the number of qubits from the L1 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel of entanglement store
            
        Returns:
            l1_num_qubits (int): number of qubits in the L1 store
        """
        
        return self._connections['memory'][_host][_store].l1_num_qubits()
    
    def l2_num_qubits(self, _host: int, _store: int) -> int:
        
        """
        Returns the number of qubits from the L2 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel of entanglement store
            
        Returns:
            l2_num_qubits (int): number of qubits in the L2 store
        """
        
        return self._connections['memory'][_host][_store].l2_num_qubits()
    
    def l3_num_qubits(self, _host: int, _store: int) -> int:
        
        """
        Returns the number of qubits from the L3 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel of entanglement store
            
        Returns:
            l3_num_qubits (int): number of qubits in the L3 store
        """
        
        return self._connections['memory'][_host][_store].l3_num_qubits()
    
    def l0_store_qubit(self, _qubit: Qubit, _host: int, _store: int, index: int=-1) -> None:
        
        """
        Stores a qubit in the L0 memory
        
        Args:
            _qubits (Qubit): list of qubits
            _store (int): SEND or RECEIVE store
            _channel (str): _channel to listen on
            index (int): index to qubits at
            
        Returns:
            /
        """
        
        self._connections['memory'][_host][_store].l0_store_qubits(_qubit, index, self._sim._sim_time)
        
    def l1_store_qubit(self, _qubit: Qubit, _host: int, _store: int, index: int=-1) -> None:
        
        """
        Stores a qubit in the L1 memory
        
        Args:
            _qubits (Qubit): list of qubits
            _store (int): SEND or RECEIVE store
            _channel (str): _channel to listen on
            index (int): index to qubits at
            
        Returns:
            /
        """
        
        self._connections['memory'][_host][_store].l1_store_qubits(_qubit, index, self._sim._sim_time)
        
    def l2_store_qubit(self, _qubit: Qubit, _host: int, _store: int, index: int=-1) -> None:
        
        """
        Stores qubits in the L2 memory
        
        Args:
            _qubits (Qubit): list of qubits
            _store (int): SEND or RECEIVE store
            _channel (str): _channel to listen on
            index (int): index to qubits at
            
        Returns:
            /
        """
        
        self._connections['memory'][_host][_store].l2_store_qubits(_qubit, index, self._sim._sim_time)
        
    def l3_store_qubit(self, _qubit: Qubit, _host: int, _store: int, index: int=-1) -> None:
        
        """
        Stores qubits in the L3 memory
        
        Args:
            _qubits (Qubit): list of qubits
            _store (int): SEND or RECEIVE store
            _channel (str): _channel to listen on
            index (int): index to qubits at
            
        Returns:
            /
        """
        
        self._connections['memory'][_host][_store].l3_store_qubits(_qubit, index, self._sim._sim_time)
        
    def l0_retrieve_qubit(self, _host: int, _store: int, index: int=-1) -> Union[Qubit, None]:
        
        """
        Retrieves a qubit from the L0 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel to retrieve qubit from
            index (int): index to retrieve from
            
        Returns:
            _qubit (Qubit/None): retrieved qubit
        """
        
        return self._connections['memory'][_host][_store].l0_retrieve_qubit(index, self._sim._sim_time)
    
    def l1_retrieve_qubit(self, _host: int, _store: int, index: int=-1) -> Union[Qubit, None]:
        
        """
        Retrieves a qubit from the L1 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel to retrieve qubit from
            index (int): index to retrieve from
            
        Returns:
            _qubit (Qubit/None): rettrieved qubit
        """
        
        return self._connections['memory'][_host][_store].l1_retrieve_qubit(index, self._sim._sim_time)
    
    def l2_retrieve_qubit(self, _host: int, _store: int, index: int=-1) -> Union[Qubit, None]:
        
        """
        Retrieves a qubit from the L2 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel to retrieve qubit from
            index (int): index to retrieve from
            
        Returns:
            _qubit (Qubit/None): rettrieved qubit
        """
        
        return self._connections['memory'][_host][_store].l2_retrieve_qubit(index, self._sim._sim_time)
    
    def l3_retrieve_qubit(self, _host: int, _store: int, index: int=-1) -> Union[Qubit, None]:
        
        """
        Retrieves a qubit from the L0 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel to retrieve qubit from
            index (int): index to retrieve from
            
        Returns:
            _qubit (Qubit/None): rettrieved qubit
        """
        
        return self._connections['memory'][_host][_store].l3_retrieve_qubit(index, self._sim._sim_time)
    
    def l0_peek_qubit(self, _host: int, _store: int, index: int=-1) -> Union[Qubit, None]:
        
        """
        Looks at the qubit without retrieving it from the L0 memory
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel of entanglement store
            _index (int): index of qubit
            
        Returns:
            _qubit (Qubit/None): peeked at qubit
        """
        
        return self._connections['memory'][_host][_store].l0_peek_qubit(index)
    
    def l1_peek_qubit(self, _host: int, _store: int, index: int=-1) -> Union[Qubit, None]:
        
        """
        Looks at the qubit without retrieving it from the L1 memory
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel of entanglement store
            index (int): index of qubit
            
        Returns:
            _qubit (Qubit/None): peeked at qubit
        """
        
        return self._connections['memory'][_host][_store].l1_peek_qubit(index)
    
    def l2_peek_qubit(self, _host: int, _store: int, index: int=-1) -> Union[Qubit, None]:
        
        """
        Looks at the qubit without retrieving it from the L2 memory
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel of entanglement store
            index (int): index of qubit
        
        Returns:
            _qubit (Qubit/None): peeked at qubit
        """
        
        return self._connections['memory'][_host][_store].l2_peek_qubit(index)
    
    def l3_peek_qubit(self, _host: int, _store: int, index: int=-1) -> Union[Qubit, None]:
        
        """
        Looks at the qubit without retrieving it from the L3 memory
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel of entanglement store
            index (int): index of qubit
            
        Returns:
            _qubit (Qubit/None): peeked at qubit
        """
        
        return self._connections['memory'][_host][_store].l3_peek_qubit(index)
    
    def l0_move_qubits_l1(self, _host: int, _store: int, _indices: List[int]) -> None:
        
        """
        Moves qubits given the indices from L0 memory to L1 memory
        
        Args:
            _store (int): entanglement store Send or Receive
            _channel (int): channel of entanglement store
            _indices (list): indices to remove
            
        Returns:
            /
        """
        
        self._connections['memory'][_host][_store].l0_move_qubits_l1(_indices)
        
    def l1_move_qubits_l2(self, _host: int, _store: int, _indices: List[int]) -> None:
        
        """
        Moves qubits given the indices from L1 memory to L2 memory
        
        Args:
            _store (int): entanglement store Send or Receive
            _channel (int): channel of entanglement store
            _indices (list): indices to remove
            
        Returns:
            /
        """
        
        self._connections['memory'][_host][_store].l1_move_qubits_l2(_indices)
        
    def l2_move_qubits_l3(self, _host: int, _store: int, _indices: List[int]) -> None:
        
        """
        Moves qubits given the indices from L2 memory to L3 memory
        
        Args:
            _store (int): entanglement store Send or Receive
            _channel (int): channel of entanglement store
            _indices (list): indices to remove
            
        Returns:
            /
        """
        
        self._connections['memory'][_host][_store].l2_move_qubits_l3(_indices)

    def l3_move_qubits_l1(self, _host: int, _store: int, _indices: List[int]) -> None:
        
        """
        Moves qubits given the indices from L3 memory to L1 memory
        
        Args:
            _store (int): entanglement store Send or Receive
            _channel (int): channel of entanglement store
            _indices (list): indices to remove
            
        Returns:
            /
        """
        
        self._connections['memory'][_host][_store].l3_move_qubits_l1(_indices)
        
    def l0_discard_qubits(self, _host: int, _store: int) -> None:
        
        """
        Discards all qubits in L0 store
        
        Args:
            _store (int) SEND or RECEIVE store
            _channel (str): channel of entanglement store
            
        Returns:
            /
        """
        
        self._connections['memory'][_host][_store].l0_discard_qubits()
        
    def l1_discard_qubits(self, _host: int, _store: int) -> None:
        
        """
        Discards all qubits in L1 store
        
        Args:
            _store (int) SEND or RECEIVE store
            _channel (str): channel of entanglement store
            
        Returns:
            /
        """
        
        self._connections['memory'][_host][_store].l1_discard_qubits()
        
    def l2_discard_qubits(self, _host: int, _store: int) -> None:
        
        """
        Discards all qubits in L2 store
        
        Args:
            _store (int) SEND or RECEIVE store
            _channel (str): channel of entanglement store
            
        Returns:
            /
        """
        
        self._connections['memory'][_host][_store].l2_discard_qubits()
        
    def l3_discard_qubits(self, _host: int, _store: int) -> None:
        
        """
        Discards all qubits in L0 store
        
        Args:
            _store (int) SEND or RECEIVE store
            _channel (str): channel of entanglement store
            
        Returns:
            /
        """
        
        self._connections['memory'][_host][_store].l3_discard_qubits()
        
    def l2_num_purification(self, _host: int, _store: int) -> int:
        
        """
        Returns the number of purifications in the store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel of entanglement store
            
        Returns:
            l2_num_purification (int): number of available purifications
        """
        
        return int(np.floor(self.l1_num_qubits(_host, _store) / 2))
    
    async def l2_purify(self, _host: int, _store: int, _direction: bool=0, _gate: str='CNOT', _basis: str='Z', _index_src: int=None, _index_dst: int=None) -> int:
        
        """
        Purifies the two qubits in the store given the indices
        
        Args:
            _store (int): send or receive entanglement store
            _channel (int): connection id
            direction (int): whether to apply src->dst or dst->src
            _gate (str): gate to apply
            _basis (str): in which basis to measure the target qubit
            _index_src (int): index of source qubit
            _index_dst (int): index of dest qubit
            
        Returns:
            _res (int): measurement result
        """

        if _index_src is None:
            _index_src = 0
        if _index_dst is None:
            _index_dst = 0
        
        _qubit_src, _qubit_dst = self._connections['memory'][_host][_store].l2_purify(_index_src, _index_dst, self._sim._sim_time)
        
        _res = await self.apply_gate('purification', _qubit_src, _qubit_dst, _direction, _gate, _basis, combine=True, remove=True)
        
        self._connections['memory'][_host][_store].l2_store_qubit(_qubit_src, -1, self._sim._sim_time)
        
        return _res
    
    def l0_estimate_fidelity(self, _host: int, _store: int, index: int=-1) -> float:
        
        """
        Estimates the fidelity of a qubit in L0 memory
        
        Args:
            _store (int): SEND or RECEIVE memory
            _channel (str): channel
            index (int): index of qubit
            
        Returns:
            _fidelity (float): estimated fidelity
        """
        
        return self._connections['memory'][_host][_store].l0_estimate_fidelity(index, self._sim._sim_time)
    
    def l1_estimate_fidelity(self, _host: int, _store: int, index: int=-1) -> float:
        
        """
        Estimates the fidelity of a qubit in L1 memory
        
        Args:
            _store (int): SEND or RECEIVE memory
            _channel (str): channel
            index (int): index of qubit
            
        Returns:
            _fidelity (float): estimated fidelity
        """
        
        return self._connections['memory'][_host][_store].l1_estimate_fidelity(index, self._sim._sim_time)
    
    def l2_estimate_fidelity(self, _host: int, _store: int, index: int=-1) -> float:
        
        """
        Estimates the fidelity of a qubit in L2 memory
        
        Args:
            _store (int): SEND or RECEIVE memory
            _channel (str): channel
            index (int): index of qubit
            
        Returns:
            _fidelity (float): estimated fidelity
        """
        
        return self._connections['memory'][_host][_store].l2_estimate_fidelity(index, self._sim._sim_time)
    
    def l3_estimate_fidelity(self, _host: int, _store: int, index: int=-1) -> float:
        
        """
        Estimates the fidelity of a qubit in L3 memory
        
        Args:
            _store (int): SEND or RECEIVE memory
            _channel (str): channel
            index (int): index of qubit
            
        Returns:
            _fidelity (float): estimated fidelity
        """
        
        return self._connections['memory'][_host][_store].l3_estimate_fidelity(index, self._sim._sim_time)
    
    def l1_check_results(self, _host: int, _store: int) -> bool:
        
        """
        Checks if there is a L1 result
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel
            
        Returns:
            _res (bool): whether results exist or not
        """
        
        if self._layer_results[_host][_store][L1]:
            return True
        return False
    
    def l2_check_results(self, _host: int, _store: int) -> bool:
        
        """
        Checks if there is a L2 result
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel
            
        Returns:
            _res (bool): whether results exist or not
        """
        
        if self._layer_results[_host][_store][L2]:
            return True
        return False
    
    def l1_num_results(self, _host: int, _store: int) -> int:
        
        """
        Returns the number of results in L1 store
        
        Args:
            _store (int): SEND or RECEIVE
            _channel (str): channel
            
        Returns:
            l1_num_results (int): number of results in L1 store
        """
        
        return len(self._layer_results[_host][_store][L1])
    
    def l2_num_results(self, _host: int, _store: int) -> int:
        
        """
        Returns the number of results in L2 store
        
        Args:
            _store (int): SEND or RECEIVE
            _channel (str): channel
            
        Returns:
            l2_num_results (int): number of results in L2 store
        """
        
        return len(self._layer_results[_host][_store][L2])

    def l1_store_result(self, _store: int, _packet: Packet) -> None:
        
        """
        Stores the L1 result of packet
        
        Args:
            _store (int): SEND or RECEIVE store
            _packet (Packet): packet of which result to store
            
        Returns:
            /
        """
        
        self._layer_results[_packet.l2_src][_store][L1].append(_packet._l1._entanglement_success)
        
    def l2_store_result(self, _store: int, _packet: Packet) -> None:
        
        """
        Stores the L2 result of packet
        
        Args:
            _store (int): SEND or RECEIVE store
            _packet (Packet): packet of which result to store
            
        Returns:
            /
        """
        
        self._layer_results[_packet.l2_dst][_store][L2].append(_packet._l2._purification_success)
        
    def l1_retrieve_result(self, _host: int, _store: int) -> np.array:
        
        """
        Retrieves the first result in the L1 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel
            
        Returns:
            _res (np.array): L1 result
        """
        
        return self._layer_results[_host][_store][L1].pop(0)
    
    def l2_retrieve_result(self, _host: int, _store: int) -> np.array:
        
        """
        Retrieves the first result in the L2 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel
            
        Returns:
            _res (np.array): L2 result
        """
        
        return self._layer_results[_host][_store][L2].pop(0)
    
    def l1_compare_results(self, _packet: Packet) -> np.array:
        
        """
        Compares the L1 results of the packet with the result in storage
        
        Args:

            
        Returns:
            res (np.array): result of the comparison
        """
        
        stor_res = self.l1_retrieve_result(_packet.l1_ack, _packet.l2_src)
        return np.logical_and(_packet.l1_entanglement_success, stor_res)
    
    def l2_compare_results(self, _packet: Packet) -> np.array:
        
        """
        Compares the L1 results of the packet with the result in storage
        
        Args:

            
        Returns:
            res (np.array): result of the comparison
        """
        
        stor_res = self.l2_retrieve_result(_packet.l2_ack, _packet.l2_src)
        return np.logical_not(np.logical_xor(_packet.l2_purification_success, stor_res))
    
    @property
    def l1_packets(self) -> List[Packet]:
        
        """
        Returns all packets in L1 packet store
        
        Args:
            /
            
        Returns:
            l1_packets (list): packets in L1 store
        """
        
        return self._l1_packets
    
    @property
    def l2_packets(self) -> List[Packet]:
        
        """
        Returns all packets in L2 packet store
        
        Args:
            /
            
        Returns:
            l2_packets (list): packets in L2 store
        """
        
        return self._l2_packets
    
    @property
    def l3_packets(self) -> List[Packet]:
        
        """
        Returns all packets in L3 packet store
        
        Args:
            /
            
        Returns:
            l3_packets (list): packets in L3 store
        """
        
        return self._l3_packets
    
    def l1_check_packets(self) -> bool:
        
        """
        Checks whether packets are in L1 store
        
        Args:
            /
            
        Returns:
            _l1_check_packets (bool): check if packets in L1 store
        """
        
        if self._l1_packets:
            return True
        return False
    
    def l2_check_packets(self) -> bool:
        
        """
        Checks whether packets are in L2 store
        
        Args:
            /
            
        Returns:
            _l2_check_packets (bool): check if packets in L2 store
        """
        
        if self._l2_packets:
            return True
        return False
    
    def l3_check_packets(self) -> bool:
        
        """
        Checks whether packets are in L3 store
        
        Args:
            /
            
        Returns:
            _l3_check_packets (bool): check if packets in L3 store
        """
        
        if self._l3_packets:
            return True
        return False
    
    def l1_num_packets(self) -> int:
        
        """
        Returns the number of packets in L1 store
        
        Args:
            /
            
        Returns:
            _l1_num_packets (int): number of packets in L1 store
        """
        
        return len(self._l1_packets)
    
    def l2_num_packets(self) -> int:
        
        """
        Returns the number of packets in L2 store
        
        Args:
            /
            
        Returns:
            _l2_num_packets (int): number of packets in L2 store
        """
        
        return len(self._l2_packets)
    
    def l3_num_packets(self) -> int:
        
        """
        Returns the number of packets in L3 store
        
        Args:
            /
            
        Returns:
            _l3_num_packets (int): number of packets in L3 store
        """
        
        return len(self._l3_packets)
    
    def l1_store_packet(self, _packet: Packet) -> None:
        
        """
        Stores a packet in the L1 store
        
        Args:
            _packet (Packet): packet to store
            
        Returns:
            /
        """
        
        self._l1_packets.append(_packet)
        
    def l2_store_packet(self, _packet: Packet) -> None:
        
        """
        Stores a packet in the L2 store
        
        Args:
            _packet (Packet): packet to store
            
        Returns:
            /
        """
        
        self._l2_packets.append(_packet)
        
    def l3_store_packet(self, _packet: Packet) -> None:
        
        """
        Stores a packet in the L3 store
        
        Args:
            _packet (Packet): packet to store
            
        Returns:
            /
        """
        
        self._l3_packets.append(_packet)
        
    def l1_retrieve_packet(self) -> Packet:
        
        """
        Retrieves a packet from the L1 store
        
        Args:
            /
            
        Returns:
            _packet (Packet): retrieved packet
        """
        
        return self._l1_packets.pop(0)
    
    def l2_retrieve_packet(self) -> Packet:
        
        """
        Retrieves a packet from the L2 store
        
        Args:
            /
            
        Returns:
            _packet (Packet): retrieved packet
        """
        
        return self._l2_packets.pop(0)
        
    def l3_retrieve_packet(self) -> Packet:
        
        """
        Retrieves a packet from the L3 store
        
        Args:
            /
            
        Returns:
            _packet (Packet): retrieved packet
        """
        
        return self._l3_packets.pop(0)