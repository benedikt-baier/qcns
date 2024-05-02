
import numpy as np
import asyncio as asc

from copy import deepcopy
from functools import partial
from typing import List, Dict, Union, Type

from python.components.simulation import Simulation
from python.components.event import StopEvent, SendEvent, ReceiveEvent, GateEvent 
from python.components.qubit import Qubit, QSystem, combine_state, remove_qubits
from python.components.channel import QChannel, PChannel
from python.components.packet import Packet
from python.components.memory import QuantumMemory
from python.components.photon_source import SinglePhotonSource, AtomPhotonSource, TwoPhotonSource, BSM

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
        self._gates: Dict[str, QUBIT] = {k: v for k, v in Qubit.__dict__.items() if not k.startswith(('__', 'f'))}
        self._gate_duration: Dict[str, float] = _gate_duration
        self._gate_parameters: Dict[str, float] = _gate_parameters
        
        self._connections: Dict[str, Dict[str, Any]] = {}
        
        self._entanglement_connection: Dict[str, Type[PhotonSource]] = {'rs': AtomPhotonSource, 'ps': TwoPhotonSource, 'bsm': BSM}
        
        self._layer_results: Dict[str, Dict[int, Dict[int, np.array]]] = {}
        self._packets: Dict[str, Dict[int, Dict[int, Packet]]] = {}
        
        self._resume: asyncio.Event = asc.Event()
    
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
    
    def log_exceptions(self, func) -> None:
        
        """
        Wrapper to log exceptions in the run function
        
        Args:
            func (Function): function to wrap around
            
        Returns:
            /
        """
        
        try:
            res = func()
            self._sim.schedule_event(StopEvent(self._node_id))
            return res
        except Exception as e:
            traceback.print_exc()
    
    def set_sqs_connection(self, host: Host, length: float=0.0, com_errors: List[QuantumError]=None, mem_errors: List[QuantumError]=None, model_s: str='standard_model', model_r: str='standard_model', lose_qubits_s: bool=False, lose_qubits_r: bool=False, attenuation_coefficient_s: float=-0.016, attenuation_coefficient_r: float=-0.016, efficiency_s: float=1.0, efficiency_r: float=1.0) -> None:
        
        """
        Sets up a single qubit source connection
        
        Args:
            host (Host): Host to set up connection with
            length (float): length of the connection
            com_errors (list): List of communication errors
            mem_errors (list): List of memory errors
            model_s (str): model for sender sided photon source
            model_r (str): model for receiver sided photon source
            lose_qubits_s (bool): whether to lose qubits in the sender receiver channel
            lose_qubits_r (bool): whether to lose qubits in the receiver sender channel
            attenuation_coefficient_s (float): attenuation coefficient for sender receiver channel
            attenuation_coefficient_r (float): attenuation coefficient for receiver sender channel
            efficiency_s (float): efficiency of sender memory
            efficiency_r (float): efficiency of receiver memory
            
        Returns:
            /
        """
        
        if com_errors is None:
            com_errors = []
        if mem_errors is None:
            mem_errors = []
            
        for com_error in com_errors:
            com_error.add_signal_time(length)
        
        channel_s = QChannel(length, com_errors, lose_qubits_s, attenuation_coefficient_s)
        channel_r = QChannel(length, com_errors, lose_qubits_r, attenuation_coefficient_r)
        
        self._connections[host._node_id]['sqs'][SEND]['c'] = channel_s
        self._connections[host._node_id]['sqs'][RECEIVE] = channel_r
        host._connections[self._node_id]['sqs'][SEND]['c'] = channel_r
        host._connections[self._node_id]['sqs'][RECEIVE] = channel_s
        
        self._connections[host._node_id]['sqs'][SEND]['s'] = SinglePhotonSource(model_s)
        host._connections[self._node_id]['sqs'][SEND]['s'] = SinglePhotonSource(model_r)
        
        self._connections[host._node_id]['memory'][SEND] = QuantumMemory(mem_errors, efficiency_s)
        self._connections[host._node_id]['memory'][RECEIVE] = QuantumMemory(mem_errors, efficiency_s)
        host._connections[self._node_id]['memory'][SEND] = QuantumMemory(mem_errors, efficiency_r)
        host._connections[self._node_id]['memory'][RECEIVE] = QuantumMemory(mem_errors, efficiency_r)
    
    def set_eqs_connection(self, host: Host, length: float=0.0, entanglement_type: str='rs', com_errors: List[QuantumError]=None, mem_errors: List[QuantumError]=None, model_s: str='standard_model', model_r: str='standard_model', lose_qubits_s: bool=False, lose_qubits_r: bool=False, attenuation_coefficient_s: float=-0.016, attenuation_coefficient_r: float=-0.016, efficiency_s: float=1.0, efficiency_r: float=1.0) -> None:
        
        """
        Sets up a entangled qubit connection between this host and another host
        
        Args:
            host (Host): host to set up connection with
            length (float): length of connection
            entanglement_type (str): type of entanglement connection
            com_errors (list): List of communication errors
            mem_errors (list): List of memory errors
            model_s (str): model for sender sided photon source
            model_r (str): model for receiver sided photon source
            lose_qubits_s (bool): whether to lose qubits in the sender receiver channel
            lose_qubits_r (bool): whether to lose qubits in the receiver sender channel
            attenuation_coefficient_s (float): attenuation coefficient for sender receiver channel
            attenuation_coefficient_r (float): attenuation coefficient for receiver sender channel
            efficiency_s (float): efficiency of sender memory
            efficiency_r (float): efficiency of receiver memory
            
        Returns:
            /
        """
        
        if com_errors is None:
            com_errors = []
            
        if mem_errors is None:
            mem_errors = []
        
        if entanglement_type != 'rs':
            length *= 0.5
         
        for com_error in com_errors:
            com_error.add_signal_time(length)
            
        channel_s = QChannel(length, com_errors, lose_qubits_s, attenuation_coefficient_s)
        channel_r = QChannel(length, com_errors, lose_qubits_r, attenuation_coefficient_r)
        
        self._connections[host._node_id]['eqs'][SEND]['c'] = channel_s
        self._connections[host._node_id]['eqs'][RECEIVE] = channel_r
        host._connections[self._node_id]['eqs'][SEND]['c'] = channel_r
        host._connections[self._node_id]['eqs'][RECEIVE] = channel_s

        eqs_s = self._entanglement_connection[entanglement_type](self, host, length, (model_s, model_r, self._gate_parameters['prob_bsm_p']), self._sim)
        eqs_r = self._entanglement_connection[entanglement_type](host, self, length, (model_r, model_s, host._gate_parameters['prob_bsm_p']), self._sim)
        
        self._connections[host._node_id]['eqs'][SEND]['s'] = eqs_s
        host._connections[self._node_id]['eqs'][SEND]['s'] = eqs_r
        
        if self._connections[host._node_id]['memory'][SEND] is None:
            self._connections[host._node_id]['memory'][SEND] = QuantumMemory(mem_errors, efficiency_s)
            self._connections[host._node_id]['memory'][RECEIVE] = QuantumMemory(mem_errors, efficiency_s)
            host._connections[self._node_id]['memory'][SEND] = QuantumMemory(mem_errors, efficiency_r)
            host._connections[self._node_id]['memory'][RECEIVE] = QuantumMemory(mem_errors, efficiency_r)
    
    def set_pconnection(self, host: Host, length: float=0.0) -> None:
        
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
        
        self._connections[host._node_id]['packet'][SEND] = channel_s
        self._connections[host._node_id]['packet'][RECEIVE] = channel_r
        host._connections[self._node_id]['packet'][SEND] = channel_r
        host._connections[self._node_id]['packet'][RECEIVE] = channel_s
    
        self._layer_results[host._node_id] = {SEND: {L1: [], L2: [], L3: []}, RECEIVE: {L1: [], L2: [], L3: []}}
        host._layer_results[self._node_id] = {SEND: {L1: [], L2: [], L3: []}, RECEIVE: {L1: [], L2: [], L3: []}}
        
        self._packets[host._node_id] = {SEND: {L1: [], L2: [], L3: []}, RECEIVE: {L1: [], L2: [], L3: []}}
        host._packets[self._node_id] = {SEND: {L1: [], L2: [], L3: []}, RECEIVE: {L1: [], L2: [], L3: []}}
    
    def set_lose_prob(self, _host: str, _lose_prob: float) -> None:
        
        """
        Sets the probability to lose qubits for specific host connection
        
        Args:
            _host (str): connection to host
            _lose_prob (float): probability to lose qubit
            
        Returns:
            /
        """
        
        self._connections[_host]['sqs'][SEND]['c'].set_lose_prob(_lose_prob)
        self._connections[_host]['eqs'][SEND]['c'].set_lose_prob(_lose_prob)  
    
    def set_connection(self, host: Host, length: float=0.0, entanglement_type: str='rs', com_errors: List[QuantumError]=None, mem_errors: List[QuantumError]=None, s_model_s: str='standard_model', s_model_r: str='standard_model', e_model_s: str='standard_model', e_model_r: str='standard_model', lose_qubits_s: bool=False, lose_qubits_r: bool=False, attenuation_coefficient_s: float=-0.016, attenuation_coefficient_r: float=-0.016, efficiency_s: float=1.0, efficiency_r: float=1.0) -> None:
        
        """
        Sets a single photon source connection, a entangled photon source connection and a packet connection
        
        Args:
            host (Host): host to set up connection with
            length (float): length of connection
            entanglement_type (str): type of entanglement connection
            com_errors (list): List of communication errors
            mem_errors (list): List of memory errors
            s_model_s (str): single photon source model for sender
            s_model_r (str): single photon source model for receiver
            e_model_s (str): entangled photon source model for sender
            e_model_r (str): entangled photon source model for receiver
            lose_qubits_s (bool): whether to lose qubits in the sender receiver channel
            lose_qubits_r (bool): whether to lose qubits in the receiver sender channel
            attenuation_coefficient_s (float): attenuation coefficient for sender receiver channel
            attenuation_coefficient_r (float): attenuation coefficient for receiver sender channel
            efficiency_s (float): efficiency of sender memory
            efficiency_r (float): efficiency of receiver memory
            
        Returns:
            /
        """
        
        self._connections[host._node_id] = {'sqs': {SEND: {'s': None, 'c': None}, RECEIVE: None},
                                            'eqs': {SEND: {'s': None, 'c': None}, RECEIVE: None},
                                            'packet': {SEND: None, RECEIVE: None},
                                            'memory': {SEND: None, RECEIVE: None}}
        
        host._connections[self._node_id] = {'sqs': {SEND: {'s': None, 'c': None}, RECEIVE: None},
                                            'eqs': {SEND: {'s': None, 'c': None}, RECEIVE: None},
                                            'packet': {SEND: None, RECEIVE: None},
                                            'memory': {SEND: None, RECEIVE: None}}
        
        self.set_sqs_connection(host, length, com_errors, mem_errors, s_model_s, s_model_r, lose_qubits_s, lose_qubits_r, attenuation_coefficient_s, attenuation_coefficient_r, efficiency_s, efficiency_r)
        self.set_eqs_connection(host, length, entanglement_type, com_errors, mem_errors, e_model_s, e_model_r, lose_qubits_s, lose_qubits_r, attenuation_coefficient_s, attenuation_coefficient_r, efficiency_s, efficiency_r)
        self.set_pconnection(host, length)
       
    def create_qsystem(self, _num_qubits: int, _fidelity: float=1., _sparse: bool=False) -> QSystem:
        
        """
        Creates a new qsystem at the host
        
        Args:
            _num_qubits (int): number of qubits in the qsystem
            _fidelity (float): fidelity of quantum system
            _sparse (float): sparsity of qsystem
            
        Returns:
            qsys (QSystem): new qsystem
        """
        
        return Simulation.create_qsystem(_num_qubits, _fidelity, _sparse)
    
    def delete_qsystem(self, _qsys: QSystem) -> None:
        
        """
        Deletes a qsystem at the host
        
        Args:
            qsys (QSystem): qsystem to delete
            
        Returns:
            /
        """
        
        Simulation.delete_qsystem(_qsys)
    
    async def create_qubit(self, _receiver: str, _num_requested: int) -> List[Qubit]:
        
        """
        Creates a number of requested qubits from the source pointed to receiver
        
        Args:
            _receiver (str): receiver to which photon source points to
            _num_requested (int): number of requested qubits
            
        Returns:
            qubits (list): created qubits
        """
        
        self._sim.schedule_event(SendEvent(self._sim._sim_time, self._node_id))
        
        await self._resume.wait()
        self._resume.clear()
        
        return self._connections[_receiver]['sqs'][SEND]['s'].create_qubit(_num_requested)
    
    async def create_bell_pairs(self, _receiver: str, _num_requested: int) -> None:
        
        """
        Creates number of requested bell pairs
        
        Args:
            _receiver (str): receiver of bell pairs
            _num_requested (int): number of requested bell pairs
            
        Returns:
            /
        """
        
        self._sim.schedule_event(SendEvent(self._sim._sim_time, self._node_id))
        
        await self._resume.wait()
        self._resume.clear()
        
        self._connections[_receiver]['eqs'][SEND]['s'].create_bell_pairs(_num_requested, _num_requested)
    
    async def apply_gate(self, gate: str, *args: str, combine: bool=False, remove: bool=False) -> Union[int, None]:
        
        """
        Applys a gate to qubits
        
        Args:
            gate (str): gate to apply
            *args (list): variable length argument list
            combine (bool): to combine qubits that are in different qsystems
            remove (bool): to remove qubits
            
        Returns:
            res (int/None): result of the gate
        """
        
        self._sim.schedule_event(GateEvent(self._sim._sim_time + self._gate_duration.get(gate, 5e-6), self._node_id))
        
        await self._resume.wait()
        self._resume.clear()
        
        if combine and gate in ['CNOT', 'CY', 'CZ', 'CH', 'CPHASE', 'CU', 'SWAP', 'bell_state', 'bsm', 'prob_bsm', 'purification']:
            combine_state(args[:2])
        if combine and gate in ['TOFFOLI', 'CCU', 'CSWAP']:
            combine_state(args[:3])
        
        res = self._gates[gate](*args)
        
        if remove and gate == 'measure':
            remove_qubits(args[:1])
        if remove and gate in ['bsm', 'prob_bsm']:
            remove_qubits(args[:2])
        if remove and gate in ['purification']:
            remove_qubits(args[1:2])
        
        return res
    
    async def send_qubit(self, _receiver: str, _qubit: Qubit) -> None:
        
        """
        Sends a qubit to the specified receiver
        
        Args:
            _receiver (str): _receiver to send _qubit to
            _qubit (Qubit): qubit to send
            
        Returns:
            /
        """
        
        self._sim.schedule_event(SendEvent(self._sim._sim_time, self._node_id))
        self._sim.schedule_event(ReceiveEvent(self._sim._sim_time + self._connections[_receiver]['sqs'][SEND]['c']._signal_time, _receiver))
        
        await self._resume.wait()
        self._resume.clear()
        
        self._connections[_receiver]['sqs'][SEND]['c'].put(_qubit)
        
    async def receive_qubit(self, _sender: str) -> Qubit:
        
        """
        Waits until a qubit is received
        
        Args:
            _sender (str): sender to receive qubit from
            
        Returns:
            _qubit (Qubit): received qubit
        """
        
        await self._resume.wait()
        self._resume.clear()
        
        return self._connections[_sender]['sqs'][RECEIVE].get()
    
    async def receive_qubit_wait(self, _sender: str, _time_out: float=0.1) -> Union[Qubit, None]:
        
        """
        Receives a qubit from the specified host with a timeout
        
        Args:
            _sender (str): sender to receive qubit from
            _time_out (float): how long to wait for qubit
            
        Returns:
            _qubit (Qubit/None): received qubit
        """
        
        try:
            await asc.wait_for(self._resume.wait(), timeout=_time_out)
        except asc.TimeoutError:
            return None

        return self._connections[_sender]['sqs'][RECEIVE].get()
    
    async def send_packet(self, _packet: Packet) -> None:
        
        """
        Sends a packet to the specified receiver in the packet
        
        Args:
            _packet (Packet): packet to send
            
        Returns:
            /
        """
        
        self._sim.schedule_event(SendEvent(self._sim._sim_time, self._node_id))
        self._sim.schedule_event(ReceiveEvent(self._sim._sim_time + len(_packet) * self._pulse_duration + self._connections[_packet._l2._dst]['packet'][SEND]._signal_time, _packet._l2._dst))
        
        await self._resume.wait()
        self._resume.clear()
        
        self._connections[_packet._l2._dst]['packet'][SEND].put(_packet)
        
    async def receive_packet(self, _sender: str) -> Packet:
        
        """
        Receives a packet
        
        Args:
            _sender (str): channel to listen on
            
        Returns:
            _packet (Packet): received packet
        """
        
        await self._resume.wait()
        self._resume.clear()
        
        return self._connections[_sender]['packet'][RECEIVE].get()
    
    async def receive_packet_wait(self, _sender: str, _time_out: float=0.1) -> Union[Packet, None]:
        
        """
        Receives a packet with a timeout
        
        Args:
            _sender (str): channel to listen on
            _time_out (float): timeout
            
        Returns:
            _packet (Packet/None): received packet or None
        """
        
        try:
            await asc.wait_for(self._resume.wait(), timeout=_time_out)
        except asc.TimeoutError:
            return None
        
        return self._connections[_sender]['packet'][RECEIVE].get()
    
    def l0_num_qubits(self, _store: int, _channel: str) -> int:
        
        """
        Returns the number of qubits from the L0 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel of entanglement store
            
        Returns:
            l0_num_qubits (int): number of qubits in the L0 store
        """
        
        return self._connections[_channel]['memory'][_store].l0_num_qubits()
    
    def l1_num_qubits(self, _store: int, _channel: str) -> int:
        
        """
        Returns the number of qubits from the L1 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel of entanglement store
            
        Returns:
            l1_num_qubits (int): number of qubits in the L1 store
        """
        
        return self._connections[_channel]['memory'][_store].l1_num_qubits()
    
    def l2_num_qubits(self, _store: int, _channel: str) -> int:
        
        """
        Returns the number of qubits from the L2 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel of entanglement store
            
        Returns:
            l2_num_qubits (int): number of qubits in the L2 store
        """
        
        return self._connections[_channel]['memory'][_store].l2_num_qubits()
    
    def l3_num_qubits(self, _store: int, _channel: str) -> int:
        
        """
        Returns the number of qubits from the L3 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel of entanglement store
            
        Returns:
            l3_num_qubits (int): number of qubits in the L3 store
        """
        
        return self._connections[_channel]['memory'][_store].l3_num_qubits()
    
    def l0_store_qubit(self, _qubit: Qubit, _store: int, _channel: str, _index: int=-1) -> None:
        
        """
        Stores a qubit in the L0 memory
        
        Args:
            _qubits (Qubit): list of qubits
            _store (int): SEND or RECEIVE store
            _channel (str): _channel to listen on
            _index (int): index to qubits at
            
        Returns:
            /
        """
        
        self._connections[_channel]['memory'][_store].l0_store_qubits(_qubit, self._sim._sim_time, _index)
        
    def l1_store_qubit(self, _qubit: Qubit, _store: int, _channel: str, _index: int=-1) -> None:
        
        """
        Stores a qubit in the L1 memory
        
        Args:
            _qubits (Qubit): list of qubits
            _store (int): SEND or RECEIVE store
            _channel (str): _channel to listen on
            _index (int): index to qubits at
            
        Returns:
            /
        """
        
        self._connections[_channel]['memory'][_store].l1_store_qubits(_qubit, self._sim._sim_time, _index)
        
    def l2_store_qubit(self, _qubit: Qubit, _store: int, _channel: str, _index: int=-1) -> None:
        
        """
        Stores qubits in the L2 memory
        
        Args:
            _qubits (Qubit): list of qubits
            _store (int): SEND or RECEIVE store
            _channel (str): _channel to listen on
            _index (int): index to qubits at
            
        Returns:
            /
        """
        
        self._connections[_channel]['memory'][_store].l2_store_qubits(_qubit, self._sim._sim_time, _index)
        
    def l3_store_qubit(self, _qubit: Qubit, _store: int, _channel: str, _index: int=-1) -> None:
        
        """
        Stores qubits in the L3 memory
        
        Args:
            _qubits (Qubit): list of qubits
            _store (int): SEND or RECEIVE store
            _channel (str): _channel to listen on
            _index (int): index to qubits at
            
        Returns:
            /
        """
        
        self._connections[_channel]['memory'][_store].l3_store_qubits(_qubit, self._sim._sim_time, _index)
        
    def l0_retrieve_qubit(self, _store: int, _channel: str, _index: int=-1) -> Union[Qubit, None]:
        
        """
        Retrieves a qubit from the L0 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel to retrieve qubit from
            _index (int): index to retrieve from
        """
        
        self._connections[_channel]['memory'][_store].l0_retrieve_qubit(_index, self._sim._sim_time)
    
    def l1_retrieve_qubit(self, _store: int, _channel: str, _index: int=-1) -> Union[Qubit, None]:
        
        """
        Retrieves a qubit from the L1 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel to retrieve qubit from
            _index (int): index to retrieve from
        """
        
        return self._connections[_channel]['memory'][_store].l1_retrieve_qubit(_index, self._sim._sim_time)
    
    def l2_retrieve_qubit(self, _store: int, _channel: str, _index: int=-1) -> Union[Qubit, None]:
        
        """
        Retrieves a qubit from the L2 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel to retrieve qubit from
            _index (int): index to retrieve from
        """
        
        return self._connections[_channel]['memory'][_store].l2_retrieve_qubit(_index, self._sim._sim_time)
    
    def l3_retrieve_qubit(self, _store: int, _channel: str, _index: int=-1) -> Union[Qubit, None]:
        
        """
        Retrieves a qubit from the L0 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel to retrieve qubit from
            _index (int): index to retrieve from
        """
        
        return self._connections[_channel]['memory'][_store].l3_retrieve_qubit(_index, self._sim._sim_time)
    
    def l0_peek_qubit(self, _store: int, _channel: str, _index: int=-1) -> Union[Qubit, None]:
        
        """
        Looks at the qubit without retrieving it from the L0 memory
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel of entanglement store
            _index (int): index of qubit
        """
        
        return self._connections[_channel]['memory'][_store].l0_peek_qubit(_index)
    
    def l1_peek_qubit(self, _store: int, _channel: str, _index: int=-1) -> Union[Qubit, None]:
        
        """
        Looks at the qubit without retrieving it from the L1 memory
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel of entanglement store
            _index (int): index of qubit
        """
        
        return self._connections[_channel]['memory'][_store].l1_peek_qubit(_index)
    
    def l2_peek_qubit(self, _store: int, _channel: str, _index: int=-1) -> Union[Qubit, None]:
        
        """
        Looks at the qubit without retrieving it from the L2 memory
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel of entanglement store
            _index (int): index of qubit
        """
        
        return self._connections[_channel]['memory'][_store].l2_peek_qubit(_index)
    
    def l3_peek_qubit(self, _store: int, _channel: str, _index: int=-1) -> Union[Qubit, None]:
        
        """
        Looks at the qubit without retrieving it from the L3 memory
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel of entanglement store
            _index (int): index of qubit
        """
        
        return self._connections[_channel]['memory'][_store].l3_peek_qubit(_index)
    
    def l0_remove_qubits(self, _store: int, _channel: str, _indices: List[int]) -> None:
        
        """
        Removes qubits from the L0 store given indices
        
        Args:
            _store (int): entanglement store Send or Receive
            _channel (int): channel of entanglement store
            _indices (list): indices to remove
            
        Returns:
            /
        """
        
        self._connections[_channel]['memory'][_store].l0_remove_qubits(_indices)
        
    def l1_remove_qubits(self, _store: int, _channel: str, _indices: List[int]) -> None:
        
        """
        Removes qubits from the L1 store given indices
        
        Args:
            _store (int): entanglement store Send or Receive
            _channel (int): channel of entanglement store
            _indices (list): indices to remove
            
        Returns:
            /
        """
        
        self._connections[_channel]['memory'][_store].l1_remove_qubits(_indices)
        
    def l2_remove_qubits(self, _store: int, _channel: str, _indices: List[int]) -> None:
        
        """
        Removes qubits from the L2 store given indices
        
        Args:
            _store (int): entanglement store Send or Receive
            _channel (int): channel of entanglement store
            _indices (list): indices to remove
            
        Returns:
            /
        """
        
        self._connections[_channel]['memory'][_store].l2_remove_qubits(_indices)
        
    def l3_remove_qubits(self, _store: int, _channel: str, _indices: List[int]) -> None:
        
        """
        Removes qubits from the L3 store given indices
        
        Args:
            _store (int): entanglement store Send or Receive
            _channel (int): channel of entanglement store
            _indices (list): indices to remove
            
        Returns:
            /
        """
        
        self._connections[_channel]['memory'][_store].l3_remove_qubits(_indices)
        
    def l0_discard_qubits(self, _store: int, _channel: str) -> None:
        
        """
        Discards all qubits in L0 store
        
        Args:
            _store (int) SEND or RECEIVE store
            _channel (str): channel of entanglement store
            
        Returns:
            /
        """
        
        self._connections[_channel]['memory'][_store].l0_discard_qubits()
        
    def l1_discard_qubits(self, _store: int, _channel: str) -> None:
        
        """
        Discards all qubits in L1 store
        
        Args:
            _store (int) SEND or RECEIVE store
            _channel (str): channel of entanglement store
            
        Returns:
            /
        """
        
        self._connections[_channel]['memory'][_store].l1_discard_qubits()
        
    def l2_discard_qubits(self, _store: int, _channel: str) -> None:
        
        """
        Discards all qubits in L2 store
        
        Args:
            _store (int) SEND or RECEIVE store
            _channel (str): channel of entanglement store
            
        Returns:
            /
        """
        
        self._connections[_channel]['memory'][_store].l2_discard_qubits()
        
    def l3_discard_qubits(self, _store: int, _channel: str) -> None:
        
        """
        Discards all qubits in L0 store
        
        Args:
            _store (int) SEND or RECEIVE store
            _channel (str): channel of entanglement store
            
        Returns:
            /
        """
        
        self._connections[_channel]['memory'][_store].l3_discard_qubits()
        
    def l2_num_purification(self, _store: int, _channel: str) -> int:
        
        """
        Returns the number of purifications in the store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel of entanglement store
            
        Returns:
            l2_num_purification (int): number of available purifications
        """
        
        return int(np.floor(self.l1_num_qubits(_store, _channel) / 2))
    
    def l2_purify(self, _store: int, _channel: int, _direction: bool=0, _gate: str='CNOT', _basis: str='Z', _index_src: int=None, _index_dst: int=None) -> int:
        
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
            index_dst = 0
        
        _qubit_src, _qubit_dst = self._connections[_channel]['memory'][_store].l2_purify(_index_src, _index_dst, self._sim._sim_time)
        
        _res = self.apply_gate('purification', _qubit_src, _qubit_dst, _direction, _gate, _basis, combine=True, remove=True)
        
        self._connections[_channel]['memory'][_store].l2_store_qubit(_qubit_src, self._sim._sim_time, 0)
        
        return _res
    
    def estimate_fidelity(self, _store: int, _channel: str, _index: int=-1) -> float:
        
        """
        Estimates the fidelity of a qubit in memory
        
        Args:
            _store (int): SEND or RECEIVE memory
            _channel (str): channel
            _index (int): _index of qubit
            
        Returns:
            _fidelity (float): estimated fidelity
        """
        
        return self._connections[_channel]['memory'][_store].estimate_fidelity(_index, self._sim._sim_time)
    
    def l1_check_results(self, _store: int, _channel: str) -> bool:
        
        """
        Checks if there is a L1 result
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel
            
        Returns:
            _res (bool): whether results exist or not
        """
        
        if self._layer_results[_channel][_store][L1]:
            return True
        return False
    
    def l2_check_results(self, _store: int, _channel: str) -> bool:
        
        """
        Checks if there is a L2 result
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel
            
        Returns:
            _res (bool): whether results exist or not
        """
        
        if self._layer_results[_channel][_store][L2]:
            return True
        return False
    
    def l1_num_results(self, _store: int, _channel: str) -> int:
        
        """
        Returns the number of results in L1 store
        
        Args:
            _store (int): SEND or RECEIVE
            _channel (str): channel
            
        Returns:
            l1_num_results (int): number of results in L1 store
        """
        
        return len(self._layer_results[_channel][_store][L1])
    
    def l2_num_results(self, _store: int, _channel: str) -> int:
        
        """
        Returns the number of results in L2 store
        
        Args:
            _store (int): SEND or RECEIVE
            _channel (str): channel
            
        Returns:
            l2_num_results (int): number of results in L2 store
        """
        
        return len(self._layer_results[_channel][_store][L2])

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
        
        self._layer_results[_packet.l2_src][_store][L2].append(_packet._l2._purification_success)
        
    def l1_retrieve_result(self, _store: int, _channel: str) -> np.array:
        
        """
        Retrieves the first result in the L1 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel
            
        Returns:
            _res (np.array): L1 result
        """
        
        return self._layer_results[_channel][_store][L1].pop(0)
    
    def l2_retrieve_result(self, _store: int, _channel: str) -> np.array:
        
        """
        Retrieves the first result in the L2 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel
            
        Returns:
            _res (np.array): L2 result
        """
        
        return self._layer_results[_channel][_store][L2].pop(0)
    
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
    
    def l1_check_packets(self, _store: int, _channel: str) -> bool:
        
        """
        Checks whether packets are in L1 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel
            
        Returns:
            _l1_check_packets (bool): check if packets in L1 store
        """
        
        if self._packets[_channel][_store][L1]:
            return True
        return False
    
    def l2_check_packets(self, _store: int, _channel: str) -> bool:
        
        """
        Checks whether packets are in L2 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel
            
        Returns:
            _l2_check_packets (bool): check if packets in L2 store
        """
        
        if self._packets[_channel][_store][L2]:
            return True
        return False
    
    def l3_check_packets(self, _store: int, _channel: str) -> bool:
        
        """
        Checks whether packets are in L3 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel
            
        Returns:
            _l3_check_packets (bool): check if packets in L3 store
        """
        
        if self._packets[_channel][_store][L3]:
            return True
        return False
    
    def l1_num_packets(self, _store: int, _channel: str) -> int:
        
        """
        Returns the number of packets in L1 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel
            
        Returns:
            _l1_num_packets (int): number of packets in L1 store
        """
        
        return len(self._packet[_channel][_store][L1])
    
    def l2_num_packets(self, _store: int, _channel: str) -> int:
        
        """
        Returns the number of packets in L2 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel
            
        Returns:
            _l2_num_packets (int): number of packets in L2 store
        """
        
        return len(self._packet[_channel][_store][L2])
    
    def l3_num_packets(self, _store: int, _channel: str) -> int:
        
        """
        Returns the number of packets in L3 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel
            
        Returns:
            _l3_num_packets (int): number of packets in L3 store
        """
        
        return len(self._packet[_channel][_store][L3])
    
    def l1_store_packet(self, _store: int, _packet: Packet) -> None:
        
        """
        Stores a packet in the L1 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _packet (Packet): packet to store
            
        Returns:
            /
        """
        
        self._packets[_packet.l2_src][_store][L1].append(_packet)
        
    def l2_store_packet(self, _store: int, _packet: Packet) -> None:
        
        """
        Stores a packet in the L2 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _packet (Packet): packet to store
            
        Returns:
            /
        """
        
        self._packets[_packet.l2_src][_store][L2].append(_packet)
        
    def l3_store_packet(self, _store: int, _packet: Packet) -> None:
        
        """
        Stores a packet in the L3 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _packet (Packet): packet to store
            
        Returns:
            /
        """
        
        self._packets[_packet.l3_src][_store][L3].append(_packet)
        
    def l1_retrieve_packet(self, _store: int, _channel: str) -> Packet:
        
        """
        Retrieves a packet from the L1 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel
            
        Returns:
            _packet (Packet): retrieved packet
        """
        
        return self._packets[_channel][_store][L1].pop(0)
    
    def l2_retrieve_packet(self, _store: int, _channel: str) -> Packet:
        
        """
        Retrieves a packet from the L2 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel
            
        Returns:
            _packet (Packet): retrieved packet
        """
        
        return self._packets[_channel][_store][L2].pop(0)
        
    def l3_retrieve_packet(self, _store: int, _channel: str) -> Packet:
        
        """
        Retrieves a packet from the L3 store
        
        Args:
            _store (int): SEND or RECEIVE store
            _channel (str): channel
            
        Returns:
            _packet (Packet): retrieved packet
        """
        
        return self._packets[_channel][_store][L3].pop(0)