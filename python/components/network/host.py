import logging
import traceback
import numpy as np
import asyncio as asc
from functools import partial
from typing import List, Dict, Set, Union, Any

from qcns.python.components.simulation import Simulation
from qcns.python.components.simulation.event import StopEvent, SendEvent, ReceiveEvent, GateEvent, WaitEvent
from qcns.python.components.qubit.qubit import Qubit, combine_state, remove_qubits
from qcns.python.components.connection.channel import PChannel
from qcns.python.components.packet import Packet
from qcns.python.components.hardware.memory import QuantumMemory
from qcns.python.components.connection import SingleQubitConnection, SenderReceiverConnection, TwoPhotonSourceConnection, BellStateMeasurementConnection, FockStateConnection, L3Connection
from qcns.python.components.network.qprogram import QProgram  

__all__ = ['Host']

class QuantumError:
    pass

_GATE_DURATION = {'X': 5e-6, 'Y': 5e-6, 'Z': 3e-6, 'H': 6e-6, 'S': 2e-6, 'T': 1e-6, 'bsm': 1e-5, 'prob_bsm': 1e-5, 'CNOT': 12e-5, 'measure': 1e-6}

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

class Host:
    
    pass

class Host:
    
    """
    Represents a Host
    
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
    
    def __init__(self, node_id: int, sim: Simulation, stop: bool=True, l1_qprogram: QProgram=QProgram(), l2_qprogram: QProgram=QProgram(), l3_qprogram: QProgram=QProgram(),
                 l4_qprogram: QProgram=QProgram(), l7_qprogram: QProgram=QProgram(), gate_duration: Dict[str, float]=_GATE_DURATION, 
                 gate_parameters: Dict[str, float]=_GATE_PARAMETERS, pulse_duration: float=10**(-11)) -> None:
        
        """
        Initializes a Host
        
        Args:
            node_id (int): ID of Host
            sim (Simulation): Simulation
            stop (bool): whether the Host has a finite number of events
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
        
        self._qprograms: Dict[int, QProgram] = {layer: qprogram for layer, qprogram in {L1: l1_qprogram, L2: l2_qprogram, L3: l3_qprogram, L4: l4_qprogram, L7: l7_qprogram}.items() if qprogram.layer}
        
        if not all([layer == qprogram.layer for layer, qprogram in self._qprograms.items()]):
            raise ValueError('Cannot have a quantum program without the program of the previous layer')
        
        for layer, qprogram in self._qprograms.items():
            qprogram.host = self
            
            if (layer - 1) in self._qprograms:
                qprogram.prev_protocol = self._qprograms[layer - 1]
            if (layer + 1) in self._qprograms:
                qprogram.next_protocol = self._qprograms[layer + 1]
            
            if not (layer + 1):
                pass
        
        self._pulse_duration: float = pulse_duration
        self._gates: Dict[str, Qubit] = {k: v for k, v in Qubit.__dict__.items() if not k.startswith(('__', 'f'))}
        self._gate_duration: Dict[str, float] = gate_duration
        self._gate_parameters: Dict[str, float] = gate_parameters
        
        self._connections: Dict[str, Dict[str, Any]] = {'sqs': {}, 'eqs': {}, 'packet': {}, 'memory': {}}
        self._neighbors: Set[int] = set()
        
        self._time: float = 0.
        
        self._layer_results: Dict[int, Dict[int, Dict[int, List[np.array]]]] = {}
        
        self._packets: Dict[int, Dict[int, Dict[int, List[Packet]]]] = {}
        
        self._resume: asc.Event = asc.Event()
        self.stop: bool = stop
    
        self.run = partial(self.log_exceptions, self.run)
        self._sim.add_host(self)
    
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
        
        if not self.stop:
            self._sim.schedule_event(StopEvent(self.id))
        
        try:
            await func()
            if self.stop:
                self._sim.schedule_event(StopEvent(self.id))
        except Exception as e:
            logging.error(f'Host {self.id} has Exception')
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
        self._resume.clear()
        
        for neighbor in self.neighbors:
            self._connections['memory'][neighbor][SEND].discard_qubits()
            self._connections['memory'][neighbor][RECEIVE].discard_qubits()
            self._layer_results[neighbor] = {SEND: {L1: [], L2: [], L3: []}, RECEIVE: {L1: [], L2: [], L3: []}}
            self._packets[neighbor] = {SEND: {L1: [], L2: [], L3: []}, RECEIVE: {L1: [], L2: [], L3: []}}
        
    def set_sqs_connection(self, host: Host, sender_source: str='perfect', receiver_source: str='perfect',
                           sender_num_sources: int=-1, receiver_num_sources: int=-1,
                           sender_length: float=0., sender_attenuation: float=-0.016, sender_in_coupling_prob: float=1., sender_out_coupling_prob: float=1., sender_lose_qubits: bool=False, sender_channel_errors: List[QuantumError]=None,
                           receiver_length: float=0., receiver_attenuation: float=-0.016, receiver_in_coupling_prob: float=1., receiver_out_coupling_prob: float=1., receiver_lose_qubits: bool=False, receiver_channel_errors: List[QuantumError]=None) -> None:
        
        """
        Sets up a single qubit source connection
        
        Args:
            host (Host): other host to establish connection with
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
        
        if sender_length < 0. or receiver_length < 0.:
            raise ValueError('Length of connection cannot be negative')
        
        self._neighbors.add(host.id)
        host._neighbors.add(self.id)
            
        if sender_channel_errors is None:
            sender_channel_errors = []
        if receiver_channel_errors is None:
            receiver_channel_errors = []
            
        if not isinstance(sender_channel_errors, list):
            sender_channel_errors = [sender_channel_errors]
            
        if not isinstance(receiver_channel_errors, list):
            receiver_channel_errors = [receiver_channel_errors]
            
        for com_error in sender_channel_errors:
            com_error.add_signal_time(sender_length + receiver_length)
        
        for com_error in receiver_channel_errors:
            com_error.add_signal_time(sender_length + receiver_length)
        
        connection_s_r = SingleQubitConnection(self, host.id, self._sim, sender_source, sender_num_sources, sender_length + receiver_length, 
                                               sender_attenuation, sender_in_coupling_prob, sender_out_coupling_prob, sender_lose_qubits, sender_channel_errors)
        connection_r_s = SingleQubitConnection(host, self.id, self._sim, receiver_source, receiver_num_sources, sender_length + receiver_length, 
                                               receiver_attenuation, receiver_in_coupling_prob, receiver_out_coupling_prob, receiver_lose_qubits, receiver_channel_errors)
        
        self._connections['sqs'][host.id] = {SEND: connection_s_r, RECEIVE: connection_r_s._channel}
        host._connections['sqs'][self.id] = {SEND: connection_r_s, RECEIVE: connection_s_r._channel}
    
    def set_eqs_connection(self, host: Host, sender_type: str='sr', sender_model: str='perfect', 
                           receiver_type: str='sr', receiver_model: str='perfect',
                           sender_source: str='perfect', receiver_source: str='perfect', 
                           sender_detector: str='perfect', receiver_detector: str='perfect',
                           sender_num_sources: int=-1, receiver_num_sources: int=-1,
                           sender_length: float=0., sender_attenuation: float=-0.016, sender_in_coupling_prob: float=1., sender_out_coupling_prob: float=1., sender_lose_qubits: bool=False,
                           receiver_length: float=0., receiver_attenuation: float=-0.016, receiver_in_coupling_prob: float=1., receiver_out_coupling_prob: float=1., receiver_lose_qubits: bool=False,
                           sender_memory_mode: str='lifo', sender_memory_size: int=-1, sender_efficiency: float=1., sender_memory_errors: List[QuantumError]=None, 
                           receiver_memory_mode: str='lifo', receiver_memory_size: int=-1, receiver_efficiency: float=1., receiver_memory_errors: List[QuantumError]=None) -> None:
        
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
        
        if sender_length < 0. or receiver_length < 0.:
            raise ValueError('Length of connection cannot be negative')
        
        self._neighbors.add(host.id)
        host._neighbors.add(self.id)
        
        if sender_memory_errors is None:
            sender_memory_errors = []
        if receiver_memory_errors is None:
            receiver_memory_errors = []
            
        if not isinstance(sender_memory_errors, list):
            sender_memory_errors = [sender_memory_errors]
            
        if not isinstance(receiver_memory_errors, list):
            receiver_memory_errors = [receiver_memory_errors]
        
        sender_memory_send = QuantumMemory(sender_memory_mode, sender_memory_size, sender_efficiency, sender_memory_errors)
        sender_memory_receive = QuantumMemory(receiver_memory_mode, -1, sender_efficiency, sender_memory_errors)
        receiver_memory_send = QuantumMemory(receiver_memory_mode, receiver_memory_size, receiver_efficiency, receiver_memory_errors)
        receiver_memory_receive = QuantumMemory(sender_memory_mode, -1, receiver_efficiency, receiver_memory_errors)
        
        if sender_type == 'sr':
            connection_s_r = SenderReceiverConnection(self, host.id, self._sim, sender_model, sender_source, receiver_detector, sender_num_sources,
                                                      sender_length + receiver_length, sender_attenuation, sender_in_coupling_prob, sender_out_coupling_prob, sender_lose_qubits, 
                                                      sender_memory_send, receiver_memory_receive)
        
        if sender_type == 'tps':
            connection_s_r = TwoPhotonSourceConnection(self, host.id, self._sim, sender_model, sender_source, sender_detector, receiver_detector, sender_num_sources, 
                                                       sender_length, sender_attenuation, sender_in_coupling_prob, sender_out_coupling_prob, sender_lose_qubits, 
                                                       receiver_length, receiver_attenuation, receiver_in_coupling_prob, receiver_out_coupling_prob, receiver_lose_qubits, 
                                                       sender_memory_send, receiver_memory_receive)
        
        if sender_type == 'bsm':
            connection_s_r = BellStateMeasurementConnection(self, host.id, self._sim, sender_model, sender_source, receiver_source, sender_detector, receiver_detector, sender_num_sources, 
                                                            sender_length, sender_attenuation, sender_in_coupling_prob, sender_out_coupling_prob, sender_lose_qubits, 
                                                            receiver_length, receiver_attenuation, receiver_in_coupling_prob, receiver_out_coupling_prob, receiver_lose_qubits, 
                                                            sender_memory_send, receiver_memory_receive)
            
        if sender_type == 'fs':
            connection_s_r = FockStateConnection(self, host.id, self._sim, sender_model, sender_source, receiver_source, sender_detector, receiver_detector, sender_num_sources, 
                                                            sender_length, sender_attenuation, sender_in_coupling_prob, sender_out_coupling_prob, sender_lose_qubits, 
                                                            receiver_length, receiver_attenuation, receiver_in_coupling_prob, receiver_out_coupling_prob, receiver_lose_qubits, 
                                                            sender_memory_send, receiver_memory_receive)
        
        if receiver_type == 'sr':
            connection_r_s = SenderReceiverConnection(host, self.id, self._sim, receiver_model, receiver_source, sender_detector, receiver_num_sources,
                                                      sender_length + receiver_length, receiver_attenuation, receiver_in_coupling_prob, receiver_out_coupling_prob, receiver_lose_qubits, 
                                                      receiver_memory_send, sender_memory_receive)
            
        if receiver_type == 'tps':
            connection_r_s = TwoPhotonSourceConnection(host, self.id, self._sim, receiver_model, receiver_source, receiver_detector, sender_detector, receiver_num_sources,
                                                       receiver_length, receiver_attenuation, receiver_in_coupling_prob, receiver_out_coupling_prob, receiver_lose_qubits, 
                                                       sender_length, sender_attenuation, sender_in_coupling_prob, sender_out_coupling_prob, sender_lose_qubits, 
                                                       receiver_memory_send, sender_memory_receive)
        
        if receiver_type == 'bsm':
            connection_r_s = BellStateMeasurementConnection(host, self.id, self._sim, receiver_model, receiver_source, sender_source, receiver_detector, sender_detector, receiver_num_sources, 
                                                            receiver_length, receiver_attenuation, receiver_in_coupling_prob, receiver_out_coupling_prob, receiver_lose_qubits, 
                                                            sender_length, sender_attenuation, sender_in_coupling_prob, sender_out_coupling_prob, sender_lose_qubits, 
                                                            receiver_memory_send, sender_memory_receive)
            
        if receiver_type == 'fs':
            connection_r_s = FockStateConnection(host, self.id, self._sim, receiver_model, receiver_source, sender_source, receiver_detector, sender_detector, receiver_num_sources, 
                                                            receiver_length, receiver_attenuation, receiver_in_coupling_prob, receiver_out_coupling_prob, receiver_lose_qubits, 
                                                            sender_length, sender_attenuation, sender_in_coupling_prob, sender_out_coupling_prob, sender_lose_qubits, 
                                                            receiver_memory_send, sender_memory_receive)
        
        self._connections['eqs'][host.id] = connection_s_r
        host._connections['eqs'][self.id] = connection_r_s
        
        self._connections['memory'][host.id] = {SEND: sender_memory_send, RECEIVE: sender_memory_receive}
        host._connections['memory'][self.id] = {SEND: receiver_memory_send, RECEIVE: receiver_memory_receive}
    
    def set_pconnection(self, host: Host, length: float=0.) -> None:
        
        """
        Sets up a packet connection between this host an another
        
        Args:
            host (Host): host to set up connection with
            length (float): length of connection
            
        Returns:
            /
        """
        
        if length < 0.:
            raise ValueError('Length of connection cannot be negative')
        
        self._neighbors.add(host.id)
        host._neighbors.add(self.id)
        
        channel_s = PChannel(length)
        channel_r = PChannel(length)
        
        self._connections['packet'][host.id] = {SEND: channel_s, RECEIVE: channel_r}
        host._connections['packet'][self.id] = {SEND: channel_r, RECEIVE: channel_s}
    
        self._layer_results[host.id] = {SEND: {L1: [], L2: [], L3: []}, RECEIVE: {L1: [], L2: [], L3: []}}
        host._layer_results[self.id] = {SEND: {L1: [], L2: [], L3: []}, RECEIVE: {L1: [], L2: [], L3: []}}
        
        self._packets[host.id] = {SEND: {L1: [], L2: [], L3: []}, RECEIVE: {L1: [], L2: [], L3: []}}
        host._packets[self.id] = {SEND: {L1: [], L2: [], L3: []}, RECEIVE: {L1: [], L2: [], L3: []}}
    
    def set_connection(self, host: Host, sender_type: str='sr', sender_model: str='perfect', 
                        receiver_type: str='sr', receiver_model: str='perfect',
                        sp_sender_source: str='perfect', sp_receiver_source: str='perfect',
                        he_sender_source: str='perfect', he_receiver_source: str='perfect', 
                        sender_detector: str='perfect', receiver_detector: str='perfect',
                        sp_sender_num_sources: int=-1, sp_receiver_num_sources: int=-1,
                        he_sender_num_sources: int=-1, he_receiver_num_sources: int=-1,
                        sender_length: float=0., sender_attenuation: float=-0.016, sender_in_coupling_prob: float=1., sender_out_coupling_prob: float=1., sender_lose_qubits: bool=False, sender_channel_errors: List[QuantumError]=None,
                        receiver_length: float=0., receiver_attenuation: float=-0.016, receiver_in_coupling_prob: float=1., receiver_out_coupling_prob: float=1., receiver_lose_qubits: bool=False, receiver_channel_errors: List[QuantumError]=None,
                        sender_memory_mode: str='lifo', sender_memory_size: int=-1, sender_efficiency: float=1., sender_memory_errors: List[QuantumError]=None, 
                        receiver_memory_mode: str='lifo', receiver_memory_size: int=-1, receiver_efficiency: float=1., receiver_memory_errors: List[QuantumError]=None) -> None:
        
        """
        Sets a single photon source connection, an entangled photon source connection and a packet connection
        
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
            sp_sender_num_sources (int): number of single photon sources at sender side
            sp_receiver_num_sources (int): number of single photon sources at receiver
            he_sender_num_sources (int): number of heralded photon sources at sender
            he_receiver_num_sources (int): number of heralded photon sources at receiver
            sender_length (float): length from sender to receiver
            sender_attenuation (float): sender attenuation coefficient
            sender_in_coupling_prob (float): probability of coupling qubit into fiber
            sender_out_coupling_prob (float): probability of coupling qubit out of fiber
            sender_lose_qubits (bool): whether to lose qubits of sender channel
            sender_channel_errors (list): list of errors on sender channel
            receiver_length (float): length from receiver to sender
            receiver_attenuation (float): receiver attenuation coefficient
            receiver_in_coupling_prob (float): probability of coupling qubit into fiber
            receiver_out_coupling_prob (float): probability of coupling qubit out of fiber
            receiver_lose_qubits (bool): whether to lose qubits of receiver channel
            receiver_channel_errors (list): list of errors on receiver channel
            sender_memory_size (int): size of sender memory
            sender_efficiency (float): efficiency of sender memory
            sender_memory_errors (list): list of memory errors at sender
            receiver_memory_size (int): size of receiver memory
            receiver_efficiency (float): efficiency of receiver memory
            receiver_memory_errors (list): list of memory errors at receiver
            
        Returns:
            /
        """
        
        self._neighbors.add(host.id)
        host._neighbors.add(self.id)
        
        self.set_sqs_connection(host, sp_sender_source, sp_receiver_source, sp_sender_num_sources, sp_receiver_num_sources,
                                sender_length, sender_attenuation, sender_in_coupling_prob, sender_out_coupling_prob, sender_lose_qubits, sender_channel_errors, 
                                receiver_length, receiver_attenuation, receiver_in_coupling_prob, receiver_out_coupling_prob, receiver_lose_qubits, receiver_channel_errors)
        self.set_eqs_connection(host, sender_type, sender_model, 
                                receiver_type, receiver_model, 
                                he_sender_source, he_receiver_source, 
                                sender_detector, receiver_detector, 
                                he_sender_num_sources, he_receiver_num_sources,
                                sender_length, sender_attenuation, sender_in_coupling_prob, sender_out_coupling_prob, sender_lose_qubits, 
                                receiver_length, receiver_attenuation, receiver_in_coupling_prob, receiver_out_coupling_prob, receiver_lose_qubits, 
                                sender_memory_mode, sender_memory_size, sender_efficiency, sender_memory_errors, 
                                receiver_memory_mode, receiver_memory_size, receiver_efficiency, receiver_memory_errors)
        self.set_pconnection(host, sender_length + receiver_length)
    
    def set_l3_connection(self, host: Host, length: float=0.,
                          sender_num_sources: int=-1, sender_source_duration: float=0.,
                          sender_success_probability: float=1., sender_fidelity: float=1., sender_fidelity_variance: float=0.,
                          receiver_num_sources: int=-1, receiver_source_duration: float=0.,
                          receiver_success_probability: float=1., receiver_fidelity: float=1., receiver_fidelity_variance: float=0.,
                          sender_memory_mode: str='lifo', sender_memory_size: int=-1, sender_efficiency: float=1., sender_memory_errors: List[QuantumError]=None, 
                          receiver_memory_mode: str='lifo', receiver_memory_size: int=-1, receiver_efficiency: float=1., receiver_memory_errors: List[QuantumError]=None) -> None:
        
        """
        Sets an abstract L3 Connection between this host and other host
        
        Args:
            host (Host): host to set up connection with
            length (float): length of connection
            sender_num_sources (int): number of sources at the sender
            sender_source_duration (float): duration of the source at the sender
            sender_success_probability (float): probability to successfully have entanglement between this host and other host
            sender_fidelity (float): fidelity of the entanglement between this host and other host
            sender_fidelity_variance (float): variance of the fidelity between this host and other host
            receiver_num_sources (int): number of sources at the receiver
            receiver_source_duration (float): duration of the source at the receiver
            receiver_success_probability (float): probability to successfully have entanglement between other host and this host
            receiver_fidelity (float): fidelity of the entanglement between other host and this host
            receiver_fidelity_variance (float): variance of the fidelity between other host and this host
            sender_memory_mode (str): mode how to extract qubits at sender, fifo or lifo
            sender_memory_size (int): memory size of this host
            sender_efficiency (float): efficiency of the memory of this host
            sender_memory_errors (list): errors to apply to qubits in memory of this host
            receiver_memory_mode (str): mode how to extract qubits at receiver, fifo or lifo
            receiver_memory_size (int): memory size of other host
            receiver_efficiency (float): efficiency of the memory of other host
            receiver_memory_errors (list): errors to apply to qubits in memory of other host
            
        Returns:
            /
        """
        
        if length < 0.:
            raise ValueError('Length of connection cannot be negative')
        
        self._neighbors.add(host.id)
        host._neighbors.add(self.id)
        
        if sender_memory_errors is None:
            sender_memory_errors = []
        if receiver_memory_errors is None:
            receiver_memory_errors = []
        
        if not isinstance(sender_memory_errors, list):
            sender_memory_errors = [sender_memory_errors]
            
        if not isinstance(receiver_memory_errors, list):
            receiver_memory_errors = [receiver_memory_errors]
        
        sender_memory_send = QuantumMemory(sender_memory_mode, sender_memory_size, sender_efficiency, sender_memory_errors)
        sender_memory_receive = QuantumMemory(receiver_memory_mode, -1, sender_efficiency, sender_memory_errors)
        receiver_memory_send = QuantumMemory(receiver_memory_mode, receiver_memory_size, receiver_efficiency, receiver_memory_errors)
        receiver_memory_receive = QuantumMemory(sender_memory_mode, -1, receiver_efficiency, receiver_memory_errors)
        
        connection_s_r = L3Connection(self, host.id, self._sim, length, sender_num_sources, sender_source_duration, sender_success_probability, sender_fidelity, sender_fidelity_variance, sender_memory_send, receiver_memory_receive)
        connection_r_s = L3Connection(host, self.id, self._sim, length, receiver_num_sources, receiver_source_duration, receiver_success_probability, receiver_fidelity, receiver_fidelity_variance, receiver_memory_send, sender_memory_receive)
        
        self._connections['eqs'][host.id] = connection_s_r
        host._connections['eqs'][self.id] = connection_r_s
        
        self._connections['memory'][host.id] = {SEND: sender_memory_send, RECEIVE: sender_memory_receive}
        host._connections['memory'][self.id] = {SEND: receiver_memory_send, RECEIVE: receiver_memory_receive}
        
        self.set_pconnection(host, length)
    
    async def attempt_qubit(self, receiver: int, num_requested: int=1, estimate: bool=True) -> None:
        
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

    async def create_qubit(self, receiver: int, num_requested: int=1) -> None:
        
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
    
    async def attempt_bell_pairs(self, receiver: int, num_requested: int=1, estimate: bool=False) -> None:
        
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
            raise ValueError('Attempting 0 qubits')
        
        self._sim.schedule_event(SendEvent(self._time, self.id))
        
        _num_needed = num_requested
        if estimate:
            _num_needed = int(np.ceil(_num_needed / self._connections['eqs'][receiver]._success_prob))
        
        if not self.has_space(receiver, 0, _num_needed):
            _num_needed = self.remaining_space(receiver, 0)
        
        if not _num_needed:
            return
        
        if _num_needed < num_requested:
            num_requested = _num_needed
        
        self._connections['eqs'][receiver].attempt_bell_pairs(num_requested, _num_needed)
    
    async def create_bell_pairs(self, receiver: int, num_requested: int=1) -> None:
        
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
        
        if not num_requested:
            return
        
        self._connections['eqs'][receiver].create_bell_pairs(num_requested)
    
    def apply_gate(self, gate: str, *args: List[Any], combine: bool=False, remove: bool=False) -> Union[int, None]:
        
        """
        Applys a gate to qubits
        
        Args:
            gate (str): gate to apply
            *args (list): variable length argument list
            combine (bool): to combine qubits that are in different qsystems
            remove (bool): to remove qubits
            
        Returns:
            res (int/np.array/None): result of the gate
        """
        
        self._time += self._gate_duration.get(gate, 5e-6)
        self._sim.schedule_event(GateEvent(self._time, self.id))
        
        if combine and gate in ['CNOT', 'CY', 'CZ', 'CH', 'CPHASE', 'CU', 'SWAP', 'bell_state', 'bsm', 'prob_bsm']:
            combine_state(args[:2])
        if combine and gate in ['TOFFOLI', 'CCU', 'CSWAP']:
            combine_state(args[:3])
        
        res = self._gates[gate](*args)
        
        if remove and gate == 'measure':
            remove_qubits(args[:1])
        if remove and gate in ['bsm', 'prob_bsm']:
            remove_qubits(args[:2])
        
        return res
    
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
        
        _qubit_src, _qubit_dst = self._connections['memory'][host][store].purify(index_src, index_dst, self._time)
        
        combine_state([_qubit_src, _qubit_dst])
        
        _res = self._gates['purification'](_qubit_src, _qubit_dst, direction, gate, basis)
        
        remove_qubits([_qubit_dst])
        
        self._connections['memory'][host][store].store_qubit(L2, _qubit_src, -1, self._time)
        
        return _res
    
    async def send_qubit(self, receiver: int, qubit: Qubit) -> None:
        
        """
        Sends a qubit to the specified receiver
        
        Args:
            receiver (int): receiver to send qubit to
            qubit (Qubit): qubit to send
            
        Returns:
            /
        """
        
        
        self._sim.schedule_event(SendEvent(self._time, self.id))
        self._sim.schedule_event(ReceiveEvent(self._time + self._connections['sqs'][receiver][SEND]._channel._sending_time, receiver))
        
        self._connections['sqs'][receiver][SEND].put(qubit)
    
    async def receive_qubit(self, sender: int=None, time_out: float=None) -> Union[Qubit, None]:
        
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
            return self._connections['sqs'][sender][RECEIVE].get()
        
        for _sender in self.neighbors:
            if not self._connections['sqs'][_sender][RECEIVE].empty():
                return self._connections['sqs'][_sender][RECEIVE].get()
         
    async def send_packet(self, _packet: Packet) -> None:
        
        """
        Sends a packet to the specified receiver in the packet
        
        Args:
            _packet (Packet): packet to send
            
        Returns:
            /
        """
        
        self._time += len(_packet) * self._pulse_duration
        self._sim.schedule_event(SendEvent(self._time, self.id))
        self._sim.schedule_event(ReceiveEvent(self._time + self._connections['packet'][_packet.l2_dst][SEND]._signal_time, _packet.l2_dst))
        
        self._connections['packet'][_packet.l2_dst][SEND].put(_packet)
        
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
        
        if sender is not None:
            return self._connections['packet'][sender][RECEIVE].get()
        
        for _sender in self.neighbors:
            if not self._connections['packet'][_sender][RECEIVE].empty():
                return self._connections['packet'][_sender][RECEIVE].get()
    
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
        
        return self._connections['memory'][host][store].has_space(num_qubits)
    
    def remaining_space(self, host: int, store: int) -> int:
        
        """
        Returns the remaining size of the memory
        
        Args:
            host (int): the host the memory points to 
            store (int): SEND or RECEIVE store
            
        Returns:
            remaining_space (int): remaining size of the memory
        """
        
        return self._connections['memory'][host][store].remaining_space()
    
    def memory_size(self, host: int, store: int) -> int:
        
        """
        Returns the size of a given memory
        
        Args:
            host (int): host the memory points to
            store (int): SEND or RECEIVE store
            
        Returns:
            memory_size (int): size of the memory
        """
        
        return self._connections['memory'][host][store].size
    
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
        
        self._connections['memory'][host][store].change_size(size)
    
    def l0_num_qubits(self, host: int, store: int) -> int:
        
        """
        Returns the number of qubits from the L0 store
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            
        Returns:
            l0_num_qubits (int): number of qubits in the L0 store
        """
        
        return self._connections['memory'][host][store].num_qubits(L0)
    
    def l1_num_qubits(self, host: int, store: int) -> int:
        
        """
        Returns the number of qubits from the L1 store
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            
        Returns:
            l1_num_qubits (int): number of qubits in the L1 store
        """
        
        return self._connections['memory'][host][store].num_qubits(L1)
    
    def l2_num_qubits(self, host: int, store: int) -> int:
        
        """
        Returns the number of qubits from the L2 store
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            
        Returns:
            l2_num_qubits (int): number of qubits in the L2 store
        """
        
        return self._connections['memory'][host][store].num_qubits(L2)
    
    def l3_num_qubits(self, host: int, store: int) -> int:
        
        """
        Returns the number of qubits from the L3 store
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            
        Returns:
            l3_num_qubits (int): number of qubits in the L3 store
        """
        
        return self._connections['memory'][host][store].num_qubits(L3)
    
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
        
        self._connections['memory'][host][store].store_qubit(L0, qubit, index, self._time)
        
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
        
        self._connections['memory'][host][store].store_qubit(L1, qubit, index, self._time)
        
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
        
        self._connections['memory'][host][store].store_qubit(L2, qubit, index, self._time)
        
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
        
        self._connections['memory'][host][store].store_qubit(L3, qubit, index, self._time)
        
    def l0_retrieve_qubit(self, host: int, store: int, index: int=None) -> Union[Qubit, None]:
        
        """
        Retrieves a qubit from the L0 store
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to retrieve from
            
        Returns:
            qubit (Qubit/None): retrieved qubit
        """
        
        return self._connections['memory'][host][store].retrieve_qubit(L0, index, self._time)
    
    def l1_retrieve_qubit(self, host: int, store: int, index: int=None) -> Union[Qubit, None]:
        
        """
        Retrieves a qubit from the L1 store
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to retrieve from
            
        Returns:
            qubit (Qubit/None): retrieved qubit
        """
        
        return self._connections['memory'][host][store].retrieve_qubit(L1, index, self._time)
    
    def l2_retrieve_qubit(self, host: int, store: int, index: int=None) -> Union[Qubit, None]:
        
        """
        Retrieves a qubit from the L2 store
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to retrieve from
            
        Returns:
            qubit (Qubit/None): retrieved qubit
        """
        
        return self._connections['memory'][host][store].retrieve_qubit(L2, index, self._time)
    
    def l3_retrieve_qubit(self, host: int, store: int, index: int=None, offset_index: int=None) -> Union[Qubit, None]:
        
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
        
        return self._connections['memory'][host][store].retrieve_qubit(L3, index, self._time, offset_index)
    
    def l0_peek_qubit(self, host: int, store: int, index: int=None) -> Union[Qubit, None]:
        
        """
        Looks at the qubit without retrieving it from the L0 memory
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to retrieve from
            
        Returns:
            _qubit (Qubit/None): peeked at qubit
        """
        
        return self._connections['memory'][host][store].peek_qubit(L0, index)
    
    def l1_peek_qubit(self, host: int, store: int, index: int=None) -> Union[Qubit, None]:
        
        """
        Looks at the qubit without retrieving it from the L1 memory
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to retrieve from
            
        Returns:
            qubit (Qubit/None): peeked at qubit
        """
        
        return self._connections['memory'][host][store].peek_qubit(L1, index)
    
    def l2_peek_qubit(self, host: int, store: int, index: int=None) -> Union[Qubit, None]:
        
        """
        Looks at the qubit without retrieving it from the L2 memory
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to retrieve from
            
        Returns:
            qubit (Qubit/None): peeked at qubit
        """
        
        return self._connections['memory'][host][store].peek_qubit(L2, index)
    
    def l3_peek_qubit(self, host: int, store: int, index: int=None) -> Union[Qubit, None]:
        
        """
        Looks at the qubit without retrieving it from the L3 memory
        
        Args:
            host (int): the host the memory points to
            store (int): SEND or RECEIVE store
            index (int): index to retrieve from
            
        Returns:
            qubit (Qubit/None): peeked at qubit
        """
        
        return self._connections['memory'][host][store].peek_qubit(L3, index)
    
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
        
        self._connections['memory'][host][store].move_qubits(L0, L1, indices)
        
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
        
        self._connections['memory'][host][store].move_qubits(L1, L2, indices)
    
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
        
        self._connections['memory'][host][store].move_qubits(L1, L3, indices)
    
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
        
        self._connections['memory'][host][store].move_qubits(L2, L3, indices)

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
        
        self._connections['memory'][host][store].move_qubits(L3, L1, indices)
    
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
        
        self._connections['memory'][host][store].move_qubits(L3, L3, indices)
      
    def l0_discard_qubits(self, host: int, store: int) -> None:
        
        """
        Discards all qubits in L0 store
        
        Args:
            host (int): the host the memory points to
            store (int) SEND or RECEIVE store
            
        Returns:
            /
        """
        
        self._connections['memory'][host][store].discard_qubits(L0)
        
    def l1_discard_qubits(self, host: int, store: int) -> None:
        
        """
        Discards all qubits in L1 store
        
        Args:
            host (int): the host the memory points to
            store (int) SEND or RECEIVE store
            
        Returns:
            /
        """
        
        self._connections['memory'][host][store].discard_qubits(L1)
        
    def l2_discard_qubits(self, host: int, store: int) -> None:
        
        """
        Discards all qubits in L2 store
        
        Args:
            host (int): the host the memory points to
            store (int) SEND or RECEIVE store
            
        Returns:
            /
        """
        
        self._connections['memory'][host][store].discard_qubits(L2)
        
    def l3_discard_qubits(self, host: int, store: int) -> None:
        
        """
        Discards all qubits in L0 store
        
        Args:
            host (int): the host the memory points to
            store (int) SEND or RECEIVE store
            
        Returns:
            /
        """
        
        self._connections['memory'][host][store].discard_qubits(L3)
    
    def discard_all_qubits(self, host: int, store: int) -> None:
        
        """
        Discards all qubits in all the stores
        
        Args:
            host (int): the host the memory points to
            store (int) SEND or RECEIVE store
            
        Returns:
            /
        """
        
        self._connections['memory'][host][store].discard_qubits()
    
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
        
        return self._connections['memory'][host][store].estimate_fidelity(L0, index, self._time)
    
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
        
        return self._connections['memory'][host][store].estimate_fidelity(L1, index, self._time)
    
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
        
        return self._connections['memory'][host][store].estimate_fidelity(L2, index, self._time)
    
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
        
        return self._connections['memory'][host][store].estimate_fidelity(L3, index, self._time)
    
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
        
    def l1_retrieve_result(self, host: int, store: int) -> np.array:
        
        """
        Retrieves the first result in the L1 store
        
        Args:
            host (int): connected host
            store (int): SEND or RECEIVE store
            
        Returns:
            res (np.array): L1 result
        """
        
        return self._layer_results[host][store][L1].pop(0)
    
    def l2_retrieve_result(self, host: int, store: int) -> np.array:
        
        """
        Retrieves the first result in the L2 store
        
        Args:
            host (int): connected host
            store (int): SEND or RECEIVE store
            
        Returns:
            res (np.array): L2 result
        """
        
        return self._layer_results[host][store][L2].pop(0)
    
    def l1_compare_results(self, packet: Packet) -> np.array:
        
        """
        Compares the L1 results of the packet with the result in storage
        
        Args:
            packet (Packet): packet to compare results of
            
        Returns:
            res (np.array): result of the comparison
        """
        
        stor_res = self.l1_retrieve_result(packet.l2_src, not packet.l1_ack)
        return np.logical_and(packet.l1_success, stor_res)
    
    def l2_compare_results(self, packet: Packet) -> np.array:
        
        """
        Compares the L1 results of the packet with the result in storage
        
        Args:
            packet (Packet): packet to compare results of
            
        Returns:
            res (np.array): result of the comparison
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
        
        return len(self._packets[host][store][int])
    
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
        
        return self._connections['memory'][host][store].add_offset(offset)
    
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
        
        return self._connections['memory'][host][store].remove_offset(index)