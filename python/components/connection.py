
import numpy as np
from typing import List
from __future__ import annotations

from python.components.photon_source import SinglePhotonSource, AtomPhotonSource, PhotonPhotonSource
from python.components.photon_detector import PhotonDetector
from python.components.channel import QChannel
from python.components.memory import QuantumMemory
from python.components.packet import Packet
from python.components.qubit import remove_qubits
from python.components.event import ReceiveEvent
from python.components.simulation import Simulation

__all__ = ['SingleQubitConnection', 'SenderReceiverConnection', 'TwoPhotonSourceConnection', 'BellStateMeasurementConnection']

class Host:
    pass

class QuantumError:
    pass

SEND = 0
RECEIVE = 1

B_0 = np.array([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]], dtype=np.complex128)
I_0 = np.array([[0.25, 0, 0, 0], [0, 0.25, 0, 0], [0, 0, 0.25, 0], [0, 0, 0, 0.25]], dtype=np.complex128)
M_0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)
A_0 = np.array([[0, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 0]], dtype=np.complex128)

SENDER_RECEIVER_MODELS = {'perfect': (0., 1., 1.), 'standard': (8.95e-7, 0.04, 0.14)}
TWO_PHOTON_MODELS = {'perfect': (0., 1., 1., 1., 1.), 'standard': (8.95e-7, 0.04, 0.14, 0.04, 0.14)}
BSM_MODELS = {'perfect': (0., 1., 1., 0., 0.), 'standard': (3e-6, 0.66, 0.313, 0.355, 0.313)}

# use tau as a parameter as detection window depends on it

class SingleQubitConnection:
    
    """
    Represents a Single Qubit Connection
    
    Attr:
        _sender (Host): sender of connection
        _receiver (Host): receiver of connection
        _source (SinglePhotonSource): sender photon source of connection
        _channel (QChannel): quantum channel of connection
        _sim (Simulation): simulation object
        _success_prob (float): probability to successfully create a photon
        _duration (float): duration of creating a qubit
    """
    
    def __init__(self, _sender: Host, _receiver: Host, _sim: Simulation, _sender_source: str='perfect', 
                 _length: float=0., _attenuation_coefficient: float=-0.016, _coupling_prob: float=1., _com_errors: List[QuantumError]=None) -> None:
        
        """
        Initializes a Single Qubit Connection
        
        Args:
            _sender (Host): sender of connection
            _receiver (Host): receiver of connection
            _sim (Simulation): simulation object
            _sender_source (str): name of model to use for single photon source
            _length (float): length of connection
            _attenuation_coefficient (float): attenuation coefficient of fiber
            _coupling_prob (float): probability of coupling photon into fiber
            _com_errors (list): list of errors on the channel
            
        Returns:
            /
        """
        
        self._sender: Host = _sender
        self._receiver: Host = _receiver
        self._source: SinglePhotonSource = SinglePhotonSource(_sender_source)
        self._channel: QChannel = QChannel(_length, _attenuation_coefficient, _coupling_prob, _com_errors)
        
        self._sim: Simulation = _sim
        
        self._success_prob: float = self._channel._coupling_prob * self._channel._lose_prob
        self._duration: float = self._channel._sending_time + self._source._duration
    
    def create_qubit(self, _num_needed: int=1) -> None:
        
        """
        Creates the number of requested qubits and sends it to the receiver
        
        Args:
            _num_needed (int): number of needed qubits
            
        Returns:
            /
        """
        
        for _ in range(_num_needed):
            
            qsys = Simulation.create_qsystem(1)
            qsys._state = np.array([[self._source._fidelity, np.sqrt(self._source._fidelity * (1 - self._source._fidelity))], [np.sqrt(self._source._fidelity * (1 - self._source._fidelity)), 1 - self._source._fidelity]])
            q_1 = qsys.qubits
            self._channel.put_prob(q_1)
            self._sim.schedule_event(ReceiveEvent(self._sim._sim_time + self._duration, self._receiver._node_id))
                  
class SenderReceiverConnection:
    
    """
    Represents a sender receiver connection
    
    Attr:
        _sender (Host): sender of connection
        _receiver (Host): receiver of connection
        _duration (float): duration of interaction of a photon with an atom
        _interaction_prob (float): probability of a photon interacting with an atom
        _state_transfer_fidelity (float): fidelity of transfering the state from the photon to the atom
        _source (AtomPhotonSource): sender atom photon source
        _detector (PhotonSource): Photon Detector at receiver
        _channel (QChannel): quantum channel of connection
        _sender_memory (QuantumMemory): quantum memory of sender
        _receiver_memory (QuantumMemory): quantum memory of receiver
        _sim (Simulation): simulation object
        _success_prob (float): probability of successfully create a bell pair
        _false_prob (float): probability of creating a non entangled qubit pair
        _total_prob (float): overall total probability 
        _inital_state (np.array): initial state
        _duration (float): duration of creating a bell pair
    """
    
    def __init__(self, _sender: Host, _receiver: Host, _sim: Simulation, 
                 _model_name: str='perfect', _sender_source: str='perfect', _receiver_detector: str='perfect', 
                 _length: float=0., _attenuation_coefficient: float=-0.016, _coupling_prob: float=1., _com_errors: List[QuantumError]=None, 
                 _sender_memory: QuantumMemory=None, _receiver_memory: QuantumMemory=None) -> None:
        
        """
        Initializes a sender receiver connection
        
        Args:
            _sender (Host): sender of connection
            _receiver (Host): receiver of connection
            _sim (Simulation): simulation object
            _model_name (str): model to use for connection
            _sender_source (str): model to use for sender source
            _receiver_detector (str): model to use for receiver detector
            _length (float): length of connection
            _attenuation_coefficient (float): attenuation coefficient of fiber
            _coupling_prob (float): probability of coupling photon into fiber
            _com_errors (list): list of errors on connection
            _sender_memory (QuantumMemory): quantum memory of sender
            _receiver_memory (QuantumMemory): quantum memory of receiver
            
        Returns:
            /
        """
        
        model = SENDER_RECEIVER_MODELS[_model_name]
        
        self._sender: Host = _sender
        self._receiver: Host = _receiver
        self._duration: float = model[0]
        self._interaction_prob: float = model[1]
        self._state_transfer_fidelity: float = model[2]
        
        self._source: AtomPhotonSource = AtomPhotonSource(_sender_source)
        self._detector: PhotonDetector = PhotonDetector(_receiver_detector)
        self._channel: QChannel = QChannel(_length, _attenuation_coefficient, _coupling_prob, _com_errors)
        self._sender_memory: QuantumMemory = _sender_memory
        self._receiver_memory: QuantumMemory = _receiver_memory
        
        self._sim: Simulation = _sim
        
        self._success_prob: float = self._channel._coupling_prob * self._channel._lose_prob * self._detector._efficiency * self._interaction_prob * (1 - self._detector._dark_count_prob)
        self._false_prob: float = self._channel._coupling_prob * self._channel._lose_prob * self._detector._efficiency * self._interaction_prob * self._detector._dark_count_prob
        self._total_prob: float = self._success_prob + self._false_prob + (1 - self._state_transfer_fidelity)
        self._inital_state: np.array = np.kron((4 * self._source._fidelity - 1)/3 * B_0 + 4 * (1 - self._source._fidelity)/3 * I_0, M_0)
        
        self._sender_duration: float = self._source._duration + self._channel._sending_time + self._duration + self._detector._duration

    def create_bell_pairs(self, _num_requested: int=1, _num_needed: int=1) -> None:
        
        """
        Creates the number of needed bell pairs based on the models in the connection
        
        Args:
            _num_requested (int): number of requested bell pairs
            _num_needed (int): number of needed bell pairs
            
        Returns:
            /
        """
        
        packet = Packet(self._sender._node_id, self._receiver._node_id, _num_requested, _num_needed)
        
        for i in range(_num_needed):
            
            _curr_time = self._sim._sim_time + self._source._duration
            
            if np.random.uniform(0, 1) > self._success_prob:
                q_1 = Simulation.create_qsystem(1).qubits
                q_2 = Simulation.create_qsystem(1).qubits
                self._sender_memory.l0_store_qubit(q_1, -1, _curr_time)
                self._receiver_memory.l0_store_qubit(q_2, -1, _curr_time)
                continue
            
            qsys = Simulation.create_qsystem(3)
            qsys._state = self._inital_state
            
            q_1, q_2, q_3 = qsys.qubits
            self._sender_memory.l0_store_qubit(q_1, -1, _curr_time)
            self._receiver_memory.l0_store_qubit(q_3, -1, _curr_time)
            
            self._channel.put(q_2)
            
            _curr_time += self._channel._sending_time
            
            q_2 = self._channel.get()
            
            q_1 = self._sender_memory.l0_retrieve_qubit_prob(-1, _curr_time)
            q_3 = self._receiver_memory.l0_retrieve_qubit_prob(-1, _curr_time)
            
            q_3.state_transfer(q_2)
            
            # maybe wrong incorporate depolarization prob as (4F - 1) /3
            # q_1._qsystem._state = (success_prob * total_depolar_prob * q_1._qsystem._state + (success_prob * (1 - total_depolar_prob) + false_prob + (1 - self._state_transfer_fidelity)) * I_0) / self._total_prob
            q_1._qsystem._state = (self._success_prob * q_1._qsystem._state + (self._false_prob + (1 - self._state_transfer_fidelity)) * I_0) / self._total_prob
            
            _curr_time += self._detector._duration
            
            self._sender_memory.l0_store_qubit(q_1, -1, _curr_time)
            self._receiver_memory.l1_store_qubit(q_3, -1, _curr_time)
            packet.update_l1_success(i)
            
        self._sim.schedule_event(ReceiveEvent(self._sim._sim_time + self._sender_duration, self._receiver._node_id))
        self._sender._connections['packet'][self._receiver._node_id][SEND].put(packet)
            
class TwoPhotonSourceConnection:
    
    """
    Represents a Two Photon Source Connection
    
    Attr:
        _sender (Host): sender of connection
        _receiver (Host): receiver of connection
        _duration (float): duration of interaction of a photon with an atom
        _sender_interaction_prob (float): probability of a photon interacting with an atom at sender
        _sender_state_transfer_fidelity (float): fidelity of transfering the state from the photon to the atom at the sender
        _receiver_interaction_prob (float): probability of a photon interacting with an atom at receiver
        _receiver_state_transfer_fidelity (float): fidelity of transfering the state from the photon to the atom at the receiver
        _source (PhotonPhotonSource): Photon Photon Source
        _sender_detector (PhotonDetector): Photon Detector at sender
        _receiver_detector (PhotonDetector): Photon Detector at receiver
        _sender_channel (QChannel): channel from source to sender
        _receiver_channel (QChannel): channel from source to receiver
        _sender_memory (QuantumMemory): quantum memory of sender
        _receiver_memory (QuantumMemory): quantum memory of receiver
        _sim (Simulation): simulation object
        _success_prob (float): probability of successfully create a bell pair
        _sender_false_prob (float): probability of creating a non entangled qubit pair at the sender
        _receiver_false_prob (float): probability of creating a non entangled qubit pair at the receiver
        _total_prob (float): overall total probability
        _inital_state (np.array): initial state
        _sender_duration (float): sender sided overall duration
        _receiver_duration (float): receiver sided overall duration
    """
    
    def __init__(self, _sender: Host, _receiver: Host, _sim: Simulation, 
                 _model_name: str='perfect', _source: str='perfect', _sender_detector: str='perfect', _receiver_detector: str='perfect',
                 _sender_length: float=0., _sender_attenuation: float=-0.016, _sender_coupling_prob: float=1., _sender_com_errors: List[QuantumError]=None,
                 _receiver_length: float=0., _receiver_attenuation: float=-0.016, _receiver_coupling_prob: float=1., _receiver_com_errors: List[QuantumError]=None,
                 _sender_memory: QuantumMemory=None, _receiver_memory: QuantumMemory=None) -> None:
        
        """
        Initializes a Two Photon Source Connection
        
        Args:
            _sender (Host): sender of connection
            _receiver (Host): receiver of connection
            _sim (Simulation): simulation object
            _model_name (str): model to use for connection
            _source (str): model to use for photon photon source
            _sender_detector (str): model to use for sender detector
            _receiver_detector (str): model to use for receiver detector
            _sender_length (float): length of connection from source to sender
            _sender_attenuation (float): attenuation coefficient of fiber from source to sender
            _sender_coupling_prob (float): probability of coupling photon into fiber from source to sender
            _sender_com_errors (list): list of errors on connection from source to sender
            _receiver_length (float): length of connection from source to receiver
            _receiver_attenuation (float): attenuation coefficient of fiber from source to receiver
            _receiver_coupling_prob (float): probability of coupling photon into fiber from source to receiver
            _receiver_com_errors (list): list of errors on connection from source to receiver
            _sender_memory (QuantumMemory): quantum memory of sender
            _receiver_memory (QuantumMemory): quantum memory of receiver
            
        Returns:
            /
        """
        
        model = TWO_PHOTON_MODELS[_model_name]
        
        self._sender: Host = _sender
        self._receiver: Host = _receiver
        self._duration: float = model[0]
        self._sender_interaction_prob: float = model[1]
        self._sender_state_transfer_fidelity: float = model[2]
        self._receiver_interaction_prob: float = model[3]
        self._receiver_state_transfer_fidelity: float = model[4]
        
        self._source: PhotonPhotonSource = PhotonPhotonSource(_source)
        self._sender_detector: PhotonDetector = PhotonDetector(_sender_detector)
        self._receiver_detector: PhotonDetector = PhotonDetector(_receiver_detector)
        self._sender_channel: QChannel = QChannel(_sender_length, _sender_attenuation, _sender_coupling_prob, _sender_com_errors)
        self._receiver_channel: QChannel = QChannel(_receiver_length, _receiver_attenuation, _receiver_coupling_prob, _receiver_com_errors)
        self._sender_memory: QuantumMemory = _sender_memory
        self._receiver_memory: QuantumMemory = _receiver_memory
        
        self._sim: Simulation = _sim
        
        _sender_success_prob = self._sender_channel._coupling_prob * self._sender_channel._lose_prob * self._sender_detector._efficiency * self._sender_interaction_prob * (1 - self._sender_detector._dark_count_prob)
        _receiver_success_prob = self._receiver_channel._coupling_prob * self._receiver_channel._lose_prob * self._receiver_detector._efficiency * self._receiver_interaction_prob * (1 - self._receiver_detector._dark_count_prob)
        
        self._success_prob: float = _sender_success_prob * _receiver_success_prob
        
        self._sender_false_prob: float = self._sender_channel._coupling_prob * self._sender_channel._lose_prob * self._sender_detector._efficiency * self._sender_interaction_prob * self._sender_detector._dark_count_prob
        self._receiver_false_prob: float = self._receiver_channel._coupling_prob * self._receiver_channel._lose_prob * self._receiver_detector._efficiency * self._receiver_interaction_prob * self._receiver_detector._dark_count_prob
        self._total_prob: float = self._success_prob + self._sender_false_prob + self._receiver_false_prob + (1 - self._sender_state_transfer_fidelity) + (1 - self._receiver_state_transfer_fidelity)
        
        self._inital_state: np.array = np.kron(M_0, np.kron((4 * self._source._fidelity - 1)/3 * B_0 + 4 * (1 - self._source._fidelity)/3 * I_0, M_0))
        
        self._sender_duration: float = self._source._duration + self._sender_channel._sending_time + self._duration + self._sender_detector._duration
        self._receiver_duration: float = self._source._duration + self._receiver_channel._sending_time + self._duration + self._receiver_detector._duration

    def create_bell_pairs(self, _num_requested: int=1, _num_needed: int=1) -> None:
        
        """
        Creates the number of needed bell pairs based on the models in the connection
        
        Args:
            _num_requested (int): number of requested bell pairs
            _num_needed (int): number of needed bell pairs
            
        Returns:
            /
        """
        
        packet_s = Packet(self._receiver._node_id, self._sender._node_id, _num_requested, _num_needed)
        packet_r = Packet(self._sender._node_id, self._receiver._node_id, _num_requested, _num_needed)
        
        for i in range(_num_needed):
            
            _curr_time = self._sim._sim_time + self._source._duration
            
            if np.random.uniform(0, 1) > self._success_prob:
                q_1 = Simulation.create_qsystem(1).qubits
                q_2 = Simulation.create_qsystem(1).qubits
                self._sender_memory.l0_store_qubit(q_1, -1, _curr_time)
                self._receiver_memory.l0_store_qubit(q_2, -1, _curr_time)
                continue
            
            qsys = Simulation.create_qsystem(4)
            qsys._state = self._inital_state
            
            q_1, q_2, q_3, q_4 = qsys.qubits
            
            self._sender_memory.l0_store_qubit(q_1, -1, _curr_time)
            self._receiver_memory.l0_store_qubit(q_4, -1, _curr_time)
            
            self._sender_channel.put(q_2)
            self._receiver_channel.put(q_3)
            
            _sender_time = _curr_time + self._sender_channel._sending_time
            _receiver_time = self._receiver_channel._sending_time + _curr_time
            
            q_2 = self._sender_channel.get()
            q_3 = self._receiver_channel.get()
            
            q_1 = self._sender_memory.l0_retrieve_qubit_prob(-1, _sender_time)
            q_4 = self._receiver_memory.l0_retrieve_qubit_prob(-1, _receiver_time)
            
            _sender_time += self._duration + self._sender_detector._duration
            _receiver_time += self._duration + self._receiver_detector._duration
            
            q_1.state_transfer(q_2)
            q_4.state_transfer(q_3)
            
            # maybe wrong incorporate depolarization prob
            q_1._qsystem._state = (self._success_prob * q_1._qsystem._state + (self._sender_false_prob + self._receiver_false_prob + (1 - self._sender_state_transfer_fidelity) + (1 - self._receiver_state_transfer_fidelity)) * I_0) / self._total_prob
            
            self._sender_memory.l0_store_qubit(q_1, -1, _sender_time)
            self._receiver_memory.l0_store_qubit(q_4, -1, _receiver_time)
            packet_s.update_l1_success(i)
            packet_r.update_l1_success(i)
            
        packet_s.set_l1_ps()
        packet_r.set_l1_ps()
        packet_r.set_l1_ack()
        
        self._sim.schedule_event(ReceiveEvent(self._sim._sim_time + self._sender_duration, self._sender._node_id))
        self._sim.schedule_event(ReceiveEvent(self._sim._sim_time + self._receiver_duration, self._receiver._node_id))
        self._sender._connections['packet'][self._receiver._node_id][RECEIVE].put(packet_s)
        self._receiver._connections['packet'][self._sender._node_id][RECEIVE].put(packet_s)

class BellStateMeasurementConnection:
    
    """
    Represents a Bell State Measurement Connection
    
    Attr:
        _sender (Host): sender of connection
        _receiver (Host): receiver of connection
        _duration (float): duration of bell state measurement
        _coin_ph_ph (float): coincidence probability of two photons at the BSM
        _coin_ph_dc (float): coincidence probability of a photon and a dark count
        _coin_dc_dc (float): coincidence probability of two dark counts
        _sender_source (AtomPhotonSource): atom photon source at sender
        _receiver_source (AtomPhotonSource): atom photon source at receiver
        _sender_detector (PhotonDetector): sender sided photon detector
        _receiver_detector (PhotonDetector): receiver sided photon detector
        _sender_channel (QChannel): quantum channel from sender to BSM
        _receiver_channel (QChannel): quantum channel from receiver to BSM
        _sender_memory (QuantumMemory): quantum memory of sender
        _receiver_memory (QuantumMemory): quantum memory of receiver
        _sim (Simulation): simulation object
        _success_prob (float): overall success probability of creating a bell pair
        _false_prob_1 (float): first probability of creating a non entangled pair
        _false_prob_3 (float): second probability of creating a non entangled pair
        _false_prob_4 (float): fourth probability of creating a non entangled pair
        _total_prob (float): overall total probability
        _inital_state (np.array): initial state
        _sender_duration (float): sender sided overall duration
        _receiver_duration (float): receiver sided overall duration
    """
    
    def __init__(self, _sender: Host, _receiver: Host, _sim: Simulation, 
                 _model_name: str='perfect', _sender_source: str='perfect', _receiver_source: str='perfect', _sender_detector: str='perfect', _receiver_detector: str='perfect',
                 _sender_length: float=0., _sender_attenuation_coefficient: float=-0.016, _sender_coupling_prob: float=1., _sender_com_errors: List[QuantumError]=None,
                 _receiver_length: float=0., _receiver_attenuation_coefficient: float=-0.016, _receiver_coupling_prob: float=1., _receiver_com_errors: List[QuantumError]=None,
                 _sender_memory: QuantumMemory=None, _receiver_memory: QuantumMemory=None) -> None:
        
        """
        Initializes a Bell State Measurement Connection
        
        Args:
            _sender (Host): sender of connection
            _receiver (Host): receiver of connection
            _sim (Simulation): simulation object
            _model_name (str): model to use for connection
            _sender_source (str): model to use for atom photon source at sender
            _receiver_source (str): model to use for atom photon source at receiver
            _sender_detector (str): model to use for sender detector
            _receiver_detector (str): model to use for receiver detector
            _sender_length (float): length of connection from source to sender
            _sender_attenuation_coefficient (float): attenuation coefficient of fiber from source to sender
            _sender_coupling_prob (float): probability of coupling photon into fiber from source to sender
            _sender_com_errors (list): list of errors on connection from source to sender
            _receiver_length (float): length of connection from source to receiver
            _receiver_attenuation_coefficient (float): attenuation coefficient of fiber from source to receiver
            _receiver_coupling_prob (float): probability of coupling photon into fiber from source to receiver
            _receiver_com_errors (list): list of errors on connection from source to receiver
            _sender_memory (QuantumMemory): quantum memory of sender
            _receiver_memory (QuantumMemory): quantum memory of receiver
        """
        
        model = BSM_MODELS[_model_name]
        
        self._sender: Host = _sender
        self._receiver: Host = _receiver
        self._duration: float = model[0]
        self._visibility: float = model[1]
        self._coin_ph_ph: float = model[2]
        self._coin_ph_dc: float = model[3]
        self._coin_dc_dc: float = model[4]
        
        self._sender_source: AtomPhotonSource = AtomPhotonSource(_sender_source)
        self._receiver_source: AtomPhotonSource = AtomPhotonSource(_receiver_source)
        self._sender_detector: PhotonDetector = PhotonDetector(_sender_detector)
        self._receiver_detector: PhotonDetector = PhotonDetector(_receiver_detector)
        self._sender_channel: QChannel = QChannel(_sender_length, _sender_attenuation_coefficient, _sender_coupling_prob, _sender_com_errors)
        self._receiver_channel: QChannel = QChannel(_receiver_length, _receiver_attenuation_coefficient, _receiver_coupling_prob, _receiver_com_errors)
        self._sender_memory: QuantumMemory = _sender_memory
        self._receiver_memory: QuantumMemory = _receiver_memory
        
        self._sim: Simulation = _sim
        
        sender_arrival_prob = self._sender_channel._coupling_prob * self._sender_channel._lose_prob * self._sender_detector._efficiency
        receiver_arrival_prob = self._receiver_channel._coupling_prob * self._receiver_channel._lose_prob * self._receiver_detector._efficiency
        
        self._success_prob: float = 0.5 * sender_arrival_prob * receiver_arrival_prob * self._coin_ph_ph * self._visibility * (1 - self._sender_detector._dark_count_prob) ** 2 * (1 - self._receiver_detector._dark_count_prob) ** 2
        self._false_prob_1: float = 0.5 * sender_arrival_prob * receiver_arrival_prob * self._coin_ph_ph * (1 - self._visibility) * (1 - self._sender_detector._dark_count_prob) ** 2 * (1 - self._receiver_detector._dark_count_prob) ** 2
        self._false_prob_3: float = (sender_arrival_prob * (1 - receiver_arrival_prob) + (1 - sender_arrival_prob) * receiver_arrival_prob) * self._coin_ph_dc * (self._sender_detector._dark_count_prob * (1 - self._sender_detector._dark_count_prob) * (1 - self._receiver_detector._dark_count_prob)**2 + self._receiver_detector._dark_count_prob * (1 - self._receiver_detector._dark_count_prob) * (1 - self._sender_detector._dark_count_prob)**2)
        self._false_prob_4: float = (1 - sender_arrival_prob) * (1 - receiver_arrival_prob) * self._coin_dc_dc * (self._sender_detector._dark_count_prob**2 * (1 - self._receiver_detector._dark_count_prob)**2 + 2 * self._sender_detector._dark_count_prob * self._receiver_detector._dark_count_prob * (1 - self._sender_detector._dark_count_prob) * (1 - self._receiver_detector._dark_count_prob) + self._receiver_detector._dark_count_prob**2 * (1 - self._sender_detector._dark_count_prob)**2)
        
        self._total_prob: float = self._success_prob  + self._false_prob_1 + self._false_prob_3 + self._false_prob_4
        
        self._inital_state: np.array = np.kron((4 * self._sender_source._fidelity - 1)/3 * B_0 + 4 * (1 - self._sender_source._fidelity)/3 * I_0, (4 * self._receiver_source._fidelity - 1)/3 * B_0 + 4 * (1 - self._receiver_source._fidelity)/3 * I_0)
        
        self._sender_duration: float = self._sender_source._duration + self._sender_channel._sending_time + self._duration + self._sender_detector._duration
        self._receiver_duration: float = self._receiver_source._duration + self._receiver_channel._sending_time + self._duration + self._receiver_detector._duration
        
    def create_bell_pairs(self, _num_requested: int=1, _num_needed: int=1) -> None:
        
        """
        Creates the number of needed bell pairs based on the models in the connection
        
        Args:
            _num_requested (int): number of requested bell pairs
            _num_needed (int): number of needed bell pairs
            
        Returns:
            /
        """
        
        packet_s = Packet(self._receiver._node_id, self._sender._node_id, _num_requested, _num_needed)
        packet_r = Packet(self._sender._node_id, self._receiver._node_id, _num_requested, _num_needed)
        
        for i in range(_num_needed):

            _curr_time = self._sim._sim_time + max(self._sender_source._duration, self._receiver_source._duration)

            if np.random.uniform(0, 1) > self._success_prob:
                q_1 = Simulation.create_qsystem(1).qubits
                q_2 = Simulation.create_qsystem(1).qubits
                self._sender_memory.l0_store_qubit(q_1, -1, _curr_time)
                self._receiver_memory.l0_store_qubit(q_2, -1, _curr_time)
                continue
            
            qsys = Simulation.create_qsystem(4)
            qsys._state = self._inital_state
            
            q_1, q_2, q_3, q_4 = qsys.qubits
            
            self._sender_memory.l0_store_qubit(q_1, -1, _curr_time)
            self._receiver_memory.l0_store_qubit(q_4, -1, _curr_time)
            
            self._sender_channel.put(q_2)
            self._receiver_channel.put(q_3)
            
            _sender_time = _curr_time + self._sender_channel._sending_time
            _receiver_time = _curr_time + self._receiver_channel._sending_time
            
            q_2 = self._sender_channel.get()
            q_3 = self._receiver_channel.get()
            
            q_1 = self._sender_memory.l0_retrieve_qubit_prob(-1, _sender_time)
            q_4 = self._receiver_memory.l0_retrieve_qubit_prob(-1, _receiver_time)
            
            res = q_2.bsm(q_3)
            
            remove_qubits([q_2, q_3])
            
            if res == 1:
                q_1.X()
            if res == 2:
                q_1.Z()
            if res == 3:
                q_1.Y()
            
            _sender_time += self._duration + self._sender_detector._duration
            _receiver_time += self._duration + self._receiver_detector._duration
            
            # maybe wrong need to add depolarization prob
            # q_1._qsystem._state = (success_prob * total_depolar_prob * q_1._qsystem._state + false_prob_1 * total_depolar_prob * A_0 + ((success_prob + false_prob_1) * (1 - total_depolar_prob) + false_prob_3 + false_prob_4) * I_0) / total_prob
            q_1._qsystem._state = (self._success_prob * q_1._qsystem._state + self._false_prob_1 * A_0 + (self._false_prob_3 + self._false_prob_4) * I_0) / self._total_prob
            
            self._sender_memory.l0_store_qubit(q_1, -1, _sender_time)
            self._receiver_memory.l0_store_qubit(q_4, -1, _receiver_time)
                
            packet_s.update_l1_success(i)
            packet_r.update_l1_success(i)
            
        packet_r.set_l1_ack()
        self._sim.schedule_event(ReceiveEvent(self._sim._sim_time + self._sender_duration, self._sender._node_id))
        self._sim.schedule_event(ReceiveEvent(self._sim._sim_time + self._receiver_duration, self._receiver._node_id))
        self._sender._connections['packet'][self._receiver._node_id][RECEIVE].put(packet_s)
        self._receiver._connections['packet'][self._sender._node_id][RECEIVE].put(packet_s)
        
class FockConnection:
    
    def __init__(self) -> None:
        
        pass
    
    def create_bell_pairs(self, _num_requested: int=1, _num_needed: int=1):
        
        pass
