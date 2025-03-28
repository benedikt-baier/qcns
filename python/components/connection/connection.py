
import numpy as np
import scipy as sp
from typing import List

from qcns.python.components.hardware.photon_source import SinglePhotonSource, AtomPhotonSource, PhotonPhotonSource, FockPhotonSource
from qcns.python.components.hardware.photon_detector import PhotonDetector
from qcns.python.components.connection.channel import QChannel
from qcns.python.components.hardware.memory import QuantumMemory
from qcns.python.components.packet import Packet
from qcns.python.components.simulation.event import ReceiveEvent
from qcns.python.components.simulation import Simulation

__all__ = ['SingleQubitConnection', 'SenderReceiverConnection', 'TwoPhotonSourceConnection', 'BellStateMeasurementConnection', 'FockStateConnection', 'L3Connection']

class Host:
    pass

class QuantumError:
    pass

L0 = 0
L1 = 1
L2 = 2
L3 = 3

SEND = 0
RECEIVE = 1

B_0 = np.array([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]], dtype=np.complex128)
I_0 = np.array([[0.25, 0, 0, 0], [0, 0.25, 0, 0], [0, 0, 0.25, 0], [0, 0, 0, 0.25]], dtype=np.complex128)
A_0 = np.array([[0.5, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0.5]], dtype=np.complex128)
B_I_0 = np.array([[0.25, 0., 0., 0.5], [0., -0.25, 0., 0.], [0., 0., -0.25, 0.], [0.5, 0., 0., 0.25]], dtype=np.complex128)

# Sender Receiver: duration, interaction probability, state transfer fidelity
# Two photon source: duration, interaction probability sender, state transfer fidelity sender, interaction probability receiver, state transfer fidelity receiver
# Bell State measurement: duration, visibility, coin ph, ph, coin ph dc, coin dc dc
# Fock State: duration, visibility, coherent phase, spin photon correlation

SENDER_RECEIVER_MODELS = {'perfect': (0., 1., 1.), 'standard': (8.95e-7, 0.14, 0.85)}
TWO_PHOTON_MODELS = {'perfect': (0., 1., 1., 1., 1.), 'standard': (8.95e-7, 0.37, 0.85, 0.37, 0.85)}
BSM_MODELS = {'perfect': (0., 1., 1., 0., 0.), 'standard': (70e-9, 0.955, 1., 1., 1.), 'leent70ns': (70e-9, 0.955, 1., 1., 1.), 'krut3mus': (3e-6, 0.66, 0.313, 0.355, 0.313), 'krut5mus': (5e-6, 0.45, 0.487, 0.546, 0.49), 
              'krut7mus': (7e-6, 0.37, 0.635, 0.698, 0.64), 'krut10mus': (10e-6, 0.36, 0.811, 0.86, 0.816), 'krut15mus': (15e-6, 0.25, 0.98, 0.988, 0.98)}

FOCK_STATE_MODELS = {'perfect': (0., 1., 0., 0.), 'standard': (3.8e-6, 0.9, 0., 0.01)}

# sender receiver model:
#   standard: An elementary quantum network of single atoms in optical cavities

# two photon model:
#   standard: An elementary quantum network of single atoms in optical cavities

# bsm model:
#   standard: Entangling single atoms over 33 km telecom fibre
#   Krutyanskiy (krut3mus): Entanglement of Trapped-Ion Qubits Separated by 230 Meters

# fock state model:
#   standard: Realization of a multinode quantum network of remote solidstate qubits

class SingleQubitConnection:
    
    """
    Represents a Single Qubit Connection
    
    Attr:
        _sender (Host): sender of connection
        _receiver (Host): receiver of connection
        _num_sources (int): number of available sources
        _source (SinglePhotonSource): sender photon source of connection
        _channel (QChannel): quantum channel of connection
        _sim (Simulation): simulation object
        _success_prob (float): probability to successfully create a photon
        _state (np.array): precalculated state
        _creation_functions (dict): function to call whether sending or receiving was successful
    """
    
    def __init__(self, _sender: Host, _receiver: int, _sim: Simulation, _sender_source: str='perfect', _num_sources: int=-1,
                 _length: float=0., _attenuation_coefficient: float=-0.016, _in_coupling_prob: float=1., _out_coupling_prob: float=1., _lose_qubits: bool=False, _com_errors: List[QuantumError]=None) -> None:
        
        """
        Initializes a Single Qubit Connection
        
        Args:
            _sender (Host): sender of connection
            _receiver (Host): receiver of connection
            _sim (Simulation): simulation object
            _sender_source (str): name of model to use for single photon source
            _num_sources (int): number of available sources
            _length (float): length of connection
            _attenuation_coefficient (float): attenuation coefficient of fiber
            _in_coupling_prob (float): probability of coupling photon into fiber
            _out_coupling_prob (float): probability of coupling photon out of fiber
            _lose_qubits (float): whether to lose qubits or not
            _com_errors (list): list of communication errors
            
        Returns:
            /
        """
        
        self._sender: Host = _sender
        self._receiver_id: int = _receiver
        self._num_sources: int = _num_sources
        self._source: SinglePhotonSource = SinglePhotonSource(_sender_source)
        self._channel: QChannel = QChannel(_length, _attenuation_coefficient, _in_coupling_prob, _out_coupling_prob, _com_errors)
        
        self._sim: Simulation = _sim
        
        self._success_prob: float = self._source._emission_prob * self._channel._in_coupling_prob * self._channel._out_coupling_prob
        
        if _lose_qubits:
            self._success_prob *= self._channel._lose_prob
        
        if -np.log10(self._success_prob) >= 6:
            raise ValueError('Too low success probability')
        
        self._state: np.array = np.array([[self._source._fidelity, np.sqrt(self._source._fidelity * (1 - self._source._fidelity))], [np.sqrt(self._source._fidelity * (1 - self._source._fidelity)), 1 - self._source._fidelity]], dtype=np.complex128)
    
        self._creation_functions = {0: self.failure_creation, 1: self.success_creation}
    
    def success_creation(self) -> None:
        
        """
        Successful creation of a qubit
        
        Args:
            /
            
        Returns:
            /
        """
        
        _sample = sp.stats.truncnorm.rvs(-1, 1, loc=0, scale=self._source._fidelity_variance)
        qsys = Simulation.create_qsystem(1)
        qsys._state = self._state + np.array([[_sample, -2 * _sample * self._source._fidelity],
                                                [-2 * _sample * self._source._fidelity, -_sample]], dtype=np.complex128)
        q_1 = qsys.qubits
        self._channel._channel.put(q_1)
        
    def failure_creation(self) -> None:
        
        """
        Unsuccessful creation of a qubit
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._channel.put(None)
      
    def attempt_qubit(self, _needed: int=1) -> None:
        
        """
        Attempts to create the number of qubits
        
        Args:
            _needed (int): number of needed qubits
            
        Returns:
            /
        """
        
        _num_tries = 1
        if self._num_sources + 1 and self._num_sources < _needed:
            _num_tries = int(np.ceil(_needed / self._num_sources))
        
        _curr_time = self._sender._time + self._source._duration
        _curr_time_samples = np.zeros(_needed) + _curr_time
        if self._num_sources + 1:
            _curr_time_samples += np.repeat(np.arange(1, _num_tries + 1).reshape(-1, 1), self._num_sources, axis=1).flatten()[:_needed] * self._source._duration
        
        _success_samples = np.random.uniform(0, 1) < self._success_prob
        
        for success, _curr_time in zip(_success_samples, _curr_time_samples):
            self._creation_functions[success]()
            self._sim.schedule_event(ReceiveEvent(_curr_time + self._channel._sending_time, self._receiver_id))
            
    def create_qubit(self, _requested: int=1) -> None:
        
        """
        Creates the requested number of qubits, no matter how long it takes
        
        Args:
            _requested (int): number of requested qubits
            
        Returns:
            /
        """
        
        _curr_time = self._sender._time + self._source._duration
        
        _curr_time_samples = np.zeros(_requested + self._num_sources + 1) + _curr_time
        _num_successfull = 0
        _current_try = 0
        
        while _num_successfull < _requested and self._num_sources + 1:
            _successes = np.count_nonzero(np.random.uniform(0, 1, self._num_sources) < self._success_prob)
            _current_try += 1
            _curr_time_samples[_num_successfull:_num_successfull + _successes] += _current_try * self._source._duration
            _num_successfull += _successes
        
        _curr_time_samples = _curr_time_samples[:_requested]
        
        for _curr_time in _curr_time_samples:
            self.success_creation(_curr_time)
            self._sim.schedule_event(ReceiveEvent(_curr_time + self._channel._sending_time, self._receiver_id))
        
class SenderReceiverConnection:
    
    """
    Represents a sender receiver connection
    
    Attr:
        _sender (Host): sender of connection
        _receiver (Host): receiver of connection
        _num_sources (int): number of available sources
        _state_transfer_fidelity (float): fidelity of transfering the state from the photon to the atom
        _source_duration (float): duration of the photon source
        _source_variance (float): fidelity variance of the source
        _sender_memory (QuantumMemory): quantum memory of sender
        _receiver_memory (QuantumMemory): quantum memory of receiver
        _sim (Simulation): simulation object
        _success_prob (float): probability of successfully create a bell pair
        _total_prob (float): overall total probability
        _sending_duration (float): duration of creating a bell pair
        _state (np.array): precalculated state
        _creation_functions (dict): function to call whether sending or receiving was successful
    """
    
    def __init__(self, _sender: Host, _receiver: int, _sim: Simulation, 
                 _model_name: str='perfect', _sender_source: str='perfect', _receiver_detector: str='perfect', _num_sources: int=-1,
                 _length: float=0., _attenuation_coefficient: float=-0.016, _in_coupling_prob: float=1., _out_coupling_prob: float=1., _lose_qubits: bool=False, 
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
            _num_sources: (int): number of available sources
            _length (float): length of connection
            _attenuation_coefficient (float): attenuation coefficient of fiber
            _in_coupling_prob (float): probability of coupling photon into fiber
            _out_coupling_prob (float): probability of coupling photon out of fiber
            _lose_qubits (bool): whether to lose qubits
            _sender_memory (QuantumMemory): quantum memory of sender
            _receiver_memory (QuantumMemory): quantum memory of receiver
            
        Returns:
            /
        """
        
        if not _num_sources:
            raise ValueError('Number of Sources should not be 0')
        
        model = SENDER_RECEIVER_MODELS[_model_name]
        
        self._sender: Host = _sender
        self._receiver_id: int = _receiver
        self._num_sources: int = _num_sources
        
        _duration: float = model[0]
        _interaction_prob: float = model[1]
        self._state_transfer_fidelity: float = model[2]
        
        _source = AtomPhotonSource(_sender_source)
        _detector = PhotonDetector(_receiver_detector)
        _channel = QChannel(_length, _attenuation_coefficient, _in_coupling_prob, _out_coupling_prob)
        
        self._source_duration: float = _source._duration
        self._source_variance: float = _source._fidelity_variance
        self._sender_memory: QuantumMemory = _sender_memory
        self._receiver_memory: QuantumMemory = _receiver_memory
        
        self._sim: Simulation = _sim
        
        _total_depolar_prob: float = (4 * _source._fidelity - 1) / 3
        
        _arrival_prob = _source._emission_prob * _channel._in_coupling_prob * _channel._out_coupling_prob * _detector._efficiency * _interaction_prob
        
        if _lose_qubits:
            _arrival_prob *= _channel._lose_prob
        
        self._success_prob: float = _arrival_prob * (1 - _detector._dark_count_prob)
        
        _false_prob = (1 - _arrival_prob) * _detector._dark_count_prob
        
        if -np.log10(self._success_prob) >= 6:
            raise ValueError('Too low success probability')
        
        self._total_prob: float = self._success_prob + _false_prob
        
        self._sending_duration: float = self._source_duration + _channel._sending_time + _duration + _detector._duration

        self._state: np.array = (self._success_prob * _total_depolar_prob * np.sqrt(self._state_transfer_fidelity) * B_0 + 
                                 (self._success_prob * (1 - _total_depolar_prob * np.sqrt(self._state_transfer_fidelity)) + _false_prob) * I_0) / self._total_prob

        self._creation_functions = {0: self.failure_creation, 1: self.success_creation}
 
    def success_creation(self, _curr_time: float) -> None:
        
        """
        Successful creation of a entangled pair
        
        Args:
            _curr_time (float): current time of creating qubit
            
        Returns:
            /
        """
        
        qsys = Simulation.create_qsystem(2)
        qsys._state = self._state + (self._success_prob * ((4 * sp.stats.truncnorm.rvs(-1, 1, loc=0, scale=self._source_variance)) / 3) * np.sqrt(self._state_transfer_fidelity) * B_I_0) / self._total_prob
        q_1, q_2 = qsys.qubits

        self._sender_memory.store_qubit(L0, q_1, -1, _curr_time)
        self._receiver_memory.store_qubit(L0, q_2, -1, _curr_time)
        
    def failure_creation(self, _curr_time: float) -> None:
        
        """
        Unsuccessful creation of a qubit
        
        Args:
            _curr_time (float): current time of creating qubit
            
        Returns:
            /
        """
        
        q_1 = Simulation.create_qsystem(1).qubits
        q_2 = Simulation.create_qsystem(1).qubits
        self._sender_memory.store_qubit(L0, q_1, -1, _curr_time)
        self._receiver_memory.store_qubit(L0, q_2, -1, _curr_time)
    
    def attempt_bell_pairs(self, _requested: int=1, _needed: int=1) -> None:
        
        """
        Attempts to create the needed number of qubits
        
        Args:
            _requested (int): number of requested bell pairs
            _needed (int): number of needed bell pairs
            
        Returns:
            /
        """
        
        packet = Packet(self._sender.id, self._receiver_id, _requested, _needed)
        
        _num_tries = 1
        if self._num_sources + 1 and self._num_sources < _needed:
            _num_tries = int(np.ceil(_needed / self._num_sources))
        
        _curr_time = self._sender._time + self._source_duration
        _curr_time_samples = np.zeros(_needed) + _curr_time
        if self._num_sources + 1:
            _curr_time_samples += np.repeat(np.arange(1, _num_tries + 1).reshape(-1, 1), self._num_sources, axis=1).flatten()[:_needed] * self._source_duration
        
        _success_samples = np.random.uniform(0, 1, _needed) < self._success_prob
        
        for success, _curr_time in zip(_success_samples, _curr_time_samples):
            self._creation_functions[success](_curr_time)
        
        packet.l1_success = _success_samples
        packet.l1_protocol = 1
        
        self._sim.schedule_event(ReceiveEvent(self._sender._time + self._sending_duration + (_num_tries - 1) * self._source_duration, self._receiver_id))
        self._sender._connections['packet'][self._receiver_id][SEND].put(packet)
    
    def create_bell_pairs(self, _requested: int=1) -> None:
        
        """
        Creates the number of requested qubits, no matter how long it takes
        
        Args:
            _requested (int): number of requested qubits
            
        Returns:
            /
        """
        
        packet = Packet(self._sender._node_id, self._receiver_id, _requested, _requested)
        
        _curr_time = self._sender._time + self._source_duration
        
        _curr_time_samples = np.zeros(_requested + self._num_sources + 1) + _curr_time
        _num_successfull = 0
        _current_try = 0
        
        while _num_successfull < _requested and self._num_sources + 1:
            _successes = np.count_nonzero(np.random.uniform(0, 1, self._num_sources) < self._success_prob)
            _current_try += 1
            _curr_time_samples[_num_successfull:_num_successfull + _successes] += _current_try * self._source_duration
            _num_successfull += _successes
        
        _curr_time_samples = _curr_time_samples[:_requested]
        
        for _curr_time in _curr_time_samples:
            self.success_creation(_curr_time)
        
        packet.l1_success = np.ones(_requested, dtype=np.bool_)
        packet.l1_protocol = 1
        
        self._sim.schedule_event(ReceiveEvent(self._sender._time + self._sending_duration + (_current_try - 1) * self._source_duration, self._receiver_id))
        self._sender._connections['packet'][self._receiver_id][SEND].put(packet)
        
class TwoPhotonSourceConnection:
    
    """
    Represents a Two Photon Source Connection
    
    Attr:
        _sender (Host): sender of connection
        _receiver (Host): receiver of connection
        _num_sources (int): number of available sources
        _sender_state_transfer_fidelity (float): fidelity of transfering the state from the photon to the atom at the sender
        _receiver_state_transfer_fidelity (float): fidelity of transfering the state from the photon to the atom at the receiver
        _source_duration (float): duration of the source
        _source_fidelity (float): fidelity of the source
        _source_variance (float): fidelity variance of the source
        _sender_memory (QuantumMemory): quantum memory of sender
        _receiver_memory (QuantumMemory): quantum memory of receiver
        _sim (Simulation): simulation object
        _sender_success_prob (float): sender sided success probability
        _receiver_success_prob (float): receiver sided success probability
        _success_prob (float): probability of successfully create a bell pair
        _total_prob (float): overall total probability
        _sender_duration (float): sender sided overall duration
        _receiver_duration (float): receiver sided overall duration
        _total_duration (float): duration of the total transmission
        _state (np.array): precalculated state
        _creation_functions (dict): function to call whether sending or receiving was successful
    """
    
    def __init__(self, _sender: Host, _receiver: int, _sim: Simulation, 
                 _model_name: str='perfect', _source: str='perfect', _sender_detector: str='perfect', _receiver_detector: str='perfect', _num_sources: int=-1,
                 _sender_length: float=0., _sender_attenuation: float=-0.016, _sender_in_coupling_prob: float=1., _sender_out_coupling_prob: float=1., _sender_lose_qubits: bool=False,
                 _receiver_length: float=0., _receiver_attenuation: float=-0.016, _receiver_in_coupling_prob: float=1., _receiver_out_coupling_prob: float=1., _receiver_lose_qubits: bool=False,
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
            _num_sources (int): number of available sources
            _sender_length (float): length of connection from source to sender
            _sender_attenuation (float): attenuation coefficient of fiber from source to sender
            _sender_in_coupling_prob (float): probability of coupling photon into fiber from source to sender
            _sender_out_coupling_prob (float): probability of coupling photon out of fiber from source to sender
            _sender_lose_qubits (bool): whether to lose qubits of sender channel
            _receiver_length (float): length of connection from source to receiver
            _receiver_attenuation (float): attenuation coefficient of fiber from source to receiver
            _receiver_in_coupling_prob (float): probability of coupling photon into fiber from source to receiver
            _receiver_out_coupling_prob (float): probability of coupling photon out of fiber from source to receiver
            _receiver_lose_qubits (bool): whether to lose qubits of receiver channel
            _sender_memory (QuantumMemory): quantum memory of sender
            _receiver_memory (QuantumMemory): quantum memory of receiver
            
        Returns:
            /
        """
        
        if not _num_sources:
            raise ValueError('Number of Sources should not be 0')
        
        model = TWO_PHOTON_MODELS[_model_name]
        
        self._sender: Host = _sender # fix
        self._receiver_id: int = _receiver # fix
        self._num_sources: int = _num_sources # fix
        
        _duration: float = model[0]
        _sender_interaction_prob: float = model[1]
        self._sender_state_transfer_fidelity: float = model[2] # fix
        _receiver_interaction_prob: float = model[3]
        self._receiver_state_transfer_fidelity: float = model[4] # fix
        
        _source = PhotonPhotonSource(_source)
        _sender_detector = PhotonDetector(_sender_detector)
        _receiver_detector = PhotonDetector(_receiver_detector)
        _sender_channel = QChannel(_sender_length, _sender_attenuation, _sender_in_coupling_prob, _sender_out_coupling_prob)
        _receiver_channel = QChannel(_receiver_length, _receiver_attenuation, _receiver_in_coupling_prob, _receiver_out_coupling_prob)
        
        self._source_duration: float = _source._duration
        self._source_fidelity: float = _source._fidelity
        self._source_variance: float = _source._fidelity_variance
        
        self._sender_memory: QuantumMemory = _sender_memory
        self._receiver_memory: QuantumMemory = _receiver_memory
        
        self._sim: Simulation = _sim
        
        _sender_arrival_prob = np.sqrt(_source._visibility * _source._emission_prob) * _sender_channel._in_coupling_prob * _sender_channel._out_coupling_prob * _sender_detector._efficiency * _sender_interaction_prob
        _receiver_arrival_prob = np.sqrt(_source._visibility * _source._emission_prob) * _receiver_channel._in_coupling_prob * _receiver_channel._out_coupling_prob * _receiver_detector._efficiency * _receiver_interaction_prob
        
        if _sender_lose_qubits:
            _sender_arrival_prob *= _sender_channel._lose_prob 

        if _receiver_lose_qubits:
            _receiver_arrival_prob *= _receiver_channel._lose_prob
        
        self._sender_success_prob: float = _sender_arrival_prob * (1 - _sender_detector._dark_count_prob) # fix
        self._receiver_success_prob: float = _receiver_arrival_prob * (1 - _receiver_detector._dark_count_prob) # fix
        _sender_false_prob = (1 - _sender_arrival_prob) * _sender_detector._dark_count_prob
        _receiver_false_prob = (1 - _receiver_arrival_prob) * _receiver_detector._dark_count_prob
        
        _total_depolar_prob: float = (4 * self._source_fidelity - 1)**2 / 9
        
        self._success_prob: float = self._sender_success_prob * self._receiver_success_prob
        
        if -np.log10(self._success_prob) >= 6:
            raise ValueError('Too low success probability')
        
        self._total_prob: float = self._success_prob + _sender_false_prob + _receiver_false_prob # fix
        
        self._sender_duration: float = self._source_duration + _sender_channel._sending_time + _duration + _sender_detector._duration # fix
        self._receiver_duration: float = self._source_duration + _receiver_channel._sending_time + _duration + _receiver_detector._duration # fix
        self._total_duration: float = 2 * (_sender_channel._sending_time + _receiver_channel._sending_time)

        self._state: np.array = (self._success_prob * _total_depolar_prob * np.sqrt(self._sender_state_transfer_fidelity * self._receiver_state_transfer_fidelity) * B_0 + 
                                 (self._success_prob * (1 - _total_depolar_prob * np.sqrt(self._sender_state_transfer_fidelity * self._receiver_state_transfer_fidelity)) + _sender_false_prob + _receiver_false_prob) * I_0) / self._total_prob # fix

        self._creation_functions = {0: self.failure_creation, 1: self.success_creation} # fix

    def success_creation(self, _curr_time: float) -> None:
        
        """
        Successful creation of a entangled pair
        
        Args:
            _curr_time (float): current time of creating qubit
            
        Returns:
            /
        """
        
        qsys = Simulation.create_qsystem(2)
        depol = 8 * (4 * self._source_fidelity - 1) * sp.stats.truncnorm.rvs(-1, 1, loc=0, scale=self._source_variance) / 9
        qsys._state = self._state + (self._success_prob * depol * np.sqrt(self._sender_state_transfer_fidelity * self._receiver_state_transfer_fidelity) * B_I_0) / self._total_prob
    
        q_1, q_2 = qsys.qubits
        
        self._sender_memory.store_qubit(L0, q_1, -1, _curr_time)
        self._receiver_memory.store_qubit(L0, q_2, -1, _curr_time)
    
    def failure_creation(self, _curr_time: float) -> None:
        
        """
        Unsuccessful creation of a qubit
        
        Args:
            _curr_time (float): current time of creating qubit
            
        Returns:
            /
        """
        
        q_1 = Simulation.create_qsystem(1).qubits
        q_2 = Simulation.create_qsystem(1).qubits
        self._sender_memory.store_qubit(L0, q_1, -1, _curr_time)
        self._receiver_memory.store_qubit(L0, q_2, -1, _curr_time)

    def attempt_bell_pairs(self, _requested: int=1, _needed: int=1) -> None:
        
        """
        Attempts to create the needed number of qubits
        
        Args:
            _requested (int): number of requested bell pairs
            _needed (int): number of needed bell pairs
            
        Returns:
            /
        """
        
        packet_s = Packet(self._receiver_id, self._sender.id, _requested, _needed)
        packet_r = Packet(self._sender.id, self._receiver_id, _requested, _needed)
        
        _num_tries = 1
        if self._num_sources + 1 and self._num_sources < _needed:
            _num_tries = int(np.ceil(_needed / self._num_sources))
        
        _curr_time = self._sender._time + self._source_duration    
        _curr_time_samples = np.zeros(_needed) + _curr_time
        if self._num_sources + 1:
            _curr_time_samples += np.repeat(np.arange(1, _num_tries + 1).reshape(-1, 1), self._num_sources, axis=1).flatten()[:_needed] * self._source_duration
        
        _sender_success_samples = np.random.uniform(0, 1, _needed) < self._sender_success_prob
        _receiver_success_samples = np.random.uniform(0, 1, _needed) < self._receiver_success_prob
        
        packet_s.l1_success = _sender_success_samples
        packet_r.l1_success = _receiver_success_samples
        
        _success_samples = np.logical_and(_sender_success_samples, _receiver_success_samples)
        
        for _success, _curr_time in zip(_success_samples, _curr_time_samples):
            self._creation_functions[_success](_curr_time)
        
        packet_s.l1_set_ps()
        packet_r.l1_set_ps()
        packet_r.l1_set_ack()
        packet_s.l1_protocol = 2
        packet_r.l1_protocol = 2
        
        self._sim.schedule_event(ReceiveEvent(self._sender._time + self._total_duration + self._sender_duration + (_num_tries - 1) * self._source_duration, self._sender.id))
        self._sim.schedule_event(ReceiveEvent(self._sender._time + self._total_duration + self._receiver_duration + (_num_tries - 1) * self._source_duration, self._receiver_id))
        self._sender._connections['packet'][self._receiver_id][SEND].put(packet_r)
        self._sender._connections['packet'][self._receiver_id][RECEIVE].put(packet_s)

    def create_bell_pairs(self, _requested: int=1):
        
        """
        Creates the number of requested qubits, no matter how long it takes
        
        Args:
            _requested (int): number of requested qubits
            
        Returns:
            /
        """
        
        packet_s = Packet(self._receiver_id, self._sender.id, _requested, _requested)
        packet_r = Packet(self._sender.id, self._receiver_id, _requested, _requested)
        
        _curr_time = self._sender._time + self._source_duration
        
        _curr_time_samples = np.zeros(_requested + self._num_sources + 1) + _curr_time
        _num_successfull = 0
        _current_try = 0
        
        while _num_successfull < _requested and self._num_sources + 1:
            _sender_successes = np.random.uniform(0, 1, self._num_sources) < self._sender_success_prob
            _receiver_successes = np.random.uniform(0, 1, self._num_sources) < self._receiver_success_prob
            _successes = np.count_nonzero(np.logical_and(_sender_successes, _receiver_successes))
            _current_try += 1
            _curr_time_samples[_num_successfull:_num_successfull + _successes] += _current_try * self._source_duration
            _num_successfull += _successes
        
        _curr_time_samples = _curr_time_samples[:_requested]
        
        for _curr_time in _curr_time_samples:
            self.success_creation(_curr_time)
        
        packet_s.l1_success = np.ones(_requested, dtype=np.bool_)
        packet_r.l1_success = np.ones(_requested, dtype=np.bool_)
        
        packet_s.l1_set_ps()
        packet_r.l1_set_ps()
        packet_r.l1_set_ack()
        packet_s.l1_protocol = 2
        packet_r.l1_protocol = 2
        
        self._sim.schedule_event(ReceiveEvent(self._sender._time + self._total_duration + self._sender_duration + (_current_try - 1) * self._source_duration, self._sender.id))
        self._sim.schedule_event(ReceiveEvent(self._sender._time + self._total_duration + self._receiver_duration + (_current_try - 1) * self._source_duration, self._receiver_id))
        self._sender._connections['packet'][self._receiver_id][SEND].put(packet_r)
        self._sender._connections['packet'][self._receiver_id][RECEIVE].put(packet_s)

class BellStateMeasurementConnection:
    
    """
    Represents a Bell State Measurement Connection
    
    Attr:
        _sender (Host): sender of connection
        _receiver (Host): receiver of connection
        _num_sources (int): number of available sources
        _sender_source_duration (float): duration of the sender source
        _receiver_source_duration (float): duration of the receiver source
        _sender_source_variance (float): fidelity variance of the sender source
        _receiver_source_variance (float): fidelity variance of the receiver source
        _sender_memory (QuantumMemory): quantum memory of sender
        _receiver_memory (QuantumMemory): quantum memory of receiver
        _sim (Simulation): simulation object
        _sender_depolar (float): depolarization probability of sender photon
        _receiver_depolar (float): depolarization probability of receiver photon
        _success_prob (float): overall success probability of creating a bell pair
        _false_prob_1 (float): first probability of creating a non entangled pair
        _total_prob (float): overall total probability
        _sender_duration (float): sender sided overall duration
        _receiver_duration (float): receiver sided overall duration
        _total_duration (float): total duration
        _state (np.array): precalculated state
        _creation_functions (dict): function to call whether sending or receiving was successful
    """
    
    def __init__(self, _sender: Host, _receiver: int, _sim: Simulation, 
                 _model_name: str='perfect', _sender_source: str='perfect', _receiver_source: str='perfect', _sender_detector: str='perfect', _receiver_detector: str='perfect', _num_sources: int=-1,
                 _sender_length: float=0., _sender_attenuation_coefficient: float=-0.016, _sender_in_coupling_prob: float=1., _sender_out_coupling_prob: float=1., _sender_lose_qubits: bool=False,
                 _receiver_length: float=0., _receiver_attenuation_coefficient: float=-0.016, _receiver_in_coupling_prob: float=1., _receiver_out_coupling_prob: float=1., _receiver_lose_qubits: bool=False,
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
            _num_sources (int): number of available sources
            _sender_length (float): length of connection from source to sender
            _sender_attenuation_coefficient (float): attenuation coefficient of fiber from source to sender
            _sender_in_coupling_prob (float): probability of coupling photon into fiber from source to sender
            _sender_out_coupling_prob (float): probability of coupling photon out of fiber from source to sender
            _sender_lose_qubits (bool): whether to lose qubits of sender channel
            _receiver_length (float): length of connection from source to receiver
            _receiver_attenuation_coefficient (float): attenuation coefficient of fiber from source to receiver
            _receiver_in_coupling_prob (float): probability of coupling photon into fiber from source to receiver
            _receiver_out_coupling_prob (float): probability of coupling photon out of fiber from source to receiver
            _receiver_lose_qubits (bool): whether to lose qubits of receiver channel
            _sender_memory (QuantumMemory): quantum memory of sender
            _receiver_memory (QuantumMemory): quantum memory of receiver
            
        Returns:
            /
        """
        
        if not _num_sources:
            raise ValueError('Number of Sources should not be 0')
        
        model = BSM_MODELS[_model_name]
        
        self._sender: Host = _sender
        self._receiver_id: int = _receiver
        self._num_sources: int = _num_sources
        
        _duration: float = model[0]
        _visibility: float = model[1]
        _coin_ph_ph: float = model[2]
        _coin_ph_dc: float = model[3]
        _coin_dc_dc: float = model[4]
        
        _sender_source = AtomPhotonSource(_sender_source)
        _receiver_source = AtomPhotonSource(_receiver_source)
        _sender_detector = PhotonDetector(_sender_detector)
        _receiver_detector = PhotonDetector(_receiver_detector)
        _sender_channel = QChannel(_sender_length, _sender_attenuation_coefficient, _sender_in_coupling_prob, _sender_out_coupling_prob)
        _receiver_channel = QChannel(_receiver_length, _receiver_attenuation_coefficient, _receiver_in_coupling_prob, _receiver_out_coupling_prob)
        
        self._sender_source_duration: float = _sender_source._duration
        self._receiver_source_duration: float = _receiver_source._duration
        self._sender_source_variance: float = _sender_source._fidelity_variance
        self._receiver_source_variance: float = _receiver_source._fidelity_variance
        self._sender_memory: QuantumMemory = _sender_memory
        self._receiver_memory: QuantumMemory = _receiver_memory
        
        self._sim: Simulation = _sim
        
        _sender_arrival_prob = _sender_source._emission_prob * _sender_channel._in_coupling_prob * _sender_channel._out_coupling_prob * _sender_detector._efficiency
        _receiver_arrival_prob = _receiver_source._emission_prob * _receiver_channel._in_coupling_prob * _receiver_channel._out_coupling_prob * _receiver_detector._efficiency
        
        if _sender_lose_qubits:
            _sender_arrival_prob *= _sender_channel._lose_prob
            
        if _receiver_lose_qubits:
            _receiver_arrival_prob *= _receiver_channel._lose_prob
        
        self._sender_depolar: float = (4 * _sender_source._fidelity - 1) / 3
        self._receiver_depolar: float = (4 * _receiver_source._fidelity - 1) / 3
        _total_depolar_prob: float = self._sender_depolar * self._receiver_depolar
        
        self._success_prob: float = 0.5 * _sender_arrival_prob * _receiver_arrival_prob * _coin_ph_ph * _visibility * (1 - _sender_detector._dark_count_prob) ** 2 * (1 - _receiver_detector._dark_count_prob) ** 2
        
        if -np.log10(self._success_prob) >= 6:
            raise ValueError('Too low success probability')
        
        self._false_prob_1: float = 0.5 * _sender_arrival_prob * _receiver_arrival_prob * _coin_ph_ph * (1 - _visibility) * (1 - _sender_detector._dark_count_prob) ** 2 * (1 - _receiver_detector._dark_count_prob) ** 2
        _false_prob_3: float = (_sender_arrival_prob * (1 - _receiver_arrival_prob) + (1 - _sender_arrival_prob) * _receiver_arrival_prob) * _coin_ph_dc * (_sender_detector._dark_count_prob * (1 - _sender_detector._dark_count_prob) * (1 - _receiver_detector._dark_count_prob)**2 + _receiver_detector._dark_count_prob * (1 - _receiver_detector._dark_count_prob) * (1 - _sender_detector._dark_count_prob)**2)
        _false_prob_4: float = (1 - _sender_arrival_prob) * (1 - _receiver_arrival_prob) * _coin_dc_dc * (_sender_detector._dark_count_prob**2 * (1 - _receiver_detector._dark_count_prob)**2 + 2 * _sender_detector._dark_count_prob * _receiver_detector._dark_count_prob * (1 - _sender_detector._dark_count_prob) * (1 - _receiver_detector._dark_count_prob) + _receiver_detector._dark_count_prob**2 * (1 - _sender_detector._dark_count_prob)**2)
        
        self._total_prob: float = self._success_prob + self._false_prob_1 + _false_prob_3 + _false_prob_4
        
        self._sender_duration: float = self._sender_source_duration + _sender_channel._sending_time + _duration + _sender_detector._duration
        self._receiver_duration: float = self._receiver_source_duration + _receiver_channel._sending_time + _duration + _receiver_detector._duration
        self._total_duration: float = 2 * (_sender_channel._sending_time + _receiver_channel._sending_time)
        
        self._state: np.array = (self._success_prob * _total_depolar_prob * B_0 + 
                            self._false_prob_1 * _total_depolar_prob * A_0 + 
                            ((self._success_prob + self._false_prob_1) * (1 - _total_depolar_prob) + _false_prob_3 + _false_prob_4) * I_0) / self._total_prob
        
        self._creation_functions = {0: self.failure_creation, 1: self.success_creation}
    
    def success_creation(self, _sender_time: float, _receiver_time: float) -> None:
        
        """
        Successful creation of a entangled pair
        
        Args:
            _sender_time (float): current time of creating sender sided qubit
            _receiver_time (float): current time of creating receiver sided qubit
            
        Returns:
            /
        """
        
        _sender_sample = 4 * sp.stats.truncnorm.rvs(-1, 1, loc=0, scale=self._sender_source_variance) / 3
        _receiver_sample = 4 * sp.stats.truncnorm.rvs(-1, 1, loc=0, scale=self._receiver_source_variance) / 3
        _sample = self._sender_depolar * _receiver_sample + self._receiver_depolar * _sender_sample + _sender_sample * _receiver_sample
        
        qsys = Simulation.create_qsystem(2)
        qsys._state = self._state + _sample * (self._success_prob * B_0 + self._false_prob_1 * A_0 - (self._success_prob + self._false_prob_1) * I_0) / self._total_prob
        
        q_1, q_2 = qsys.qubits
        
        self._sender_memory.store_qubit(L0, q_1, -1, _sender_time)
        self._receiver_memory.store_qubit(L0, q_2, -1, _receiver_time)
    
    def failure_creation(self, _sender_time: float, _receiver_time: float) -> None:
        
        """
        Unsuccessful creation of a entangled pair
        
        Args:
            _sender_time (float): current time of creating sender sided qubit
            _receiver_time (float): current time of creating receiver sided qubit
            
        Returns:
            /
        """
    
        q_1 = Simulation.create_qsystem(1).qubits
        q_2 = Simulation.create_qsystem(1).qubits
        self._sender_memory.store_qubit(L0, q_1, -1, _sender_time)
        self._receiver_memory.store_qubit(L0, q_2, -1, _receiver_time)
    
    def attempt_bell_pairs(self, _requested: int=1, _needed: int=1) -> None:
        
        """
        Attempts to create the number of needed qubits
        
        Args:
            _requested (int): number of requested bell pairs
            _needed (int): number of needed bell pairs
            
        Returns:
            /
        """
        
        packet_s = Packet(self._receiver_id, self._sender.id, _requested, _needed)
        packet_r = Packet(self._sender.id, self._receiver_id, _requested, _needed)

        _num_tries = 1
        if self._num_sources + 1 and self._num_sources < _needed:
            _num_tries = int(np.ceil(_needed / self._num_sources))

        _sender_time = self._sender._time + self._sender_source_duration
        _receiver_time = self._sender._time + self._receiver_source_duration
        
        _sender_time_samples = np.zeros(_needed) + _sender_time
        _receiver_time_samples = np.zeros(_needed) + _receiver_time
        
        if self._num_sources + 1:
            _sender_time_samples += np.repeat(np.arange(1, _num_tries + 1).reshape(-1, 1), self._num_sources, axis=1).flatten()[:_needed] * self._sender_source_duration
            _receiver_time_samples += np.repeat(np.arange(1, _num_tries + 1).reshape(-1, 1), self._num_sources, axis=1).flatten()[:_needed] * self._receiver_source_duration
        
        _success_samples = np.random.uniform(0, 1, _needed) < self._success_prob
        
        for success, _sender_time, _receiver_time in zip(_success_samples, _sender_time_samples, _receiver_time_samples):
            self._creation_functions[success](_sender_time, _receiver_time)
        
        packet_s.l1_success = _success_samples
        packet_r.l1_success = _success_samples
        
        packet_r.l1_set_ack()
        packet_s.l1_protocol = 3
        packet_r.l1_protocol = 3
        
        self._sim.schedule_event(ReceiveEvent(self._sender._time + self._total_duration + self._sender_duration + (_num_tries - 1) * self._sender_source_duration, self._sender.id))
        self._sim.schedule_event(ReceiveEvent(self._sender._time + self._total_duration + self._receiver_duration + (_num_tries - 1) * self._receiver_source_duration, self._receiver_id))
        self._sender._connections['packet'][self._receiver_id][SEND].put(packet_r)
        self._sender._connections['packet'][self._receiver_id][RECEIVE].put(packet_s)        
    
    def create_bell_pairs(self, _requested: int=1) -> None:
        
        """
        Creates the number of requested qubits, no matter how long it takes
        
        Args:
            _requested (int): number of requested qubits
            
        Returns:
            /
        """
        
        packet_s = Packet(self._receiver_id, self._sender.id, _requested, _requested)
        packet_r = Packet(self._sender.id, self._receiver_id, _requested, _requested)
        
        _sender_time = self._sender._time + self._sender_source_duration
        _receiver_time = self._sender._time + self._receiver_source_duration
        
        _sender_time_samples = np.zeros(_requested + self._num_sources + 1) + _sender_time
        _receiver_time_samples = np.zeros(_requested + self._num_sources + 1) + _receiver_time
        
        _num_successfull = 0
        _current_try = 0
        
        while _num_successfull < _requested and self._num_sources + 1:
            _successes = np.count_nonzero(np.random.uniform(0, 1, self._num_sources) < self._success_prob)
            _current_try += 1
            _sender_time_samples[_num_successfull:_num_successfull + _successes] += _current_try * self._sender_source_duration
            _receiver_time_samples[_num_successfull:_num_successfull + _successes] += _current_try * self._receiver_source_duration
            _num_successfull += _successes
        
        _sender_time_samples = _sender_time_samples[:_requested]
        _receiver_time_samples = _receiver_time_samples[:_requested]
        
        for _sender_time, _receiver_time in zip(_sender_time_samples, _receiver_time_samples):
            self.success_creation(_sender_time, _receiver_time)
        
        packet_s.l1_success = np.ones(_requested, dtype=np.bool_)
        packet_r.l1_success = np.ones(_requested, dtype=np.bool_)
        
        packet_r.l1_set_ack()
        packet_s.l1_protocol = 3
        packet_r.l1_protocol = 3
    
        self._sim.schedule_event(ReceiveEvent(self._sender._time + self._total_duration + self._sender_duration + (_current_try - 1) * self._sender_source_duration, self._sender.id))
        self._sim.schedule_event(ReceiveEvent(self._sender._time + self._total_duration + self._receiver_duration + (_current_try - 1) * self._receiver_source_duration, self._receiver_id))
        self._sender._connections['packet'][self._receiver_id][SEND].put(packet_r)
        self._sender._connections['packet'][self._receiver_id][RECEIVE].put(packet_s)
    
class FockStateConnection:
    
    """
    Represents a Fock state connection
    
    Attrs:
        _sender (Host): sender of connection
        _receiver (Host): receiver of connection
        _num_sources (int): number of available sources
        _visibility (float): visibility/indistinguishability of arriving photons
        _coherent_phase (float): phase of the emitted atom photon state
        _spin_photon_correlation (float): correlation between spin of atom and polarization of photon
        _sender_source_duration (float): duration of the sender source
        _receiver_source_duration (float): duration of the receiver source
        _sender_source_alpha (float): alpha of the sender source
        _receiver_source_alpha (float): alpha of the receiver source
        _sender_source_variance (float): alpha variance of the sender source
        _recevier_source_variance (float): alpha variance of the receiver source
        _sender_memory (QuantumMemory): quantum memory of sender
        _receiver_memory (QuantumMemory): quantum memory of receiver
        _sim (Simulation): simulation object
        _detector_dark_count_prob (float): overall detector dark count prob
        _case_up_up (float): case for up up 
        _case_up_down (float): case for up down
        _case_down_up (float): case for down up
        _success_prob (float): success probability of establishing entanglement
        _sender_duration (float): sender sided overall duration
        _receiver_duration (float): receiver sided overall duration
        _total_duration (float): total duration
        _state (np.array): precalculated state
        _creation_functions (dict): function to call whether sending or receiving was successful
    """
    
    def __init__(self, _sender: Host, _receiver: int, _sim: Simulation,
                 _model_name: str='perfect', _sender_source: str='perfect', _receiver_source: str='perfect', _sender_detector: str='perfect', _receiver_detector: str='perfect', _num_sources: int=-1,
                 _sender_length: float=0., _sender_attenuation_coefficient: float=-0.016, _sender_in_coupling_prob: float=1., _sender_out_coupling_prob: float=1., _sender_lose_qubits: bool=False,
                 _receiver_length: float=0., _receiver_attenuation_coefficient: float=-0.016, _receiver_in_coupling_prob: float=1., _receiver_out_coupling_prob: float=1., _receiver_lose_qubits: bool=False,
                 _sender_memory: QuantumMemory=None, _receiver_memory: QuantumMemory=None) -> None:
        
        """
        Initializes a Fock state connection
        
        Args:
            _sender (Host): sender of connection
            _receiver (Host): receiver of connection
            _sim (Simulation): simulation object
            _model_name (str): model to use for connection
            _sender_source (str): model to use for atom photon source at sender
            _receiver_source (str): model to use for atom photon source at receiver
            _sender_detector (str): model to use for sender detector
            _receiver_detector (str): model to use for receiver detector
            _num_sources (int): number of available sources
            _sender_length (float): length of connection from source to sender
            _sender_attenuation_coefficient (float): attenuation coefficient of fiber from source to sender
            _sender_in_coupling_prob (float): probability of coupling photon into fiber from source to sender
            _sender_out_coupling_prob (float): probability of coupling photon out of fiber from source to sender
            _sender_lose_qubits (bool): whether to lose qubits of sender channel
            _receiver_length (float): length of connection from source to receiver
            _receiver_attenuation_coefficient (float): attenuation coefficient of fiber from source to receiver
            _receiver_in_coupling_prob (float): probability of coupling photon into fiber from source to receiver
            _receiver_out_coupling_prob (float): probability of coupling photon out of fiber from source to receiver
            _receiver_lose_qubits (bool): whether to lose qubits of receiver channel
            _sender_memory (QuantumMemory): quantum memory of sender
            _receiver_memory (QuantumMemory): quantum memory of receiver
            
        Returns:
            /
        """
        
        if not _num_sources:
            raise ValueError('Number of Sources should not be 0')
        
        _model = FOCK_STATE_MODELS[_model_name]
        
        self._sender: Host = _sender
        self._receiver_id: int = _receiver
        self._num_sources: int = _num_sources
        
        _duration: float = _model[0]
        self._visibility: float = _model[1]
        self._coherent_phase: float = _model[2]
        self._spin_photon_correlation: float = _model[3]
        
        _sender_source = FockPhotonSource(_sender_source)
        _receiver_source = FockPhotonSource(_receiver_source)
        _sender_detector = PhotonDetector(_sender_detector)
        _receiver_detector = PhotonDetector(_receiver_detector)
        _sender_channel = QChannel(_sender_length, _sender_attenuation_coefficient, _sender_in_coupling_prob, _sender_out_coupling_prob)
        _receiver_channel = QChannel(_receiver_length, _receiver_attenuation_coefficient, _receiver_in_coupling_prob, _receiver_out_coupling_prob)
        
        self._sender_source_duration: float = _sender_source._duration
        self._receiver_source_duration: float = _receiver_source._duration
        self._sender_source_alpha: float = _sender_source._alpha
        self._receiver_source_alpha: float = _receiver_source._alpha
        self._sender_source_variance: float = _sender_source._alpha_variance
        self._recevier_source_variance: float = _receiver_source._alpha_variance
        
        self._sender_memory: QuantumMemory = _sender_memory
        self._receiver_memory: QuantumMemory = _receiver_memory
        
        self._sim: Simulation = _sim
    
        _sender_arrival_prob = _sender_source._emission_prob * _sender_channel._in_coupling_prob * _sender_channel._out_coupling_prob * _sender_detector._efficiency
        _receiver_arrival_prob = _receiver_source._emission_prob * _receiver_channel._in_coupling_prob * _receiver_channel._out_coupling_prob * _receiver_detector._efficiency
        
        if _sender_lose_qubits:
            _sender_arrival_prob *= _sender_channel._lose_prob
            
        if _receiver_lose_qubits:
            _receiver_arrival_prob *= _receiver_channel._lose_prob
        
        self._detector_dark_count_prob: float = _sender_detector._dark_count_prob + _receiver_detector._dark_count_prob - 2 * _sender_detector._dark_count_prob * _receiver_detector._dark_count_prob
        
        case_up_up_a = (1 - _sender_detector._dark_count_prob) * (1 - _receiver_detector._dark_count_prob) * (_sender_arrival_prob + _receiver_arrival_prob - 2 * _sender_arrival_prob * _receiver_arrival_prob)
        case_up_up_b = (1 - _sender_arrival_prob) * (1 - _receiver_arrival_prob) * self._detector_dark_count_prob
        
        self._case_up_up: float = case_up_up_a + case_up_up_b # fix
        
        up_up_prob = self._sender_source_alpha * self._receiver_source_alpha * self._case_up_up
        
        case_up_down_a = (1 - _sender_detector._dark_count_prob) * (1 - _receiver_detector._dark_count_prob) * _sender_arrival_prob
        case_up_down_b = (1 - _sender_arrival_prob) * self._detector_dark_count_prob
        
        self._case_up_down: float = case_up_down_a + case_up_down_b # fix
        
        up_down_prob = self._sender_source_alpha * (1 - self._receiver_source_alpha) * self._case_up_down
        
        case_down_up_a = (1 - _sender_detector._dark_count_prob) * (1 - _receiver_detector._dark_count_prob) * _receiver_arrival_prob
        case_down_up_b = (1 - _receiver_arrival_prob) * self._detector_dark_count_prob
        
        self._case_down_up: float = case_down_up_a + case_down_up_b # fix
        
        down_up_prob = self._receiver_source_alpha * (1 - self._sender_source_alpha) * self._case_down_up
        
        down_down_prob = (1 - self._sender_source_alpha) * (1 - self._receiver_source_alpha) * self._detector_dark_count_prob
    
        self._success_prob: float = up_up_prob + up_down_prob + down_up_prob + down_down_prob
    
        if -np.log10(self._success_prob) >= 6:
            raise ValueError('Too low success probability')
    
        _s = np.sqrt(self._visibility * up_down_prob * down_up_prob)
        
        self._sender_duration: float = self._sender_source_duration + _sender_channel._sending_time + _duration + _sender_detector._duration
        self._receiver_duration: float = self._receiver_source_duration + _receiver_channel._sending_time + _duration + _receiver_detector._duration

        self._total_duration: float = 2 * (_sender_channel._sending_time + _receiver_channel._sending_time)
    
        self._state: np.array = np.array([[(1 - self._spin_photon_correlation) * up_down_prob, 0., 0., (1 - self._spin_photon_correlation) * np.exp(1j * self._coherent_phase) * _s], 
                                          [0., 0.5 * self._spin_photon_correlation * (up_down_prob + down_up_prob) + up_up_prob, 0., 0.],
                                          [0., 0., 0.5 * self._spin_photon_correlation * (up_down_prob + down_up_prob) + down_down_prob, 0.],
                                          [(1 - self._spin_photon_correlation) * np.exp(-1j * self._coherent_phase) * _s, 0., 0., (1 - self._spin_photon_correlation) * down_up_prob]], dtype=np.complex128) / self._success_prob
    
        self._creation_functions = {0: self.failure_creation, 1: self.success_creation}
        
    def success_creation(self, _sender_time: float, _receiver_time: float) -> None:
        
        """
        Successful creation of a entangled pair
        
        Args:
            _sender_time (float): current time of creating sender sided qubit
            _receiver_time (float): current time of creating receiver sided qubit
            
        Returns:
            /
        """
        
        _sample_sender = sp.stats.truncnorm.rvs(-1, 1, loc=0, scale=self._sender_source_variance)
        _sample_receiver = sp.stats.truncnorm.rvs(-1, 1, loc=0, scale=self._recevier_source_variance)
        
        _sample_up_up = (self._sender_source_alpha * _sample_receiver + self._receiver_source_alpha * _sample_sender + _sample_sender * _sample_receiver) * self._case_up_up
        _sample_up_down = (-self._sender_source_alpha * _sample_receiver + (1 - self._receiver_source_alpha) * _sample_sender - _sample_sender * _sample_receiver) * self._case_up_down
        _sample_down_up = ((1 - self._sender_source_alpha) * _sample_receiver - self._receiver_source_alpha * _sample_sender - _sample_sender * _sample_receiver) * self._case_down_up
        _sample_down_down = (-(1 - self._sender_source_alpha) * _sample_receiver - (1 - self._receiver_source_alpha) * _sample_sender + _sample_sender * _sample_receiver) * self._detector_dark_count_prob
        
        qsys = Simulation.create_qsystem(2)
        qsys._state = self._state + np.array([[(1 - self._spin_photon_correlation) * _sample_up_down, 0., 0., (1 - self._spin_photon_correlation) * np.exp(1j * self._coherent_phase) * np.sqrt(self._visibility) * (_sample_up_down * _sample_down_up)],
                                            [0., 0.5 * self._spin_photon_correlation * (_sample_up_down + _sample_down_up) + _sample_up_up, 0., 0.],
                                            [0., 0., 0.5 * self._spin_photon_correlation * (_sample_up_down + _sample_down_up) + _sample_down_down, 0.],
                                            [(1 - self._spin_photon_correlation) * np.exp(-1j * self._coherent_phase) * np.sqrt(self._visibility) * (_sample_up_down * _sample_down_up), 0., 0., (1 - self._spin_photon_correlation) * _sample_down_up]], dtype=np.complex128)
        
        q_1, q_2 = qsys.qubits
        
        self._sender_memory.store_qubit(L0, q_1, -1, _sender_time)
        self._receiver_memory.store_qubit(L0, q_2, -1, _receiver_time)
    
    def failure_creation(self, _sender_time: float, _receiver_time: float) -> None:
        
        """
        Unsuccessful creation of a entangled pair
        
        Args:
            _sender_time (float): current time of creating sender sided qubit
            _receiver_time (float): current time of creating receiver sided qubit
            
        Returns:
            /
        """
        
        q_1 = Simulation.create_qsystem(1).qubits
        q_2 = Simulation.create_qsystem(1).qubits
        self._sender_memory.store_qubit(L0, q_1, -1, _sender_time)
        self._receiver_memory.store_qubit(L0, q_2, -1, _receiver_time)
    
    def attempt_bell_pairs(self, _requested: int=1, _needed: int=1) -> None:
        
        """
        Attempts to create the needed number of qubits
        
        Args:
            _requested (int): number of requested bell pairs
            _needed (int): number of needed bell pairs
            
        Returns:
            /
        """
        
        packet_s = Packet(self._receiver_id, self._sender.id, _requested, _needed)
        packet_r = Packet(self._sender.id, self._receiver_id, _requested, _needed)

        _num_tries = 1
        if self._num_sources + 1 and self._num_sources < _needed:
            _num_tries = int(np.ceil(_needed / self._num_sources))

        _sender_time = self._sender._time + self._sender_source_duration
        _receiver_time = self._sender._time + self._receiver_source_duration
        
        _sender_time_samples = np.zeros(_needed) + _sender_time
        _receiver_time_samples = np.zeros(_needed) + _receiver_time
        
        if self._num_sources + 1:
            _sender_time_samples += np.repeat(np.arange(1, _num_tries + 1).reshape(-1, 1), self._num_sources, axis=1).flatten()[:_needed] * self._sender_source_duration
            _receiver_time_samples += np.repeat(np.arange(1, _num_tries + 1).reshape(-1, 1), self._num_sources, axis=1).flatten()[:_needed] * self._receiver_source_duration
        
        _success_samples = np.random.uniform(0, 1, _needed) < self._success_prob
        
        for success, _sender_time, _receiver_time in zip(_success_samples, _sender_time_samples, _receiver_time_samples):
            self._creation_functions[success](_sender_time, _receiver_time)
        
        packet_s.l1_success = _success_samples
        packet_r.l1_success = _success_samples
        
        packet_r.l1_set_ack()
        packet_s.l1_protocol = 4
        packet_r.l1_protocol = 4
        
        self._sim.schedule_event(ReceiveEvent(self._sender._time + self._total_duration + self._sender_duration + (_num_tries - 1) * self._sender_source_duration, self._sender.id))
        self._sim.schedule_event(ReceiveEvent(self._sender._time + self._total_duration + self._receiver_duration + (_num_tries - 1) * self._receiver_source_duration, self._receiver_id))
        self._sender._connections['packet'][self._receiver_id][SEND].put(packet_r)
        self._sender._connections['packet'][self._receiver_id][RECEIVE].put(packet_s)

    def create_bell_pairs(self, _requested: int=1) -> None:
        
        """
        Creates the number of requested qubits, no matter how long it takes
        
        Args:
            _requested (int): number of requested qubits
            
        Returns:
            /
        """
        
        packet_s = Packet(self._receiver_id, self._sender.id, _requested, _requested)
        packet_r = Packet(self._sender.id, self._receiver_id, _requested, _requested)
        
        _sender_time = self._sender._time + self._sender_source_duration
        _receiver_time = self._sender._time + self._receiver_source_duration
        
        _sender_time_samples = np.zeros(_requested + self._num_sources + 1) + _sender_time
        _receiver_time_samples = np.zeros(_requested + self._num_sources + 1) + _receiver_time
        
        _num_successfull = 0
        _current_try = 0
        
        while _num_successfull < _requested and self._num_sources + 1:
            _successes = np.count_nonzero(np.random.uniform(0, 1, self._num_sources) < self._success_prob)
            _current_try += 1
            _sender_time_samples[_num_successfull:_num_successfull + _successes] += _current_try * self._sender_source_duration
            _receiver_time_samples[_num_successfull:_num_successfull + _successes] += _current_try * self._receiver_source_duration
            _num_successfull += _successes
        
        _sender_time_samples = _sender_time_samples[:_requested]
        _receiver_time_samples = _receiver_time_samples[:_requested]
        
        for _sender_time, _receiver_time in zip(_sender_time_samples, _receiver_time_samples):
            self.success_creation(_sender_time, _receiver_time)
        
        packet_s.l1_success = np.ones(_requested, dtype=np.bool_)
        packet_r.l1_success = np.ones(_requested, dtype=np.bool_)
        
        packet_r.l1_set_ack()
        packet_s.l1_protocol = 4
        packet_r.l1_protocol = 4
        
        self._sim.schedule_event(ReceiveEvent(self._sender._time + self._total_duration + self._sender_duration + (_current_try - 1) * self._sender_source_duration, self._sender.id))
        self._sim.schedule_event(ReceiveEvent(self._sender._time + self._total_duration + self._receiver_duration + (_current_try - 1) * self._receiver_source_duration, self._receiver_id))
        self._sender._connections['packet'][self._receiver_id][SEND].put(packet_r)
        self._sender._connections['packet'][self._receiver_id][RECEIVE].put(packet_s)
        
class L3Connection:
    
    """
    Represents a Layer 3 connection
    
    Attrs:
        _sender (Host): sender of connection
        _receiver (Host): receiver of connection
        _num_sources (int): number of sources available
        _sim (Simulation): Simulation connection lives in
        _source_duration (float): duration of the source
        _sending_time (float): sending time of qubits
        _success_prob (float): probability of sucessfully establishing entanglement
        _fidelity_variance (float): variance of state fidelity
        _state (np.array): resulting state
        _sender_memory (QuantumMemory): sender sided quantum memory
        _receiver_memory (QuantumMemory): receiver sided quantum memory
        _creation_functions (dict): function to call whether sending or receiving was successful
    """
    
    def __init__(self, _sender: Host, _receiver: int, _sim: Simulation,_length: float=0.,
                 _num_sources: int=-1, _source_duration: float=0.,
                 _success_prob: float=1., _fidelity: float=1., _fidelity_variance: float=0.,
                 _sender_memory: QuantumMemory=None, _receiver_memory: QuantumMemory=None) -> None:
        
        """
        Initializes a L3 connection
        
        Args:
            _sender (Host): sender of connection
            _receiver (Host): receiver of connection
            _sim (Simulation): simulation connection lives in
            _num_sources (int): number of sources available
            _length (float): length of connection
            _success_prob (float): probability of sucessfully establishing entanglement
            _fidelity (float): fidelity of resulting state
            _fidelity_variance (float): variance of fidelity
            _sender_memory (QuantumMemory): sender sided quantum memory
            _receiver_memory (QuantumMemory): receiver sided quantum memory
            
        Returns:
            /
        """
        
        if -np.log10(_success_prob) >= 6:
            raise ValueError('Too low success probability')
        
        self._sender: Host = _sender
        self._receiver_id: int = _receiver
        self._num_sources: int = _num_sources
        self._sim: Simulation = _sim
        
        self._source_duration: float = _source_duration
        self._sending_time: float = _length * 5e-6
        self._success_prob: float = _success_prob
        self._fidelity_variance: float = _fidelity_variance
        
        self._state: np.array = (4 * _fidelity - 1) / 3 * B_0 + 4 * (1 - _fidelity) / 3 * I_0
        
        self._sender_memory: QuantumMemory = _sender_memory
        self._receiver_memory: QuantumMemory = _receiver_memory
        
        self._creation_functions = {0: self.failure_creation, 1: self.success_creation}
        
    def success_creation(self, _time: float) -> None:
        
        """
        Sucessful creation of entanglement
        
        Args:
            _time (float): time at which qubits are created
            
        Returns:
            /
        """
        
        _sample = sp.stats.truncnorm.rvs(-1, 1, loc=0, scale=self._fidelity_variance)
        
        qsys = Simulation.create_qsystem(2)
        qsys._state = self._state + (4 * _sample) / 3 * B_I_0
        q_1, q_2 = qsys.qubits
        
        self._sender_memory.store_qubit(L3, q_1, -1, _time)
        self._receiver_memory.store_qubit(L3, q_2, -1, _time)
    
    def failure_creation(self, _time: float) -> None:
        
        """
        Unsuccessful creation of qubits
        
        Args:
            _time (float): time at which qubits are created
            
        Returns:
            /
        """
        
        q_1 = Simulation.create_qsystem(1).qubits
        self._receiver_memory.store_qubit(L3, q_1, -1, _time)
    
    def attempt_bell_pairs(self, _requested: int=1, _needed: int=1) -> None:
        
        """
        Attempt the number of bell pairs
        
        Args:
            _requested (int): number of requested qubits
            _needed (int): number of needed qubits
            
        Returns:
            /
        """
        
        packet = Packet(self._sender.id, self._receiver_id, _requested, _needed)

        _num_tries = 1
        if self._num_sources + 1 and self._num_sources < _needed:
            _num_tries = int(np.ceil(_needed / self._num_sources))

        _time_samples = np.zeros(_needed) + self._sender._time + self._sending_time
        _success_samples = np.random.uniform(0, 1, _needed) < self._success_prob
        
        if self._num_sources + 1:
            _time_samples += np.repeat(np.arange(1, _num_tries + 1).reshape(-1, 1), self._num_sources, axis=1).flatten()[:_needed] * self._source_duration
        
        for success, _time_sample in zip(_success_samples, _time_samples):
            self._creation_functions[success](_time_sample)
        
        packet.l1_success = _success_samples
        
        self._sim.schedule_event(ReceiveEvent(self._sender._time + self._sending_time + (_num_tries - 1) * self._source_duration, self._receiver_id))
        self._sender._connections['packet'][self._receiver_id][SEND].put(packet)
    
    def create_bell_pairs(self, _requested: int=1) -> None:
        
        """
        Creates the number of requested qubits
        
        Args:
            _requested (int): number of requested qubits
        
        Returns:
            /
        """
        
        packet = Packet(self._sender.id, self._receiver_id, _requested, _requested)
        
        _time_samples = np.zeros(_requested + self._num_sources + 1) + self._sender._time + self._sending_time
        
        _num_successfull = 0
        _current_try = 0
        
        while _num_successfull < _requested and self._num_sources + 1:
            _successes = np.count_nonzero(np.random.uniform(0, 1, self._num_sources) < self._success_prob)
            _current_try += 1
            _time_samples[_num_successfull:_num_successfull + _successes] += _current_try * self._source_duration
            _num_successfull += _successes
        
        _time_samples = _time_samples[:_requested]
        
        for _time_sample in _time_samples:
            self.success_creation(_time_sample)
        
        packet.l1_success = np.ones(_requested, dtype=np.bool_)
        
        self._sim.schedule_event(ReceiveEvent(self._sender._time + self._sending_time + (_current_try - 1) * self._source_duration, self._receiver_id))
        self._sender._connections['packet'][self._receiver_id][SEND].put(packet)