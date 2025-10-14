
import numpy as np
import scipy as sp
from typing import List

from .channel import QChannel
from qcns.python.components.hardware.photon_source import *
from qcns.python.components.hardware.photon_detector import *
from qcns.python.components.hardware.memory import QuantumMemory
from qcns.python.components.hardware.connection import *
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
# Bell State measurement: duration, visibility, coin ph ph, coin ph dc, coin dc dc
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
    
    def __init__(self, _sender: Host, _receiver: int, _sim: Simulation, _model: SQC_Model=SQC_Model()) -> None:
        
        """
        Initializes a Single Qubit Connection
        
        Args:
            _sender (Host): sender of connection
            _receiver (Host): receiver of connection
            _sim (Simulation): simulation object
            _sender_source (str): name of model to use for single photon source
            _num_sources (int): number of available sources
            _length (float): length of connection
            _attenuation (float): attenuation coefficient of fiber
            _in_coupling_prob (float): probability of coupling photon into fiber
            _out_coupling_prob (float): probability of coupling photon out of fiber
            _lose_qubits (float): whether to lose qubits or not
            _com_errors (list): list of communication errors
            
        Returns:
            /
        """
        
        self._sender: Host = _sender
        self._receiver_id: int = _receiver
        self._num_sources: int = _model._num_sources
        self._source: SinglePhotonSource = _model._source_model

        self._sim: Simulation = _sim

        self._success_prob: float = self._source._visibility * self._source._pmf[1] * _model._channel_model._in_coupling

        if -np.log10(self._success_prob) >= 6:
            raise ValueError('Too low success probability')

        self._state: np.ndarray = np.array([[self._source._fidelity, np.sqrt(self._source._fidelity * (1 - self._source._fidelity))], [np.sqrt(self._source._fidelity * (1 - self._source._fidelity)), 1 - self._source._fidelity]], dtype=np.complex128)

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
        self._sender._channels['qc'][self._receiver][SEND].put(q_1)
        
    def failure_creation(self) -> None:
        
        """
        Unsuccessful creation of a qubit
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._sender._channels['qc'][self._receiver][SEND].put(None)
      
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
    
    def __init__(self, _sender: Host, _receiver: int, _sim: Simulation, _sender_memory: QuantumMemory, _receiver_memory: QuantumMemory,
                 _model: SRC_Model) -> None:
        
        """
        Initializes a sender receiver connection
        
        Args:
            _sender (Host): sender of connection
            _receiver (Host): receiver of connection
            _sim (Simulation): simulation object
            _connection_model (str): model to use for connection
            _sender_source (str): model to use for sender source
            _receiver_detector (str): model to use for receiver detector
            _num_sources: (int): number of available sources
            _length (float): length of connection
            _attenuation (float): attenuation coefficient of fiber
            _in_coupling_prob (float): probability of coupling photon into fiber
            _out_coupling_prob (float): probability of coupling photon out of fiber
            _lose_qubits (bool): whether to lose qubits
            _sender_memory (QuantumMemory): quantum memory of sender
            _receiver_memory (QuantumMemory): quantum memory of receiver
            
        Returns:
            /
        """
        
        self._sender: Host = _sender
        self._receiver_id: int = _receiver
        self._num_sources: int = _model._num_sources

        _source = _model._source
        _detector = _model._detector
        _device = _model._device
        _channel = _model._qchannel

        self._state_transfer_fidelity: float = _device._state_transfer_fidelity
        self._source_duration: float = _source._duration
        self._source_fidelity: float = _source._fidelity
        self._source_std: float = np.sqrt(_source._fidelity_variance)
        self._sender_memory: QuantumMemory = _sender_memory
        self._receiver_memory: QuantumMemory = _receiver_memory

        self._sim: Simulation = _sim

        _p0, _p1, _p2 = _source._pmf[:3]

        _visibility = _source._visibility
        _dark_count = _detector._dark_count
        _eta_channel = _channel._in_coupling * 10 ** (-_channel._attenuation * _channel._length / 10.0) * _channel._out_coupling
        _eta = _eta_channel * _detector._efficiency * _device._interaction_probability * self._receiver_memory._efficiency

        _true_1 = _p1 * _visibility * _eta * (1 - _dark_count)
        _true_2 = _p2 * _visibility * 2 * _eta * (1 - _eta) * (1 - _dark_count)
        
        _false_0 = _p0 * _visibility * _dark_count
        _false_1 = _p1 * _visibility * (1 - _eta) * _dark_count
        _false_2 = _p2 * _visibility * (1 - _eta) ** 2 * _dark_count
        
        if isinstance(_detector, ThresholdDetector):
            _false_2 += _p2 * _visibility * _eta ** 2
        
        self._success_prob = _true_1 + _true_2
        _false_prob = _false_0 + _false_1 + _false_2
        
        _total_prob = self._success_prob + _false_prob
        
        self._state_weight: float = self._success_prob / _total_prob
        
        _depolar_prob = (4 * _source._fidelity - 1) / 3 * self._state_weight * ((4 * self._state_transfer_fidelity - 1) / 3)

        if isinstance(_detector, ThresholdDetector):
            _depolar_prob += (1 - self._state_weight) / 3

        _click_prob = _p0 * _dark_count + _p1 * (1 - (1 - _dark_count) * (1 - _eta)) + _p2 * (1 - (1 - _dark_count) * (1 - _eta) ** 2)

        self._state = _depolar_prob * B_0 + (1 - _depolar_prob) * I_0

        _detector_cycle = _detector._duration + _click_prob * (_detector._quench_time + _detector._dead_time + _detector._after_pulse_duration)

        self._sending_duration = self._source_duration + _channel._propagation_time + _device._duration + _detector_cycle
        self._total_duration = 2 * _channel._propagation_time

        self._creation_functions = {0: self.failure_creation, 1: self.success_creation}
 
    def success_creation(self, _curr_time: float) -> None:
        
        """
        Successful creation of a entangled pair
        
        Args:
            _curr_time (float): current time of creating qubit
            
        Returns:
            /
        """
        
        _sample = 0
        if self._source_std > 0.:
            _sample = 4 * sp.stats.truncnorm.rvs(-self._source_fidelity / self._source_std, 1 - self._source_fidelity / self._source_std, loc=0, scale=self._source_std) / 3
        
        qsys = Simulation.create_qsystem(2)
        qsys._state = self._state + self._state_weight * ((4 * self._state_transfer_fidelity - 1) / 3) * _sample * B_I_0
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
        self._sender._channels['pc'][self._receiver_id][SEND].put(packet)
    
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
        self._sender._channels['pc'][self._receiver_id][SEND].put(packet)
        
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
    
    def __init__(self, _sender: Host, _receiver: int, _sim: Simulation, _sender_memory: QuantumMemory, _receiver_memory: QuantumMemory,
                 _model: TPSC_Model) -> None:

        self._sender = _sender
        self._receiver_id = _receiver
        self._num_sources = _model._num_sources

        _source = _model._source
        _sender_detector = _model._sender_detector
        _sender_channel = _model._sender_qchannel
        _sender_device = _model._sender_device
        _receiver_detector = _model._receiver_detector
        _receiver_channel = _model._receiver_qchannel
        _receiver_device = _model._receiver_device

        self._sender_state_transfer_fidelity = _sender_device._state_transfer_fidelity
        self._receiver_state_transfer_fidelity = _receiver_device._state_transfer_fidelity

        self._source_duration = _source._duration
        self._source_fidelity = _source._fidelity
        self._source_std = np.sqrt(_source._fidelity_variance)

        self._sender_memory = _sender_memory
        self._receiver_memory = _receiver_memory

        self._sim = _sim

        _p0, _p1, _p2 = _source._pmf[:3]

        _eta_sender_channel = _sender_channel._in_coupling * 10 ** (-_sender_channel._attenuation * _sender_channel._length / 10.0) * _sender_channel._out_coupling
        _eta_receiver_channel = _receiver_channel._in_coupling * 10 ** (-_receiver_channel._attenuation * _receiver_channel._length / 10.0) * _receiver_channel._out_coupling
        
        _sender_dark = _sender_detector._dark_count
        _receiver_dark = _receiver_detector._dark_count
        
        _source_visibility = _source._visibility
        
        _eta_sender = _eta_sender_channel * _sender_detector._efficiency * _sender_device._interaction_probability * self._sender_memory._efficiency
        _eta_receiver = _eta_receiver_channel * _receiver_detector._efficiency * _receiver_device._interaction_probability * self._receiver_memory._efficiency
        
        _sender_true_1 = np.sqrt(_p1 * _source_visibility) * _eta_sender * (1 - _sender_dark)
        _sender_true_2 = np.sqrt(0.5 * _p2 * _source_visibility) * 2 * (1 - _eta_sender) * _eta_sender * (1 - _sender_dark)
        
        _sender_false_0 = np.sqrt(_p0) * _sender_dark
        _sender_false_1 = np.sqrt(_p1 * _source_visibility) * (1 - _eta_sender) * _sender_dark
        _sender_false_2 = (1 - _eta_sender) ** 2 * _sender_dark
        if isinstance(_sender_detector, ThresholdDetector):
            _sender_false_2 += _eta_sender ** 2
        _sender_false_2 *= np.sqrt(_p2)
        
        _receiver_true_1 = np.sqrt(_p1 * _source_visibility) * _eta_receiver * (1 - _receiver_dark)
        _receiver_true_2 = np.sqrt(0.5 * _p2 * _source_visibility) * 2 * (1 - _eta_receiver) * _eta_receiver * (1 - _receiver_dark)
        
        _receiver_false_0 = np.sqrt(_p0) * _receiver_dark
        _receiver_false_1 = np.sqrt(_p1 * _source_visibility) * (1 - _eta_receiver) * _receiver_dark
        _receiver_false_2 = (1 - _eta_receiver) ** 2 * _receiver_dark
        if isinstance(_receiver_detector, ThresholdDetector):
            _receiver_false_2 += _eta_receiver ** 2
        _receiver_false_2 *= np.sqrt(_p2)
        
        self._success_prob: float = _sender_true_1 * _receiver_true_1 + _sender_true_2 * _receiver_true_2
        
        self._p_10 = 0
        if self._success_prob < 1.:
            self._p_10 = (_sender_true_1 + _sender_true_2 - self._success_prob) / (1 - self._success_prob)
            
        self._p_01 = 0
        if self._success_prob < 1.:
            self._p_01 = (_receiver_true_1 + _receiver_true_2 - self._success_prob) / (1 - self._success_prob)
        
        self._p_00 = 1 - self._success_prob - self._p_01 - self._p_10
        
        if self._success_prob <= 0.0 or -np.log10(self._success_prob) >= 6:
            raise ValueError('Too low success probability')
        
        _false_0 = _sender_false_0 * _receiver_false_0
        _false_1 = _sender_true_1 * _receiver_false_1 + _sender_false_1 * _receiver_true_1 + _sender_false_1 * _receiver_false_1
        _false_2 = _sender_true_2 * _receiver_false_2 + _sender_false_2 * _receiver_true_2 + _sender_false_2 * _receiver_false_2
        
        _false_prob = _false_0 + _false_1 + _false_2

        _total_prob = self._success_prob + _false_prob

        self._state_weight: float = self._success_prob / _total_prob

        _depol_prob = ((4 * self._source_fidelity - 1) ** 2 / 9) * self._state_weight * ((4 * self._sender_state_transfer_fidelity - 1) / 3) * ((4 * self._receiver_state_transfer_fidelity - 1) / 3)
        
        if isinstance(_sender_detector, ThresholdDetector) and isinstance(_receiver_detector, ThresholdDetector):
            _alpha_acc = 1 / 3
        
        if isinstance(_sender_detector, ThresholdDetector) and isinstance(_receiver_detector, PNRDetector):
            _l_z = np.sqrt(_p2) * _eta_sender ** 2
            _false_z = _l_z * (_receiver_true_2 + _receiver_false_2)
            _alpha_acc = _false_z / (3 * _false_prob)
            
        if isinstance(_sender_detector, PNRDetector) and isinstance(_receiver_detector, ThresholdDetector):
            _l_z = np.sqrt(_p2) * _eta_receiver ** 2
            _false_z = _l_z * (_sender_true_2 + _sender_false_2)
            _alpha_acc = _false_z / (3 * _false_prob)
            
        if isinstance(_sender_detector, PNRDetector) and isinstance(_receiver_detector, PNRDetector):
            _alpha_acc = 0
            
        _depol_prob += (1 - self._state_weight) * _alpha_acc

        self._state: np.ndarray = _depol_prob * B_0 + (1 - _depol_prob) * I_0

        _sender_click_prob = _p0 * _sender_dark + _p1 * (1 - (1 - _sender_dark) * (1 - _eta_sender)) + _p2 * (1 - (1 - _sender_dark) * (1 - _eta_sender) ** 2)
        _receiver_click_prob = _p0 * _receiver_dark + _p1 * (1 - (1 - _receiver_dark) * (1 - _eta_receiver)) + _p2 * (1 - (1 - _receiver_dark) * (1 - _eta_receiver) ** 2)

        _sender_detector_cycle = _sender_detector._duration + _sender_click_prob * (_sender_detector._quench_time + _sender_detector._dead_time + _sender_detector._after_pulse_duration)
        _receiver_detector_cycle = _receiver_detector._duration + _receiver_click_prob * (_receiver_detector._quench_time + _receiver_detector._dead_time + _receiver_detector._after_pulse_duration)

        self._sender_duration = self._source_duration + _sender_channel._propagation_time + _sender_device._duration + _sender_detector_cycle
        self._receiver_duration = self._source_duration + _receiver_channel._propagation_time + _receiver_device._duration + _receiver_detector_cycle
        self._total_duration = 2 * (_sender_channel._propagation_time + _receiver_channel._propagation_time)

        self._creation_functions = {0: self.failure_creation, 1: self.success_creation}
        
    def success_creation(self, _curr_time: float) -> None:
        
        """
        Successful creation of a entangled pair
        
        Args:
            _curr_time (float): current time of creating qubit
            
        Returns:
            /
        """
        
        _sample = 0
        if self._source_std > 0.:
            _sample = 8 * (4 * self._source_fidelity + 1) * sp.stats.truncnorm.rvs(-self._source_fidelity / self._source_std, 1 - self._source_fidelity / self._source_std, loc=0, scale=self._source_std) / 9

        qsys = Simulation.create_qsystem(2)
        qsys._state = self._state + self._state_weight * ((4 * self._sender_state_transfer_fidelity - 1) / 3) * ((4 * self._receiver_state_transfer_fidelity - 1) / 3) * _sample * B_I_0
    
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
        
        _success_samples = np.random.uniform(0, 1, _needed) < self._success_prob
        
        _trials = np.random.uniform(0, 1, _needed)
        _m_10 = (~_success_samples) & (_trials <  self._p_10)
        _m_01 = (~_success_samples) & (_trials >= self._p_10) & (_trials < self._p_10 + self._p_01)
        
        _sender_success_samples = np.copy(_success_samples)
        _receiver_success_samples = np.copy(_success_samples)
        
        _sender_success_samples[_m_10] = 1
        _receiver_success_samples[_m_01] = 1
        
        packet_s.l1_success = _sender_success_samples
        packet_r.l1_success = _receiver_success_samples
        
        for _success, _curr_time in zip(_success_samples, _curr_time_samples):
            self._creation_functions[_success](_curr_time)
        
        packet_s.l1_set_ps()
        packet_r.l1_set_ps()
        packet_r.l1_set_ack()
        packet_s.l1_protocol = 2
        packet_r.l1_protocol = 2
        
        self._sim.schedule_event(ReceiveEvent(self._sender._time + self._total_duration + self._sender_duration + (_num_tries - 1) * self._source_duration, self._sender.id))
        self._sim.schedule_event(ReceiveEvent(self._sender._time + self._total_duration + self._receiver_duration + (_num_tries - 1) * self._source_duration, self._receiver_id))
        self._sender._channels['pc'][self._receiver_id][SEND].put(packet_r)
        self._sender._channels['pc'][self._receiver_id][RECEIVE].put(packet_s)

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
        self._sender._channels['pc'][self._receiver_id][SEND].put(packet_r)
        self._sender._channels['pc'][self._receiver_id][RECEIVE].put(packet_s)

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
    
    def __init__(self, _sender: Host, _receiver: int, _sim: Simulation, _sender_memory: QuantumMemory, _receiver_memory: QuantumMemory,
                 _model: BSMC_Model) -> None:
        
        """
        Initializes a Bell State Measurement Connection
        
        Args:
            _sender (Host): sender of connection
            _receiver (Host): receiver of connection
            _sim (Simulation): simulation object
            _connection_model (str): model to use for connection
            _sender_source (str): model to use for atom photon source at sender
            _receiver_source (str): model to use for atom photon source at receiver
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
        
        self._sender: Host = _sender
        self._receiver_id: int = _receiver
        self._num_sources: int = _model._num_sources

        _sender_source = _model._sender_source
        _sender_detector = _model._sender_detector
        _sender_channel = _model._sender_qchannel
        _receiver_source = _model._receiver_source
        _receiver_detector = _model._receiver_detector
        _receiver_channel = _model._receiver_qchannel
        _device = _model._device

        _duration = _device._duration
        _split_eff = 0.5 * (1 - _device._visibility) + _device._visibility * _device._signal_prob

        self._sender_source_duration: float = _sender_source._duration
        self._receiver_source_duration: float = _receiver_source._duration
        self._sender_fidelity: float = _sender_source._fidelity
        self._receiver_fidelity: float = _receiver_source._fidelity
        self._sender_std: float = np.sqrt(_sender_source._fidelity_variance)
        self._receiver_std: float = np.sqrt(_receiver_source._fidelity_variance)
        self._sender_memory: QuantumMemory = _sender_memory
        self._receiver_memory: QuantumMemory = _receiver_memory

        self._sim: Simulation = _sim

        _s_p0, _s_p1, _s_p2 = _sender_source._pmf[:3]
        _r_p0, _r_p1, _r_p2 = _receiver_source._pmf[:3]

        _sender_eta = self._sender_memory._efficiency * _sender_channel._in_coupling * 10 ** (-_sender_channel._attenuation * _sender_channel._length / 10.0) * _sender_channel._out_coupling
        _receiver_eta = self._receiver_memory._efficiency * _receiver_channel._in_coupling * 10 ** (-_receiver_channel._attenuation * _receiver_channel._length / 10.0) * _receiver_channel._out_coupling
        _sender_efficiency = _sender_detector._efficiency
        _sender_dark = _sender_detector._dark_count
        _receiver_efficiency = _receiver_detector._efficiency
        _receiver_dark = _receiver_detector._dark_count
        
        _sender_arrival_0 = _s_p0 + _s_p1 * _sender_source._visibility * (1 - _sender_eta) + _s_p2 * _sender_source._visibility * (1 - _sender_eta) ** 2
        _sender_arrival_1 = _s_p1 * _sender_source._visibility * _sender_eta + _s_p2 * _sender_source._visibility * 2 * _sender_eta * (1 - _sender_eta)
        _sender_arrival_2 = _s_p2 * _sender_source._visibility * _sender_eta ** 2
        
        _receiver_arrival_0 = _r_p0 + _r_p1 * _receiver_source._visibility * (1 - _receiver_eta) + _r_p2 * _receiver_source._visibility * (1 - _receiver_eta) ** 2
        _receiver_arrival_1 = _r_p1 * _receiver_source._visibility * _receiver_eta + _r_p2 * _receiver_source._visibility * 2 * _receiver_eta * (1 - _receiver_eta)
        _receiver_arrival_2 = _r_p2 * _receiver_source._visibility * _receiver_eta ** 2
        
        _p_00 = _sender_arrival_0 * _receiver_arrival_0
        _p_01 = _sender_arrival_0 * _receiver_arrival_1
        _p_10 = _sender_arrival_1 * _receiver_arrival_0
        _p_11 = _sender_arrival_1 * _receiver_arrival_1
        _p_02 = _sender_arrival_0 * _receiver_arrival_2
        _p_20 = _sender_arrival_2 * _receiver_arrival_0
        _p_12 = _sender_arrival_1 * _receiver_arrival_2
        _p_21 = _sender_arrival_2 * _receiver_arrival_1
        _p_22 = _sender_arrival_2 * _receiver_arrival_2
        
        def __TD_rate(_eta, _k, _dark_count):
            
            if _k == 0:
                return _dark_count
            
            if _eta == 0.:
                return _dark_count
            
            if _eta == 1.:
                return 1.
            
            return 1 - (1 - _dark_count) * (1 - _eta) ** _k
        
        def __PNR_rate(_eta, _k, _dark_count):
            
            if _k == 0:
                return _dark_count
            
            if _eta == 0.:
                return _dark_count
            
            if _eta == 1.:
                return (1.0 - _dark_count) * (1.0 if _k == 1 else 0.0)
            
            return (1 - _dark_count) * _k * _eta * (1 - _eta) ** (_k - 1) + _dark_count * (1 - _eta) ** _k
        
        __D_rate = {0: __TD_rate, 1: __PNR_rate}
        
        _s_det = isinstance(_sender_detector, PNRDetector)
        _r_det = isinstance(_receiver_detector, PNRDetector)
            
        C_00 = __D_rate[_s_det](_sender_efficiency, 0, _sender_dark) * __D_rate[_r_det](_receiver_efficiency, 0, _receiver_dark)
        C_01 = 0.5 * (__D_rate[_s_det](_sender_efficiency, 1, _sender_dark) * __D_rate[_r_det](_receiver_efficiency, 0, _receiver_dark) + __D_rate[_s_det](_sender_efficiency, 0, _sender_dark) * __D_rate[_r_det](_receiver_efficiency, 1, _receiver_dark))
        C_10 = C_01
        C_11_t = _split_eff * _sender_efficiency * _receiver_efficiency
        if _s_det:
            C_11_t *= (1 - _sender_dark)
        if _r_det:
            C_11_t *= (1 - _receiver_dark)
        C_11_total = _split_eff * __D_rate[_s_det](_sender_efficiency, 1, _sender_dark) * __D_rate[_r_det](_receiver_efficiency, 1, _receiver_dark) + 0.5 * (1 - _split_eff) * (__D_rate[_s_det](_sender_efficiency, 2, _sender_dark) * __D_rate[_r_det](_receiver_efficiency, 0, _receiver_dark) + __D_rate[_s_det](_sender_efficiency, 0, _sender_dark) * __D_rate[_r_det](_receiver_efficiency, 2, _receiver_dark))
        C_11_f = C_11_total - C_11_t
        C_02 = 0.5 * __D_rate[_s_det](_sender_efficiency, 1, _sender_dark) * __D_rate[_r_det](_receiver_efficiency, 1, _receiver_dark) + 0.25 * (__D_rate[_s_det](_sender_efficiency, 2, _sender_dark) * __D_rate[_r_det](_receiver_efficiency, 0, _receiver_dark) + __D_rate[_s_det](_sender_efficiency, 0, _sender_dark) * __D_rate[_r_det](_receiver_efficiency, 2, _receiver_dark))
        C_20 = C_02
        C_12 = (3 / 8) * (__D_rate[_s_det](_sender_efficiency, 0, _sender_dark) * __D_rate[_r_det](_receiver_efficiency, 3, _receiver_dark) + __D_rate[_s_det](_sender_efficiency, 3, _sender_dark) * __D_rate[_r_det](_receiver_efficiency, 0, _receiver_dark)) + 0.125 * (__D_rate[_s_det](_sender_efficiency, 1, _sender_dark) * __D_rate[_r_det](_receiver_efficiency, 2, _receiver_dark) + __D_rate[_s_det](_sender_efficiency, 2, _sender_dark) * __D_rate[_r_det](_receiver_efficiency, 1, _receiver_dark))
        C_21 = C_12
        C_22 = (3 / 8) * (__D_rate[_s_det](_sender_efficiency, 0, _sender_dark) * __D_rate[_r_det](_receiver_efficiency, 4, _receiver_dark) + __D_rate[_s_det](_sender_efficiency, 4, _sender_dark) * __D_rate[_r_det](_receiver_efficiency, 0, _receiver_dark)) + 0.25 * __D_rate[_s_det](_sender_efficiency, 2, _sender_dark) * __D_rate[_r_det](_receiver_efficiency, 2, _receiver_dark)
        
        self._success_prob = C_11_t * _p_11
        
        if self._success_prob <= 0.0 or -np.log10(self._success_prob) >= 6:
            raise ValueError('Too low success probability')
        
        _false_prob = C_11_f * _p_11 + C_00 * _p_00 + C_01 * _p_01 + C_10 * _p_10 + C_02 * _p_02 + C_20 * _p_20 + C_12 * _p_12 + C_21 * _p_21 + C_22 * _p_22
        
        _total_prob = self._success_prob + _false_prob

        self._state_weight: float = self._success_prob / _total_prob

        _td_00 = __D_rate[_s_det](_sender_efficiency, 0, 0) * __D_rate[_r_det](_receiver_efficiency, 0, 0)
        _td_01 = 0.5 * (__D_rate[_s_det](_sender_efficiency, 1, 0) * __D_rate[_r_det](_receiver_efficiency, 0, 0) + __D_rate[_s_det](_sender_efficiency, 0, 0) * __D_rate[_r_det](_receiver_efficiency, 1, 0))
        _td_10 = _td_01
        _td_11 = _split_eff * __D_rate[_s_det](_sender_efficiency, 1, 0) * __D_rate[_r_det](_receiver_efficiency, 1, 0) + 0.5 * (1 - _split_eff) * (__D_rate[_s_det](_sender_efficiency, 2, 0) * __D_rate[_r_det](_receiver_efficiency, 0, 0) + __D_rate[_s_det](_sender_efficiency, 0, 0) * __D_rate[_r_det](_receiver_efficiency, 2, 0))
        _td_02 = 0.5 * __D_rate[_s_det](_sender_efficiency, 1, 0) * __D_rate[_r_det](_receiver_efficiency, 1, 0) + 0.25 * (__D_rate[_s_det](_sender_efficiency, 2, 0) * __D_rate[_r_det](_receiver_efficiency, 0, 0) + __D_rate[_s_det](_sender_efficiency, 0, 0) * __D_rate[_r_det](_receiver_efficiency, 2, 0))
        _td_20 = _td_02
        _td_12 = (3 / 8) * (__D_rate[_s_det](_sender_efficiency, 0, 0) * __D_rate[_r_det](_receiver_efficiency, 3, 0) + __D_rate[_s_det](_sender_efficiency, 3, 0) * __D_rate[_r_det](_receiver_efficiency, 0, 0)) + 0.125 * (__D_rate[_s_det](_sender_efficiency, 1, 0) * __D_rate[_r_det](_receiver_efficiency, 2, 0) + __D_rate[_s_det](_sender_efficiency, 2, 0) * __D_rate[_r_det](_receiver_efficiency, 1, 0))
        _td_21 = _td_12
        _td_22 = (3 / 8) * (__D_rate[_s_det](_sender_efficiency, 0, 0) * __D_rate[_r_det](_receiver_efficiency, 4, 0) + __D_rate[_s_det](_sender_efficiency, 4, 0) * __D_rate[_r_det](_receiver_efficiency, 0, 0)) + 0.25 * __D_rate[_s_det](_sender_efficiency, 2, 0) * __D_rate[_r_det](_receiver_efficiency, 2, 0)
        
        _pnr_11 = _split_eff * __PNR_rate(_sender_efficiency, 1, 0) * __PNR_rate(_receiver_efficiency, 1, 0) + 0.5 * (1 - _split_eff) * (__PNR_rate(_sender_efficiency, 2, 0) * __PNR_rate(_receiver_efficiency, 0, 0) + __PNR_rate(_sender_efficiency, 0, 0) * __PNR_rate(_receiver_efficiency, 2, 0))
        _pnr_02 = 0.5 * __PNR_rate(_sender_efficiency, 1, 0) * __PNR_rate(_receiver_efficiency, 1, 0) + 0.25 * (__PNR_rate(_sender_efficiency, 2, 0) * __PNR_rate(_receiver_efficiency, 0, 0) + __PNR_rate(_sender_efficiency, 0, 0) * __PNR_rate(_receiver_efficiency, 2, 0))
        _pnr_20 = _pnr_02
        _pnr_12 = (3 / 8) * (__PNR_rate(_sender_efficiency, 0, 0) * __PNR_rate(_receiver_efficiency, 3, 0) + __PNR_rate(_sender_efficiency, 3, 0) * __PNR_rate(_receiver_efficiency, 0, 0)) + 0.125 * (__PNR_rate(_sender_efficiency, 1, 0) * __PNR_rate(_receiver_efficiency, 2, 0) + __PNR_rate(_sender_efficiency, 2, 0) * __PNR_rate(_receiver_efficiency, 1, 0))
        _pnr_21 = _pnr_12
        _pnr_22 = (3 / 8) * (__PNR_rate(_sender_efficiency, 0, 0) * __PNR_rate(_receiver_efficiency, 4, 0) + __PNR_rate(_sender_efficiency, 4, 0) * __PNR_rate(_receiver_efficiency, 0, 0)) + 0.25 * __PNR_rate(_sender_efficiency, 2, 0) * __PNR_rate(_receiver_efficiency, 2, 0)
        
        if _false_prob > 0.:
            _alpha_acc = (_p_00 * (_td_00) + _p_01 * (_td_01) + _p_10 * (_td_10) + _p_11 * (_td_11 - _pnr_11) + _p_02 * (_td_02 - _pnr_02) + _p_20 * (_td_20 - _pnr_20) + _p_12 * (_td_12 - _pnr_12) + _p_21 * (_td_21 - _pnr_21) + _p_22 * (_td_22 - _pnr_22)) / (3 * _false_prob)
        else:
            _alpha_acc = 0.

        self._sender_depol: float = (4 * self._sender_fidelity - 1) / 3
        self._receiver_depol: float = (4 * self._receiver_fidelity - 1) / 3

        _depol_prob = self._sender_depol * self._receiver_depol * self._state_weight + (1 - self._state_weight) * _alpha_acc
        
        self._state: np.ndarray = _depol_prob * B_0 + (1 - _depol_prob) * I_0

        _A_0 = _p_00 + 0.5 * (_p_01 + _p_10) + 0.5 * (1 - _split_eff) * _p_11 + 0.25 * (_p_02 + _p_20) + 0.375 * (_p_12 + _p_21) + 0.375 * _p_22
        _A_1 = 0.5 * (_p_01 + _p_10) + _split_eff * _p_11 + 0.5 * (_p_02 + _p_20) + 0.125 * (_p_12 + _p_21)
        _A_2 = 0.5 * (1 - _split_eff) * _p_11 + 0.25 * (_p_02 + _p_20) + 0.125 * (_p_12 + _p_21) + 0.25 * _p_22
        _A_3 = 0.375 * (_p_12 + _p_21)
        _A_4 = 0.375 * _p_22

        _sender_click_prob = _A_0 * __D_rate[_s_det](_sender_efficiency, 0, _sender_dark) + _A_1 * __D_rate[_s_det](_sender_efficiency, 1, _sender_dark) + _A_2 * __D_rate[_s_det](_sender_efficiency, 2, _sender_dark) + _A_3 * __D_rate[_s_det](_sender_efficiency, 3, _sender_dark) + _A_4 * __D_rate[_s_det](_sender_efficiency, 4, _sender_dark)
        _receiver_click_prob = _A_0 * __D_rate[_s_det](_receiver_efficiency, 0, _receiver_dark) + _A_1 * __D_rate[_s_det](_receiver_efficiency, 1, _receiver_dark) + _A_2 * __D_rate[_s_det](_receiver_efficiency, 2, _receiver_dark) + _A_3 * __D_rate[_s_det](_receiver_efficiency, 3, _receiver_dark) + _A_4 * __D_rate[_s_det](_receiver_efficiency, 4, _receiver_dark)

        _sender_detector_cycle = _sender_detector._duration + _sender_click_prob * (_sender_detector._quench_time + _sender_detector._dead_time + _sender_detector._after_pulse_duration)
        _receiver_detector_cycle = _receiver_detector._duration + _receiver_click_prob * (_receiver_detector._quench_time + _receiver_detector._dead_time + _receiver_detector._after_pulse_duration)

        self._sender_duration = self._sender_source_duration + _sender_channel._propagation_time + _duration + _sender_detector_cycle
        self._receiver_duration = self._receiver_source_duration + _receiver_channel._propagation_time + _duration + _receiver_detector_cycle
        self._total_duration = 2 * (_sender_channel._propagation_time + _receiver_channel._propagation_time)

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
        
        _sender_sample = 0
        if self._sender_std > 0.:
            _sender_sample = 4 * sp.stats.truncnorm.rvs(-self._sender_fidelity / self._sender_std, 1 - self._sender_fidelity / self._sender_std, loc=0, scale=self._sender_std) / 3
            
        _receiver_sample = 0
        if self._receiver_std > 0.:
            _receiver_sample = 4 * sp.stats.truncnorm.rvs(-self._receiver_fidelity / self._receiver_std, 1 - self._receiver_fidelity / self._receiver_std, loc=0, scale=self._receiver_std) / 3
        _sample = self._sender_depol * _receiver_sample + self._receiver_depol * _sender_sample + _sender_sample * _receiver_sample
        
        qsys = Simulation.create_qsystem(2)
        qsys._state = self._state + self._state_weight * _sample * B_I_0
        
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
        self._sender._channels['pc'][self._receiver_id][SEND].put(packet_r)
        self._sender._channels['pc'][self._receiver_id][RECEIVE].put(packet_s)
    
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
        self._sender._channels['pc'][self._receiver_id][SEND].put(packet_r)
        self._sender._channels['pc'][self._receiver_id][RECEIVE].put(packet_s)
    
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
        _detector_dark_count (float): overall detector dark count prob
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
    
    def __init__(self, _sender: Host, _receiver: int, _sim: Simulation, _sender_memory: QuantumMemory, _receiver_memory: QuantumMemory,
                 _model: FSC_Model) -> None:
        
        """
        Initializes a Fock state connection
        
        Args:
            _sender (Host): sender of connection
            _receiver (Host): receiver of connection
            _sim (Simulation): simulation object
            _connection_model (str): model to use for connection
            _sender_source (str): model to use for atom photon source at sender
            _receiver_source (str): model to use for atom photon source at receiver
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
        
        self._sender: Host = _sender
        self._receiver_id: int = _receiver
        self._num_sources: int = _model._num_sources

        _device = _model._device
        _sender_source = _model._sender_source
        _receiver_source = _model._receiver_source
        _sender_detector = _model._sender_detector
        _receiver_detector = _model._receiver_detector
        _sender_channel = _model._sender_qchannel
        _receiver_channel = _model._receiver_qchannel

        _duration = _device._duration
        self._visibility: float = _device._visibility
        self._coherent_phase: float = _device._coherent_phase
        self._spin_photon_correlation: float = _device._spin_photon_correlation

        self._sender_source_duration: float = _sender_source._duration
        self._receiver_source_duration: float = _receiver_source._duration
        self._sender_fidelity: float = _sender_source._fidelity
        self._receiver_fidelity: float = _receiver_source._fidelity
        self._sender_std: float = np.sqrt(_sender_source._fidelity_variance)
        self._receiver_std: float = np.sqrt(_receiver_source._fidelity_variance)

        self._sender_memory: QuantumMemory = _sender_memory
        self._receiver_memory: QuantumMemory = _receiver_memory

        self._sim: Simulation = _sim

        _s_p0, _s_p1, _s_p2 = _sender_source._pmf[:3]
        _r_p0, _r_p1, _r_p2 = _receiver_source._pmf[:3]

        _sender_eta = self._sender_memory._efficiency * _sender_channel._in_coupling * 10 ** (-_sender_channel._attenuation * _sender_channel._length / 10.0) * _sender_channel._out_coupling
        _receiver_eta = self._receiver_memory._efficiency * _receiver_channel._in_coupling * 10 ** (-_receiver_channel._attenuation * _receiver_channel._length / 10.0) * _receiver_channel._out_coupling
        _sender_efficiency = _sender_detector._efficiency
        _sender_dark = _sender_detector._dark_count
        _receiver_efficiency = _receiver_detector._efficiency
        _receiver_dark = _receiver_detector._dark_count
        
        _sender_arrival_0 = _s_p0 + _s_p1 * (1 - _sender_eta) + _s_p2 * (1 - _sender_eta) ** 2
        _sender_arrival_1 = _s_p1 * _sender_eta + _s_p2 * 2 * _sender_eta * (1 - _sender_eta)
        _sender_arrival_2 = _s_p2 * _sender_eta ** 2
        
        _receiver_arrival_0 = _r_p0 + _r_p1 * (1 - _receiver_eta) + _r_p2 * (1 - _receiver_eta) ** 2
        _receiver_arrival_1 = _r_p1 * _receiver_eta + _r_p2 * 2 * _receiver_eta * (1 - _receiver_eta)
        _receiver_arrival_2 = _r_p2 * _receiver_eta ** 2
        
        _p_00 = _sender_arrival_0 * _receiver_arrival_0
        _p_01 = _sender_arrival_0 * _receiver_arrival_1
        _p_10 = _sender_arrival_1 * _receiver_arrival_0
        _p_11 = _sender_arrival_1 * _receiver_arrival_1
        _p_02 = _sender_arrival_0 * _receiver_arrival_2
        _p_20 = _sender_arrival_2 * _receiver_arrival_0
        _p_12 = _sender_arrival_1 * _receiver_arrival_2
        _p_21 = _sender_arrival_2 * _receiver_arrival_1
        _p_22 = _sender_arrival_2 * _receiver_arrival_2
        
        _p_plus = 0.5 * (1 + self._visibility * self._coherent_phase)
        _p_minus = 1 - _p_plus
        
        def __TD_rate(_eta, _k, _dark_count):
            
            if _k == 0:
                return _dark_count
            
            if _eta == 0.:
                return _dark_count
            
            if _eta == 1.:
                return 1.
            
            return 1 - (1 - _dark_count) * (1 - _eta) ** _k
        
        def __PNR_rate(_eta, _k, _dark_count):
            
            if _k == 0:
                return _dark_count
            
            if _eta == 0.:
                return _dark_count
            
            if _eta == 1.:
                return (1.0 - _dark_count) * (1.0 if _k == 1 else 0.0)
            
            return (1 - _dark_count) * _k * _eta * (1 - _eta) ** (_k - 1) + _dark_count * (1 - _eta) ** _k
        
        __D_rate = {0: __TD_rate, 1: __PNR_rate}
        
        def __S(_p_s, _p_r):
            
            return _p_s + _p_r - 2 * _p_s * _p_r
        
        def __S_true(_p_s, _p_r):
            
            return _p_s * (1 - _p_r)
        
        _s_det = isinstance(_sender_detector, PNRDetector)
        _r_det = isinstance(_receiver_detector, PNRDetector)
        
        C_00 = __D_rate[_s_det](_sender_efficiency, 0, _sender_dark) + __D_rate[_r_det](_receiver_efficiency, 0, _receiver_dark) - 2 * __D_rate[_s_det](_sender_efficiency, 0, _sender_dark) * __D_rate[_r_det](_receiver_efficiency, 0, _receiver_dark)

        _sender_true = _sender_efficiency
        if _s_det:
            _sender_true *= (1 - _sender_dark)
        _receiver_true = _receiver_efficiency
        if _r_det:
            _receiver_true *= (1 - _receiver_dark)

        C_01_t = _p_plus * __S_true(_receiver_true, __D_rate[_s_det](_sender_efficiency, 0, _sender_dark)) + _p_minus * __S_true(_sender_true, __D_rate[_r_det](_receiver_efficiency, 0, _receiver_dark))
        C_01_total = _p_plus * __S(__D_rate[_s_det](_sender_efficiency, 0, _sender_dark), __D_rate[_r_det](_receiver_efficiency, 1, _receiver_dark)) + _p_minus * __S(__D_rate[_s_det](_sender_efficiency, 1, _sender_dark), __D_rate[_r_det](_receiver_efficiency, 0, _receiver_dark))
        C_01_f = C_01_total - C_01_t

        C_10_t = _p_plus * __S_true(_sender_true, __D_rate[_r_det](_receiver_efficiency, 0, _receiver_dark)) + _p_minus * __S_true(_receiver_true, __D_rate[_s_det](_sender_efficiency, 0, _sender_dark))
        C_10_total = _p_plus * __S(__D_rate[_s_det](_sender_efficiency, 1, _sender_dark), __D_rate[_r_det](_receiver_efficiency, 0, _receiver_dark)) + _p_minus * __S(__D_rate[_s_det](_sender_efficiency, 0, _sender_dark), __D_rate[_r_det](_receiver_efficiency, 1, _receiver_dark))
        C_10_f = C_10_total - C_10_t

        if _s_det:
            C_01_t *= (1 - _sender_dark)
            C_10_t *= (1 - _sender_dark)
        if _r_det:
            C_01_t *= (1 - _receiver_dark)
            C_10_t *= (1 - _receiver_dark)
        
        C_11 = 0.5 * __S(__D_rate[_s_det](_sender_efficiency, 1, _sender_dark), __D_rate[_r_det](_receiver_efficiency, 1, _receiver_dark)) + 0.25 * (__S(__D_rate[_s_det](_sender_efficiency, 2, _sender_dark), __D_rate[_r_det](_receiver_efficiency, 0, _receiver_dark)) + __S(__D_rate[_s_det](_sender_efficiency, 0, _sender_dark), __D_rate[_r_det](_receiver_efficiency, 2, _receiver_dark)))
        C_02 = 0.5 * __S(__D_rate[_s_det](_sender_efficiency, 1, _sender_dark), __D_rate[_r_det](_receiver_efficiency, 1, _receiver_dark)) + 0.25 * (__S(__D_rate[_s_det](_sender_efficiency, 2, _sender_dark), __D_rate[_r_det](_receiver_efficiency, 0, _receiver_dark)) + __S(__D_rate[_s_det](_sender_efficiency, 0, _sender_dark), __D_rate[_r_det](_receiver_efficiency, 2, _receiver_dark)))
        C_20 = C_02
        C_12 = 0.375 * (__S(__D_rate[_s_det](_sender_efficiency, 0, _sender_dark), __D_rate[_r_det](_receiver_efficiency, 3, _receiver_dark)) + __S(__D_rate[_s_det](_sender_efficiency, 3, _sender_dark), __D_rate[_r_det](_receiver_efficiency, 0, _receiver_dark))) + 0.125 * (__S(__D_rate[_s_det](_sender_efficiency, 1, _sender_dark), __D_rate[_r_det](_receiver_efficiency, 2, _receiver_dark)) + __S(__D_rate[_s_det](_sender_efficiency, 2, _sender_dark), __D_rate[_r_det](_receiver_efficiency, 1, _receiver_dark)))
        C_21 = C_12
        C_22 = 0.25 * (__S(__D_rate[_s_det](_sender_efficiency, 2, _sender_dark), __D_rate[_r_det](_receiver_efficiency, 2, _receiver_dark))) + 0.375 * (__S(__D_rate[_s_det](_sender_efficiency, 0, _sender_dark), __D_rate[_r_det](_receiver_efficiency, 4, _receiver_dark)) + __S(__D_rate[_s_det](_sender_efficiency, 4, _sender_dark), __D_rate[_r_det](_receiver_efficiency, 0, _receiver_dark)))
        
        self._success_prob: float = _p_01 * C_01_t + _p_10 * C_10_t
        
        if self._success_prob <= 0.0 or -np.log10(self._success_prob) >= 6:
            raise ValueError('Too low success probability')

        _false_prob = _p_00 * C_00 + _p_01 * C_01_f + _p_10 * C_10_f + _p_11 * C_11 + _p_02 * C_02 + _p_20 * C_20 + _p_12 * C_12 + _p_21 * C_21 + _p_22 * C_22

        _total_prob = self._success_prob + _false_prob

        self._state_weight: float = self._success_prob / _total_prob

        _td_00 = __S(__D_rate[_s_det](_sender_efficiency, 0, 0), __D_rate[_r_det](_receiver_efficiency, 0, 0))
        _td_01 = _p_plus * __S(__D_rate[_s_det](_sender_efficiency, 0, 0), __D_rate[_r_det](_receiver_efficiency, 1, 0)) + _p_minus * __S(__D_rate[_s_det](_sender_efficiency, 1, 0), __D_rate[_r_det](_receiver_efficiency, 0, 0))
        _td_10 = _p_plus * __S(__D_rate[_s_det](_sender_efficiency, 1, 0), __D_rate[_r_det](_receiver_efficiency, 0, 0)) + _p_minus * __S(__D_rate[_s_det](_sender_efficiency, 0, 0), __D_rate[_r_det](_receiver_efficiency, 1, 0))
        _td_11 = 0.5 * __S(__D_rate[_s_det](_sender_efficiency, 1, 0), __D_rate[_r_det](_receiver_efficiency, 1, 0)) + 0.25 * (__S(__D_rate[_s_det](_sender_efficiency, 2, 0), __D_rate[_r_det](_receiver_efficiency, 0, 0)) + __S(__D_rate[_s_det](_sender_efficiency, 0, 0), __D_rate[_r_det](_receiver_efficiency, 2, 0)))
        _td_02 = 0.5 * __S(__D_rate[_s_det](_sender_efficiency, 1, 0), __D_rate[_r_det](_receiver_efficiency, 1, 0)) + 0.25 * (__S(__D_rate[_s_det](_sender_efficiency, 2, 0), __D_rate[_r_det](_receiver_efficiency, 0, 0)) + __S(__D_rate[_s_det](_sender_efficiency, 0, 0), __D_rate[_r_det](_receiver_efficiency, 2, 0)))
        _td_20 = _td_02
        _td_12 = (3 / 8) * (__S(__D_rate[_s_det](_sender_efficiency, 0, 0), __D_rate[_r_det](_receiver_efficiency, 3, 0)) + __S(__D_rate[_s_det](_sender_efficiency, 3, 0), __D_rate[_r_det](_receiver_efficiency, 0, 0))) + 0.125 * (__S(__D_rate[_s_det](_sender_efficiency, 1, 0), __D_rate[_r_det](_receiver_efficiency, 2, 0)) + __S(__D_rate[_s_det](_sender_efficiency, 2, 0), __D_rate[_r_det](_receiver_efficiency, 1, 0)))
        _td_21 = _td_12
        _td_22 = (3 / 8) * (__S(__D_rate[_s_det](_sender_efficiency, 0, 0), __D_rate[_r_det](_receiver_efficiency, 4, 0)) + __S(__D_rate[_s_det](_sender_efficiency, 4, 0), __D_rate[_r_det](_receiver_efficiency, 0, 0))) + 0.25 * __S(__D_rate[_s_det](_sender_efficiency, 2, 0), __D_rate[_r_det](_receiver_efficiency, 2, 0))
        
        _pnr_11 = 0.5 * __S(__PNR_rate(_sender_efficiency, 1, 0), __PNR_rate(_receiver_efficiency, 1, 0)) + 0.25 * (__S(__PNR_rate(_sender_efficiency, 2, 0), __PNR_rate(_receiver_efficiency, 0, 0)) + __S(__PNR_rate(_sender_efficiency, 0, 0), __PNR_rate(_receiver_efficiency, 2, 0)))
        _pnr_02 = 0.5 * __S(__PNR_rate(_sender_efficiency, 1, 0), __PNR_rate(_receiver_efficiency, 1, 0)) + 0.25 * (__S(__PNR_rate(_sender_efficiency, 2, 0), __PNR_rate(_receiver_efficiency, 0, 0)) + __S(__PNR_rate(_sender_efficiency, 0, 0), __PNR_rate(_receiver_efficiency, 2, 0)))
        _pnr_20 = _pnr_02
        _pnr_12 = (3 / 8) * (__S(__PNR_rate(_sender_efficiency, 0, 0), __PNR_rate(_receiver_efficiency, 3, 0)) + __S(__PNR_rate(_sender_efficiency, 3, 0), __PNR_rate(_receiver_efficiency, 0, 0))) + 0.125 * (__S(__PNR_rate(_sender_efficiency, 1, 0), __PNR_rate(_receiver_efficiency, 2, 0)) + __S(__PNR_rate(_sender_efficiency, 2, 0), __PNR_rate(_receiver_efficiency, 1, 0)))
        _pnr_21 = _pnr_12
        _pnr_22 = (3 / 8) * (__S(__PNR_rate(_sender_efficiency, 0, 0), __PNR_rate(_receiver_efficiency, 4, 0)) + __S(__PNR_rate(_sender_efficiency, 4, 0), __PNR_rate(_receiver_efficiency, 0, 0))) + 0.25 * __S(__PNR_rate(_sender_efficiency, 2, 0), __PNR_rate(_receiver_efficiency, 2, 0))

        if _false_prob > 0.:
            _alpha_acc = (_p_00 * (_td_00) + _p_01 * (_td_01) + _p_10 * (_td_10) + _p_11 * (_td_11 - _pnr_11) + _p_02 * (_td_02 - _pnr_02) + _p_20 * (_td_20 - _pnr_20) + _p_12 * (_td_12 - _pnr_12) + _p_21 * (_td_21 - _pnr_21) + _p_22 * (_td_22 - _pnr_22)) / (3 * _false_prob)
        else:
            _alpha_acc = 0.

        self._sender_depol: float = (4 * self._sender_fidelity - 1) / 3
        self._receiver_depol: float = (4 * self._receiver_fidelity - 1) / 3

        _depol_prob = self._spin_photon_correlation * self._sender_depol * self._receiver_depol * 2 * np.sqrt(_p_01 * _p_10) / (_p_01 + _p_10) * self._state_weight + (1 - self._state_weight) * _alpha_acc

        self._state: np.ndarray = _depol_prob * B_0 + (1 - _depol_prob) * I_0

        _A_0 = _p_00 + 0.5 * (_p_01 + _p_10) + 0.25 * _p_11 + 0.25 * (_p_02 + _p_20) + 0.375 * (_p_12 + _p_21) + 0.375 * _p_22
        _A_1 = 0.5 * (_p_01 + _p_10) + 0.5 * _p_11 + 0.5 * (_p_02 + _p_20) + 0.125 * (_p_12 + _p_21)
        _A_2 = 0.25 * _p_11 + 0.25 * (_p_02 + _p_20) + 0.125 * (_p_12 + _p_21) + 0.25 * _p_22
        _A_3 = 0.375 * (_p_12 + _p_21)
        _A_4 = 0.375 * _p_22

        _sender_click_prob = _A_0 * __D_rate[_s_det](_sender_efficiency, 0, _sender_dark) + _A_1 * __D_rate[_s_det](_sender_efficiency, 1, _sender_dark) + _A_2 * __D_rate[_s_det](_sender_efficiency, 2, _sender_dark) + _A_3 * __D_rate[_s_det](_sender_efficiency, 3, _sender_dark) + _A_4 * __D_rate[_s_det](_sender_efficiency, 4, _sender_dark)
        _receiver_click_prob = _A_0 * __D_rate[_s_det](_receiver_efficiency, 0, _receiver_dark) + _A_1 * __D_rate[_s_det](_receiver_efficiency, 1, _receiver_dark) + _A_2 * __D_rate[_s_det](_receiver_efficiency, 2, _receiver_dark) + _A_3 * __D_rate[_s_det](_receiver_efficiency, 3, _receiver_dark) + _A_4 * __D_rate[_s_det](_receiver_efficiency, 4, _receiver_dark)

        _sender_detector_cycle = _sender_detector._duration + _sender_click_prob * (_sender_detector._quench_time + _sender_detector._dead_time + _sender_detector._after_pulse_duration)
        _receiver_detector_cycle = _receiver_detector._duration + _receiver_click_prob * (_receiver_detector._quench_time + _receiver_detector._dead_time + _receiver_detector._after_pulse_duration)

        self._sender_duration = self._sender_source_duration + _sender_channel._propagation_time + _duration + _sender_detector_cycle
        self._receiver_duration = self._receiver_source_duration + _receiver_channel._propagation_time + _duration + _receiver_detector_cycle
        self._total_duration = 2 * (_sender_channel._propagation_time + _receiver_channel._propagation_time)

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
        
        _sample_sender = 0
        if self._sender_std > 0.:
            _sample_sender = 4 * sp.stats.truncnorm.rvs(-self._sender_alpha / self._sender_std, 1 - self._sender_alpha / self._sender_std, loc=0, scale=self._sender_std) / 3
            
        _sample_receiver = 0
        if self._receiver_std > 0.:
            _sample_receiver = 4 * sp.stats.truncnorm.rvs(-self._receiver_alpha / self._receiver_std, 1 - self._receiver_alpha / self._receiver_std, loc=0, scale=self._receiver_std) / 3
        
        _depol = self._spin_photon_correlation * self._state_weight * (self._sender_depol * _sample_receiver + self._receiver_depol * _sample_sender + _sample_sender * _sample_receiver)
        
        qsys = Simulation.create_qsystem(2)
        qsys._state = self._state + _depol * B_I_0
        
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
        self._sender._channels['pc'][self._receiver_id][SEND].put(packet_r)
        self._sender._channels['pc'][self._receiver_id][RECEIVE].put(packet_s)

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
        self._sender._channels['pc'][self._receiver_id][SEND].put(packet_r)
        self._sender._channels['pc'][self._receiver_id][RECEIVE].put(packet_s)
        
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
    
    def __init__(self, _sender: Host, _receiver: int, _sim: Simulation, _sender_memory: QuantumMemory, _receiver_memory: QuantumMemory,
                 _model: L3C_Model) -> None:
        
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
        
        self._sender: Host = _sender
        self._receiver_id: int = _receiver
        self._num_sources: int = _model._num_sources
        self._sim: Simulation = _sim
        
        self._source_duration: float = _model._duration
        self._propagation_time: float = _model._qchannel._length * 5e-6
        self._success_prob: float = _model._success_prob * _sender_memory._efficiency * _receiver_memory._efficiency
        self._fidelity_variance: float = _model._fidelity_variance
        
        self._state: np.ndarray = (4 * _model._fidelity - 1) / 3 * B_0 + 4 * (1 - _model._fidelity) / 3 * I_0
        
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

        _time_samples = np.zeros(_needed) + self._sender._time + self._propagation_time
        _success_samples = np.random.uniform(0, 1, _needed) < self._success_prob
        
        if self._num_sources + 1:
            _time_samples += np.repeat(np.arange(1, _num_tries + 1).reshape(-1, 1), self._num_sources, axis=1).flatten()[:_needed] * self._source_duration
        
        for success, _time_sample in zip(_success_samples, _time_samples):
            self._creation_functions[success](_time_sample)
        
        packet.l1_success = _success_samples
        
        self._sim.schedule_event(ReceiveEvent(self._sender._time + self._propagation_time + (_num_tries - 1) * self._source_duration, self._receiver_id))
        self._sender._channels['pc'][self._receiver_id][SEND].put(packet)
    
    def create_bell_pairs(self, _requested: int=1) -> None:
        
        """
        Creates the number of requested qubits
        
        Args:
            _requested (int): number of requested qubits
        
        Returns:
            /
        """
        
        packet = Packet(self._sender.id, self._receiver_id, _requested, _requested)
        
        _time_samples = np.zeros(_requested + self._num_sources + 1) + self._sender._time + self._propagation_time
        
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
        
        self._sim.schedule_event(ReceiveEvent(self._sender._time + self._propagation_time + (_current_try - 1) * self._source_duration, self._receiver_id))
        self._sender._channels['pc'][self._receiver_id][SEND].put(packet)
