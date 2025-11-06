import numpy as np
from heapq import heappop, heappush

from typing import List, Tuple

from qcns.python.components.qubit.qubit import remove_qubits
from qcns.python.components.hardware.connection import QChannel_Model, PChannel_Model

__all__ = ['QChannel', 'PChannel']

class Qubit:
    pass

class Packet:
    pass

class QuantumError:
    pass

class QChannel:
    
    """
    Represents a quantum channel
    
    Attr:
        _length (float): length of the channel
        _sending_time (float): time to transmit qubit through channel
        _lose_prob (float): probability of losing qubit in channel
        _in_coupling_prob (float): probability of successfully inserting qubit in channel
        _out_prob (float): probability of successfully transmitting qubit through channel
        _errors (list): list of errors on the channel
        _channel (queue): queue representing the channel
    """
    
    def __init__(self, _model: QChannel_Model, _errors: List[QuantumError]) -> None:
        
        """
        Initializes a quantum channel
        
        Args:
            _length (float): length of the channel
            _attenuation_coefficient (float): attenuation coefficient of the channel
            _in_coupling_prob (float): probability of coupling photon into the channel
            _out_coupling_prob (float): probability of coupling photon out of the channel
            _errors (list): list of errors to apply to qubits
            
        Returns:
            /
        """
        
        self._propagation_time: float = _model._length * 5e-6
        self._in_coupling: float = _model._in_coupling
        self._out_prob: float = 10 ** (_model._length * _model._attenuation) * _model._out_coupling
        self._errors: List[QuantumError] = _errors
        self._channel: List[Tuple[float, Qubit]] = []
    
    def empty(self) -> bool:
        
        """
        Checks wether the channel is empty
        
        Args:
            /
            
        Returns:
            _empty (bool): whether channel is empty
        """
        
        return self._channel.empty()

    def put(self, _qubit: Qubit, _arrival_time: float) -> None:
        
        """
        Sends a qubit through the channel
        
        Args:
            _qubit (Qubit): qubit to send
            
        Returns:
            /
        """
        
        if np.random.uniform(0, 1) > self._in_coupling:
            remove_qubits([_qubit])
            heappush(self._channel, (_arrival_time, None))
            return
        
        heappush(self._channel, (_arrival_time, _qubit))

    def get(self) -> Qubit | None:
        
        """
        Receives a qubit from the channel
        
        Args:
            /
            
        Returns:
            _qubit (Qubit): received qubit
        """
        
        _, _qubit = heappop(self._channel)
        
        if _qubit is None:
            return _qubit
        
        if np.random.uniform(0, 1) > self._out_prob:
            remove_qubits([_qubit])
            return None
        
        for error in self._errors:
            _qubit = error.apply(_qubit, self._sending_time)
        
        return _qubit

class PChannel:
    
    """
    Represents a packet channel
    
    Attr:
        _signal_time (float): time to send packet through channel
        _channel (queue): queue representing the channel
    """
    
    def __init__(self, _model: PChannel_Model) -> None:
        
        """
        Initializesa packet channel
        
        Args:
            _length (float): length of the channel
            
        Returns:
            /
        """
        
        self._propagation_time: float = _model._length * 5e-6
        self._data_rate: float = _model._data_rate
        self._channel: List[Tuple[float, Packet]] = []
    
    def empty(self) -> bool:
        
        """
        Checks wether the channel is empty
        
        Args:
            /
            
        Returns:
            _empty (bool): whether channel is empty
        """
        
        return not self._channel
    
    def _sending_time(self, _packet_length: float) -> float:
        
        if self._data_rate < 0:
            return 0.
        
        return _packet_length / self._data_rate
    
    def put(self, _packet: Packet, _arrival_time: float) -> None:
        
        """
        Sends a packet through the channel
        
        Args:
            _packet (Packet): packet to send
            
        Returns:
            /
        """
        
        heappush(self._channel, (_arrival_time, _packet))
        
    def get(self) -> Packet:
        
        """
        Receives a packet from the channel
        
        Args:
            /
            
        Returns:
            _packet (Packet): received packet
        """
        
        _, _packet = heappop(self._channel)
        
        return _packet