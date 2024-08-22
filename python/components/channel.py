import numpy as np
from queue import Queue
from python.components.qubit import remove_qubits

from typing import List, Union

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
        _out_coupling_prob (float): probability of successfully coupling qubit out of channel
        _out_prob (float): probability of successfully transmitting qubit through channel
        _errors (list): list of errors on the channel
        _channel (queue): queue representing the channel
    """
    
    def __init__(self, _length: float=0., _attenuation_coefficient: float=-0.016, _in_coupling_prob: float=1., _out_coupling_prob: float=1., _errors: List[QuantumError]=None) -> None:
        
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
        
        self._length: float = _length
        self._sending_time: float = _length * 5e-6
        self._lose_prob: float = 10 ** (_length * _attenuation_coefficient)
        self._in_coupling_prob: float = _in_coupling_prob
        self._out_coupling_prob: float = _out_coupling_prob
        self._out_prob: float = self._lose_prob * self._out_coupling_prob
        self._errors: List[QuantumError] = _errors
        if _errors is None:
            self._errors = []
        self._channel: Queue = Queue()
    
    def empty(self) -> bool:
        
        """
        Checks wether the channel is empty
        
        Args:
            /
            
        Returns:
            _empty (bool): whether channel is empty
        """
        
        return self._channel.empty()

    def put(self, _qubit: Qubit) -> None:
        
        """
        Sends a qubit through the channel
        
        Args:
            _qubit (Qubit): qubit to send
            
        Returns:
            /
        """
        
        if np.random.uniform(0, 1) < self._in_coupling_prob:
            remove_qubits([_qubit])
            self._channel.put(None)
            return
        
        self._channel.put(_qubit)

    def get(self) -> Union[Qubit, None]:
        
        """
        Receives a qubit from the channel
        
        Args:
            /
            
        Returns:
            _qubit (Qubit): received qubit
        """
        
        _qubit = self._channel.get()
        
        if _qubit is None:
            return _qubit
        
        if np.random.uniform(0, 1) < self._out_prob:
            remove_qubits([_qubit])
            return None
        
        for error in self._errors:
            _qubit = error.apply(_qubit)
        
        return _qubit

class PChannel:
    
    """
    Represents a packet channel
    
    Attr:
        _signal_time (float): time to send packet through channel
        _channel (queue): queue representing the channel
    """
    
    def __init__(self, _length: float=0.) -> None:
        
        """
        Initializesa packet channel
        
        Args:
            _length (float): length of the channel
            
        Returns:
            /
        """
        
        self._signal_time: float = _length * (5e-6)
        self._channel: Queue = Queue()
    
    def empty(self) -> bool:
        
        """
        Checks wether the channel is empty
        
        Args:
            /
            
        Returns:
            _empty (bool): whether channel is empty
        """
        
        return self._channel.empty()
    
    def put(self, _packet: Packet) -> None:
        
        """
        Sends a packet through the channel
        
        Args:
            _packet (Packet): packet to send
            
        Returns:
            /
        """
        
        self._channel.put(_packet)
        
    def get(self) -> Packet:
        
        """
        Receives a packet from the channel
        
        Args:
            /
            
        Returns:
            _packet (Packet): received packet
        """
        
        return self._channel.get()