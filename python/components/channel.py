import numpy as np
from queue import Queue

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
        _coupling_prob (float): probability of successfully inserting qubit in channel
        _errors (list): list of errors on the channel
        _channel (queue): queue representing the channel
    """
    
    def __init__(self, _length: float=0., _attenuation_coefficient: float=-0.016, _coupling_prob: float=1., _errors: List[QuantumError]=None) -> None:
        
        """
        Initializes a quantum channel
        
        Args:
            _length (float): length of the channel
            _attenuation_coefficient (float): attenuation coefficient of the channel
            _coupling_prob (float): probability of coupling photon into the channel
            _errors (list): list of errors to apply to qubits
            
        Returns:
            /
        """
        
        self._length: float = _length
        self._sending_time: float = _length * 5e-6
        self._lose_prob: float = 10 ** (_length * _attenuation_coefficient)
        self._coupling_prob: float = _coupling_prob
        self._errors: List[QuantumError] = _errors
        self._channel: Queue = Queue()
    
    def set_coupling_prob(self, _prob: float=0.) -> None:
        
        """
        Sets the coupling probability of the channel
        
        Args:
            _prob (float): new coupling probability
            
        Returns:
            /
        """
        
        if not (0 <= _prob <= 1.):
            raise ValueError('Probability should be between 0 and 1')
        
        self._coupling_prob = _prob
    
    def set_lose_prob(self, _prob: float=0.) -> None:
        
        """
        Sets the lose probability of the channel
        
        Args:
            _prob (float): new coupling probability
            
        Returns:
            /
        """
        
        if not (0 <= _prob <= 1.):
            raise ValueError('Probability should be between 0 and 1')
        
        self._lose_prob = _prob
    
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
        Sends a qubit through the channel without coupling errors
        
        Args:
            _qubit (Qubit): qubit to send
            
        Returns:
            /
        """
        
        self._channel.put(_qubit)

    def put_prob(self, _qubit: Qubit) -> None:
        
        """
        Sends a qubit through the channel with coupling errors
        
        Args:
            _qubit (Qubit): qubit to send
            
        Returns:
            /
        """
        
        if np.random.uniform(0, 1) > self._coupling_prob:
            remove_qubits([_qubit])
            self._channel.put(None)
            return
        
        self._channel.put(_qubit)

    def get(self) -> Union[Qubit, None]:
        
        """
        Receives a qubit from the channel without losing the qubit
        
        Args:
            /
            
        Returns:
            _qubit (Qubit): received qubit
        """
        
        _qubit = self._channel.get()
        
        if _qubit is None:
            return _qubit
        
        for error in self._errors:
            _qubit = error.apply(_qubit)
        
        return _qubit

    def get_prob(self) -> Union[Qubit, None]:
        
        """
        Receives a qubit from the channel with losing the qubit
        
        Args:
            /
            
        Returns:
            _qubit (Qubit): received qubit
        """
        
        _qubit = self._channel.get()
        
        if _qubit is None:
            return _qubit
        
        if np.random.uniform(0, 1) < self._lose_prob:
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
        
        self._signal_time: float = _length * (5e-6)# + 16e-6
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