import numpy as np
from queue import Queue
from typing import Union, List

from python.components.qubit import Qubit, remove_qubits
from code.qcns.python.components.photon_source import *
from python.components.error import *

__all__ = ['QChannel', 'PChannel']

class QuantumError:
    pass

class QChannel:
    
    def __init__(self, _length: float=0.0, _errors: List[QuantumError]=None, _lose_qubits: bool=False, _attenuation_coefficient: float=-0.016) -> None:
        
        """
        Instantiates a quantum channel
        
        Args:
            _length (float): length of the channel in Km
            _errors (list): list of errors to apply to qubits in channel
            
        Returns:
            /
        """
        
        self._signal_time: float = _length * (5e-6)
        self._lose_prob: float = 0.
        self._lose_qubits: bool = _lose_qubits
        if self._lose_qubits:
            self._lose_prob = 1 - 10 ** (_length * _attenuation_coefficient)
        self._queue = Queue()
        self._errors = _errors
        if _errors is None:
            self._errors = []
    
    def set_lose_prob(self, _lose_prob: float=0.) -> None:
        
        """
        Sets the lose probability for the channel
        
        Args:
            _lose_prob (float): probability to lose qubit
            
        Returns:
            /
        """
        
        if not (0. <= _lose_prob <= 1.0):
            raise ValueError('Probability should be between 0 and 1')
        
        self._lose_prob = _lose_prob
        self._lose_qubits = True
        
    def lose_qubit(self, _qubit: Qubit) -> Union[Qubit, None]:
        
        """
        Loses a qubit based on the lose probability
        
        Args:
            _qubit (Qubit): _qubit to lose
            
        Returns:
            _qubit (Qubit): lost qubit
        """
        
        if np.random.uniform(0, 1) < self._lose_prob:
            remove_qubits([_qubit])
            return None
        return _qubit
    
    def put(self, _qubit: Qubit) -> None:
        
        """
        Sends a qubit through the channel
        
        Args:
            _qubit (Qubit): qubit to send
            
        Returns:
            /
        """
        
        self._queue.put(_qubit)
        
    def get(self) -> Union[Qubit, None]:

        """
        Receives a qubit from the channel
        
        Args:
            /
            
        Returns:
            _qubit (Qubit/None): qubit to receive
        """
        
        _qubit = self._queue.get()
        
        if self._lose_qubits:
            _qubit = self.lose_qubit(_qubit)
            
        if _qubit is None:
            return _qubit
        
        for error in self._errors:
            _qubit = error.apply(_qubit)
    
        return _qubit
       
class PChannel:
    
    def __init__(self, _length: float=0.0) -> None:
        
        """
        Instaniates a classical packet channel
        
        Args:
            _length (float): length of channel in Km
            pulse_length (float): pulse length

        Returns:
            /
        """
        
        self._signal_time = _length * (5e-6) + 16e-6
        self._queue = Queue()
        
    def put(self, _packet: Packet) -> None:
        
        """
        Sends a packet through the channel
        
        Args:
            _packet (Packet): packet to transmit
            
        Returns:
            /
        """
        
        self._queue.put(_packet)
        
    def get(self) -> Packet:
        
        """
        Receives a packet from the channel
        
        Args:
            /
            
        Returns:
            packet (Packet): received packet
        """
            
        return self._queue.get()
        