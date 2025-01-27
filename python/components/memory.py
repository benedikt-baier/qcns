
import numpy as np
from typing import List, Dict, Tuple, Union, Callable

__all__ = ['QuantumMemory']

L0 = 0
L1 = 1
L2 = 2
L3 = 3

class Qubit:
    pass

class QuantumError:
    pass

class QuantumMemory:
    
    """
    Represents a Quantum Memory
    
    Attr:
        _l0_memory (list): memory section of qubits, which have no confirmed corresponding qubit
        _l1_memory (list): memory section of qubits, which have a confirmed corresponding qubit, ready for purification
        _l2_memory (list): memory section of qubits, which are purified, but the purification was not confirmed successful
        _l3_memory (list): memory section of qubits, which are confirmed purified
        _size (int): size in qubits of memory of all four memory sections
        _efficiency (float): efficiency of extracting qubits from the memory
        _errors (list): list of errors to apply to qubits at extraction 
    """
    
    def __init__(self, _mode: str='lifo', _size: int=-1, _efficiency: float=1., _errors: List[QuantumError]=None) -> None:
        
        """
        Initializes a Quantum Memory
        
        Args:
            _size (int): size of the memory
            _efficiency (float): probability to extract qubits out of memory
            _errors (list): list of quantum errors to apply to extracted qubits
            
        Returns:
            / 
        """
        
        if not _size:
            raise ValueError('Memory size should not be 0')
        
        self._mode: str = _mode
        
        self._transform = {'fifo': self._fifo_transform, 'lifo': self._lifo_transform}

        self._size: int = _size
        self._offsets: List[int] = []
        self._efficiency: float = _efficiency
        self._errors: List[QuantumError] = _errors

        self._memory: List[List[Qubit]] = [[], [], [], []]
     
    def __len__(self) -> int:
        
        """
        Custom Length function
        
        Args:
            /
            
        Returns:
            _len (int): number of elements in the memory
        """
        
        return len(self._memory[0]) + len(self._memory[1]) + len(self._memory[2]) + len(self._memory[3])

    @property
    def size(self) -> int:
        
        """
        Returns the size of the memory
        
        Args:
            /
            
        Returns:
            size (int): size of memory
        """
        
        return self._size
    
    def change_size(self, _size: int) -> None:
        
        """
        Changes the size of the memory
        
        Args:
            _size (int): new size of memory
            
        Returns:
            /
        """
        
        if not _size:
            raise ValueError('Size cannot be zero')
        
        self._size = _size
    
    def has_space(self, _num_qubits: int=1) -> bool:
        
        """
        Check whether the number of qubits can fit into the memory
        
        Args:
            _num_qubits (int): number of qubits to fit into space
            
        Returns:
            _has_space (bool): whether qubits can fit into memory
        """
        
        if not self._size + 1:
            return True
        
        return self.remaining_space() >= _num_qubits
    
    def remaining_space(self) -> int:
        
        """
        Returns the remaining size of the memory
        
        Args:
            /
            
        Returns:
            _size (int): the remaining size left in the memory
        """
        
        if not self._size + 1:
            return 2147483647
        
        return self._size - len(self)
    
    def num_qubits(self, _store: int) -> int:
        
        """
        Returns the number of qubits in the L0 store
        
        Args:
            /
            
        Returns:
            num_qubits (int): number of qubits in the L0 store
        """
        
        return len(self._memory[_store])
    
    def store_qubit(self, _store: int, _qubit: Qubit, _index: int, _time: float) -> None:
        
        """
        Stores a qubit in the L0 memory
        
        Args:
            _qubit (Qubit): qubit to store
            _index (int): index at which to store qubit
            _time (float): time stamp
            
        Returns:
            /
        """
        
        if self._size + 1 > 0 and len(self) + 1 > self._size:
            raise ValueError('Memory exceeds size limit')
                
        if not (_index + 1):
            self._memory[_store].append((_qubit, _time))
            return
        
        self._memory[_store].insert(_index, (_qubit, _time))
        
    def retrieve_qubit(self, _store: int, _index: int, _time: float, _offset_index: int=None) -> Union[Qubit, None]:
    
        """
        Retrieves a qubit from the L0 memory
        
        Args:
            _index (int): index of qubit to retrieve
            _time (float): time of retrieval
            
        Returns:
            _qubit (Qubit/None): qubit to retrieve
        """
    
        if not self._memory[_store]:
            raise ValueError('No Qubit in memory')
        
        if _index is None:
            _index = 0
        
        _index = self._transform[self._mode](_index + sum(self._offsets[:_offset_index]))
        
        _qubit, _store_time = self._memory[_store].pop(_index)
        
        if np.random.uniform(0, 1) > self._efficiency:
            return None
        
        _diff = _time - _store_time
        if _diff <= 0.:
            return _qubit
        
        for error in self._errors:
            _qubit = error.apply(_qubit, _diff)
    
        return _qubit
    
    def peek_qubit(self, _store: int, _index: int) -> Qubit:
        
        """
        Takes a look at the qubit in L0 memory at index without extracting it
        
        Args:
            _index (int): index of qubit
            
        Returns:
            _qubit (Qubit): qubit
        """
        
        if not self._memory[_store]:
            raise ValueError('No Qubit in memory')
        
        if _index is None:
            _index = self._transform[self._mode](0) # + or - offset[_offset_index] depending on the mode
        
        _qubit, _ = self._memory[_store][_index]
        
        return _qubit
    
    def move_qubits(self, _src_store: int, _dst_store: int, _indices: List[bool]) -> None:
 
        """
        Moves qubits given indices from the L0 memory to the L1 memory
        
        Args:
            _indices (list): list of bool whether to remove qubit or not
            
        Returns:
            /
        """
 
        for index in _indices:
            qubit, time = self._memory[_src_store].pop(0)
            if index:
                self.store_qubit(_dst_store, qubit, -1, time)
    
    def discard_qubits(self, _store: int) -> None:
        
        """
        Discards all qubits in L0 memory
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._memory[_store] = []
    
    def purify(self, _index_src: int, _index_dst: int, _time: float) -> Tuple[Qubit, Qubit]:
        
        """
        Retrieves the two qubits for purification given the indices
        
        Args:
            _index_src (int): index of src qubit
            _index_dst (int): index of dst qubit
            _time (float): time to retrieve qubits
        
        Returns:
            qubit_src (Qubit): src qubit
            qubit_dst (Qubit): dst qubit
        """
        
        if len(self._memory[1]) < 2:
            raise ValueError('Purification needs at most 2 qubits')
        
        if _index_src is None:
            _index_src = self._transform[self._mode](0)
        if _index_dst is None:
            _index_dst = self._transform[self._mode](0)
        
        return self.retrieve_qubit(L1, _index_src, _time), self.retrieve_qubit(L1, _index_dst, _time)
    
    def estimate_fidelity(self, _store: int, _index: int, _time: float) -> float:
        
        """
        Estimates the fidelity of a qubit in the L0 memory given current time
        
        Args:
            _index (int): index of qubit
            _time (float): time of storage access
            
        Returns:
            _fidelity (float): estimated fidelity
        """
        
        if not self._errors:
            return 1
        
        if _index is None:
            _index = self._indices[self._mode]
        
        _, _store_time = self._memory[_store][_index]

        _diff = _time - _store_time

        depolar = 1 - np.exp(-(_diff / self._errors[0].depolar_time))
        dephase = 0.5 * (1 - np.exp(-_diff * (1/self._errors[0].dephase_time - 1/(2*self._errors[0].depolar_time))))
        
        return (1 - depolar / 2) * (1 - dephase)

    def add_offset(self, _offset: int) -> None:
        
        """
        Adds an offset to the existing offset
        
        Args:
            _offset (int): L3 offset to add
            
        Returns:
            /
        """
        
        self._offsets.append(_offset)
        
        return len(self._offsets) - 1
    
    def remove_offset(self, _index: int) -> None:
        
        """
        Removes an offset from the offset array
        
        Args:
            _index (int): index to remove
            
        Returns:
            /
        """
        
        self._offsets.pop(_index)
    
    def _fifo_transform(self, _index: int) -> int:
        
        return _index

    def _lifo_transform(self, _index: int) -> int:
        
        return -1 * _index - 1
    