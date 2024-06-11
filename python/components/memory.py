
import numpy as np
from typing import List, Tuple, Union

__all__ = ['QuantumMemory']

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
    
    def __init__(self, _size: int=-1, _efficiency: float=1., _errors: List[QuantumError]=None) -> None:
        
        self._l0_memory: List[Qubit] = []
        self._l1_memory: List[Qubit] = []
        self._l2_memory: List[Qubit] = []
        self._l3_memory: List[Qubit] = []
        
        self._size: int = _size
        self._efficiency: float = _efficiency
        self._errors: List[QuantumError] = _errors
     
    def __len__(self) -> int:
        
        """
        Custom Length function
        
        Args:
            /
            
        Returns:
            _len (int): number of elements in the memory
        """
        
        return len(self._l0_memory) + len(self._l1_memory) + len(self._l2_memory) + len(self._l3_memory)
        
    def l0_num_qubits(self) -> int:
        
        """
        Returns the number of qubits in the L0 store
        
        Args:
            /
            
        Returns:
            l0_num_qubits (int): number of qubits in the L0 store
        """
        
        return len(self._l0_memory)
    
    def l1_num_qubits(self) -> int:
        
        """
        Returns the number of qubits in the L1 store
        
        Args:
            /
            
        Returns:
            l1_num_qubits (int): number of qubits in the L1 store
        """
        
        return len(self._l1_memory)
    
    def l2_num_qubits(self) -> int:
        
        """
        Returns the number of qubits in the L2 store
        
        Args:
            /
            
        Returns:
            l2_num_qubits (int): number of qubits in the L2 store
        """
        
        return len(self._l2_memory)
    
    def l3_num_qubits(self) -> int:
        
        """
        Returns the number of qubits in the L3 store
        
        Args:
            /
            
        Returns:
            l3_num_qubits (int): number of qubits in the L3 store
        """
        
        return len(self._l3_memory)
    
    def l0_store_qubit(self, _qubit: Qubit, _index: int, _time: float) -> None:
        
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
            self._l0_memory.append((_qubit, _time))
            return
        
        self._l0_memory.insert(_index, (_qubit, _time))
    
    def l1_store_qubit(self, _qubit: Qubit, _index: int, _time: float) -> None:
        
        """
        Stores a qubit in the L1 memory
        
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
            self._l1_memory.append((_qubit, _time))
            return
        
        self._l1_memory.insert(_index, (_qubit, _time))
    
    def l2_store_qubit(self, _qubit: Qubit, _index: int, _time: float) -> None:
        
        """
        Stores a qubit in the L2 memory
        
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
            self._l2_memory.append((_qubit, _time))
            return
        
        self._l2_memory.insert(_index, (_qubit, _time))
    
    def l3_store_qubit(self, _qubit: Qubit, _index: int, _time: float) -> None:
        
        """
        Stores a qubit in the L3 memory
        
        Args:
            _qubit (Qubit): qubit to store
            index (int): index at which to store qubit
            _time (float): time stamp
            
        Returns:
            /
        """
        
        if self._size + 1 > 0 and len(self) + 1 > self._size:
            raise ValueError('Memory exceeds size limit')
        
        if not (_index + 1):
            self._l3_memory.append((_qubit, _time))
            return
        
        self._l3_memory.insert(_index, (_qubit, _time))
        
    def l0_retrieve_qubit(self, _index: int, _time: float) -> Union[Qubit, None]:
    
        """
        Retrieves a qubit from the L0 memory
        
        Args:
            _index (int): index of qubit to retrieve
            _time (float): time of retrieval
            
        Returns:
            _qubit (Qubit/None): qubit to retrieve
        """
    
        if not self._l0_memory:
            raise ValueError('No Qubit in memory')
            
        _qubit, _store_time = self._l0_memory.pop(_index)
        
        if np.random.uniform(0, 1) > self._efficiency:
            return None
        
        _diff = _time - _store_time
        if _diff <= 0.:
            return _qubit
        
        for error in self._errors:
            _qubit = error.apply(_qubit, _diff)
    
        return _qubit
          
    def l1_retrieve_qubit(self, _index: int, _time: float) -> Union[Qubit, None]:
    
        """
        Retrieves a qubit from the L1 memory
        
        Args:
            _index (int): index of qubit to retrieve
            _time (float): time of retrieval
            
        Returns:
            _qubit (Qubit/None): qubit to retrieve
        """
    
        if not self._l1_memory:
            raise ValueError('No Qubit in memory')
            
        _qubit, _store_time = self._l1_memory.pop(_index)
        
        if np.random.uniform(0, 1) > self._efficiency:
            return None
        
        _diff = _time - _store_time
        if _diff <= 0.:
            return _qubit
        
        for error in self._errors:
            _qubit = error.apply(_qubit, _diff)
    
        return _qubit
    
    def l2_retrieve_qubit(self, _index: int, _time: float) -> Union[Qubit, None]:
    
        """
        Retrieves a qubit from the L2 memory
        
        Args:
            _index (int): index of qubit to retrieve
            _time (float): time of retrieval
            
        Returns:
            _qubit (Qubit/None): qubit to retrieve
        """
    
        if not self._l2_memory:
            raise ValueError('No Qubit in memory')
            
        _qubit, _store_time = self._l2_memory.pop(_index)
        
        if np.random.uniform(0, 1) > self._efficiency:
            return None
        
        _diff = _time - _store_time
        if _diff <= 0.:
            return _qubit
        
        for error in self._errors:
            _qubit = error.apply(_qubit, _diff)
    
        return _qubit
    
    def l3_retrieve_qubit(self, _index: int, _time: float) -> Union[Qubit, None]:
    
        """
        Retrieves a qubit from the L3 memory
        
        Args:
            _index (int): index of qubit to retrieve
            _time (float): time of retrieval
            
        Returns:
            _qubit (Qubit/None): qubit to retrieve
        """
    
        if not self._l3_memory:
            raise ValueError('No Qubit in memory')
            
        _qubit, _store_time = self._l3_memory.pop(_index)
        
        if np.random.uniform(0, 1) > self._efficiency:
            return None
        
        _diff = _time - _store_time
        if _diff <= 0.:
            return _qubit
        
        for error in self._errors:
            _qubit = error.apply(_qubit, _diff)
    
        return _qubit  
    
    def l0_retrieve_qubit_prob(self, _index: int, _time: float) -> Qubit:
        
        """
        Retrieves a qubit without storage inefficiencies
        
        Args:
            /
            
        Returns:
            /
        """
        
        _qubit, _store_time = self._l0_memory.pop(_index)
        
        _diff = _time - _store_time
        if _diff <= 0.:
            return _qubit
        
        for error in self._errors:
            _qubit = error.apply(_qubit, _diff)
            
        return _qubit
    
    def l0_peek_qubit(self, _index: int) -> Qubit:
        
        """
        Takes a look at the qubit in L0 memory at index without extracting it
        
        Args:
            _index (int): index of qubit
            
        Returns:
            _qubit (Qubit): qubit
        """
        
        if not self._l0_memory:
            return None
        
        _qubit, _ = self._l0_memory[_index]
        
        return _qubit
    
    def l1_peek_qubit(self, _index: int) -> Qubit:
        
        """
        Takes a look at the qubit in L1 memory at index without extracting it
        
        Args:
            _index (int): index of qubit
            
        Returns:
            _qubit (Qubit): qubit
        """
        
        if not self._l1_memory:
            return None
        
        _qubit, _ = self._l1_memory[_index]
        
        return _qubit
    
    def l2_peek_qubit(self, _index: int) -> Qubit:
        
        """
        Takes a look at the qubit in L2 memory at index without extracting it
        
        Args:
            _index (int): index of qubit
            
        Returns:
            _qubit (Qubit): qubit
        """
        
        if not self._l2_memory:
            return None
        
        _qubit, _ = self._l2_memory[_index]
        
        return _qubit
    
    def l3_peek_qubit(self, _index: int) -> Qubit:
        
        """
        Takes a look at the qubit in L3 memory at index without extracting it
        
        Args:
            _index (int): index of qubit
            
        Returns:
            _qubit (Qubit): qubit
        """
        
        if not self._l3_memory:
            return None
        
        _qubit, _ = self._l3_memory[_index]
        
        return _qubit
    
    def l0_move_qubits_l1(self, _indices: List[bool]) -> None:
 
        """
        Moves qubits given indices from the L0 memory to the L1 memory
        
        Args:
            _indices (list): list of bool whether to remove qubit or not
            
        Returns:
            /
        """
 
        for index in _indices:
            qubit, time = self._l0_memory.pop(0)
            if index:
                self.l1_store_qubit(qubit, -1, time)
    
    def l1_move_qubits_l2(self, _indices: List[bool]) -> None:
        
        """
        Moves qubits given indices from the L1 memory to the L2 memory
        
        Args:
            _indices (list): list of bool whether to remove qubit or not
            
        Returns:
            /
        """
        
        for index in _indices:
            qubit, time = self._l1_memory.pop(0)
            if index:
                self.l2_store_qubit(qubit, -1, time)
    
    def l2_move_qubits_l3(self, _indices: List[bool]) -> None:
        
        """
        Moves qubits given indices from the L2 memory to the L3 memory
        
        Args:
            _indices (list): list of bool whether to remove qubit or not
            
        Returns:
            /
        """
    
        for index in _indices:
            qubit, time = self._l2_memory.pop(0)
            if index:
                self.l3_store_qubit(qubit, -1, time)

    def l3_move_qubits_l1(self, _indices: List[bool]) -> None:
        
        """
        Moves qubits given indices from the L3 memory to the L1 memory
        
        Args:
            _indices (list): list of bool whether to remove qubit or not
            
        Returns:
            /
        """
        
        for index in _indices:
            qubit, time = self._l3_memory.pop(0)
            if index:
                self.l1_store_qubit(qubit, -1, time)
    
    def l0_discard_qubits(self) -> None:
        
        """
        Discards all qubits in L0 memory
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._l0_memory = []
    
    def l1_discard_qubits(self) -> None:
        
        """
        Discards all qubits in L1 memory
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._l1_memory = []
    
    def l2_discard_qubits(self) -> None:
        
        """
        Discards all qubits in L2 memory
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._l2_memory = []
        
    def l3_discard_qubits(self) -> None:
        
        """
        Discards all qubits in L3 memory
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._l3_memory = []
    
    def l2_purify(self, _index_src: int, _index_dst: int, _time: float) -> Tuple[Qubit, Qubit]:
        
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
        
        if len(self._l1_memory) < 2:
            raise ValueError('Purification needs at most 2 qubits')
        
        return self.l1_retrieve_qubit(_index_src, _time), self.l1_retrieve_qubit(_index_dst, _time)
    
    def l0_estimate_fidelity(self, _index: int, _time: float) -> float:
        
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
        
        _, _store_time = self._l0_memory[_index]

        _diff = _time - _store_time

        depolar = 1 - np.exp(-(_diff / self._errors[0].depolar_time))
        dephase = 0.5 * (1 - np.exp(-_diff * (1/self._errors[0].dephase_time - 1/(2*self._errors[0].depolar_time))))
        
        return (1 - depolar / 2) * (1 - dephase)
    
    def l1_estimate_fidelity(self, _index: int, _time: float) -> float:
        
        """
        Estimates the fidelity of a qubit in the L1 memory given current time
        
        Args:
            _index (int): index of qubit
            _time (float): time of storage access
            
        Returns:
            _fidelity (float): estimated fidelity
        """
        
        if not self._errors:
            return 1
        
        _, _store_time = self._l1_memory[_index]

        _diff = _time - _store_time

        depolar = 1 - np.exp(-(_diff / self._errors[0].depolar_time))
        dephase = 0.5 * (1 - np.exp(-_diff * (1/self._errors[0].dephase_time - 1/(2*self._errors[0].depolar_time))))
        
        return (1 - depolar / 2) * (1 - dephase)
    
    def l2_estimate_fidelity(self, _index: int, _time: float) -> float:
        
        """
        Estimates the fidelity of a qubit in the L2 memory given current time
        
        Args:
            _index (int): index of qubit
            _time (float): time of storage access
            
        Returns:
            _fidelity (float): estimated fidelity
        """
        
        if not self._errors:
            return 1
        
        _, _store_time = self._l2_memory[_index]

        _diff = _time - _store_time

        depolar = 1 - np.exp(-(_diff / self._errors[0].depolar_time))
        dephase = 0.5 * (1 - np.exp(-_diff * (1/self._errors[0].dephase_time - 1/(2*self._errors[0].depolar_time))))
        
        return (1 - depolar / 2) * (1 - dephase)
    
    def l3_estimate_fidelity(self, _index: int, _time: float) -> float:
        
        """
        Estimates the fidelity of a qubit in the L3 memory given current time
        
        Args:
            _index (int): index of qubit
            _time (float): time of storage access
            
        Returns:
            _fidelity (float): estimated fidelity
        """
        
        if not self._errors:
            return 1
        
        _, _store_time = self._l3_memory[_index]

        _diff = _time - _store_time

        depolar = 1 - np.exp(-(_diff / self._errors[0].depolar_time))
        dephase = 0.5 * (1 - np.exp(-_diff * (1/self._errors[0].dephase_time - 1/(2*self._errors[0].depolar_time))))
        
        return (1 - depolar / 2) * (1 - dephase)