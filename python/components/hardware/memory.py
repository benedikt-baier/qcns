
import numpy as np
from copy import deepcopy
from qcns.python.components.qubit.qubit import remove_qubits
from math import comb
from typing import List, Dict, Tuple, Union, Callable

from qcns.python.components.qubit.error import pauli_error

__all__ = ['PQM_Model', 'LQM_Model', 'PhysicalQuantumMemory', 'LogicalQuantumMemory']

L0 = 0
L1 = 1
L2 = 2
L3 = 3

class Qubit:
    pass

class QuantumError:
    pass

class PQM_Model:
    
    def __init__(self, size: int=-1, extraction_mode: str='lifo', efficiency: float=1., memory_errors: List[QuantumError]=None):
        
        if size == 0 or size < -2:
            raise ValueError(f'Size should either be -1 or positive: {size}')
        
        if extraction_mode not in ['lifo', 'fifo']:
            raise ValueError(f'Extraction Mode should be LiFo or FiFo')
        
        if not (0. <= efficiency <= 1.):
            raise ValueError(f'Efficiency should be between 0 and 1: {efficiency}')
        
        self._memory_type: str = 'pqm'
        
        self._size: int = size
        self._extraction_mode: str = extraction_mode
        self._efficiency: float = efficiency
        self._memory_errors: List[QuantumError] = memory_errors
        
        if self._memory_errors is None:
            self._memory_errors = []
            
        if not isinstance(self._memory_errors, list):
            self._memory_errors = [self._memory_errors]
    
class LQM_Model:
    
    def __init__(self, size: int=-1, extraction_mode: str='lifo', efficiency: float=1.0, logical_mode: str='deterministic', 
                code_length: int=5, x_errors: int=1, z_errors: int=1, 
                correction_frequency: float=1e-3, 
                depolarization_time: float=1e-3, dephasing_time: float=1e-3,
                measurement_error_rate: float=1e-3, extraction_error_rate: float=1e-3, 
                x_error_rate: float=1e-3, z_error_rate: float=1e-3, repetitions: int=1):
        
        if size == 0 or size < -2:
            raise ValueError(f'Size should either be -1 or positive: {size}')
        
        if extraction_mode not in ['lifo', 'fifo']:
            raise ValueError(f'Extraction Mode should be LiFo or FiFo')
        
        if not (0. <= efficiency <= 1.):
            raise ValueError(f'Efficiency should be between 0 and 1: {efficiency}')
        
        if logical_mode not in ['deterministic', 'stochastic']:
            raise ValueError(f'Logical Mode should be either deterministic or stochastic: {logical_mode}')
        
        if code_length < 1:
            raise ValueError(f'Code Lnegth should be positive: {code_length}')
        
        if x_errors < 0:
            raise ValueError(f'X Errors should be positive: {x_errors}')
        
        if z_errors < 0:
            raise ValueError(f'Z Errors should be positive: {z_errors}')
        
        if correction_frequency < 0.:
            raise ValueError(f'Correction Frequency should be positive: {correction_frequency}')
        
        if depolarization_time < 0.:
            raise ValueError(f'Depolarization Time should be positive: {depolarization_time}')
        
        if dephasing_time < 0.:
            raise ValueError(f'Dephasing Time should be positive: {dephasing_time}')
        
        if measurement_error_rate < 0.:
            raise ValueError(f'Measurement Error Rate should be positive: {measurement_error_rate}')
        
        if extraction_error_rate < 0.:
            raise ValueError(f'Extraction Error Rate should be positive: {extraction_error_rate}')
        
        if x_error_rate < 0.:
            raise ValueError(f'X Error Rate should be positive: {x_error_rate}')
        
        if z_error_rate < 0.:
            raise ValueError(f'Z Error Rate should be positive: {z_error_rate}')
        
        if repetitions < 1:
            raise ValueError(f'Repetitions should be at least one: {repetitions}')
         
        self._memory_type: str = 'lqm'
        
        self._size: int = size
        self._extraction_mode: str = extraction_mode
        self._efficiency: float = efficiency
        
        self._logical_mode: str = logical_mode
        
        self._code_length: int = code_length
        self._x_errors: int = x_errors
        self._z_errors: int = z_errors
        
        self._depolarization_time: float = depolarization_time
        self._dephasing_time: float = dephasing_time
        self._measurement_error_rate: float = measurement_error_rate
        self._extraction_error_rate: float = extraction_error_rate
        self._x_error_rate: float = x_error_rate
        self._z_error_rate: float = z_error_rate
        self._repetitions: int = repetitions
        
        self._correction_frequency: float = correction_frequency

class QuantumMemory:
    
    """
    Represents a Quantum Memory
    
    Attr:
        _mode (str): mode in which the qubits are extracted fifo or lifo
        _transform (dict): convenience functions for transforming indices based on the mode
        _size (int): size in qubits of memory of all four memory sections
        _offsets (list): offsets to add to L3 memory indices
        _efficiency (float): efficiency of extracting qubits from the memory
        _errors (list): list of errors to apply to qubits at extraction 
        _l0_memory (list): memory section of qubits, which have no confirmed corresponding qubit
        _l1_memory (list): memory section of qubits, which have a confirmed corresponding qubit, ready for purification
        _l2_memory (list): memory section of qubits, which are purified, but the purification was not confirmed successful
        _l3_memory (list): memory section of qubits, which are confirmed purified
    """
    
    def __init__(self, _size: int, _extraction_mode: str, _efficiency: float) -> None:
        
        """
        Initializes a Quantum Memory
        
        Args:
            _mode (str): mode in which the qubits are extracted fifo or lifo
            _size (int): size of the memory
            _efficiency (float): probability to extract qubits out of memory
            _errors (list): list of quantum errors to apply to extracted qubits
            
        Returns:
            / 
        """
        
        self._extraction_mode: str = _extraction_mode
        
        self._transform = {'fifo': self._fifo_transform, 'lifo': self._lifo_transform}

        self._size: int = _size
        self._offsets = {}
        self._next_reservation = 0
        self._efficiency: float = _efficiency
        self.rng = np.random.default_rng()

        self._memory = [[], [], [], []]
        self._sessions: dict[int, int] = {}
        self._generation_batches: dict[int, list[Qubit]] = {}
        self._tracks_generation = False
     
    def bind_rng(self, rng: np.random.Generator) -> None:
        self.rng = rng
        for error in getattr(self, "_errors", ()):
            if hasattr(error, "bind_rng"):
                error.bind_rng(rng)

    def __len__(self) -> int:
        
        """
        Custom Length function
        
        Args:
            /
            
        Returns:
            _len (int): number of elements in the memory
        """
        
        return sum(len(store) for store in self._memory)

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
        Returns the number of qubits in the store
        
        Args:
            _store (int): which store to use L0 to L3
            
        Returns:
            num_qubits (int): number of qubits in the store
        """
        
        return len(self._memory[_store])
    
    def store_qubit(self, _store: int, _qubit: Qubit, _index: int, _time: float) -> None:
        
        """
        Stores a qubit in the memory
        
        Args:
            _store (int): which store to use
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
    
    def peek_qubit(self, _store: int, _index: int) -> Qubit:
        
        """
        Takes a look at the qubit in memory at index without extracting it
        
        Args:
            _store (int): which store to use
            _index (int): index of qubit
            
        Returns:
            _qubit (Qubit): qubit
        """
        
        if not self._memory[_store]:
            raise ValueError('No Qubit in memory')
        
        if _index is None:
            _index = self._transform[self._extraction_mode](0)
        
        _qubit, _ = self._memory[_store][_index]
        
        return _qubit
    
    def move_qubits(self, _src_store: int, _dst_store: int, _indices: List[bool]) -> None:
 
        mask = np.asarray(_indices, dtype=bool)
        if mask.ndim != 1 or len(mask) > len(self._memory[_src_store]):
            raise ValueError('Invalid memory selection mask')
        entries = self._memory[_src_store][:len(mask)]
        if _src_store == 3 and any(id(entry) in self._reserved_ids() for entry in entries):
            raise ValueError('Cannot move reserved qubits')
        del self._memory[_src_store][:len(mask)]
        for keep, (qubit, time) in zip(mask, entries):
            if keep:
                self.store_qubit(_dst_store, qubit, -1, time)
            elif qubit is not None:
                remove_qubits([qubit])
    
    def discard_qubits(self, _store: int=None) -> None:
        
        if _store is None:
            self._memory = [[], [], [], []]
            self._offsets.clear()
            self._sessions.clear()
            self._generation_batches.clear()
            return
        if _store == 3:
            self._offsets.clear()
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
            _index_src = self._transform[self._extraction_mode](0)
        if _index_dst is None:
            _index_dst = self._transform[self._extraction_mode](0)
        
        return self.retrieve_qubit(L1, _index_src, _time), self.retrieve_qubit(L1, _index_dst, _time)
    
    def add_offset(self, _offset: int, *, ready: bool = False) -> int:
        
        # Stable reservation IDs identify concrete L3 entries, not shifting indices.
        if _offset < 1:
            raise ValueError('Reservation size must be positive')
        entries = self._ready_entries() if ready else self._available_entries(3)
        if len(entries) < _offset:
            raise ValueError('Not enough unreserved qubits')
        identifier = self._next_reservation
        self._next_reservation += 1
        self._offsets[identifier] = entries[:_offset]
        return identifier
    
    def remove_offset(self, _index: int) -> None:
        
        del self._offsets[_index]
    
    def retrieve_time_stamp(self, _store: int, _index: int) -> float:
        
        """
        Retrieves the time stamp of a qubit in memory
        
        Args:
            _store (int): which store to use
            _index (int): index of qubit
            
        Returns:
            _time (float): time stamp of qubit
        """
        
        if not self._memory[_store]:
            raise ValueError('No Qubit in memory')
        
        if _index is None:
            _index = self._transform[self._extraction_mode](0)
        
        _, _time = self._memory[_store][_index]
        
        return _time
    
    def _fifo_transform(self, _index: int) -> int:
        
        """
        Transforms the _index according to the fifo principle
        
        Args:
            _index (int): index to transform
            
        Returns:
            _index (int): transformed index
        """
        
        return _index

    def _lifo_transform(self, _index: int) -> int:
        
        """
        Transforms an index based on the lifo principle
        
        Args:
            _index (int): index to transform
            
        Returns:
            _index (int): transformed index
        """
        
        return -1 * _index - 1

    def _reserved_ids(self):
        return {id(entry) for entries in self._offsets.values() for entry in entries}

    def _available_entries(self, store, reservation=None):
        if reservation is not None:
            if store != 3:
                raise ValueError('Reservations apply to L3 only')
            return list(self._offsets[reservation])
        reserved = self._reserved_ids()
        entries = [entry for entry in self._memory[store]
                   if id(entry) not in reserved and id(entry[0]) not in self._sessions]
        return entries if self._extraction_mode == 'fifo' else entries[::-1]

    def available_qubits(self, store, reservation=None):
        return len(self._available_entries(store, reservation))

    def _extract_entry(self, store, index, peek, reservation):
        index = 0 if index is None else index
        if index < 0:
            raise IndexError('Extraction index cannot be negative')
        entries = self._available_entries(store, reservation)
        if index >= len(entries):
            raise ValueError('No available qubit at this index')
        entry = entries[index]
        if not peek:
            position = next(i for i, value in enumerate(self._memory[store]) if value is entry)
            self._memory[store].pop(position)
            if reservation is not None:
                self._offsets[reservation] = [value for value in self._offsets[reservation] if value is not entry]
        return entry

    def _ready_entries(self) -> list[tuple[Qubit, float]]:
        """Confirmed raw inputs first, then confirmed improved inputs."""
        return self._available_entries(L1) + self._available_entries(L3)

    def ready_count(self, reservation: int | None = None) -> int:
        return len(self._ready_entries() if reservation is None else self._offsets[reservation])

    def ready_layer(self, index: int = 0, reservation: int | None = None) -> int:
        entries = self._ready_entries() if reservation is None else self._offsets[reservation]
        entry = entries[index]
        return next(layer for layer in (L1, L3) if any(value is entry for value in self._memory[layer]))

    def session_layer(self, qubit: Qubit) -> int:
        return self._sessions[id(qubit)]

    def retrieve_ready(self, time: float, index: int | None = None, reservation: int | None = None) -> Qubit | None:
        entries = self._ready_entries() if reservation is None else self._offsets[reservation]
        if not entries:
            raise ValueError('No confirmed pair available')
        entry = entries[0 if index is None else index]
        layer = next(layer for layer in (L1, L3) if any(value is entry for value in self._memory[layer]))
        if reservation is not None:
            self._offsets[reservation].remove(entry)
        position = next(i for i, value in enumerate(self._available_entries(layer)) if value is entry)
        return self.retrieve_qubit(layer, position, time, False)

    def reserve_session(self, qubit: Qubit, layer: int) -> None:
        self._sessions[id(qubit)] = layer

    def retrieve_session(self, qubit: Qubit, time: float) -> Qubit | None:
        layer = self._sessions.pop(id(qubit))
        index = next(i for i, entry in enumerate(self._available_entries(layer)) if entry[0] is qubit)
        return self.retrieve_qubit(layer, index, time, False)

    def confirm_session(self, qubit: Qubit) -> None:
        self.stage_session(qubit, L3)

    def stage_session(self, qubit: Qubit, destination: int) -> None:
        """Move an owned input while preserving its storage-noise timestamp."""
        layer = self._sessions[id(qubit)]
        self.move_selected_qubits(layer, destination, [qubit], [True])
        self._sessions[id(qubit)] = destination

    def finish_session_operation(self, qubit: Qubit, time: float, layer: int, *, consumed: bool = False) -> None:
        """Release a consumed slot or timestamp an ideal operation's survivor."""
        source = self._sessions.pop(id(qubit))
        index = next(i for i, entry in enumerate(self._memory[source]) if entry[0] is qubit)
        self._memory[source].pop(index)
        if consumed:
            return
        self.store_qubit(layer, qubit, -1, time)
        self.reserve_session(qubit, layer)

    def discard_session_qubit(self, qubit: Qubit) -> None:
        layer = self._sessions.pop(id(qubit))
        index = next(i for i, entry in enumerate(self._memory[layer]) if entry[0] is qubit)
        self._memory[layer].pop(index)
        remove_qubits([qubit])

    def register_generation(self, identifier: int, qubits: list[Qubit]) -> None:
        if identifier in self._generation_batches:
            raise ValueError('Duplicate generation batch identifier')
        self._generation_batches[identifier] = qubits
        self._tracks_generation = True

    def register_recent_generation(self, identifier: int, count: int) -> None:
        """Bind inputs just synchronously inserted by a connection model."""
        if count < 1 or count > len(self._memory[L0]):
            raise ValueError('Generation count does not match inserted L0 resources')
        self.register_generation(identifier, [entry[0] for entry in self._memory[L0][-count:]])

    def confirm_generation(self, identifier: int, success: np.ndarray) -> bool | None:
        qubits = self._generation_batches.pop(identifier, None)
        if qubits is None:
            return None if self._tracks_generation else False
        self.move_selected_qubits(L0, L1, qubits, success)
        return True

    def move_selected_qubits(self, source, destination, qubits, success):
        """Confirm one purification batch even when results arrive out of order."""
        if len(qubits) != len(success):
            raise ValueError('Selection and result counts differ')
        entries = []
        for qubit in qubits:
            entry = next((entry for entry in self._memory[source] if entry[0] is qubit), None)
            if entry is None:
                raise ValueError('Purification survivor is no longer in memory')
            entries.append(entry)
        selected = {id(entry) for entry in entries}
        self._memory[source] = [entry for entry in self._memory[source] if id(entry) not in selected]
        for keep, (qubit, time) in zip(success, entries):
            if keep:
                self.store_qubit(destination, qubit, -1, time)
            else:
                remove_qubits([qubit])

class PhysicalQuantumMemory(QuantumMemory):
    
    def __init__(self, model: PQM_Model) -> None:
        super(PhysicalQuantumMemory, self).__init__(model._size, model._extraction_mode, model._efficiency)
        
        self._errors: List[QuantumError] = deepcopy(model._memory_errors)
        if self._errors is None:
            self._errors = []
    
    def retrieve_qubit(self, _store: int, _index: int, _time: float, _mode: bool, _offset_index: int=None) -> Qubit | None:
    
        _qubit, _store_time = self._extract_entry(_store, _index, _mode, _offset_index)
        if self.rng.uniform(0, 1) > self._efficiency:
            remove_qubits([_qubit])
            return None

        _diff = _time - _store_time
        if _diff <= 0.:
            return _qubit

        for error in self._errors:
            _qubit = error.apply(_qubit, _diff)

        return _qubit
    
class LogicalQuantumMemory(QuantumMemory):
    
    def __init__(self, model: LQM_Model) -> None:
        super(LogicalQuantumMemory, self).__init__(model._size, model._extraction_mode, model._efficiency)
        
        self._code_length: int = model._code_length
        self._x_errors: int = model._x_errors
        self._z_errors: int = model._z_errors
        
        self._depolarization_time: float = model._depolarization_time
        if self._depolarization_time is None:
            self._depolarization_time: float = np.inf
        
        self._dephasing_time: float = model._dephasing_time
        if self._dephasing_time is None:
            self._dephasing_time: float = np.inf
            
        self._measurement_error_rate: float = model._measurement_error_rate
        self._extraction_error_rate: float = model._extraction_error_rate
        self._x_error_rate: float = model._x_error_rate
        self._z_error_rate: float = model._z_error_rate
        self._repetitions: int = model._repetitions
        
        self._correction_frequency: float = model._correction_frequency
        
        if model._logical_mode == 'deterministic':
            self.apply = self.apply_deterministic
        else:
            self.apply = self.apply_stochastic
    
    def _compute_pauli_probs(self, _diff: float, ) -> Tuple[float]:
        
        zeta  = np.exp(-_diff / self._depolarization_time)
        sperp = np.exp(-_diff * (1.0/self._depolarization_time + 1.0/self._dephasing_time))
        qX = (1 - zeta) / 2.0
        qZ = (1 + zeta - 2 * sperp)/4.0 + (1 - zeta) / 4.0
        
        qX = min(0.5, max(0.0, qX + self._x_error_rate * self._extraction_error_rate))
        qZ = min(0.5, max(0.0, qZ + self._z_error_rate * self._extraction_error_rate))
        
        _SX = _binomial_tail(self._code_length, qX, self._x_errors)
        _SZ = _binomial_tail(self._code_length, qZ, self._z_errors)
        
        pmv = _compute_measurement_majority(self._measurement_error_rate, self._repetitions)
        
        _SXp = (1 - pmv) * _SX + pmv * (1 - _SX)
        _SZp = (1 - pmv) * _SZ + pmv * (1 - _SZ)
        
        pI = _SXp * _SZp
        pX = (1 - _SXp) * _SZp
        pZ = _SXp * (1 - _SZp)
        pY = (1 - _SXp) * (1 - _SZp)
        s = pI + pX + pY + pZ
        
        return pI/s, pX/s, pY/s, pZ/s
    
    def apply_deterministic(self, _qubit: Qubit, p_i: float, p_x: float, p_y: float, p_z: float) -> Qubit:
        
        return pauli_error(_qubit, p_i, p_x, p_y, p_z)
        
    def apply_stochastic(self, _qubit: Qubit, p_i: float, p_x: float, p_y: float, p_z: float) -> Qubit:
        
        num_random = self.rng.uniform(0, 1)
        
        if num_random < p_i:
            return _qubit
        if num_random < p_i + p_x:
            return pauli_error(_qubit, p_i + p_y + p_z, p_x, 0, 0)
        if num_random < p_i + p_x + p_y:
            return pauli_error(_qubit, p_i + p_x + p_z, 0, p_y, 0)
        
        return pauli_error(_qubit, p_i + p_x + p_y, 0, 0, p_z)
    
    def apply(self) -> None:
        
        pass
    
    def retrieve_qubit(self, _store: int, _index: int, _time: float, _mode: bool, _offset_index: int=None) -> Qubit | None:
        
        _qubit, _store_time = self._extract_entry(_store, _index, _mode, _offset_index)
        e_max = 2 * min(self._x_errors, self._z_errors)

        if self.rng.binomial(n=self._code_length, p=1.0 - self._efficiency) > e_max:
            return None

        _diff = _time - _store_time
        if _diff <= 0.:
            return _qubit

        p_i, p_x, p_y, p_z = self._compute_pauli_probs(_diff)

        for _ in range(int(np.floor(_diff / self._correction_frequency))):
            _qubit = self.apply(_qubit, p_i, p_x, p_y, p_z)

        return _qubit
    
def _binomial_tail(n, q, t):
    
    return sum(comb(n, i) * (q ** i) * ((1 - q) ** (n - i)) for i in range(t + 1))

def _compute_measurement_majority(measurement_error_rate, repetitions):
    
    if repetitions < 2 or measurement_error_rate <= 0: 
        return 0.0
    
    return sum(comb(repetitions, k) * (measurement_error_rate ** k) * ((1 - measurement_error_rate) ** (repetitions - k)) for k in range((repetitions + 1)//2, repetitions + 1))
