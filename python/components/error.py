import numpy as np
import scipy.sparse as sp

from typing import Union, List

from python.components.qubit import Qubit, dot, get_single_operator, remove_qubits

__all__ = ['DepolarizationError', 'DephasingError', 'TimeDependentError', 'RandomDepolarizationError', 'RandomDephasingError', 'RandomError', 'SystematicDepolarizationError',
           'SystematicDephasingError', 'SystematicError', 'DepolarizationMemoryError', 'DephasingMemoryError', 'TimeDependentMemoryError']

full_gates = {'P0': np.array([[1, 0], [0, 0]], dtype=np.complex128),
              'P1': np.array([[0, 0], [0, 1]], dtype=np.complex128),
              'P01': np.array([[0, 1], [0, 0]], dtype=np.complex128),
              'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128)
              }

sparse_gates = {'P0': sp.csr_matrix([[1, 0], [0, 0]], dtype=np.complex128),
              'P1': sp.csr_matrix([[0, 0], [0, 1]], dtype=np.complex128),
              'P01': sp.csr_matrix([[0, 1], [0, 0]], dtype=np.complex128),
              'Z': sp.csr_matrix([[1, 0], [0, -1]], dtype=np.complex128)
              }

gates = {0: full_gates, 1: sparse_gates}
                
class DepolarizationError:
    
    """
    Represents a Depolarization Error
    """
    
    def __init__(self, _depolar_time: float=1e-3) -> None:
        
        """
        Initializes a Depolarization Error
        
        Args:
            _depolar_time (float): depolarization time
            
        Returns:
            /
        """
        
        self._depolar_time: float = _depolar_time
        
    def add_signal_time(self, _length: float, _attenuation_coefficient: float=-0.016) -> None:
        
        """
        Adds the signal time to calculate Depolarization probability
        
        Args:
            _length (float): length of fiber
            _attenuation_coefficient (float): attenuation coefficient of fiber
            
        Returns:
            /
        """

        signal_time = _length * (5e-6)
        depolar_prob = np.exp(-signal_time / self._depolar_time)
        
        self._gate_e0: Union[np.array, sp.csr_matrix] = {0: full_gates['P0'] + np.sqrt(1 - depolar_prob) * full_gates['P1'], 1: sparse_gates['P0'] + np.sqrt(1 - depolar_prob) * sparse_gates['P1']}
        self._gate_e1: Union[np.array, sp.csr_matrix] = {0: np.sqrt(depolar_prob) * full_gates['P01'], 1: np.sqrt(1 - depolar_prob) * sparse_gates['P01']}
        
    def apply(self, _qubit: Qubit) -> None:
        
        """
        Applies the Error to the qubits
        
        Args:
            _qubits (Qubit): single or collection of qubits
            
        Returns:
            /
        """

        gate_e0 = get_single_operator('', _qubit._qsystem._sparse, self.gate_e0[_qubit._qsystem._sparse], _qubit._index, _qubit._qsystem._num_qubits)
        gate_e1 = get_single_operator('', _qubit._qsystem._sparse, self.gate_e1[_qubit._qsystem._sparse], _qubit._index, _qubit._qsystem._num_qubits)
        _qubit._qsystem._state = dot(_qubit._qsystem._state, gate_e0) + dot(_qubit._qsystem._state, gate_e1)
        
        return _qubit

class DephasingError:
    
    """
    Represents a Dephasing Error
    """
    
    def __init__(self, _dephase_time: float=1e-3) -> None:
        
        """
        Initializes a Dephasing Error
        
        Args:
            _dephase_time (float): dephasing time
            
        Returns:
            /
        """
        
        self._dephase_time: float = _dephase_time
        
    def add_signal_time(self, _length: float, _attenuation_coefficient: float=-0.016) -> None:
        
        """
        Adds the signal time to calculate Dephasing probability
        
        Args:
            _length (float): length of fiber
            _attenuation_coefficient (float): attenuation coefficient of fiber
            
        Returns:
            /
        """
        
        signal_time = _length * (5e-6)
        
        self._dephase_prob: float = 0.5 * (1 - np.exp(-signal_time / self._dephase_time))
        self._dephase_prob_inv: float = 1 - self._dephase_prob
        
    def apply(self, _qubit: Qubit) -> None:
        
        """
        Applies the Error to the qubits
        
        Args:
            _qubits (list/Qubit): single or collection of qubits
            
        Returns:
            /
        """

        key = f'{_qubit._qsystem._sparse}_s_z_{_qubit._qsystem._num_qubits}_{_qubit._index}'
        gate_z = get_single_operator(key, _qubit._qsystem._sparse, gates[_qubit._qsystem._sparse]['Z'], _qubit._index, _qubit._qsystem._num_qubits)
        _qubit._qsystem._state = self._dephase_prob_inv * _qubit._qsystem._state + self._dephase_prob * dot(_qubit._qsystem._state, gate_z)

        return _qubit

class TimeDependentError:
    
    """
    Represents a Time Dependent Error
    """
    
    def __init__(self, _depolar_time: float=1e-3, _dephase_time: float=1e-3) -> None:
        
        """
        Initializes a Time Dependent Error
        
        Args:
            _depolar_time (float): depolarization time
            _dephase_time (float): dephasing time
            
        Returns:
            /
        """
        
        if _dephase_time >= 2 * _depolar_time:
            raise ValueError('Depolarization Rate has to be greater than Dephasing Rate')
        
        self._depolar_time: float = _depolar_time
        self._dephase_time: float = _dephase_time
        
    def add_signal_time(self, _length: float, _attenuation_coefficient: float=-0.016) -> None:
        
        """
        Adds the signal time to calculate Depolarization and Dephasing probability
        
        Args:
            _length (float): length of fiber
            _attenuation_coefficient (float): attenuation coefficient of fiber
            
        Returns:
            /
        """
 
        signal_time = _length * (5e-6)
        
        depolar_prob = 1 - np.exp(-(signal_time / self._depolar_time))
        self._gate_e0: Union[np.array, sp.csr_matrix] = {0: full_gates['P0'] + np.sqrt(1 - depolar_prob) * full_gates['P1'], 1: sparse_gates['P0'] + np.sqrt(1 - depolar_prob) * sparse_gates['P1']}
        self._gate_e1: Union[np.array, sp.csr_matrix] = {0: np.sqrt(depolar_prob) * full_gates['P01'], 1: np.sqrt(depolar_prob) * sparse_gates['P01']}
        
        self._dephase_prob: float = 0.5 * (1 - np.exp(-signal_time * (1/self._dephase_time - 1/(2*self._depolar_time))))
        self._dephase_prob_inv: float = 1 - self._dephase_prob
        
    def apply(self, _qubit: Union[Qubit, List[Qubit]]) -> None:
        
        """
        Applies the Error to qubits
        
        Args:
            _qubits (list/Qubit): single or collection of qubits
            
        Returns:
            /
        """
        
        key = f'{_qubit._qsystem._sparse}_s_z_{_qubit._qsystem._num_qubits}_{_qubit._index}'
        gate_e0 = get_single_operator('', _qubit._qsystem._sparse, self._gate_e0[_qubit._qsystem._sparse], _qubit._index, _qubit._qsystem._num_qubits)
        gate_e1 = get_single_operator('', _qubit._qsystem._sparse, self._gate_e1[_qubit._qsystem._sparse], _qubit._index, _qubit._qsystem._num_qubits)
        gate_z = get_single_operator(key, _qubit._qsystem._sparse, gates[_qubit._qsystem._sparse]['Z'], _qubit._index, _qubit._qsystem._num_qubits)
        
        state = dot(_qubit._qsystem._state, gate_e0) + dot(_qubit._qsystem._state, gate_e1)
        _qubit._qsystem._state = self._dephase_prob_inv * state + self._dephase_prob * dot(state, gate_z)
        
        return _qubit
        
class RandomDepolarizationError:
    
    """
    Represents a Random Depolarization Error
    """
    
    def __init__(self, _variance: float, _lose_prob: float=0.0) -> None:
        
        """
        Initializes a Random Depolarization Error
        
        Args:
            _variance (float): variance of normal distributed angles
            _lose_prob (float): qubit loss probability
            
        Returns:
            /
        """
        
        if not (0. <= _lose_prob <= 1.0):
            raise ValueError('Probability should be between 0 and 1')
        
        self._variance: float = _variance
        self._lose_prob: float = _lose_prob
        self._lose_qubits: bool = self._lose_prob > 0.0
        
    def add_signal_time(self, _length: float=0.0, _attenuation_coefficient: float=-0.016) -> None:
        
        """
        Adds the signal time to calculate Depolarization and Dephasing probability
        
        Args:
            _length (float): length of fiber
            _attenuation_coefficient (float): attenuation coefficient of fiber
            
        Returns:
            /
        """
        
        pass
    
    def apply(self, _qubit: Union[Qubit, List[Qubit]]) -> None:
        
        """
        Applies the Error to qubits
        
        Args:
            _qubits (list/Qubit): single or collection of qubits
            
        Returns:
            /
        """

        theta = np.random.normal(0, self._variance)
        _qubit.Rx(theta)
        
        return _qubit    
            
class RandomDephasingError:
    
    """
    Represents a Random Dephasing Error
    """
    
    def __init__(self, _variance: float, _lose_prob: float=0.0) -> None:
        
        """
        Initializes a Random Dephasing Error
        
        Args:
            _variance (float): _variance of dephasing angle
            _lose_prob (float): probability to lose qubits
        
        Returns:
            /
        """
        
        self._variance: float = _variance
        self._lose_prob: float = _lose_prob
        self._lose_qubits: bool = self._lose_prob > 0.0
        
    def add_signal_time(self, _length: float=0.0, _attenuation_coefficient: float=-0.016) -> None:
        
        """
        Adds the signal time to calculate Depolarization and Dephasing probability
        
        Args:
            _length (float): length of fiber
            _attenuation_coefficient (float): attenuation coefficient of fiber
            
        Returns:
            /
        """
        
        pass
    
    def apply(self, _qubit: List[Qubit]) -> None:
        
        """
        Applies the Error to qubits
        
        Args:
            _qubits (list/Qubit): single or collection of qubits
            
        Returns:
            /
        """
        
        theta = np.random.normal(0, self._variance)
        _qubit.Rz(theta)
        
        return _qubit
            
class RandomError:
    
    """
    Represents a Random Error
    """
    
    def __init__(self, _x_variance: float, _z_variance: float, _lose_prob: float=0.0) -> None:
        
        """
        Initializes a Random Error
        
        Args:
            _x_variance (float): variance of depolarization angle
            _z_variance (float): variance of dephasing angle
            _lose_prob (float): probability to lose qubits
            
        Returns:
            /
        """
        
        self._x_variance: float = _x_variance
        self._z_variance: float = _z_variance
        self._lose_prob: float = _lose_prob
        self._lose_qubits: bool = _lose_qubits
        
    def add_signal_time(self, _length: float=0.0, _attenuation_coefficient: float=-0.016) -> None:
        
        """
        Adds the signal time to calculate Depolarization and Dephasing probability
        
        Args:
            _length (float): length of fiber
            _attenuation_coefficient (float): attenuation coefficient of fiber
            
        Returns:
            /
        """
        
        pass
    
    def apply(self, _qubit: List[Qubit]) -> None:
        
        """
        Applies the Error to qubits
        
        Args:
            _qubits (list/Qubit): single or collection of qubits
            
        Returns:
            /
        """

        theta_x = np.random.normal(0, self._x_variance)
        theta_z = np.random.normal(0, self._z_variance)
        _qubit.Rx(theta_x)
        _qubit.Rz(theta_z)
        
        return _qubit
        
class SystematicDepolarizationError:
    
    """
    Represents a Systematic Depolarization Error
    """
    
    def __init__(self, _variance: float, _lose_prob: float=0.0) -> None:
        
        """
        Initializes a Systematic Depolarization Error
        
        Args:
            _variance (float): variance of once drawn depolarization angle
            _lose_prob (float): probability to lose qubits
            
        Returns:
            /
        """
        
        self._theta: float = np.random.normal(0, _variance)
        self._lose_prob: float = _lose_prob
        self._lose_qubits: bool = self._lose_prob > 0.0
        
    def add_signal_time(self, _length: float=0.0, _attenuation_coefficient: float=-0.016) -> None:
        
        """
        Adds the signal time to calculate Depolarization and Dephasing probability
        
        Args:
            _length (float): length of fiber
            _attenuation_coefficient (float): attenuation coefficient of fiber
            
        Returns:
            /
        """
        
        pass
    
    def apply(self, _qubit: List[Qubit]) -> None:
        
        """
        Applies the Error to qubits
        
        Args:
            _qubits (list/Qubit): single or collection of qubits
            
        Returns:
            /
        """

        _qubit.Rx(self._theta)
        
        return _qubit
            
class SystematicDephasingError:
    
    """
    Represents a Systematic Dephasing Error
    """
    
    def __init__(self, _variance: float, _lose_prob: float=0.0) -> None:
        
        """
        Initializes a Systematic Dephasing Error
        
        Args:
            _variance (float): variance of normal distributed dephasing angle
            _lose_prob (float): probability of losing qubits
            
        Returns:
            /
        """
        
        self._theta: float = np.random.normal(0, _variance)
        self._lose_prob: float = _lose_prob
        self._lose_qubits: bool = self._lose_prob > 0.0
        
    def add_signal_time(self, _length: float=0.0, _attenuation_coefficient: float=-0.016) -> None:
        
        """
        Adds the signal time to calculate Depolarization and Dephasing probability
        
        Args:
            _length (float): length of fiber
            _attenuation_coefficient (float): attenuation coefficient of fiber
            
        Returns:
            /
        """
        
        pass
    
    def apply(self, _qubit: List[Qubit]) -> None:
        
        """
        Applies the Error to qubits
        
        Args:
            _qubits (list/Qubit): single or collection of qubits
            
        Returns:
            /
        """

        _qubit.Rz(self._theta)
        
        return _qubit

class SystematicError:
    
    """
    Represents a Systematic Error
    """

    def __init__(self, _x_variance: float, _z_variance: float, _lose_prob: float=0.0) -> None:
        
        """
        Initializes a Systematic Error
        
        Args:
            _x_variance (float): variance of normal distributed depolarization angle
            _z_variance (float): variance of normal distributed dephasing angle
            _lose_prob (float): probability of losing qubits
            
        Returns:
            /
        """
        
        self._theta_x: float = np.random.normal(0, _x_variance)
        self._theta_z: float = np.random.normal(0, _z_variance)
        self._lose_prob: float = _lose_prob
        self._lose_qubits: bool = self._lose_prob > 0.0
        
    def add_signal_time(self, _length: float=0.0, _attenuation_coefficient: float=-0.016) -> None:
        
        """
        Adds the signal time to calculate Depolarization and Dephasing probability
        
        Args:
            _length (float): length of fiber
            _attenuation_coefficient (float): attenuation coefficient of fiber
            
        Returns:
            /
        """
        
        pass
    
    def apply(self, _qubit: List[Qubit]) -> None:
        
        """
        Applies the Error to qubits
        
        Args:
            _qubits (list/Qubit): single or collection of qubits
            
        Returns:
            /
        """
        
        _qubit.Rx(self._theta_x)
        _qubit.Rz(self._theta_z)
        
        return _qubit

class DepolarizationMemoryError:
    
    """
    Represents a Depolarization Error in Memory
    """
    
    def __init__(self, _depolar_time: float=1e-3) -> None:
        
        """
        Initializes a Depolarization Error in Memory
        
        Args:
            _depolar_time (float): depolarization time
            _efficiency (float): efficiency of memory
            
        Returns:
            /
        """
        
        self._depolar_time: float = _depolar_time

    def apply(self, _qubit: Qubit, _time: float) -> None:
        
        """
        Applies the Error to Qubit
        
        Args:
            _q (Qubit): qubit to apply error to
            _time (float): time qubit is in storage
            
        Returns:
            /
        """
        
        depolar_prob = 1 - np.exp(-_time / self._depolar_time)
        gate_e0 = gates[_qubit._qsystem._sparse]['P0'] + np.sqrt(1 - depolar_prob) * gates[_qubit._qsystem._sparse]['P1']
        gate_e1 = np.sqrt(depolar_prob) * gates[_qubit._qsystem._sparse]['P01']
        
        gate_e0 = get_single_operator('', _qubit._qsystem._sparse, gate_e0, _qubit._index, _qubit._qsystem._num_qubits)
        gate_e1 = get_single_operator('', _qubit._qsystem._sparse, gate_e1, _qubit._index, _qubit._qsystem._num_qubits)
        _qubit._qsystem._state = dot(_qubit._qsystem._state, gate_e0) + dot(_qubit._qsystem._state, gate_e1)
        
        return _qubit

class DephasingMemoryError:
    
    """
    Represents a Depolarization Error in Memory
    """
    
    def __init__(self, _dephase_time: float=1e-3) -> None:
        
        """
        Initializes a Depolarization Error in Memory
        
        Args:
            _depolar_time (float): depolarization time
            
        Returns:
            /
        """
        
        self._dephase_time: float = _dephase_time

    def apply(self, _qubit: Qubit, _time: float) -> None:
        
        """
        Applies the Error to Qubit
        
        Args:
            _q (Qubit): qubit to apply error to
            _time (float): time qubit is in storage
            
        Returns:
            /
        """
        
        dephase_prob = 0.5 * (1 - np.exp(-_time / self._dephase_time))
        dephase_prob_inv = 1 - dephase_prob
         
        key = f'{_qubit._qsystem._sparse}s_z_{_qubit._qsystem._num_qubits}_{_qubit._index}'
        gate_z = get_single_operator(key, _qubit._qsystem._sparse, gates[_qubit._qsystem._sparse]['Z'], _qubit._index, _qubit._qsystem._num_qubits)
        _qubit._qsystem._state = dephase_prob_inv * _qubit._qsystem._state + dephase_prob * dot(_qubit._qsystem._state, gate_z)
        
        return _qubit

class TimeDependentMemoryError:
    
    """
    Represents a Time Dependent Errors in Memories
    """
    
    def __init__(self, _depolar_time: float=1e-3, _dephase_time: float=1e-3) -> None:
        
        """
        Initializes a Time Dependent Errors in Memories
        
        Args:
            _depolar_time (float): depolarization time
            _dephase_time (float): dephasing time
            
        Returns:
            /
        """
        
        if _dephase_time >= 2 * _depolar_time:
            raise ValueError('Depolarization Rate has to be greater than Dephasing Rate')
        
        self._depolar_time: float = _depolar_time
        self._dephase_time: float = _dephase_time
        
    def apply(self, _qubit: Qubit, _time: float) -> None:
        
        """
        Applies the Error to Qubit
        
        Args:
            _q (Qubit): qubit to apply error to
            _time (float): time qubit is in storage
            
        Returns:
            /
        """

        depolar_prob = 1 - np.exp(-_time / self._depolar_time)
        
        gate_e0 = gates[_qubit._qsystem._sparse]['P0'] + np.sqrt(1 - depolar_prob) * gates[_qubit._qsystem._sparse]['P1']
        gate_e1 = np.sqrt(depolar_prob) * gates[_qubit._qsystem._sparse]['P01']
        
        dephase_prob = 0.5 * (1 - np.exp(-_time * (1/self._dephase_time - 1/(2*self._depolar_time))))
        dephase_prob_inv = 1 - dephase_prob
        
        key = f'{_qubit._qsystem._sparse}_s_z_{_qubit._qsystem._num_qubits}_{_qubit._index}'
        gate_e0 = get_single_operator('', _qubit._qsystem._sparse, gate_e0, _qubit._index, _qubit._qsystem._num_qubits)
        gate_e1 = get_single_operator('', _qubit._qsystem._sparse, gate_e1, _qubit._index, _qubit._qsystem._num_qubits)
        gate_z = get_single_operator(key, _qubit._qsystem._sparse, gates[_qubit._qsystem._sparse]['Z'], _qubit._index, _qubit._qsystem._num_qubits)
        
        _qubit._qsystem._state = dot(_qubit._qsystem._state, gate_e0) + dot(_qubit._qsystem._state, gate_e1)
        _qubit._qsystem._state = dephase_prob_inv * _qubit._qsystem._state + dephase_prob * dot(_qubit._qsystem._state, gate_z)
        
        return _qubit