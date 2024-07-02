import numpy as np
import scipy.sparse as sp

from typing import Union, Dict

from python.components.qubit import Qubit, dot, get_single_operator

__all__ = ['DepolarizationError', 'DephasingError', 'TimeDependentError', 'RandomDepolarizationError', 'RandomDephasingError', 'RandomError', 'SystematicDepolarizationError',
           'SystematicDephasingError', 'SystematicError', 'DepolarizationMemoryError', 'DephasingMemoryError', 'TimeDependentMemoryError']

full_gates = {'P0': np.array([[1, 0], [0, 0]], dtype=np.complex128),
              'P1': np.array([[0, 0], [0, 1]], dtype=np.complex128),
              'P01': np.array([[0, 1], [0, 0]], dtype=np.complex128),
              'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128)}

sparse_gates = {'P0': sp.csr_matrix([[1, 0], [0, 0]], dtype=np.complex128),
              'P1': sp.csr_matrix([[0, 0], [0, 1]], dtype=np.complex128),
              'P01': sp.csr_matrix([[0, 1], [0, 0]], dtype=np.complex128),
              'Z': sp.csr_matrix([[1, 0], [0, -1]], dtype=np.complex128)}

gates = {0: full_gates, 1: sparse_gates}
                
class DepolarizationError:
    
    """
    Represents a Depolarization Error
    
    Attr:
        _depolar_time (float): depolarization time of error
        _gate_e0 (dict): precalculated E0 matrix representing depolarization
        _gate_e1 (dict): precalculated E1 matrix representing depolarization
    """
    
    def __init__(self, depolar_time: float=1e-3) -> None:
        
        """
        Initializes a Depolarization Error
        
        Args:
            depolar_time (float): depolarization time
            
        Returns:
            /
        """
        
        self._depolar_time: float = depolar_time
        
    def add_signal_time(self, _length: float) -> None:
        
        """
        Adds the signal time to calculate Depolarization probability
        
        Args:
            _length (float): length of fiber
            
        Returns:
            /
        """

        depolar_prob = np.exp(-(_length * (5e-6)) / self._depolar_time)
        
        self._gate_e0: Dict[int, Union[np.array, sp.csr_matrix]] = {0: full_gates['P0'] + np.sqrt(1 - depolar_prob) * full_gates['P1'], 1: sparse_gates['P0'] + np.sqrt(1 - depolar_prob) * sparse_gates['P1']}
        self._gate_e1: Dict[int, Union[np.array, sp.csr_matrix]] = {0: np.sqrt(depolar_prob) * full_gates['P01'], 1: np.sqrt(1 - depolar_prob) * sparse_gates['P01']}
        
    def apply(self, _qubit: Qubit) -> None:
        
        """
        Applies the Error to the qubits
        
        Args:
            _qubit (Qubit): qubit to apply error to
            
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
    
    Attr:
        _dephase_time (float): dephasing time of error
        _dephase_prob (float): dephasing probability
        _dephase_prob_inv (float): 1 - dephase probability
    """
    
    def __init__(self, dephase_time: float=1e-3) -> None:
        
        """
        Initializes a Dephasing Error
        
        Args:
            dephase_time (float): dephasing time
            
        Returns:
            /
        """
        
        self._dephase_time: float = dephase_time
        
    def add_signal_time(self, _length: float) -> None:
        
        """
        Adds the signal time to calculate Dephasing probability
        
        Args:
            _length (float): length of fiber
            
        Returns:
            /
        """
        
        self._dephase_prob: float = 0.5 * (1 - np.exp(-(_length * (5e-6)) / self._dephase_time))
        self._dephase_prob_inv: float = 1 - self._dephase_prob
        
    def apply(self, _qubit: Qubit) -> None:
        
        """
        Applies the Error to the qubits
        
        Args:
            _qubit (Qubit): qubit to apply error to
            
        Returns:
            /
        """

        key = f'{_qubit._qsystem._sparse}_s_z_{_qubit._qsystem._num_qubits}_{_qubit._index}'
        gate_z = get_single_operator(key, _qubit._qsystem._sparse, gates[_qubit._qsystem._sparse]['Z'], _qubit._index, _qubit._qsystem._num_qubits)
        _qubit._qsystem._state = self._dephase_prob_inv * _qubit._qsystem._state + self._dephase_prob * dot(_qubit._qsystem._state, gate_z)

        return _qubit

class TimeDependentError:
    
    """
    Represents a Time Dependent Error consisting of depolarization and dephasing
    
    Attr:
        _depolar_time (float): depolarization time
        _dephase_time (float): dephasing time
        _gate_e0 (dict): precalculated E0 matrix representing depolarization
        _gate_e1 (dict): precalculated E1 matrix representing depolarization
        _dephase_prob (float): dephasing probability
        _dephase_prob_inv (float): 1 - dephase probability
    """
    
    def __init__(self, depolar_time: float=1e-3, dephase_time: float=1e-3) -> None:
        
        """
        Initializes a Time Dependent Error
        
        Args:
            depolar_time (float): depolarization time
            dephase_time (float): dephasing time
            
        Returns:
            /
        """
        
        if dephase_time >= 2 * depolar_time:
            raise ValueError('Depolarization Rate has to be greater than Dephasing Rate')
        
        self._depolar_time: float = depolar_time
        self._dephase_time: float = dephase_time
        
    def add_signal_time(self, _length: float) -> None:
        
        """
        Adds the signal time to calculate Depolarization and Dephasing probability
        
        Args:
            _length (float): length of fiber
            
        Returns:
            /
        """
 
        signal_time = _length * (5e-6)
        
        depolar_prob = 1 - np.exp(-(signal_time / self._depolar_time))
        self._gate_e0: Dict[int, Union[np.array, sp.csr_matrix]] = {0: full_gates['P0'] + np.sqrt(1 - depolar_prob) * full_gates['P1'], 1: sparse_gates['P0'] + np.sqrt(1 - depolar_prob) * sparse_gates['P1']}
        self._gate_e1: Dict[int, Union[np.array, sp.csr_matrix]] = {0: np.sqrt(depolar_prob) * full_gates['P01'], 1: np.sqrt(depolar_prob) * sparse_gates['P01']}
        
        self._dephase_prob: float = 0.5 * (1 - np.exp(-signal_time * (1/self._dephase_time - 1/(2*self._depolar_time))))
        self._dephase_prob_inv: float = 1 - self._dephase_prob
        
    def apply(self, _qubit: Qubit) -> None:
        
        """
        Applies the Error to qubits
        
        Args:
            _qubit (Qubit): qubit to apply error to
            
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
    
    Attr:
        _variance (float): variance of normally distributed angles
    """
    
    def __init__(self, variance: float) -> None:
        
        """
        Initializes a Random Depolarization Error
        
        Args:
            variance (float): variance of normal distributed angles
            
        Returns:
            /
        """
        
        self._variance: float = variance
        
    def add_signal_time(self, _length: float=0.0) -> None:
        
        """
        Adds the signal time to calculate Depolarization and Dephasing probability
        
        Args:
            _length (float): length of fiber
            
        Returns:
            /
        """
        
        pass
    
    def apply(self, _qubit: Qubit) -> None:
        
        """
        Applies the Error to qubits
        
        Args:
            _qubit (Qubit): qubit to apply error to
            
        Returns:
            /
        """

        theta = np.random.normal(0, self._variance)
        _qubit.Rx(theta)
        
        return _qubit    
            
class RandomDephasingError:
    
    """
    Represents a Random Dephasing Error
    
    Attr:
        _variance (float): variance of normally distributed angles
    """
    
    def __init__(self, variance: float) -> None:
        
        """
        Initializes a Random Dephasing Error
        
        Args:
            variance (float): _variance of dephasing angle
        
        Returns:
            /
        """
        
        self._variance: float = variance
        
    def add_signal_time(self, _length: float=0.0) -> None:
        
        """
        Adds the signal time to calculate Depolarization and Dephasing probability
        
        Args:
            _length (float): length of fiber
            
        Returns:
            /
        """
        
        pass
    
    def apply(self, _qubit: Qubit) -> None:
        
        """
        Applies the Error to qubits
        
        Args:
            _qubit (Qubit): qubit to apply error to
            
        Returns:
            /
        """
        
        theta = np.random.normal(0, self._variance)
        _qubit.Rz(theta)
        
        return _qubit
            
class RandomError:
    
    """
    Represents a Random Error
    
    Attr:
        _variance (float): variance of normally distributed angles
    """
    
    def __init__(self, x_variance: float, z_variance: float) -> None:
        
        """
        Initializes a Random Error
        
        Args:
            x_variance (float): variance of depolarization angle
            z_variance (float): variance of dephasing angle
            
        Returns:
            /
        """
        
        self._x_variance: float = x_variance
        self._z_variance: float = z_variance
        
    def add_signal_time(self, _length: float=0.0) -> None:
        
        """
        Adds the signal time to calculate Depolarization and Dephasing probability
        
        Args:
            _length (float): length of fiber
            
        Returns:
            /
        """
        
        pass
    
    def apply(self, _qubit: Qubit) -> None:
        
        """
        Applies the Error to qubits
        
        Args:
            _qubit (Qubit): qubit to apply error to
            
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
    
    _theta (float): depolarization angle
    """
    
    def __init__(self, variance: float) -> None:
        
        """
        Initializes a Systematic Depolarization Error
        
        Args:
            variance (float): variance of once drawn depolarization angle
            
        Returns:
            /
        """
        
        self._theta: float = np.random.normal(0, variance)
        
    def add_signal_time(self, _length: float=0.0) -> None:
        
        """
        Adds the signal time to calculate Depolarization and Dephasing probability
        
        Args:
            _length (float): length of fiber
            
        Returns:
            /
        """
        
        pass
    
    def apply(self, _qubit: Qubit) -> None:
        
        """
        Applies the Error to qubits
        
        Args:
            _qubits (Qubit): qubit to apply error to
            
        Returns:
            /
        """

        _qubit.Rx(self._theta)
        
        return _qubit
            
class SystematicDephasingError:
    
    """
    Represents a Systematic Dephasing Error
    
    Attr:
        _theta (float): dephasing angle
    """
    
    def __init__(self, variance: float) -> None:
        
        """
        Initializes a Systematic Dephasing Error
        
        Args:
            variance (float): variance of normal distributed dephasing angle
            
        Returns:
            /
        """
        
        self._theta: float = np.random.normal(0, variance)
        
    def add_signal_time(self, _length: float=0.0) -> None:
        
        """
        Adds the signal time to calculate Depolarization and Dephasing probability
        
        Args:
            _length (float): length of fiber
            
        Returns:
            /
        """
        
        pass
    
    def apply(self, _qubit: Qubit) -> None:
        
        """
        Applies the Error to qubits
        
        Args:
            _qubit (Qubit): qubit to apply error to
            
        Returns:
            /
        """

        _qubit.Rz(self._theta)
        
        return _qubit

class SystematicError:
    
    """
    Represents a Systematic Error
    
    _theta_x (float): depolarization angle
    _theta_z (float): dephasing angle
    """

    def __init__(self, x_variance: float, z_variance: float) -> None:
        
        """
        Initializes a Systematic Error
        
        Args:
            x_variance (float): variance of normal distributed depolarization angle
            z_variance (float): variance of normal distributed dephasing angle
            
        Returns:
            /
        """
        
        self._theta_x: float = np.random.normal(0, x_variance)
        self._theta_z: float = np.random.normal(0, z_variance)
        
    def add_signal_time(self, _length: float=0.0) -> None:
        
        """
        Adds the signal time to calculate Depolarization and Dephasing probability
        
        Args:
            _length (float): length of fiber
            _attenuation_coefficient (float): attenuation coefficient of fiber
            
        Returns:
            /
        """
        
        pass
    
    def apply(self, _qubit: Qubit) -> None:
        
        """
        Applies the Error to qubits
        
        Args:
            _qubit (Qubit): qubit to apply error to
            
        Returns:
            /
        """
        
        _qubit.Rx(self._theta_x)
        _qubit.Rz(self._theta_z)
        
        return _qubit

class DepolarizationMemoryError:
    
    """
    Represents a Depolarization Error in Memory
    
    Attr:
        _depolar_time (float): depolarization time
    """
    
    def __init__(self, depolar_time: float=1e-3) -> None:
        
        """
        Initializes a Depolarization Error in Memory
        
        Args:
            depolar_time (float): depolarization time
            
        Returns:
            /
        """
        
        self._depolar_time: float = depolar_time

    def apply(self, _qubit: Qubit, _time: float) -> None:
        
        """
        Applies the Error to Qubit
        
        Args:
            _qubit (Qubit): qubit to apply error to
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
    
    Attr:
        _dephase_time (float): dephasing time
    """
    
    def __init__(self, dephase_time: float=1e-3) -> None:
        
        """
        Initializes a Depolarization Error in Memory
        
        Args:
            dephase_time (float): depolarization time
            
        Returns:
            /
        """
        
        self._dephase_time: float = dephase_time

    def apply(self, _qubit: Qubit, _time: float) -> None:
        
        """
        Applies the Error to Qubit
        
        Args:
            _qubit (Qubit): qubit to apply error to
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
    
    Attr:
        _depolar_time (float): depolarization time
        _dephase_time (float): dephasing time
    """
    
    def __init__(self, depolar_time: float=1e-3, dephase_time: float=1e-3) -> None:
        
        """
        Initializes a Time Dependent Errors in Memories
        
        Args:
            depolar_time (float): depolarization time
            dephase_time (float): dephasing time
            
        Returns:
            /
        """
        
        if dephase_time >= 2 * depolar_time:
            raise ValueError('Depolarization Rate has to be greater than Dephasing Rate')
        
        self._depolar_time: float = depolar_time
        self._dephase_time: float = dephase_time
        
    def apply(self, _qubit: Qubit, _time: float) -> None:
        
        """
        Applies the Error to Qubit
        
        Args:
            _qubit (Qubit): qubit to apply error to
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