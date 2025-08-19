import numpy as np
import copy
from typing import Union, Dict

from qcns.python.components.qubit.qubit import Qubit, dot, get_single_operator

__all__ = ['DepolarizationError', 'DephasingError', 'TimeDependentError', 'RandomDepolarizationError', 'RandomDephasingError', 'RandomError', 'SystematicDepolarizationError',
           'SystematicDephasingError', 'SystematicError', 'DepolarizationMemoryError', 'DephasingMemoryError', 'TimeDependentMemoryError']

full_gates = {'P0': np.array([[1, 0], [0, 0]], dtype=np.complex128),
              'P1': np.array([[0, 0], [0, 1]], dtype=np.complex128),
              'P01': np.array([[0, 1], [0, 0]], dtype=np.complex128),
              'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128)}
   
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
        
    def apply(self, _qubit: Qubit, _time: float) -> None:
        
        """
        Applies the Error to Qubit
        
        Args:
            _qubit (Qubit): qubit to apply error to
            _time (float): time qubit is in storage
            
        Returns:
            /
        """
        
        depolar_prob = np.exp(-_time / self._depolar_time)
        gate_e0 = full_gates['P0'] + np.sqrt(depolar_prob) * full_gates['P1']
        gate_e1 = np.sqrt(1 - depolar_prob) * full_gates['P01']
        
        gate_e0 = get_single_operator('', gate_e0, _qubit._index, _qubit.num_qubits)
        gate_e1 = get_single_operator('', gate_e1, _qubit._index, _qubit.num_qubits)
        _qubit.state = dot(_qubit.state, gate_e0) + dot(_qubit.state, gate_e1)
        
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
         
        key = f's_z_{_qubit.num_qubits}_{_qubit._index}'
        gate_z = get_single_operator(key, full_gates['Z'], _qubit._index, _qubit.num_qubits)
        _qubit.state = dephase_prob_inv * _qubit.state + dephase_prob * dot(_qubit.state, gate_z)
        
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
        
    def apply(self, _qubit: Qubit, _time: float) -> None:
        
        """
        Applies the Error to Qubit
        
        Args:
            _qubit (Qubit): qubit to apply error to
            _time (float): time qubit is in storage
            
        Returns:
            /
        """

        depolar_prob = np.exp(-_time / self._depolar_time)
        
        gate_e0 = full_gates['P0'] + np.sqrt(depolar_prob) * full_gates['P1']
        gate_e1 = np.sqrt(1 - depolar_prob) * full_gates['P01']
        
        dephase_prob = 0.5 * (1 - np.exp(-_time * (1/self._dephase_time - 1/(2 * self._depolar_time))))
        dephase_prob_inv = 1 - dephase_prob
        
        key = f's_z_{_qubit.num_qubits}_{_qubit._index}'
        gate_e0 = get_single_operator('', gate_e0, _qubit._index, _qubit.num_qubits)
        gate_e1 = get_single_operator('', gate_e1, _qubit._index, _qubit.num_qubits)
        gate_z = get_single_operator(key, full_gates['Z'], _qubit._index, _qubit.num_qubits)
        
        _qubit.state = dot(_qubit.state, gate_e0) + dot(_qubit.state, gate_e1)
        _qubit.state = dephase_prob_inv * _qubit.state + dephase_prob * dot(_qubit.state, gate_z)
        
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
    
    def apply(self, _qubit: Qubit, _time: float=None) -> None:
        
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
    
    def apply(self, _qubit: Qubit, _time: float=None) -> None:
        
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
    
    def apply(self, _qubit: Qubit, _time: float=None) -> None:
        
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
    
    def apply(self, _qubit: Qubit, _time: float=None) -> None:
        
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
    
    def apply(self, _qubit: Qubit, _time: float=None) -> None:
        
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
    
    def apply(self, _qubit: Qubit, _time: float=None) -> None:
        
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
