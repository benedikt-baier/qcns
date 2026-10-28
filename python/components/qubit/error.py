import numpy as np

from qcns.python.components.qubit.qubit import Qubit

__all__ = ['DepolarizationError', 'DephasingError', 'DepolarizationDephasingError', 'RandomDepolarizationError', 'RandomDephasingError', 'RandomError', 'SystematicDepolarizationError', 'SystematicDephasingError', 'SystematicError', 'pauli_error']

full_gates = {'P0': np.array([[1, 0], [0, 0]], dtype=np.complex128),
              'P1': np.array([[0, 0], [0, 1]], dtype=np.complex128),
              'P01': np.array([[0, 1], [0, 0]], dtype=np.complex128),
              'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
              'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
              'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128)}

class AmplitudeDampingError:
    
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
        
        probability = 1 - np.exp(-_time / self._depolar_time)
        _qubit.state = _qubit.qsystem.backend.amplitude_damping(_qubit.state, _qubit._index, probability)
        
        return _qubit

class DepolarizationError(AmplitudeDampingError):
    """Legacy name for amplitude damping; retained for existing experiments.

    Use DepolarizingError for the isotropic, unital Pauli channel.
    """


class DepolarizingError:
    """Local isotropic noise: Bloch-vector contraction exp(-elapsed/time).

    Applies I with probability (1+3*exp(-t/T))/4 and each nonidentity
    Pauli with (1-exp(-t/T))/4, including on entangled inputs.
    """

    def __init__(self, depolar_time: float = 1e-3) -> None:
        if np.isnan(depolar_time) or depolar_time <= 0:
            raise ValueError('depolar_time must be positive')
        self.depolar_time = depolar_time

    def apply(self, qubit: Qubit, elapsed: float) -> Qubit:
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Noise elapsed time must be finite and nonnegative')
        contraction = np.exp(-elapsed / self.depolar_time)
        probability = (1 - contraction) / 4
        qubit.state = qubit.qsystem.backend.depolarizing(qubit.state, qubit._index, 3 * probability)
        return qubit


__all__ += ['AmplitudeDampingError', 'DepolarizingError']


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
        _qubit.state = _qubit.qsystem.backend.dephasing(_qubit.state, _qubit._index, dephase_prob)
        
        return _qubit

class DepolarizationDephasingError:
    
    """
    Represents a depolarization and dephasing Error
    
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
        
        dephase_prob = 0.5 * (1 - np.exp(-_time * (1/self._dephase_time - 1/(2 * self._depolar_time))))
        backend = _qubit.qsystem.backend
        _qubit.state = backend.amplitude_damping(_qubit.state, _qubit._index, 1 - depolar_prob)
        _qubit.state = backend.dephasing(_qubit.state, _qubit._index, dephase_prob)
        
        return _qubit
        
class _SampledNoise:
    def bind_rng(self, rng: np.random.Generator) -> None:
        self._rng = rng


class RandomDepolarizationError(_SampledNoise):
    
    """
    Represents a Random Depolarization Error
    
    Attr:
        _variance (float): variance of normally distributed angles
    """
    
    def __init__(self, variance: float, *, rng: np.random.Generator | None = None) -> None:
        
        """
        Initializes a Random Depolarization Error
        
        Args:
            variance (float): variance of normal distributed angles
            
        Returns:
            /
        """
        self._rng = np.random.default_rng() if rng is None else rng
        
        self._variance: float = variance
    
    def apply(self, _qubit: Qubit, _time: float=None) -> None:
        
        """
        Applies the Error to qubits
        
        Args:
            _qubit (Qubit): qubit to apply error to
            
        Returns:
            /
        """

        theta = self._rng.normal(0, self._variance)
        _qubit.Rx(theta)
        
        return _qubit    
            
class RandomDephasingError(_SampledNoise):
    
    """
    Represents a Random Dephasing Error
    
    Attr:
        _variance (float): variance of normally distributed angles
    """
    
    def __init__(self, variance: float, *, rng: np.random.Generator | None = None) -> None:
        
        """
        Initializes a Random Dephasing Error
        
        Args:
            variance (float): _variance of dephasing angle
        
        Returns:
            /
        """
        self._rng = np.random.default_rng() if rng is None else rng
        
        self._variance: float = variance
    
    def apply(self, _qubit: Qubit, _time: float=None) -> None:
        
        """
        Applies the Error to qubits
        
        Args:
            _qubit (Qubit): qubit to apply error to
            
        Returns:
            /
        """
        
        theta = self._rng.normal(0, self._variance)
        _qubit.Rz(theta)
        
        return _qubit
            
class RandomError(_SampledNoise):
    
    """
    Represents a Random Error
    
    Attr:
        _variance (float): variance of normally distributed angles
    """
    
    def __init__(self, x_variance: float, z_variance: float, *, rng: np.random.Generator | None = None) -> None:
        
        """
        Initializes a Random Error
        
        Args:
            x_variance (float): variance of depolarization angle
            z_variance (float): variance of dephasing angle
            
        Returns:
            /
        """
        self._rng = np.random.default_rng() if rng is None else rng
        
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

        theta_x = self._rng.normal(0, self._x_variance)
        theta_z = self._rng.normal(0, self._z_variance)
        _qubit.Rx(theta_x)
        _qubit.Rz(theta_z)
        
        return _qubit
        
class SystematicDepolarizationError:
    
    """
    Represents a Systematic Depolarization Error
    
    _theta (float): depolarization angle
    """
    
    def __init__(self, variance: float, *, rng: np.random.Generator | None = None) -> None:
        
        """
        Initializes a Systematic Depolarization Error
        
        Args:
            variance (float): variance of once drawn depolarization angle
            
        Returns:
            /
        """
        self._rng = np.random.default_rng() if rng is None else rng
        
        self._theta: float = self._rng.normal(0, variance)
    
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
    
    def __init__(self, variance: float, *, rng: np.random.Generator | None = None) -> None:
        
        """
        Initializes a Systematic Dephasing Error
        
        Args:
            variance (float): variance of normal distributed dephasing angle
            
        Returns:
            /
        """
        self._rng = np.random.default_rng() if rng is None else rng
        
        self._theta: float = self._rng.normal(0, variance)
    
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

    def __init__(self, x_variance: float, z_variance: float, *, rng: np.random.Generator | None = None) -> None:
        
        """
        Initializes a Systematic Error
        
        Args:
            x_variance (float): variance of normal distributed depolarization angle
            z_variance (float): variance of normal distributed dephasing angle
            
        Returns:
            /
        """
        self._rng = np.random.default_rng() if rng is None else rng
        
        self._theta_x: float = self._rng.normal(0, x_variance)
        self._theta_z: float = self._rng.normal(0, z_variance)
    
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

def pauli_error(_qubit: Qubit, p_i: float, p_x: float, p_y: float, p_z: float) -> Qubit:
        
    key_x = f's_x_{_qubit.num_qubits}_{_qubit._index}'
    key_y = f's_y_{_qubit.num_qubits}_{_qubit._index}'
    key_z = f's_z_{_qubit.num_qubits}_{_qubit._index}'
    
    backend = _qubit.qsystem.backend
    x_state = backend.single(_qubit.state, full_gates['X'], _qubit._index, key=key_x)
    y_state = backend.single(_qubit.state, full_gates['Y'], _qubit._index, key=key_y)
    z_state = backend.single(_qubit.state, full_gates['Z'], _qubit._index, key=key_z)
    _qubit.state = p_i * _qubit.state + p_x * x_state + p_y * y_state + p_z * z_state
    
    return _qubit
