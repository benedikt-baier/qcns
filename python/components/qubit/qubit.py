import numpy as np
from collections import OrderedDict
from functools import wraps, reduce
from typing import List, Union, Callable
from .backend import PythonBackend, get_backend

__all__ = ['Qubit', 'QSystem', 'tensor_operator', 'dot', 'get_single_operator', 'depolarization_error', 'combine_state', 'combine_gates', 'remove_qubits']

full_gates = {'P0': np.array([[1, 0], [0, 0]], dtype=np.complex128),
              'P1': np.array([[0, 0], [0, 1]], dtype=np.complex128),
              'P01': np.array([[0, 1], [0, 0]], dtype=np.complex128),
              'P10': np.array([[0, 0], [1, 0]], dtype=np.complex128),
              'I': np.array([[1, 0], [0, 1]], dtype=np.complex128),
              'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
              'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
              'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128),
              'H': np.array([[1, 1], [1, -1]], dtype=np.complex128) / (np.sqrt(2)),
              'K': 0.5 * np.array([[1 + 1j, 1 - 1j], [-1 + 1j, -1 - 1j]], dtype=np.complex128),
              'T': np.array([[1, 0], [0, np.exp(1j * 0.25 * np.pi)]], dtype=np.complex128),
              'SX':  0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=np.complex128),
              'SY': 0.5 * np.array([[1 + 1j, -1 - 1j], [1 + 1j, 1 + 1j]], dtype=np.complex128),
              'SZ': np.array([[1, 0], [0, 1j]], dtype=np.complex128),
              'iK': 0.5 * np.array([[1 - 1j, -1 - 1j], [1 + 1j, -1 + 1j]], dtype=np.complex128),
              'iT': np.array([[1, 0], [0, np.exp(-1j * 0.25 * np.pi)]], dtype=np.complex128),
              'iSX': 0.5 * np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]], dtype=np.complex128),
              'iSY': 0.5 * np.array([[1 - 1j, 1 - 1j], [-1 + 1j, 1 - 1j]], dtype=np.complex128),
              'iSZ': np.array([[1, 0], [0, -1j]], dtype=np.complex128)}

purification_gates = {'CNOT': 'X', 'CY': 'Y', 'CZ': 'Z', 'CH': 'H'}

class Qubit:
    pass

class QSystem:
    pass

DEFAULT_DENSE_BUDGET = 256 * 1024 * 1024


def check_dense_allocation(qubits: int, max_bytes: int) -> int:
    """Conservative four-array working-set estimate for complex128 kernels."""
    if not isinstance(qubits, int) or isinstance(qubits, bool) or qubits < 1:
        raise ValueError('num_qubits must be a positive integer')
    if not isinstance(max_bytes, int) or max_bytes < 1:
        raise ValueError('max_bytes must be a positive integer')
    required = 4 * 16 * 4**qubits
    if required > max_bytes:
        raise MemoryError(f'Dense {qubits}-qubit operation needs approximately {required} bytes; '
                          f'budget is {max_bytes}. Increase max_bytes explicitly or use a validated reduced backend.')
    return required

def cache(func: Callable) -> Callable:
    
    """
    Caches the applied gates to reduce calculation times

    Args:
        func (Function): function which input to cache

    Returns:
        wrapper (Function): wrapper function
    """

    _cache: OrderedDict[str, np.ndarray] = OrderedDict()
    limit = 8 * 1024 * 1024
    retained = 0

    def configure(max_bytes: int = 8 * 1024 * 1024) -> None:
        nonlocal limit, retained
        if not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError('Cache max_bytes must be a nonnegative integer')
        limit = max_bytes
        while retained > limit:
            retained -= _cache.popitem(last=False)[1].nbytes

    def clear() -> None:
        nonlocal retained
        _cache.clear()
        retained = 0

    def info() -> dict[str, int]:
        return {'entries': len(_cache), 'bytes': retained, 'max_bytes': limit}
    
    @wraps(func)
    def wrapper(*args: object) -> np.ndarray:
        nonlocal retained
        
        key = args[0]
        if not key:
            return func(*(args[1:]))
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
        value = func(*(args[1:]))
        if value.nbytes <= limit:
            while retained + value.nbytes > limit:
                retained -= _cache.popitem(last=False)[1].nbytes
            _cache[key] = value
            retained += value.nbytes
        return value
    wrapper.cache_configure = configure
    wrapper.cache_clear = clear
    wrapper.cache_info = info
    return wrapper

def kronecker_product(_op_1: np.ndarray, _op_2: np.ndarray) -> np.array:
    
    """
    Computes the kronecker product of two full 2d matrices

    Args:
        _op_1 (np.array): first matrix
        _op_2 (np.array): second matrix

    Returns:
        op (np.array): result of the kronecker product
    """

    _op = _op_1[:, None, :, None] * _op_2[None, :, None, :]
    
    return _op.reshape(_op_1.shape[0] * _op_2.shape[0], _op_1.shape[1] * _op_2.shape[1])

def tensor_operator(_operator_l: np.ndarray) -> np.ndarray:
    
    """
    Creates the tensor operator out of a operator list
    
    Args:
        _operator_l (list): list of 2x2 np.arrays
        
    Returns:
        res (np.array): resulting tensor operator
    """

    return reduce(kronecker_product, _operator_l)

def dot(_state: np.ndarray, _gate: np.ndarray) -> np.ndarray:
    
    """
    Performs the linalg dot for quantum gates
    
    Args:
        _state (np.array): state to apply the gate to
        _gate (np.array): gate to apply
        
    Returns:
        _state (np.array): resulting state
    """
    
    return _gate.dot(_state).dot(_gate.conj().T)

def sqrt_matrix(_state: np.ndarray) -> np.ndarray:
    
    """
    Calculates the square root of a matrix
    
    Args:
        _state (np.array): _state to calculate square root of
        
    Returns:
        _sqrt (np.array): square root of matrix
    """
    
    evs, vecs = np.linalg.eigh(_state)
    return dot(np.diag(np.sqrt(np.abs(evs))), vecs)

@cache
def get_single_operator(_gate: np.ndarray, _index: int, _num_qubits: int) -> np.ndarray:

    """
    Creates the tensor operator for a single qubit gate
    
    Args:
        _gate (np.array): 2x2 gate to apply to the target qubit
        _index (int): index of target qubit in QSystem
        _num_qubits (int): number of qubits in QSystem
        
    Returns:
        _op (np.array): resulting tensor for the single qubit
    """
    
    if _num_qubits < 2:
        return _gate
    
    if _num_qubits < 3:
        operator_l = np.array([full_gates['I']] * _num_qubits)  
        operator_l[_index] = _gate
        
        return tensor_operator(operator_l)

    if _index == 0:
        return kronecker_product(_gate, np.eye(2**(_num_qubits - 1), 2**(_num_qubits - 1), dtype=np.complex128))
        
    if _index == _num_qubits - 1:
        return kronecker_product(np.eye(2**(_num_qubits - 1), 2**(_num_qubits - 1), dtype=np.complex128), _gate)
    
    left_array = np.eye(2**_index, 2**_index, dtype=np.complex128)
    right_array = np.eye(2**(_num_qubits - _index - 1), 2**(_num_qubits - _index - 1), dtype=np.complex128)
    
    return kronecker_product(left_array, kronecker_product(_gate, right_array))

@cache
def get_double_operator(_gate: np.ndarray, _c_index: int, _t_index: int, _t_num_qubits: int) -> np.ndarray:
    
    """
    Creates the tensor operator for a controlled qubit gate
    
    Args:
        _gate (np.array): 2x2 gate to apply to the target qubit
        _c_index (int): index of control qubit
        _t_index (int): index of target index
        _t_num_qubits (int): number of qubits in target QSystem
        
    Returns:
        _op (np.array): resulting tensor for the controlled qubit gate
    """
    
    key = f's_m0_{1}_{1}_{_t_num_qubits}_{_c_index}'
    proj0 = get_single_operator(key, full_gates['P0'], _c_index, _t_num_qubits)

    gate_dict = {_c_index: full_gates['P1'], _t_index: _gate}
    index_1, index_2 = sorted([_c_index, _t_index])
    
    left_index = index_1
    middle_index = index_2 - index_1 - 1
    right_index = _t_num_qubits - index_2 - 1
    
    proj1 = gate_dict[index_1]
    if left_index:
        proj1 = kronecker_product(np.eye(2**left_index, 2**left_index, dtype=np.complex128), proj1)
    if middle_index:
        proj1 = kronecker_product(proj1, np.eye(2**middle_index, 2**middle_index, dtype=np.complex128))
    proj1 = kronecker_product(proj1, gate_dict[index_2])
    if right_index:
        proj1 = kronecker_product(proj1, np.eye(2**right_index, 2**right_index, dtype=np.complex128))
    
    return proj0 + proj1

@cache
def get_triple_operator(_gate_l: np.ndarray, _c1_index: int, _c2_index: int, _t_index: int, _t_num_qubits: int) -> np.ndarray:
    
    """
    Generates the operator for a 3 qubit gate
    
    Args:
        _gate_l (np.array): gate list to apply to target
        _c1_index (int): index of first control qubit
        _c2_index (int): index of second control qubit
        _t_index (int): index of target qubit
        _t_num_qubits (int): number of qubits in target QSystem
        
    Returns:
        _op (np.array): resulting tensor for the controlled controlled qubit gate
    """
    
    gate_dict = {_c1_index: [full_gates['P0'], full_gates['P0'], full_gates['P1'], full_gates['P1']],
                  _c2_index: [full_gates['P0'], full_gates['P1'], full_gates['P0'], full_gates['P1']],
                  _t_index: _gate_l}
    
    index_1, index_2, index_3 = sorted([_c1_index, _c2_index, _t_index])
    indices = [index_1, index_2 - index_1 - 1, index_3 - index_2 - 1, _t_num_qubits - index_3 - 1]
    
    gate_list = [[0] * 7 for _ in range(4)]
    
    for i in range(4):
        gate_list[i][::2] = [np.eye(2**(index), 2**(index), dtype=np.complex128) if index else None for index in indices]
        gate_list[i][1::2] = [gate_dict[index_1][i], gate_dict[index_2][i], gate_dict[index_3][i]]
        gate_list[i] = [gate for gate in gate_list[i] if gate is not None]
    
    return tensor_operator(gate_list[0]) + tensor_operator(gate_list[1]) + tensor_operator(gate_list[2]) + tensor_operator(gate_list[3])
    
@cache
def get_bell_operator(_bell_state: int, _c_index: int, _t_index: int, _t_num_qubits: int) -> np.ndarray:
    
    """
    Generates the bell operator
    
    Args:
        _bell_state (int): bell state to calculate
        _c_index (int): index of control qubit
        _t_index (int): index of target qubit
        _t_num_qubits (int): number of qubits in target QSystem
        
    Returns:
        _op (np.array): resulting tensor for the bell state operator
    """
    
    h_gate = get_single_operator(f's_h_{_t_num_qubits}_{_c_index}', full_gates['H'], _c_index, _t_num_qubits)
    cnot_gate = get_double_operator(f'd_x_{_t_num_qubits}_{_c_index}_{_t_index}', full_gates['X'], _c_index, _t_index, _t_num_qubits)
    
    if _bell_state == 0:
        return cnot_gate.dot(h_gate)
    if _bell_state == 1:
        x_gate = get_single_operator(f's_x_{_t_num_qubits}_{_t_index}', full_gates['X'], _t_index, _t_num_qubits)
        return cnot_gate.dot(h_gate).dot(x_gate)
    if _bell_state == 2:
        x_gate = get_single_operator(f's_x_{_t_num_qubits}_{_c_index}', full_gates['X'], _c_index, _t_num_qubits)
        return cnot_gate.dot(h_gate).dot(x_gate)
    if _bell_state == 3:
        x_gate_c = get_single_operator(f's_x_{_t_num_qubits}_{_c_index}', full_gates['X'], _c_index, _t_num_qubits)
        x_gate_t = get_single_operator(f's_x_{_t_num_qubits}_{_t_index}', full_gates['X'], _t_index, _t_num_qubits)
        return cnot_gate.dot(h_gate).dot(x_gate_c).dot(x_gate_t)

@cache
def get_bsm_operator(_c_index: int, _t_index: int, _t_num_qubits: int) -> np.ndarray:
    
    """
    Generates the bsm operator
    
    Args:
        _c_index (int): index of control qubit
        _t_index (int): index of target qubit
        _t_num_qubits (int): number of qubits in target QSystem
        
    Returns:
        _op (np.array): resulting tensor for the bsm operator
    """
    
    cnot_gate = get_double_operator(f'd_x_{_t_num_qubits}_{_c_index}_{_t_index}', full_gates['X'], _c_index, _t_index, _t_num_qubits)
    h_gate = get_single_operator(f's_h_{_t_num_qubits}_{_c_index}', full_gates['H'], _c_index, _t_num_qubits)
    
    return h_gate.dot(cnot_gate) 

@cache
def get_swap_operator(_index_1: int, _index_2: int, _num_qubits: int) -> np.ndarray:
    
    """
    Generates the swap operator
    
    Args:
        _index_1 (int): index of first qubit
        _index_2 (int): index of second qubit
        _num_qubits (int):  number of qubits in QSystem
        
    Returns:
        _op (np.array): resulting tensor for the bsm operator
    """
    
    _op = np.array([full_gates['I']] * _num_qubits)
    
    _op[_index_1] = full_gates['P0']
    _op[_index_2] = full_gates['P0']
    
    _res = tensor_operator(_op)
    
    _op[_index_1] = full_gates['P01']
    _op[_index_2] = full_gates['P10']
    
    _res += tensor_operator(_op)
    
    _op[_index_1] = full_gates['P10']
    _op[_index_2] = full_gates['P01']
    
    _res += tensor_operator(_op)
    
    _op[_index_1] = full_gates['P1']
    _op[_index_2] = full_gates['P1']
    
    return _res + tensor_operator(_op)

def depolarization_error(_qubit: Qubit, _fidelity: float) -> None:
    
    """
    Applies a depolarization error to the qubit given a fidelity
    
    Args:
        _qubit (Qubit): qubit to apply error to
        _fidelity (float): fidelity of the depolarization error
        
    Returns:
        /
    """
    
    _qubit.state = _qubit.qsystem.backend.depolarizing(_qubit.state, _qubit._index, 1 - _fidelity)

def combine_state(q_l: List[Qubit | QSystem]) -> QSystem:
        
    """
    Combines multiple qsystems into one qsystem
    
    Args:
        q_l (list): List of qsystems
        
    Returns:
        qsys_n (QSystem): new qsystem
    """
    
    q_l = [q.qsystem for q in q_l]

    qsystem_ids = set()
    qsys_l = []
    for qsys in q_l:
        if qsys.qsystem_id in qsystem_ids:
            continue
        qsystem_ids.add(qsys.qsystem_id)
        qsys_l.append(qsys)

    qsys_n = qsys_l[0]

    if len(qsys_l) < 2:
        return qsys_n

    total = sum(qsys.num_qubits for qsys in qsys_l)
    if any(type(qsys.backend) is not type(qsys_n.backend) for qsys in qsys_l):
        raise ValueError('Cannot combine states from different backends; create them with the same backend')
    qsys_n.backend.validate_size(total)
    budget = min(qsys.max_bytes for qsys in qsys_l)
    check_dense_allocation(total, budget)
    qsys_n.max_bytes = budget
    qsys_n._qubits = [q for qsys in qsys_l for q in qsys._qubits]
    qsys_n._num_qubits = sum(qsys.num_qubits for qsys in qsys_l)
    qsys_n._state = tensor_operator([qsys._state for qsys in qsys_l]).astype(np.complex128, copy=False)

    for i, qubit in enumerate(qsys_n._qubits):
        qubit._index = i
        qubit._qsystem = qsys_n

    return qsys_n

def combine_gates(gate_l: List[np.ndarray]) -> np.ndarray:
    
    """
    Combines multiple gates into one gate
    
    Args:
        gate_l (list): List of gates
        
    Returns:
        gate_n (np.array): new gate
    """
    
    if len(gate_l) < 2:
        return gate_l[0]
    
    return reduce(np.dot, gate_l[::-1])
    
def ptrace_full(_q: Qubit) -> None:
    
    """
    Traces out a qubit from the state, adjusts the QSystem accordingly
    
    Args:
        _q (Qubit): qubit to trace out
        
    Returns:
        /
    """

    _q.state = _q.qsystem.backend.partial_trace(_q.state, _q._index)
    _q.num_qubits -= 1
    
    del _q._qsystem._qubits[_q._index]  
    for i in range(_q.num_qubits):
        _q._qsystem._qubits[i]._index = i

def remove_qubits(q_l: List[Qubit]) -> None:

    """
    Traces out the given qubits from the state, adjusts the QSystem accordingly
    
    Args:
        _q_l (list): list of qubits to trace out
        
    Returns:
        /
    """
    
    for qubit in q_l:
        ptrace_full(qubit)

class Qubit:
    
    """
    Represents a single qubit
    
    Attr:
        _qsystem (QSystem): qsystem qubit belongs to
        _index (int): index of qubit in qsystem
    """
    
    def __init__(self, _qsystem: QSystem, _index: int) -> None:
        
        """
        Instantiate a single qubit
        
        Args:
            _qsystem (QSystem): reference to parent qsystem, which the qubit is part of
            _index (int): index of the qubit in the parent qsystem
            
        Returns:
            /
        """
    
        self._qsystem: QSystem = _qsystem
        self._index: int = _index
    
    def __repr__(self) -> str:
        
        """
        Custom print method
        
        Args:
            /
        
        Returns:
            state (str): state of the overall qsystem
        """
        
        return str(self.state)
    
    @property
    def state(self) -> np.array:
        
        """
        Returns the full state of the qubit
        
        Args:
            /
            
        Returns:
            _state (np.array): full state of the qubit
        """
        
        return self._qsystem._state
    
    @state.setter
    def state(self, state: np.ndarray) -> None:
        
        """
        Sets the state of QSystem
        
        Args:
            state (np.array): new state of the qsystem
            
        Returns:
            /
        """
        
        self._qsystem._state = state
    
    @property
    def num_qubits(self) -> int:
        
        """
        Returns the number of qubits in the qsystem
        
        Args:
            /
            
        Returns:
            num_qubits (int): number of qubits in the qsystem
        """
        
        return self._qsystem._num_qubits
    
    @num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        
        """
        Sets the number of qubits in the qsystem
        
        Args:
            num_qubits (int): new number of qubits
            
        Returns:
            /
        """
        
        self._qsystem._num_qubits = num_qubits
    
    @property
    def qsystem(self) -> QSystem:
        
        """
        Returns the Qsystem the qubit is a part of
        
        Args:
            /
            
        Returns:
            qsystem (QSystem): qsystem the qubit is a part of
        """
        
        return self._qsystem
    
    @property
    def qubit_id(self) -> int:
        
        """
        Returns the ID of the qubit
        
        Args:
            /
            
        Returns:
            _id (int): id of qubit
        """
        
        return id(self)
    
    @property
    def qsystem_id(self) -> int:
        
        """
        Returns the ID of the qsystem the qubit is in
        
        Args:
            /
            
        Returns:
            _id (int): id of the qsystem the qubit is in
        """
        
        return id(self._qsystem)
    
    def _single_gate(self, gate: np.ndarray, fidelity: float, apply: bool, *, key: str) -> np.ndarray | None:
        if not apply:
            return get_single_operator(key, gate, self._index, self.num_qubits)
        self.state = self.qsystem.backend.single(self.state, gate, self._index, key=key)
        if fidelity < 1.:
            depolarization_error(self, fidelity)
        return None

    def _controlled_gate(self, target: Qubit, gate: np.ndarray, fidelity: float, apply: bool,
                         *, key: str, cnot: bool = False) -> np.ndarray | None:
        if self.qsystem is not target.qsystem or self is target:
            raise ValueError('Control and target must be distinct qubits in the same QSystem; use combine_state first')
        if not apply:
            return get_double_operator(key, gate, self._index, target._index, self.num_qubits)
        backend = self.qsystem.backend
        if cnot:
            self.state = backend.cnot(self.state, self._index, target._index, key=key)
        else:
            self.state = backend.controlled(self.state, gate, self._index, target._index, key=key)
        if fidelity < 1.:
            depolarization_error(self, fidelity)
            depolarization_error(target, fidelity)
        return None

    def X(self, fidelity: float=1., apply: bool=True) -> np.ndarray | None:

        """
        Applies the X gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f's_x_{self.num_qubits}_{self._index}'
        return self._single_gate(full_gates['X'], fidelity, apply, key=key)
        
    def Y(self, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Applies the Y gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f's_y_{self.num_qubits}_{self._index}'
        return self._single_gate(full_gates['Y'], fidelity, apply, key=key)
        
    def Z(self, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Applies the Z gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f's_z_{self.num_qubits}_{self._index}'
        return self._single_gate(full_gates['Z'], fidelity, apply, key=key)
        
    def H(self, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Applies the Hadamard gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f's_h_{self.num_qubits}_{self._index}'
        return self._single_gate(full_gates['H'], fidelity, apply, key=key)

    def SX(self, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Applies the square root X gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f's_sx_{self.num_qubits}_{self._index}'
        return self._single_gate(full_gates['SX'], fidelity, apply, key=key)
    
    def SY(self, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Applies the square root Y gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f's_sy_{self.num_qubits}_{self._index}'
        return self._single_gate(full_gates['SY'], fidelity, apply, key=key)
    
    def SZ(self, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Applies the square root Z gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f's_sz_{self.num_qubits}_{self._index}'
        return self._single_gate(full_gates['SZ'], fidelity, apply, key=key)
    
    def T(self, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Applies the T gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
    
        key = f's_t_{self.num_qubits}_{self._index}'
        return self._single_gate(full_gates['T'], fidelity, apply, key=key)
    
    def K(self, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Applies the K gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f's_k_{self.num_qubits}_{self._index}'
        return self._single_gate(full_gates['K'], fidelity, apply, key=key)
    
    def iSX(self, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Applies the inverse square root X gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f's_isx_{self.num_qubits}_{self._index}'
        return self._single_gate(full_gates['iSX'], fidelity, apply, key=key)
    
    def iSY(self, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Applies the inverse square root Y gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f's_isy_{self.num_qubits}_{self._index}'
        return self._single_gate(full_gates['iSY'], fidelity, apply, key=key)
    
    def iSZ(self, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Applies the inverse square root Z gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f's_isz_{self.num_qubits}_{self._index}'
        return self._single_gate(full_gates['iSZ'], fidelity, apply, key=key)
    
    def iT(self, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Applies the inverse of T gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f's_it_{self.num_qubits}_{self._index}'
        return self._single_gate(full_gates['iT'], fidelity, apply, key=key)
    
    def iK(self, fidelity: float=1., apply: bool=True):
        
        """
        Applies the inverse of K gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f's_ik_{self.num_qubits}_{self._index}'
        return self._single_gate(full_gates['iK'], fidelity, apply, key=key)
    
    def Rx(self, theta: float, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Applies the rotation gate around the x axis to the qubit
        
        Args:
            theta (float): angle for the rotation matrix
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f's_rx_{theta}_{self.num_qubits}_{self._index}'
        gate_s = np.array([[np.cos(theta/2), -1j * np.sin(theta/2)], [-1j * np.sin(theta/2), np.cos(theta/2)]], dtype=np.complex128)
        return self._single_gate(gate_s, fidelity, apply, key=key)
    
    def Ry(self, theta: float, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Applies the rotation gate around the y axis to the qubit
        
        Args:
            theta (float): angle for the rotation matrix
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f's_ry_{theta}_{self.num_qubits}_{self._index}'
        gate_s = np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]], dtype=np.complex128)
        return self._single_gate(gate_s, fidelity, apply, key=key)
        
    def Rz(self, theta: float, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Applies the rotation gate around the z axis to the qubit
        
        Args:
            theta (float): angle for the rotation matrix
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f's_rz_{theta}_{self.num_qubits}_{self._index}'
        gate_s = np.array([[np.exp(-1j * theta/2), 0], [0, np.exp(1j * theta/2)]], dtype=np.complex128)
        return self._single_gate(gate_s, fidelity, apply, key=key)
        
    def PHASE(self, theta: float, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Applies the PHASE gate to the qubit
        
        Args:
            theta (float): angle for the rotation matrix
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f's_p_{theta}_{self.num_qubits}_{self._index}'
        gate_s = np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=np.complex128)
        return self._single_gate(gate_s, fidelity, apply, key=key)
    
    def general_rotation(self, theta: float, phi: float, psi: float, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Applies a general rotation to a single qubit
        
        Args:
            theta (float): angle theta
            phi (float): angle phi
            psi (float): angle psi
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f's_gr_{theta}_{phi}_{psi}_{self.num_qubits}_{self._index}'
        gate_s = np.array([[np.cos(theta/2), -np.exp(1j*psi)*np.sin(theta/2)], [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+psi))*np.cos(theta/2)]], dtype=np.complex128)
        return self._single_gate(gate_s, fidelity, apply, key=key)
    
    def custom_single_gate(self, gate: np.ndarray, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Applies a custom gate to the qubit
        
        Args:
            gate (np.array): unitary gate to apply
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        return self._single_gate(gate, fidelity, apply, key='')
     
    def CNOT(self, target: Qubit, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Applies the CNOT gate to a target qubit, with this qubit as the controll qubit
        
        Args:
            target (qubit): target qubit to apply the CNOT gate to
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'd_x_{target.num_qubits}_{self._index}_{target._index}'
        return self._controlled_gate(target, full_gates['X'], fidelity, apply, key=key, cnot=True)

    def CX(self, target: Qubit, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Applies the CNOT gate to a target qubit, with this qubit as the controll qubit
        
        Args:
            target (qubit): target qubit to apply the CNOT gate to
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'd_x_{target.num_qubits}_{self._index}_{target._index}'
        return self._controlled_gate(target, full_gates['X'], fidelity, apply, key=key, cnot=True)

    def CY(self, target: Qubit, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Applies the CY gate to a target qubit, with this qubit as the control qubit
        
        Args:
            target (qubit): target qubit to apply the CNOT gate to
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'd_y_{target.num_qubits}_{self._index}_{target._index}'
        return self._controlled_gate(target, full_gates['Y'], fidelity, apply, key=key)
    
    def CZ(self, target: Qubit, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Applies the CZ gate to a target qubit, with this qubit as the control qubit
        
        Args:
            target (qubit): target qubit to apply the CNOT gate to
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'd_z_{target.num_qubits}_{self._index}_{target._index}'
        return self._controlled_gate(target, full_gates['Z'], fidelity, apply, key=key)
    
    def CH(self, target: Qubit, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Applies the CH gate to a target qubit, with this qubit as the control qubit
        
        Args:
            target (qubit): target qubit to apply the CNOT gate to
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'd_h_{target.num_qubits}_{self._index}_{target._index}'
        return self._controlled_gate(target, full_gates['H'], fidelity, apply, key=key)
    
    def CPHASE(self, target: Qubit, theta: float, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Applies the CPHASE gate to a target qubit, with this qubit as the controll qubit
        
        Args:
            target (qubit): target qubit to apply the CPHASE gate to
            theta (float): angle for the rotation matrix
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'd_p_{theta}_{target.num_qubits}_{self._index}_{target._index}'
        gate_s = np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=np.complex128)
        return self._controlled_gate(target, gate_s, fidelity, apply, key=key)
    
    def CU(self, target: Qubit, gate: np.ndarray, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Applies a custom gate to a target qubit, with this qubit as the controll qubit
        
        Args:
            target (qubit): target qubit to apply a custom gate to
            gate (np.array): 2x2 unitary array
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        return self._controlled_gate(target, gate, fidelity, apply, key='')
    
    def SWAP(self, target: Qubit, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Swaps the state of this qubit with the target qubit by applying CNOT gates
        
        Args:
            target (qubit): target qubit to swap state with
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'd_sw_{self.num_qubits}_{self._index}_{target._index}'
        if not apply:
            return get_swap_operator(key, self._index, target._index, self.num_qubits)
        
        self.CNOT(target)
        target.CNOT(self)
        self.CNOT(target)
        self._qsystem._qubits[self._index], self._qsystem._qubits[target._index] = self._qsystem._qubits[target._index], self._qsystem._qubits[self._index]
        self._index, target._index = target._index, self._index
        
        if fidelity < 1.:
            depolarization_error(self, fidelity)
            depolarization_error(target, fidelity)
    
    def iSWAP(self, target: Qubit, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applies the Imaginary Swap gate to this qubit and target qubit
        
        Args:
            target (qubit): target qubit to apply the CPHASE gate to
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        s_t_swap = self.SWAP(target, apply=apply)
        s_sz = self.SZ(apply=apply)
        t_sz = target.SZ(apply=apply)
        t_h = target.H(apply=apply)
        s_t_cx = self.CNOT(target, apply=apply)
        t_h = target.H(apply=apply)
        
        if not apply:
            return combine_gates([s_t_swap, s_sz, t_sz, t_h, s_t_cx, t_h])
        
        if fidelity < 1.:
            depolarization_error(self, fidelity)
            depolarization_error(target, fidelity)
    
    def QAND(self, control: Qubit, target: Qubit, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applies the quantum AND gate to the target qubit with this qubit and control qubit as control qubits
        
        Args:
            control (qubit): second control qubit
            target (qubit): target qubit
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f't_iiix_{target.num_qubits}_{self._index}_{control._index}_{target._index}'
        gate = get_triple_operator(key, 
                                    [full_gates['I'], full_gates['I'], full_gates['I'], full_gates['X']], 
                                    self._index, control._index, target._index, target.num_qubits)
        
        if not apply:
            return gate    
    
        target.state = dot(target.state, gate)
        if fidelity < 1.:
            depolarization_error(self, fidelity)
            depolarization_error(control, fidelity)
            depolarization_error(target, fidelity)
    
    def QOR(self, control: Qubit, target: Qubit, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applies the quantum OR gate to the target qubit with this qubit and control qubit as control qubits
        
        Args:
            control (qubit): second control qubit
            target (qubit): target qubit
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f't_ixxx_{target.num_qubits}_{self._index}_{control._index}_{target._index}'
        gate = get_triple_operator(key, 
                                    [full_gates['I'], full_gates['X'], full_gates['X'], full_gates['X']], 
                                    self._index, control._index, target._index, target.num_qubits)
        
        if not apply:
            return gate
        
        target.state = dot(target.state, gate)
        if fidelity < 1.:
            depolarization_error(self, fidelity)
            depolarization_error(control, fidelity)
            depolarization_error(target, fidelity)
    
    def QXOR(self, control: Qubit, target: Qubit, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applies the quantum XOR gate to the target qubit with this qubit and control qubit as control qubits
        
        Args:
            control (qubit): second control qubit
            target (qubit): target qubit
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f't_ixxi_{target.num_qubits}_{self._index}_{control._index}_{target._index}'
        gate = get_triple_operator(key, 
                                    [full_gates['I'], full_gates['X'], full_gates['X'], full_gates['I']], 
                                    self._index, control._index, target._index, target.num_qubits)
        
        if not apply:
            return gate
        
        target.state = dot(target.state, gate)
        if fidelity < 1.:
            depolarization_error(self, fidelity)
            depolarization_error(control, fidelity)
            depolarization_error(target, fidelity)
    
    def QNAND(self, control: Qubit, target: Qubit, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applies the quantum NAND gate to the target qubit with this qubit and control qubit as control qubits
        
        Args:
            control (qubit): second control qubit
            target (qubit): target qubit
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f't_xxxi_{target.num_qubits}_{self._index}_{control._index}_{target._index}'
        gate = get_triple_operator(key, 
                                    [full_gates['X'], full_gates['X'], full_gates['X'], full_gates['I']], 
                                    self._index, control._index, target._index, target.num_qubits)
        
        if not apply:
            return gate
        
        target.state = dot(target.state, gate)
        if fidelity < 1.:
            depolarization_error(self, fidelity)
            depolarization_error(control, fidelity)
            depolarization_error(target, fidelity)
    
    def QNOR(self, control: Qubit, target: Qubit, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applies the quantum NOR gate to the target qubit with this qubit and control qubit as control qubits
        
        Args:
            control (qubit): second control qubit
            target (qubit): target qubit
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f't_xiii_{target.num_qubits}_{self._index}_{control._index}_{target._index}'
        gate = get_triple_operator(key, 
                                    [full_gates['X'], full_gates['I'], full_gates['I'], full_gates['I']], 
                                    self._index, control._index, target._index, target.num_qubits)
        
        if not apply:
            return gate
        
        target.state = dot(target.state, gate)
        if fidelity < 1.:
            depolarization_error(self, fidelity)
            depolarization_error(control, fidelity)
            depolarization_error(target, fidelity)
    
    def QXNOR(self, control: Qubit, target: Qubit, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applies the quantum XNOR gate to the target qubit with this qubit and control qubit as control qubits
        
        Args:
            control (qubit): second control qubit
            target (qubit): target qubit
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f't_xiix_{target.num_qubits}_{self._index}_{control._index}_{target._index}'
        gate = get_triple_operator(key, 
                                    [full_gates['X'], full_gates['I'], full_gates['I'], full_gates['X']], 
                                    self._index, control._index, target._index, target.num_qubits)
        
        if not apply:
            return gate
        
        target.state = dot(target.state, gate)
        if fidelity < 1.:
            depolarization_error(self, fidelity)
            depolarization_error(control, fidelity)
            depolarization_error(target, fidelity)
    
    def CCU(self, control: Qubit, target: Qubit, gate: np.ndarray, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applies a custom unitary gate to controled by this qubit and control qubit
        
        Args:
            control (qubit): second control qubit
            target (qubit): target qubit
            gate (np.array): gate to apply
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        gate = get_triple_operator('', 
                                        [full_gates['I'], full_gates['I'], full_gates['I'], gate],
                                        self._index, control._index, target._index, target.num_qubits)
        
        if not apply:
            return gate
        
        target.state = dot(target.state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, fidelity)
            depolarization_error(control, fidelity)
            depolarization_error(target, fidelity)
    
    def CSWAP(self, target1: Qubit, target2: Qubit, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applies the CSWAP gate with self as control and target1 and target2 as targets
        
        Args:
            target1 (qubit): first target qubit
            target2 (qubit): second target qubit
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
            
        t2_t1 = self.TOFFOLI(target2, target1, apply=apply)
        t1_t2 = self.TOFFOLI(target1, target2, apply=apply)
        t2_t1 = self.TOFFOLI(target2, target1, apply=apply)
        
        if not apply:
            return combine_gates([t2_t1, t1_t2, t2_t1])
        
        if fidelity < 1.:
            depolarization_error(self, fidelity)
            depolarization_error(target1, fidelity)
            depolarization_error(target2, fidelity)
    
    def exp_pauli(self, theta: float, gates: List[int], fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applies a exponential of a Pauli gate sequence with custom angle 
        
        Args:
            theta (float): angle to rotate
            gates (list): list of pauli gates
            fidelity (float): fidelity of depolarization error
            
        Returns:
            /
        """
        
        identity_seq = tensor_operator([full_gates['I']] * len(gates))
        pauli_gates = [full_gates['I'], full_gates['X'], full_gates['Y'], full_gates['Z']]
        pauli_seq = tensor_operator([pauli_gates[gate] for gate in gates])
        gate = np.cos(theta/2) * identity_seq - 1j * np.sin(theta/2) * pauli_seq
        
        if not apply:
            return gate
        
        self.state = dot(self.state, gate)
        if fidelity < 1.:
            depolarization_error(self, fidelity)
    
    def bell_state(self, target: Qubit, bell_state: int=0, fidelity: float=1., apply: bool=True) -> np.ndarray | None:
        
        """
        Transforms the state into a bell state
        
        Args:
            target (Qubit): target qubit
            bell_state (int): bell state to put state into
            fidelity (float): fidelity of the depolarization error
            apply (bool): whether to apply the gate or return it
            
        Returns:
            /
        """
        
        if not (0 <= bell_state <= 3):
            raise ValueError('Bell state should be between 0 and 3')
        
        key = f'bs_{bell_state}_{target.num_qubits}_{self._index}_{target._index}'
        if not apply:
            return get_bell_operator(key, bell_state, self._index, target._index, target.num_qubits)
        
        if self.qsystem is not target.qsystem or self is target:
            raise ValueError('Bell preparation requires two distinct qubits in the same QSystem')
        if bell_state & 1:
            target.X()
        if bell_state & 2:
            self.X()
        self.H()
        self.CNOT(target)
        if fidelity < 1.:
            depolarization_error(self, fidelity)
            depolarization_error(target, fidelity)
    
    def measure(self, basis: str='z', fid_0: float=1., fid_1: float=1., *, rng: np.random.Generator | None = None) -> int:
        
        """
        Measures this qubit and modifies the state accordingly
        
        Args:
            qubit (Qubit): qubit to measure
            fid_0 (float): fidelity of measuring 0
            fid_1 (float): fidelity of measuring 1
            
        Returns:
            measurement (int): measurement outcome
        """
        
        if not 0 <= fid_0 <= 1 or not 0 <= fid_1 <= 1:
            raise ValueError('Measurement fid_0 and fid_1 must be probabilities in [0,1]')
        if basis not in ('x', 'X', 'y', 'Y', 'z', 'Z'):
            raise ValueError('Measurement basis must be x, y, or z')
        if basis in ['x', 'X']:
            self.H()
        if basis in ['y', 'Y']:
            self.iSZ()
            self.H()
        
        P0 = np.array([[np.sqrt(fid_0), 0], [0, np.sqrt(1 - fid_1)]])
        P1 = np.array([[np.sqrt(1 - fid_0), 0], [0, np.sqrt(fid_1)]])
        
        key = f's_m0_{fid_0}_{fid_1}_{self.num_qubits}_{self._index}'
        branch_0 = self.qsystem.backend.single(self.state, P0, self._index, key=key)
        prob = float(np.trace(branch_0).real)
        if not np.isfinite(prob) or not -1e-12 <= prob <= 1 + 1e-12:
            raise ValueError('Invalid quantum state: measurement probability outside [0,1]')
        draw = np.random.uniform(0, 1) if rng is None else rng.random()
        if draw < prob:
            self.state = branch_0 / prob
            return 0
        key = f's_m1_{fid_0}_{fid_1}_{self.num_qubits}_{self._index}'
        branch_1 = self.qsystem.backend.single(self.state, P1, self._index, key=key)
        norm = float(np.trace(branch_1).real)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError('Invalid quantum state: selected a zero-probability measurement branch')
        self.state = branch_1 / norm
        return 1

    def state_transfer(self, source: Qubit, fidelity: float=1.) -> None:
        
        """
        Transfers a state from the source qubit to this qubit
        
        Args:
            source (Qubit): qubit from which state is taken
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        self.CNOT(source)
        res = source.measure()
        
        remove_qubits([source])
        
        if res:
            self.X()
        
        if fidelity < 1.:
            depolarization_error(self, fidelity)

    def bsm(self, target: Qubit, basis_0: str='z', basis_1: str='z', fid_0: float=1., fid_1: float=1., *, rng: np.random.Generator | None = None) -> int:
        
        """
        Performs a Bell state measurement on this qubit and another qubit
        
        Args:
            target (Qubit): target qubit
            basis_0 (str): basis to measure qubit 1 in
            basis_1 (str): basis to measure qubit 2 in
            fid_0 (float): fidelity of measuring 0
            fid_1 (flaot): fidelity of measuring 1
            
        Returns:
            res (int): measurement result
        """
        
        self.CNOT(target)
        self.H()
        
        return 2 * self.measure(basis=basis_0, fid_0=fid_0, fid_1=fid_1, rng=rng) + target.measure(basis=basis_1, fid_0=fid_0, fid_1=fid_1, rng=rng)

    def custom_gate(self, gate: np.ndarray, fidelity: float=1., apply:bool=True) -> None:
        
        """
        Applies a custom gate to the qubit
        
        Args:
            gate (np.array): unitary gate to apply
        
        Returns:
            /
        """
        
        if not apply:
            return gate
        
        self.state = dot(self.state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, fidelity)

    def fidelity(self, _op: Qubit | np.ndarray) -> float:

        """
        Computes the quantum fidelity of this qubit state and a operator

        Args:
            _op (np.array): operator to compare the state of this qubit to

        Returns:
            fidelity (float): fidelity of the quantum state
        """
        
        if isinstance(_op, Qubit):
            _op = _op.state
        
        _sqrt_mat = sqrt_matrix(self.state)
        return float(np.clip((np.real(np.trace(sqrt_matrix(np.dot(_sqrt_mat, np.dot(_op, _sqrt_mat)))))**2), 0, 1))
        
class QSystem:
    
    """
    Represents a system consisting of multiple entanglable qubits
    
    Attr:
        _num_qubits (int): number of qubits in the qsystem
        _qubits (list): qubits in the qsystem
        _state (np.array): density matrix of qsystem
    """
    
    def __init__(self, num_qubits: int=1, fidelity: float=1., *, max_bytes: int = DEFAULT_DENSE_BUDGET,
                 backend: str | PythonBackend = 'python') -> None:
        
        """
        Instantiates the qubit state for a qubit system, initializes to the zero state
        
        Args:
            num_qubits (int): number of qubits in the system
            fidelity (float): fidelity of the initial quantum system
            
        Returns:
            /
        """

        self._backend = get_backend(backend)
        self._backend.validate_size(num_qubits)
        check_dense_allocation(num_qubits, max_bytes)
        self.max_bytes = max_bytes
        if isinstance(fidelity, float):
            fidelity = [fidelity] * num_qubits
        
        if len(fidelity) != num_qubits:
            raise ValueError('There should be a fidelity for all qubits')
        
        if not all([0 <= fid <= 1. for fid in fidelity]):
            raise ValueError('Fidelities should be between 0 and 1')
        
        self._num_qubits: int = num_qubits
        self._qubits: List[Qubit] = [Qubit(self, i) for i in range(self._num_qubits)]
            
        init_state = [np.array([[fid, 0], [0, 1 - fid]]) for fid in fidelity]

        if self._num_qubits > 1:
            self._state: np.ndarray = tensor_operator(init_state)
        else:
            self._state: np.ndarray = init_state[0]
    
    @property
    def backend(self) -> PythonBackend:
        """Numerical kernels used by this state; never changes global selection."""
        return self._backend

    def __len__(self) -> int:
        
        """
        Custom length method for the QSystem
        
        Args:
            /
            
        Returns:
            num_qubits (int): number of qubits in the system
        """
        
        return self._num_qubits
    
    def __repr__(self) -> str:
        
        """
        Custom print method
        
        Args:
            /
        
        Returns:
            state (str): state of the overall qsystem
        """
        
        return str(self._state)
    
    @property
    def num_qubits(self) -> int:
        
        """
        Returns the number of qubits in the qsystem
        
        Args:
            /
            
        Returns:
            num_qubits (int): number of qubits in the qsystem
        """
        
        return self._num_qubits
    
    @property
    def state(self) -> np.ndarray:
        
        """
        Returns the full state of the QSystem
        
        Args:
            /
            
        Returns:
            state (np.array): full state of the QSystem
        """
        
        return self._state
    
    @state.setter
    def state(self, _state: np.ndarray) -> None:
        
        """
        Sets the state of the QSystem
        
        Args:
            _state (np.array): new state of the Qsystem
            
        Returns:
            /
        """
        
        self._state = _state
    
    @property
    def qsystem(self) -> QSystem:
        
        """
        Returns itself
        
        Args:
            /
            
        Returns:
            qsystem (QSystem): the qsystem
        """
        
        return self
    
    @property    
    def qubits(self) -> Union[Qubit, List[Qubit]]:
        
        """
        Makes the retrieving of qubits out of a QSystem more convenient
        
        Args:
            /
            
        Returns:
            qubits (list/Qubit): returns a Qubit or a list of Qubits, depending on the number of qubits
        """
        
        if self._num_qubits > 1:
            return self._qubits
        return self._qubits[0]
    
    @property
    def qsystem_id(self) -> int:
        
        """
        Returns the id of the qsystem
        
        Args:
            /
            
        Returns:
            qsystem_id (int): id of qsystem
        """
        
        return id(self)
    
    def qubit(self, _index: int) -> Qubit:
        
        """
        Returns the qubit at the specified index
        
        Args:
            index (int): index of qubit
            
        Returns:
            qubit (qubit): qubit at specified index
        """
        
        return self._qubits[_index]
    
    def fidelity(self, _op: np.ndarray) -> float:

        """
        Computes the quantum fidelity of this qubit state and a operator

        Args:
            _op (np.array): operator to compare the state of this qubit to

        Returns:
            fidelity (float): fidelity of the quantum state
        """
        
        if isinstance(_op, Qubit):
            _op = _op.state
        
        _sqrt_mat = sqrt_matrix(self._state)
        return float((np.real(np.trace(sqrt_matrix(np.dot(_sqrt_mat, np.dot(_op, _sqrt_mat)))))**2))
    
