import numpy as np
import scipy.sparse as sp
from functools import wraps, partial, reduce
from typing import List, Union, Callable

__all__ = ['Qubit', 'QSystem', 'tensor_operator', 'dot', 'get_single_operator', 'depolarization_error', 'combine_state', 'remove_qubits']

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

sparse_gates = {'P0': sp.csr_matrix([[1, 0], [0, 0]], dtype=np.complex128),
                'P1': sp.csr_matrix([[0, 0], [0, 1]], dtype=np.complex128),
                'P01': sp.csr_matrix([[0, 1], [0, 0]], dtype=np.complex128),
                'P10': sp.csr_matrix([[0, 0], [1, 0]], dtype=np.complex128),
                'I': sp.csr_matrix([[1, 0], [0, 1]], dtype=np.complex128),
                'X': sp.csr_matrix([[0, 1], [1, 0]], dtype=np.complex128),
                'Y': sp.csr_matrix([[0, -1j], [1j, 0]], dtype=np.complex128),
                'Z': sp.csr_matrix([[1, 0], [0, -1]], dtype=np.complex128),
                'H': sp.csr_matrix([[1, 1], [1, -1]], dtype=np.complex128) / (np.sqrt(2)),
                'K': 0.5 * sp.csr_matrix([[1 + 1j, 1 - 1j], [-1 + 1j, -1 - 1j]], dtype=np.complex128),
                'T': sp.csr_matrix([[1, 0], [0, np.exp(1j * 0.25 * np.pi)]], dtype=np.complex128),
                'SX': 0.5 * sp.csr_matrix([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=np.complex128),
                'SY': 0.5 * sp.csr_matrix([[1 + 1j, -1 - 1j], [1 + 1j, 1 + 1j]], dtype=np.complex128),
                'SZ': sp.csr_matrix([[1, 0], [0, 1j]], dtype=np.complex128),
                'iK': 0.5 * sp.csr_matrix([[1 - 1j, -1 - 1j], [1 + 1j, -1 + 1j]], dtype=np.complex128),
                'iT': sp.csr_matrix([[1, 0], [0, np.exp(-1j * 0.25 * np.pi)]], dtype=np.complex128),
                'iSX': 0.5 * sp.csr_matrix([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]], dtype=np.complex128),
                'iSY': 0.5 * sp.csr_matrix([[1 - 1j, 1 - 1j], [-1 + 1j, 1 - 1j]], dtype=np.complex128),
                'iSZ': sp.csr_matrix([[1, 0], [0, -1j]], dtype=np.complex128)}

gates = {0: full_gates, 1: sparse_gates}
identity_gates = {0: np.eye, 1: sp.eye}

purification_gates = {'CNOT': 'X', 'CY': 'Y', 'CZ': 'Z', 'CH': 'H'}

_PROB_BSM_MAPPING = {0: {0: 0, 1: 1, 2: 2, 3: 3}, 
                    1: {0: 1, 1: 0, 2: 3, 3: 2}, 
                    2: {0: 2, 1: 3, 2: 0, 3: 1}, 
                    3: {0: 3, 1: 2, 2: 1, 3: 0}}

class Qubit:
    pass

class QSystem:
    pass

def cache(func: Callable) -> Callable:
    
    """
    Caches the applied gates to reduce calculation times

    Args:
        func (Function): function which input to cache

    Returns:
        wrapper (Function): wrapper function
    """

    _cache = {}
    
    @wraps(func)
    def wrapper(*args):
        
        key = args[0]
        if not key:
            return func(*(args[1:]))
        if key not in _cache:
            _cache[key] = func(*(args[1:]))
        return _cache[key]
    return wrapper

def kronecker_product_full(_op_1: np.array, _op_2: np.array) -> np.array:
    
    """
    Computes the kronecker product of two full 2d matrices

    Args:
        _op_1 (np.array): first matrix
        _op_2 (np.array): second matrix

    Returns:
        op (np.array): result of the kronecker product
    """

    _op = _op_1[:, None, :, None] * _op_2[None, :, None, :]
    _op.shape = (_op_1.shape[0] * _op_2.shape[0], _op_1.shape[1] * _op_2.shape[1])

    return _op

def kronecker_product_sparse(_op_1: sp.csr_matrix, _op_2: sp.csr_matrix) -> sp.csr_matrix:
    
    """
    Computes the kronecker product of two sparse 2d matrices

    Args:
        _op_1 (sp.csr_matrix): first matrix
        _op_2 (sp.csr_matrix): second matrix

    Returns:
        _op (sp.csr_matrix): result of the kronecker product
    """
    
    return sp.kron(_op_1, _op_2)

_KRON_DICT = {0: kronecker_product_full, 1: kronecker_product_sparse}

def tensor_operator(_sparse: bool, _operator_l: np.array) -> Union[np.array, sp.csr_matrix]:
    
    """
    Creates the tensor operator out of a operator list
    
    Args:
        _sparse (bool): indicator whether the operators are full or sparse
        _operator_l (list): list of 2x2 np.arrays
        
    Returns:
        res (np.array): resulting tensor operator
    """

    return reduce(_KRON_DICT[_sparse], _operator_l)

def dot(_state: Union[np.array, sp.csr_matrix], _gate: Union[np.array, sp.csr_matrix]) -> Union[np.array, sp.csr_matrix]:
    
    """
    Performs the linalg dot for quantum gates
    
    Args:
        _state (np.array): state to apply the gate to
        _gate (np.array): gate to apply
        
    Returns:
        _state (np.array): resulting state
    """
    
    return _gate.dot(_state).dot(_gate.conj().T)

def sqrt_matrix(_state: Union[np.array, sp.csr_matrix]) -> Union[np.array, sp.csr_matrix]:
    
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
def get_single_operator(_sparse: bool, _gate: Union[np.array, sp.csr_matrix], _index: int, _num_qubits: int) -> np.array:

    """
    Creates the tensor operator for a single qubit gate
    
    Args:
        _sparse (bool): indicator whether the operators are full or sparse
        _gate (np.array): 2x2 gate to apply to the target qubit
        _index (int): index of target qubit in QSystem
        _num_qubits (int): number of qubits in QSystem
        
    Returns:
        _op (np.array): resulting tensor for the single qubit
    """
    
    if _num_qubits < 2:
        return _gate
    
    if _num_qubits < 3:
        operator_l = np.array([gates[_sparse]['I']] * _num_qubits)  
        operator_l[_index] = _gate
        
        return tensor_operator(_sparse, operator_l)

    if _index == 0:
        return _KRON_DICT[_sparse](_gate, identity_gates[_sparse](2**(_num_qubits - 1), 2**(_num_qubits - 1), dtype=np.complex128))
        
    if _index == _num_qubits - 1:
        return _KRON_DICT[_sparse](identity_gates[_sparse](2**(_num_qubits - 1), 2**(_num_qubits - 1), dtype=np.complex128), _gate)
    
    left_array = identity_gates[_sparse](2**_index, 2**_index, dtype=np.complex128)
    right_array = identity_gates[_sparse](2**(_num_qubits - _index - 1), 2**(_num_qubits - _index - 1), dtype=np.complex128)
    
    return _KRON_DICT[_sparse](left_array, _KRON_DICT[_sparse](_gate, right_array))

@cache
def get_double_operator(_sparse: bool, _gate: Union[np.array, sp.csr_matrix], _c_index: int, _t_index: int, _t_num_qubits: int) -> Union[np.array, sp.csr_matrix]:
    
    """
    Creates the tensor operator for a controlled qubit gate
    
    Args:
        _sparse (bool): indicator whether the operators are full or sparse
        _gate (np.array): 2x2 gate to apply to the target qubit
        _c_index (int): index of control qubit
        _t_index (int): index of target index
        _t_num_qubits (int): number of qubits in target QSystem
        
    Returns:
        _op (np.array): resulting tensor for the controlled qubit gate
    """
    
    key = f'{_sparse}_s_m0_{1}_{1}_{_t_num_qubits}_{_c_index}'
    proj0 = get_single_operator(key, _sparse, gates[_sparse]['P0'], _c_index, _t_num_qubits)

    gate_dict = {_c_index: gates[_sparse]['P1'], _t_index: _gate}
    index_1, index_2 = sorted([_c_index, _t_index])
    
    left_index = index_1
    middle_index = index_2 - index_1 - 1
    right_index = _t_num_qubits - index_2 - 1
    
    proj1 = gate_dict[index_1]
    if left_index:
        proj1 = _KRON_DICT[_sparse](identity_gates[_sparse](2**left_index, 2**left_index, dtype=np.complex128), proj1)
    if middle_index:
        proj1 = _KRON_DICT[_sparse](proj1, identity_gates[_sparse](2**middle_index, 2**middle_index, dtype=np.complex128))
    proj1 = _KRON_DICT[_sparse](proj1, gate_dict[index_2])
    if right_index:
        proj1 = _KRON_DICT[_sparse](proj1, identity_gates[_sparse](2**right_index, 2**right_index, dtype=np.complex128))
    
    return proj0 + proj1

@cache
def get_triple_operator(_sparse: bool, _gate_l: Union[np.array, sp.csr_matrix], _c1_index: int, _c2_index: int, _t_index: int, _t_num_qubits: int) -> Union[np.array, sp.csr_matrix]:
    
    """
    Generates the operator for a 3 qubit gate
    
    Args:
        _sparse (bool): indicator whether the operators are full or sparse
        _gate_l (np.array): gate list to apply to target
        _c1_index (int): index of first control qubit
        _c2_index (int): index of second control qubit
        _t_index (int): index of target qubit
        _t_num_qubits (int): number of qubits in target QSystem
        
    Returns:
        _op (np.array): resulting tensor for the controlled controlled qubit gate
    """
    
    gate_dict = {_c1_index: [gates[_sparse]['P0'], gates[_sparse]['P0'], gates[_sparse]['P1'], gates[_sparse]['P1']],
                  _c2_index: [gates[_sparse]['P0'], gates[_sparse]['P1'], gates[_sparse]['P0'], gates[_sparse]['P1']],
                  _t_index: _gate_l}
    
    index_1, index_2, index_3 = sorted([_c1_index, _c2_index, _t_index])
    indices = [index_1, index_2 - index_1 - 1, index_3 - index_2 - 1, _t_num_qubits - index_3 - 1]
    
    gate_list = [[0] * 7 for _ in range(4)]
    
    for i in range(4):
        gate_list[i][::2] = [identity_gates[_sparse](2**(index), 2**(index), dtype=np.complex128) if index else None for index in indices]
        gate_list[i][1::2] = [gate_dict[index_1][i], gate_dict[index_2][i], gate_dict[index_3][i]]
        gate_list[i] = [gate for gate in gate_list[i] if gate is not None]
    
    return tensor_operator(_sparse, gate_list[0]) + tensor_operator(_sparse, gate_list[1]) + tensor_operator(_sparse, gate_list[2]) + tensor_operator(_sparse, gate_list[3])
    
@cache
def get_bell_operator(_sparse: bool, _bell_state: int, _c_index: int, _t_index: int, _t_num_qubits: int) -> Union[np.array, sp.csr_matrix]:
    
    """
    Generates the bell operator
    
    Args:
        _sparse (bool): indicator whether the operators are full or sparse
        _bell_state (int): bell state to calculate
        _c_index (int): index of control qubit
        _t_index (int): index of target qubit
        _t_num_qubits (int): number of qubits in target QSystem
        
    Returns:
        _op (np.array): resulting tensor for the bell state operator
    """
    
    h_gate = get_single_operator(f'{_sparse}_s_h_{_t_num_qubits}_{_c_index}', _sparse, gates[_sparse]['H'], _c_index, _t_num_qubits)
    cnot_gate = get_double_operator(f'{_sparse}_d_x_{_t_num_qubits}_{_c_index}_{_t_index}', _sparse, gates[_sparse]['X'], _c_index, _t_index, _t_num_qubits)
    
    if _bell_state == 0:
        return cnot_gate.dot(h_gate)
    if _bell_state == 1:
        x_gate = get_single_operator(f'{_sparse}_s_x_{_t_num_qubits}_{_t_index}', _sparse, gates[_sparse]['X'], _t_index, _t_num_qubits)
        return cnot_gate.dot(h_gate).dot(x_gate)
    if _bell_state == 2:
        x_gate = get_single_operator(f'{_sparse}_s_x_{_t_num_qubits}_{_c_index}', _sparse, gates[_sparse]['X'], _c_index, _t_num_qubits)
        return cnot_gate.dot(h_gate).dot(x_gate)
    if _bell_state == 3:
        x_gate_c = get_single_operator(f'{_sparse}_s_x_{_t_num_qubits}_{_c_index}', _sparse, gates[_sparse]['X'], _c_index, _t_num_qubits)
        x_gate_t = get_single_operator(f'{_sparse}_s_x_{_t_num_qubits}_{_t_index}', _sparse, gates[_sparse]['X'], _t_index, _t_num_qubits)
        return cnot_gate.dot(h_gate).dot(x_gate_c).dot(x_gate_t)

@cache
def get_bsm_operator(_sparse: bool, _c_index: int, _t_index: int, _t_num_qubits: int) -> np.array:
    
    """
    Generates the bsm operator
    
    Args:
        _sparse (bool): indicator whether the operators are full or sparse
        _c_index (int): index of control qubit
        _t_index (int): index of target qubit
        _t_num_qubits (int): number of qubits in target QSystem
        
    Returns:
        _op (np.array): resulting tensor for the bsm operator
    """
    
    cnot_gate = get_double_operator(f'{_sparse}_d_x_{_t_num_qubits}_{_c_index}_{_t_index}', _sparse, gates[_sparse]['X'], _c_index, _t_index, _t_num_qubits)
    h_gate = get_single_operator(f'{_sparse}_s_h_{_t_num_qubits}_{_c_index}', _sparse, gates[_sparse]['H'], _c_index, _t_num_qubits)
    
    return h_gate.dot(cnot_gate) 

@cache
def get_swap_operator(_sparse: bool, _index_1: int, _index_2: int, _num_qubits: int) -> np.array:
    
    """
    Generates the swap operator
    
    Args:
        _sparse (bool): whether the operators are full or sparse
        _index_1 (int): index of first qubit
        _index_2 (int): index of second qubit
        _num_qubits (int):  number of qubits in QSystem
        
    Returns:
        _op (np.array): resulting tensor for the bsm operator
    """
    
    _op = np.array([gates[_sparse]['I']] * _num_qubits)
    
    _op[_index_1] = gates[_sparse]['P0']
    _op[_index_2] = gates[_sparse]['P0']
    
    _res = tensor_operator(_sparse, _op)
    
    _op[_index_1] = gates[_sparse]['P01']
    _op[_index_2] = gates[_sparse]['P10']
    
    _res += tensor_operator(_sparse, _op)
    
    _op[_index_1] = gates[_sparse]['P10']
    _op[_index_2] = gates[_sparse]['P01']
    
    _res += tensor_operator(_sparse, _op)
    
    _op[_index_1] = gates[_sparse]['P1']
    _op[_index_2] = gates[_sparse]['P1']
    
    return _res + tensor_operator(_sparse, _op)

def depolarization_error(_sparse: bool, _qubit: Qubit, _fidelity: float) -> None:
    
    """
    Applys a depolarization error to the qubit given a fidelity
    
    Args:
        _sparse (bool): sparsity of QSystem
        _qubit (Qubit): qubit to apply error to
        _fidelity (float): fidelity of the depolarization error
        
    Returns:
        /
    """
    
    key_x = f'{_sparse}_s_x_{_qubit.num_qubits}_{_qubit._index}'
    key_y = f'{_sparse}_s_y_{_qubit.num_qubits}_{_qubit._index}'
    key_z = f'{_sparse}_s_z_{_qubit.num_qubits}_{_qubit._index}'
    
    gate_x = get_single_operator(key_x, _sparse, gates[_sparse]['X'], _qubit._index, _qubit.num_qubits)    
    gate_y = get_single_operator(key_y, _sparse, gates[_sparse]['y'], _qubit._index, _qubit.num_qubits)
    gate_z = get_single_operator(key_z, _sparse, gates[_sparse]['Z'], _qubit._index, _qubit.num_qubits)
        
    _qubit.state = _fidelity * _qubit.state + ((1 - _fidelity) / 3) * (dot(_qubit.state, gate_x) + dot(_qubit.state, gate_y) + dot(_qubit.state, gate_z))

def combine_state(q_l: List[Qubit]) -> QSystem:
        
    """
    Combines multiple qsystems into one qsystem
    
    Args:
        q_l (list): List of qsystems
        
    Returns:
        qsys_n (QSystem): new qsystem
    """
    
    if len(set([id(qubit._qsystem) for qubit in q_l])) == 1:
        return q_l[0]._qsystem
    
    qsys_n = q_l[0]._qsystem
    qsys_l = [q._qsystem for n, q in enumerate(q_l) if q not in q_l[:n]]
    num_qubits_n = sum([qsys._num_qubits for qsys in qsys_l])
    
    qsys_n._qubits = [q for qsys in qsys_l for q in qsys._qubits]
    qsys_n._num_qubits = num_qubits_n
    qsys_n._state = tensor_operator(q_l[0].sparse, np.array([qsys._state for qsys in qsys_l], dtype=object))
    qsys_n._state = qsys_n._state.astype(np.complex128)
    for i in range(num_qubits_n):
        qsys_n._qubits[i]._index = i
        qsys_n._qubits[i]._qsystem = qsys_n

    return qsys_n

def sparse_dot(op_1: sp.csr_matrix, op_2: sp.csr_matrix) -> sp.csr_matrix:
    
    """
    Calculates the sparse dot product of two matrices
    
    Args:
        op_1 (sp.csr_matrix): first matrix
        op_2 (sp.csr_matrix): second matrix
        
    Returns:
        res (sp.csr_matrix): resulting matrix
    """
    
    return op_1.dot(op_2)

_COMBINE_DICT = {0: np.dot, 1: sparse_dot}

def combine_gates(_sparse: bool, gate_l: List[np.array]) -> np.array:
    
    """
    Combines multiple gates into one gate
    
    Args:
        _sparse (bool): indicator whether the operators are full or sparse
        gate_l (list): List of gates
        
    Returns:
        gate_n (np.array): new gate
    """
    
    return reduce(_COMBINE_DICT[_sparse], gate_l[::-1])
    
def ptrace_full(_q: Qubit) -> None:
    
    """
    Traces out a qubit from the state, adjusts the QSystem accordingly
    
    Args:
        _q (Qubit): qubit to trace out
        
    Returns:
        /
    """

    dims_ = np.array([2] * 2 * _q.num_qubits, dtype=np.int8)
    state_dim = 1 << (_q.num_qubits - 1)

    reshaped_state = _q.state.reshape(dims_)

    reshaped_state = np.moveaxis(reshaped_state, _q._index, -1)
    reshaped_state = np.moveaxis(reshaped_state, _q.num_qubits + _q._index - 1, -1)

    _q.state = np.trace(reshaped_state, axis1 = -2, axis2 = -1).reshape([state_dim, state_dim])
    _q.num_qubits -= 1
    
    del _q._qsystem._qubits[_q._index]  
    for i in range(_q.num_qubits):
        _q._qsystem._qubits[i]._index = i

def ptrace_sparse(_q: Qubit) -> None:
        
    """
    Traces out a qubit from the state, adjusts the QSystem accordingly
    
    Args:
        _q (Qubit): qubit to trace out
        
    Returns:
        /
    """
    
    v1 = sp.csr_matrix([[1, 0]], dtype=np.complex128)
    v2 = sp.csr_matrix([[0, 1]], dtype=np.complex128)
    
    key_1 = f'{1}_v1_{_q.num_qubits}_{_q._index}'
    key_2 = f'{1}_v1T_{_q.num_qubits}_{_q._index}'
    key_3 = f'{1}_v2_{_q.num_qubits}_{_q._index}'
    key_4 = f'{1}_v2T_{_q.num_qubits}_{_q._index}'
    
    op_1_a = get_single_operator(key_1, 1, v1, _q._index, _q.num_qubits)
    op_1_b = get_single_operator(key_2, 1, v1.T, _q._index, _q.num_qubits)
    op_2_a = get_single_operator(key_3, 1, v2, _q._index, _q.num_qubits)
    op_2_b = get_single_operator(key_4, 1, v2.T, _q._index, _q.num_qubits)

    _q.state = sp.csr_matrix(op_1_a.dot(_q.state.dot(op_1_b)) + op_2_a.dot(_q.state.dot(op_2_b)))
    _q.state.eliminate_zeros()
    _q.num_qubits -= 1
    
    del _q._qsystem._qubits[_q._index]
    for i in range(_q.num_qubits):
        _q._qsystem._qubits[i]._index = i

_PTRACE_DICT = {0: ptrace_full, 1: ptrace_sparse}

def remove_qubits(q_l: List[Qubit]) -> None:

    """
    Traces out the given qubits from the state, adjusts the QSystem accordingly
    
    Args:
        _q_l (list): list of qubits to trace out
        
    Returns:
        /
    """
    
    for qubit in q_l:
        _PTRACE_DICT[qubit.sparse](qubit)

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
    def state(self, state: np.array) -> None:
        
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
    def sparse(self) -> int:
        
        """
        Returns whether the state of the QSystem is in sparse representation or not
        
        Args:
            /
            
        Returns:
            _sparse (int): sparsity of state
        """
        
        return self._qsystem._sparse
    
    def qubit_id(self) -> int:
        
        """
        Returns the ID of the qubit
        
        Args:
            /
            
        Returns:
            _id (int): id of qubit
        """
        
        return id(self)
    
    def qsystem_id(self) -> int:
        
        """
        Returns the ID of the qsystem the qubit is in
        
        Args:
            /
            
        Returns:
            _id (int): id of the qsystem the qubit is in
        """
        
        return id(self._qsystem)
    
    def X(self, fidelity: float=1., apply: bool=True) -> None:

        """
        Applys the X gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self.sparse}_s_x_{self.num_qubits}_{self._index}'
        gate = get_single_operator(key, self.sparse, gates[self.sparse]['X'], self._index, self.num_qubits)
        
        if not apply:
            return gate
        
        self.state = dot(self.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
        
    def Y(self, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the Y gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self.sparse}_s_y_{self.num_qubits}_{self._index}'
        gate = get_single_operator(key, self.sparse, gates[self.sparse]['Y'], self._index, self.num_qubits)
        
        if not apply:
            return gate
        
        self.state = dot(self.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
        
    def Z(self, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the Z gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self.sparse}_s_z_{self.num_qubits}_{self._index}'
        gate = get_single_operator(key, self.sparse, gates[self.sparse]['Z'], self._index, self.num_qubits)
        
        if not apply:    
            return gate
        
        self.state = dot(self.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
        
    def H(self, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the Hadamard gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self.sparse}_s_h_{self.num_qubits}_{self._index}'
        gate = get_single_operator(key, self.sparse, gates[self.sparse]['H'], self._index, self.num_qubits)
        
        if not apply:
            return gate
        
        self.state = dot(self.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)

    def SX(self, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the square root X gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self.sparse}_s_sx_{self.num_qubits}_{self._index}'
        gate = get_single_operator(key, self.sparse, gates[self.sparse]['SX'], self._index, self.num_qubits)
        
        if not apply:
            return gate
        
        self.state = dot(self.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
    
    def SY(self, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the square root Y gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self.sparse}_s_sy_{self.num_qubits}_{self._index}'
        gate = get_single_operator(key, self.sparse, gates[self.sparse]['SY'], self._index, self.num_qubits)
        
        if not apply:
            return gate
        
        self.state = dot(self.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
    
    def SZ(self, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the square root Z gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self.sparse}_s_sz_{self.num_qubits}_{self._index}'
        gate = get_single_operator(key, self.sparse, gates[self.sparse]['SZ'], self._index, self.num_qubits)
        
        if not apply:
            return gate
        
        self.state = dot(self.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
    
    def T(self, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the T gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
    
        key = f'{self.sparse}_s_t_{self.num_qubits}_{self._index}'
        gate = get_single_operator(key, self.sparse, gates[self.sparse]['T'], self._index, self.num_qubits)
        
        if not apply:
            return gate
        
        self.state = dot(self.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
    
    def K(self, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the K gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self.sparse}_s_k_{self.num_qubits}_{self._index}'
        gate = get_single_operator(key, self.sparse, gates[self.sparse]['K'], self._index, self.num_qubits)
        
        if not apply:
            return gate
        
        self.state = dot(self.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
    
    def iSX(self, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the inverse square root X gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self.sparse}_s_isx_{self.num_qubits}_{self._index}'
        gate = get_single_operator(key, self.sparse, gates[self.sparse]['iSX'], self._index, self.num_qubits)
        
        if not apply:
            return gate
        
        self.state = dot(self.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
    
    def iSY(self, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the inverse square root Y gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self.sparse}_s_isy_{self.num_qubits}_{self._index}'
        gate = get_single_operator(key, self.sparse, gates[self.sparse]['iSY'], self._index, self.num_qubits)
        
        if not apply:
            return gate
        
        self.state = dot(self.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
    
    def iSZ(self, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the inverse square root Z gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self.sparse}_s_isz_{self.num_qubits}_{self._index}'
        gate = get_single_operator(key, self.sparse, gates[self.sparse]['iSZ'], self._index, self.num_qubits)
        
        if not apply:
            return gate
        
        self.state = dot(self.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
    
    def iT(self, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the inverse of T gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self.sparse}_s_it_{self.num_qubits}_{self._index}'
        gate = get_single_operator(key, self.sparse, gates[self.sparse]['iT'], self._index, self.num_qubits)
        
        if not apply:
            return gate
        
        self.state = dot(self.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
    
    def iK(self, fidelity: float=1., apply: bool=True):
        
        """
        Applys the inverse of K gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self.sparse}_s_ik_{self.num_qubits}_{self._index}'
        gate = get_single_operator(key, self.sparse, gates[self.sparse]['iK'], self._index, self.num_qubits)
        
        if not apply:
            return gate
        
        self.state = dot(self.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
    
    def Rx(self, theta: float, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the rotation gate around the x axis to the qubit
        
        Args:
            theta (float): angle for the rotation matrix
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self.sparse}_s_rx_{theta}_{self.num_qubits}_{self._index}'
        gate_s = np.array([[np.cos(theta/2), -1j * np.sin(theta/2)], [-1j * np.sin(theta/2), np.cos(theta/2)]], dtype=np.complex128)
        if self.sparse:
            gate_s = sp.csr_matrix([[np.cos(theta/2), -1j * np.sin(theta/2)], [-1j * np.sin(theta/2), np.cos(theta/2)]], dtype=np.complex128)
        gate = get_single_operator(key, self.sparse, gate_s, self._index, self.num_qubits)
        
        if not apply:
            return gate
        
        self.state = dot(self.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
    
    def Ry(self, theta: float, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the rotation gate around the y axis to the qubit
        
        Args:
            theta (float): angle for the rotation matrix
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self.sparse}_s_ry_{theta}_{self.num_qubits}_{self._index}'
        gate_s = np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]], dtype=np.complex128)
        if self.sparse:
            gate_s = sp.csr_matrix([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]], dtype=np.complex128)
        gate = get_single_operator(key, self.sparse, gate_s, self._index, self.num_qubits)
        
        if not apply:
            return gate
        
        self.state = dot(self.state, gate)        
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
        
    def Rz(self, theta: float, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the rotation gate around the z axis to the qubit
        
        Args:
            theta (float): angle for the rotation matrix
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self.sparse}_s_rz_{theta}_{self.num_qubits}_{self._index}'
        gate_s = np.array([[np.exp(-1j * theta/2), 0], [0, np.exp(1j * theta/2)]], dtype=np.complex128)
        if self.sparse:
            sp.csr_matrix([[np.exp(-1j * theta/2), 0], [0, np.exp(1j * theta/2)]], dtype=np.complex128)
        gate = get_single_operator(key, self.sparse, gate_s, self._index, self.num_qubits)
        
        if not apply:
            return gate
        
        self.state = dot(self.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
        
    def PHASE(self, theta: float, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the PHASE gate to the qubit
        
        Args:
            theta (float): angle for the rotation matrix
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self.sparse}_s_p_{theta}_{self.num_qubits}_{self._index}'
        gate_s = np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=np.complex128)
        if self.sparse:
            gate_s = sp.csr_matrix([[1, 0], [0, np.exp(1j * theta)]], dtype=np.complex128)
        gate = get_single_operator(key, self.sparse, gate_s, self._index, self.num_qubits)
        
        if not apply:
            return gate
        
        self.state = dot(self.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
    
    def general_rotation(self, theta: float, phi: float, psi: float, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys a general rotation to a single qubit
        
        Args:
            theta (float): angle theta
            phi (float): angle phi
            psi (float): angle psi
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self.sparse}_s_gr_{theta}_{phi}_{psi}_{self.num_qubits}_{self._index}'
        gate_s = np.array([[np.cos(theta/2), -np.exp(1j*psi)*np.sin(theta/2)], [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+psi))*np.cos(theta/2)]], dtype=np.complex128)
        if self.sparse:
            gate_s = sp.csr_matrix([[np.cos(theta/2), -np.exp(1j*psi)*np.sin(theta/2)], [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+psi))*np.cos(theta/2)]], dtype=np.complex128)
        gate = get_single_operator(key, self.sparse, gate_s, self._index, self.num_qubits)
        
        if not apply:
            return gate
        
        self.state = dot(self.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
    
    def custom_single_gate(self, gate: np.array, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys a custom gate to the qubit
        
        Args:
            gate (np.array): unitary gate to apply
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        gate = get_single_operator('', self.sparse, gate, self._index, self.num_qubits)
        
        if not apply:
            return gate
        
        self.state = dot(self.state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
     
    def CNOT(self, target: Qubit, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the CNOT gate to a target qubit, with this qubit as the controll qubit
        
        Args:
            target (qubit): target qubit to apply the CNOT gate to
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self.sparse}_d_x_{target.num_qubits}_{self._index}_{target._index}'
        gate = get_double_operator(key, target.sparse, gates[self.sparse]['X'], self._index, target._index, target.num_qubits)
        
        if not apply:
            return gate
        
        target.state = dot(target.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
            depolarization_error(target, target.sparse, fidelity)

    def CX(self, target: Qubit, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the CNOT gate to a target qubit, with this qubit as the controll qubit
        
        Args:
            target (qubit): target qubit to apply the CNOT gate to
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self.sparse}_d_x_{target.num_qubits}_{self._index}_{target._index}'
        gate = get_double_operator(key, target.sparse, gates[self.sparse]['X'], self._index, target._index, target.num_qubits)
        
        if not apply:
            return gate
        
        target.state = dot(target.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
            depolarization_error(target, target.sparse, fidelity)

    def CY(self, target: Qubit, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the CY gate to a target qubit, with this qubit as the control qubit
        
        Args:
            target (qubit): target qubit to apply the CNOT gate to
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self.sparse}_d_y_{target.num_qubits}_{self._index}_{target._index}'
        gate = get_double_operator(key, target.sparse, gates[self.sparse]['Y'], self._index, target._index, target.num_qubits)
        
        if not apply:
            return gate
        
        target.state = dot(target.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
            depolarization_error(target, target.sparse, fidelity)
    
    def CZ(self, target: Qubit, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the CZ gate to a target qubit, with this qubit as the control qubit
        
        Args:
            target (qubit): target qubit to apply the CNOT gate to
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self.sparse}_d_z_{target.num_qubits}_{self._index}_{target._index}'
        gate = get_double_operator(key, target.sparse, gates[self.sparse]['Z'], self._index, target._index, target.num_qubits)
        
        if not apply:
            return gate
        
        target.state = dot(target.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
            depolarization_error(target, target.sparse, fidelity)
    
    def CH(self, target: Qubit, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the CH gate to a target qubit, with this qubit as the control qubit
        
        Args:
            target (qubit): target qubit to apply the CNOT gate to
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self.sparse}_d_h_{target.num_qubits}_{self._index}_{target._index}'
        gate = get_double_operator(key, target.sparse, gates[self.sparse]['H'], self._index, target._index, target.num_qubits)
        
        if not apply:
            return gate
        
        target.state = dot(target.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
            depolarization_error(target, target.sparse, fidelity)
    
    def CPHASE(self, target: Qubit, theta: float, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the CPHASE gate to a target qubit, with this qubit as the controll qubit
        
        Args:
            target (qubit): target qubit to apply the CPHASE gate to
            theta (float): angle for the rotation matrix
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self.sparse}_d_p_{theta}_{target.num_qubits}_{self._index}_{target._index}'
        gate_s = np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=np.complex128)
        if self.sparse:
            gate_s = sp.csr_matrix([[1, 0], [0, np.exp(1j * theta)]], dtype=np.complex128)
        gate = get_double_operator(key, target.sparse, gate_s, self._index, target._index, target.num_qubits)
        
        if not apply:
            return gate
        
        target.state = dot(target.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
            depolarization_error(target, self.sparse, fidelity)
    
    def CU(self, target: Qubit, gate: np.array, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys a custom gate to a target qubit, with this qubit as the controll qubit
        
        Args:
            target (qubit): target qubit to apply a custom gate to
            gate (np.array): 2x2 unitary array
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        gate = get_double_operator('', target.sparse, gate, self._index, target._index, target.num_qubits)
        
        if not apply:
            return gate
        
        target.state = dot(target.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
            depolarization_error(target, self.sparse, fidelity)
    
    def SWAP(self, target: Qubit, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Swaps the state of this qubit with the target qubit by applying CNOT gates
        
        Args:
            target (qubit): target qubit to swap state with
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self.sparse}_d_sw_{self.num_qubits}_{self._index}_{target._index}'
        swap_gate = get_swap_operator(key, self.sparse, self._index, target._index, self.num_qubits)
        
        if not apply:
            return swap_gate
        
        self.state = dot(self.state, swap_gate)
        self._qsystem._qubits[self._index], self._qsystem._qubits[target._index] = self._qsystem._qubits[target._index], self._qsystem._qubits[self._index]
        self._index, target._index = target._index, self._index
        
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
            depolarization_error(target, target.sparse, fidelity)
    
    def iSWAP(self, target: Qubit, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the Imaginary Swap gate to this qubit and target qubit
        
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
            depolarization_error(self, self.sparse, fidelity)
            depolarization_error(target, self.sparse, fidelity)
    
    def QAND(self, control: Qubit, target: Qubit, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the quantum AND gate to the target qubit with this qubit and control qubit as control qubits
        
        Args:
            control (qubit): second control qubit
            target (qubit): target qubit
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self.sparse}_t_iiix_{target.num_qubits}_{self._index}_{control._index}_{target._index}'
        gate = get_triple_operator(key, target.sparse, 
                                    [gates[self.sparse]['I'],gates[self.sparse]['I'], gates[self.sparse]['I'], gates[self.sparse]['X']], 
                                    self._index, control._index, target._index, target.num_qubits)
        
        if not apply:
            return gate    
    
        target.state = dot(target.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
            depolarization_error(control, self.sparse, fidelity)
            depolarization_error(target, self.sparse, fidelity)
    
    def QOR(self, control: Qubit, target: Qubit, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the quantum OR gate to the target qubit with this qubit and control qubit as control qubits
        
        Args:
            control (qubit): second control qubit
            target (qubit): target qubit
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self.sparse}_t_ixxx_{target.num_qubits}_{self._index}_{control._index}_{target._index}'
        gate = get_triple_operator(key, target.sparse, 
                                    [gates[self.sparse]['I'],gates[self.sparse]['X'], gates[self.sparse]['X'], gates[self.sparse]['X']], 
                                    self._index, control._index, target._index, target.num_qubits)
        
        if not apply:
            return gate
        
        target.state = dot(target.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
            depolarization_error(control, self.sparse, fidelity)
            depolarization_error(target, self.sparse, fidelity)
    
    def QXOR(self, control: Qubit, target: Qubit, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the quantum XOR gate to the target qubit with this qubit and control qubit as control qubits
        
        Args:
            control (qubit): second control qubit
            target (qubit): target qubit
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self.sparse}_t_ixxi_{target.num_qubits}_{self._index}_{control._index}_{target._index}'
        gate = get_triple_operator(key, target.sparse, 
                                    [gates[self.sparse]['I'],gates[self.sparse]['X'], gates[self.sparse]['X'], gates[self.sparse]['I']], 
                                    self._index, control._index, target._index, target.num_qubits)
        
        if not apply:
            return gate
        
        target.state = dot(target.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
            depolarization_error(control, self.sparse, fidelity)
            depolarization_error(target, self.sparse, fidelity)
    
    def QNAND(self, control: Qubit, target: Qubit, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the quantum NAND gate to the target qubit with this qubit and control qubit as control qubits
        
        Args:
            control (qubit): second control qubit
            target (qubit): target qubit
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self.sparse}_t_xxxi_{target.num_qubits}_{self._index}_{control._index}_{target._index}'
        gate = get_triple_operator(key, target.sparse, 
                                    [gates[self.sparse]['X'],gates[self.sparse]['X'], gates[self.sparse]['X'], gates[self.sparse]['I']], 
                                    self._index, control._index, target._index, target.num_qubits)
        
        if not apply:
            return gate
        
        target.state = dot(target.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
            depolarization_error(control, self.sparse, fidelity)
            depolarization_error(target, self.sparse, fidelity)
    
    def QNOR(self, control: Qubit, target: Qubit, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the quantum NOR gate to the target qubit with this qubit and control qubit as control qubits
        
        Args:
            control (qubit): second control qubit
            target (qubit): target qubit
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self.sparse}_t_xiii_{target.num_qubits}_{self._index}_{control._index}_{target._index}'
        gate = get_triple_operator(key, target.sparse, 
                                    [gates[self.sparse]['X'],gates[self.sparse]['I'], gates[self.sparse]['I'], gates[self.sparse]['I']], 
                                    self._index, control._index, target._index, target.num_qubits)
        
        if not apply:
            return gate
        
        target.state = dot(target.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
            depolarization_error(control, self.sparse, fidelity)
            depolarization_error(target, self.sparse, fidelity)
    
    def QXNOR(self, control: Qubit, target: Qubit, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the quantum XNOR gate to the target qubit with this qubit and control qubit as control qubits
        
        Args:
            control (qubit): second control qubit
            target (qubit): target qubit
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self.sparse}_t_xiix_{target.num_qubits}_{self._index}_{control._index}_{target._index}'
        gate = get_triple_operator(key, target.sparse, 
                                    [gates[self.sparse]['X'],gates[self.sparse]['I'], gates[self.sparse]['I'], gates[self.sparse]['X']], 
                                    self._index, control._index, target._index, target.num_qubits)
        
        if not apply:
            return gate
        
        target.state = dot(target.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
            depolarization_error(control, self.sparse, fidelity)
            depolarization_error(target, self.sparse, fidelity)
    
    def CCU(self, control: Qubit, target: Qubit, gate: np.array, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys a custom unitary gate to controled by this qubit and control qubit
        
        Args:
            control (qubit): second control qubit
            target (qubit): target qubit
            gate (np.array): gate to apply
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        gate = get_triple_operator('', target.sparse, 
                                        [gates[self.sparse]['I'], gates[self.sparse]['I'], gates[self.sparse]['I'], gate],
                                        self._index, control._index, target._index, target.num_qubits)
        
        if not apply:
            return gate
        
        target.state = dot(target.state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
            depolarization_error(control, self.sparse, fidelity)
            depolarization_error(target, self.sparse, fidelity)
    
    def CSWAP(self, target1: Qubit, target2: Qubit, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Applys the CSWAP gate with self as control and target1 and target2 as targets
        
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
            depolarization_error(self, self.sparse, fidelity)
            depolarization_error(target1, target1.sparse, fidelity)
            depolarization_error(target2, target2.sparse, fidelity)
    
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
        
        identity_seq = tensor_operator([gates[self.sparse]['I']] * len(gates))
        pauli_gates = [gates[self.sparse]['I'], gates[self.sparse]['X'], gates[self.sparse]['Y'], gates[self.sparse]['Z']]
        pauli_seq = tensor_operator([pauli_gates[gate] for gate in gates])
        gate = np.cos(theta/2) * identity_seq - 1j * np.sin(theta/2) * pauli_seq
        
        if not apply:
            return gate
        
        self.state = dot(self.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
    
    def bell_state(self, target: Qubit, bell_state: int=0, fidelity: float=1., apply: bool=True) -> None:
        
        """
        Transforms the state into a bell state
        
        Args:
            target (Qubit): target qubit
            bell_state (int): bell state to put state into
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        if not (0 <= bell_state <= 3):
            raise ValueError('Bell state should be between 0 and 3')
        
        key = f'{self.sparse}_bs_{bell_state}_{target.num_qubits}_{self._index}_{target._index}'
        gate = get_bell_operator(key, target.sparse, bell_state, self._index, target._index, target.num_qubits)
        
        if not apply:
            return gate
        
        target.state = dot(target.state, gate)
        if fidelity < 1.:
            depolarization_error(self, self.sparse, fidelity)
            depolarization_error(target, target.sparse, fidelity)
    
    def measure(self, basis: str='z', fid_0: float=1., fid_1: float=1.) -> int:
        
        """
        Measures this qubit and modifies the state accordingly
        
        Args:
            qubit (Qubit): qubit to measure
            fid_0 (float): fidelity of measuring 0
            fid_1 (float): fidelity of measuring 1
            
        Returns:
            measurement (int): measurement outcome
        """
        
        if basis == 'x' or basis == 'X':
            self.H()
        if basis == 'y' or basis == 'Y':
            self.iSZ()
            self.H()
        
        P0 = np.array([[np.sqrt(fid_0), 0], [0, np.sqrt(1 - fid_1)]])
        P1 = np.array([[np.sqrt(1 - fid_0), 0], [0, np.sqrt(fid_1)]])
        if self.sparse:
            P0 = sp.csr_matrix([[np.sqrt(fid_0), 0], [0, np.sqrt(1 - fid_1)]])
            P1 = sp.csr_matrix([[np.sqrt(1 - fid_0), 0], [0, np.sqrt(fid_1)]])
        
        key = f'{self.sparse}_s_m0_{fid_0}_{fid_1}_{self.num_qubits}_{self._index}'
        measure_0 = get_single_operator(key, self.sparse, P0, self._index, self.num_qubits)
        prob = np.real(np.trace(np.dot(measure_0, self.state)))
        
        if np.random.uniform(0, 1) <= prob:
            self.state = dot(self.state, measure_0) / prob
            return 0
        else:
            key = f'{self.sparse}_s_m1_{fid_0}_{fid_1}_{self.num_qubits}_{self._index}'
            measure_1 = get_single_operator(key, self.sparse, P1, self._index, self.num_qubits)
            self.state = dot(self.state, measure_1) / (1 - prob)
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
            depolarization_error(self, self.sparse, fidelity)

    def bsm(self, target: Qubit, basis_0: str='z', basis_1: str='z', fid_0: float=1., fid_1: float=1.) -> int:
        
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
        
        key = f'{self.sparse}_bsm_{target.num_qubits}_{self._index}_{target._index}'
        gate = get_bsm_operator(key, target.sparse, self._index, target._index, target.num_qubits)
        target.state = dot(target.state, gate)
        
        return 2 * self.measure(basis=basis_0, fid_0=fid_0, fid_1=fid_1) + target.measure(basis=basis_1, fid_0=fid_0, fid_1=fid_1)

    def prob_bsm(self, target: Qubit, success_prob: float=0.57, fid_0: float=1., fid_1: float=1.) -> int:
        
        """
        Applies a probabilistic bell state measurement to the qubits
        
        Args:
            q_src (Qubit): source qubit
            q_dst (Qubit): target qubit
            success_prob (float): success probability of measurement
            
        Returns:
            res (int): result of the bell state measurement
        """
        
        res = self.bsm(target, fid_0, fid_1)
        prob_res = np.random.uniform(0, 1)
        gate = 0
        if 2 * success_prob < 2 * prob_res <= (success_prob + 1):
            gate = 1
        elif 2 * (success_prob + 1) < 4 * prob_res <= (success_prob + 3):
            gate = 2
        elif 4 * prob_res > (success_prob + 3):
            gate = 3
        
        return _PROB_BSM_MAPPING[gate][res]

    def custom_gate(self, gate: np.array) -> None:
        
        """
        Applys a custom gate to the qubit
        
        Args:
            gate (np.array): unitary gate to apply
        
        Returns:
            /
        """
        
        self.state = dot(self.state, gate)

    def purification(self, target: Qubit, direction: bool=0, gate: str='CNOT', basis: str='Z') -> int:
        
        """
        Performs a purification on the self and target qubit
        
        Args:
            target (Qubit): target Qubit which will be measured
            direction (bool): wether s->t or t->s
            gate (str): gate to apply
            basis (str): basis to measure target in
            
        Returns:
            res (int): measurement result
        """
        
        q_1, q_2 = self, target
        if direction:
            q_1, q_2 = q_2, q_1
        
        key = f'{q_2.sparse}_d_{purification_gates[gate].lower()}_{q_2.num_qubits}_{q_1._index}_{q_2._index}'
        gate_f = get_double_operator(key, q_2.sparse, gates[q_1.sparse][purification_gates[gate]], q_1._index, q_2._index, q_2.num_qubits)
        q_2.state = dot(q_2.state, gate_f)
        
        return target.measure(basis)

    def fidelity(self, _op: np.array) -> float:

        """
        Computes the quantum fidelity of this qubit state and a operator

        Args:
            _op (np.array): operator to compare the state of this qubit to

        Returns:
            fidelity (float): fidelity of the quantum state
        """
        
        _sqrt_mat = sqrt_matrix(self.state)
        return float((np.real(np.trace(sqrt_matrix(np.dot(_sqrt_mat, np.dot(_op, _sqrt_mat)))))**2))
        
class QSystem:
    
    """
    Represents a system consisting of multiple entanglable qubits
    
    Attr:
        _num_qubits (int): number of qubits in the qsystem
        _qubits (list): qubits in the qsystem
        _sparse (bool): whether the representation of the density matrix is sparse or not
        _state (np.array): density matrix of qsystem
    """
    
    def __init__(self, num_qubits: int=1, fidelity: float=1., sparse: bool=0) -> None:
        
        """
        Instantiates the qubit state for a qubit system, initializes to the zero state
        
        Args:
            num_qubits (int): number of qubits in the system
            fidelity (float): fidelity of the initial quantum system
            sparse (bool): sparsity of quantum system
            
        Returns:
            /
        """
        
        self._num_qubits: int = num_qubits
        self._qubits: List[Qubit] = [Qubit(self, i) for i in range(self._num_qubits)]
        self._sparse: bool = sparse
        
        init_state = np.array([[fidelity, 0], [0, 1 - fidelity]], dtype=np.complex128)
        if sparse:
            init_state = sp.csr_matrix([[fidelity, 0], [0, 1 - fidelity]], dtype=np.complex128)

        self._state: Union[np.array, sp.csr_matrix] = init_state
        if self._num_qubits > 1:
            self._state: Union[np.array, sp.csr_matrix] = tensor_operator(sparse, np.array([init_state] * self._num_qubits))
    
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
    def state(self) -> np.array:
        
        """
        Returns the full state of the QSystem
        
        Args:
            /
            
        Returns:
            state (np.array): full state of the QSystem
        """
    
    @state.setter
    def state(self, _state: np.array) -> None:
        
        """
        Sets the state of the QSystem
        
        Args:
            _state (np.array): new state of the Qsystem
            
        Returns:
            /
        """
        
        self._state = _state
    
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
    
    def qubit(self, _index: int) -> Qubit:
        
        """
        Returns the qubit at the specified index
        
        Args:
            index (int): index of qubit
            
        Returns:
            qubit (qubit): qubit at specified index
        """
        
        return self._qubits[_index]
    
    def fidelity(self, _op: np.array) -> float:

        """
        Computes the quantum fidelity of this qubit state and a operator

        Args:
            _op (np.array): operator to compare the state of this qubit to

        Returns:
            fidelity (float): fidelity of the quantum state
        """
        
        _sqrt_mat = sqrt_matrix(self._state)
        return float((np.real(np.trace(sqrt_matrix(np.dot(_sqrt_mat, np.dot(_op, _sqrt_mat)))))**2))
    