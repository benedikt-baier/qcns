import numpy as np
import scipy.sparse as sp
import functools as ft
from typing import List, Union

__all__ = ['Qubit', 'QSystem', 'tensor_operator', 'dot', 'get_single_operator', 'depolarization_error', 'combine_state', 'remove_qubits']

B_0 = np.array([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]], dtype=np.complex128)

full_gates = {'P0': np.array([[1, 0], [0, 0]], dtype=np.complex128),
              'P1': np.array([[0, 0], [0, 1]], dtype=np.complex128),
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

purification_gates = {'CNOT': 'X', 'CY': 'Y', 'CZ': 'Z', 'CH': 'H'}

_PROB_BSM_MAPPING = {0: {0: 0, 1: 1, 2: 2, 3: 3}, 
                    1: {0: 1, 1: 0, 2: 3, 3: 2}, 
                    2: {0: 2, 1: 3, 2: 0, 3: 1}, 
                    3: {0: 3, 1: 2, 2: 1, 3: 0}}

class Qubit:
    pass

class QSystem:
    pass

def cache(func):
    
    """
    Caches the applied gates to reduce calculation times

    Args:
        func (Function): function which input to cache

    Returns:
        wrapper (Function): wrapper function
    """

    _cache = {}
    
    @ft.wraps(func)
    def wrapper(*args):
        
        key = args[0]
        if not key:
            return func(*(args[1:]))
        if key not in _cache:
            _cache[key] = func(*(args[1:]))
        return _cache[key]
    return wrapper

def kronecker_product_full(op_1: np.array, op_2: np.array) -> np.array:
    
    """
    Computes the kronecker product of two full 2d matrices

    Args:
        op_1 (np.array): first matrix
        op_2 (np.array): second matrix

    Returns:
        op (np.array): result of the kronecker product
    """

    op = op_1[:, None, :, None] * op_2[None, :, None, :]
    op.shape = (op_1.shape[0] * op_2.shape[0], op_1.shape[1] * op_2.shape[1])

    return op

def kronecker_product_sparse(op_1: sp.csr_matrix, op_2: sp.csr_matrix) -> sp.csr_matrix:
    
    """
    Computes the kronecker product of two sparse 2d matrices

    Args:
        op_1 (sp.csr_matrix): first matrix
        op_2 (sp.csr_matrix): second matrix

    Returns:
        op (sp.csr_matrix): result of the kronecker product
    """
    
    return sp.kron(op_1, op_2)

_KRON_DICT = {0: kronecker_product_full, 1: kronecker_product_sparse}

def tensor_operator(sparse: bool, operator_l: np.array) -> Union[np.array, sp.csr_matrix]:
    
    """
    Creates the tensor operator out of a operator list
    
    Args:
        sparse (bool): indicator whether the operators are full or sparse
        operator_l (list): list of 2x2 np.arrays
        
    Returns:
        res (np.array): resulting tensor operator
    """

    return ft.reduce(_KRON_DICT[sparse], operator_l)

def dot(state: Union[np.array, sp.csr_matrix], gate: Union[np.array, sp.csr_matrix]) -> Union[np.array, sp.csr_matrix]:
    
    """
    Performs the linalg dot for quantum gates
    
    Args:
        state (np.array): state to apply the gate to
        gate (np.array): gate to apply
        
    Returns:
        state (np.array): resulting state
    """
    
    return gate.dot(state).dot(gate.conj().T)

@cache
def get_single_operator(sparse: bool, gate: Union[np.array, sp.csr_matrix], index: int, num_qubits: int) -> np.array:

    """
    Creates the tensor operator for a single qubit gate
    
    Args:
        sparse (bool): indicator whether the operators are full or sparse
        gate (np.array): 2x2 gate to apply to the target qubit
        index (int): index of target qubit in QSystem
        num_qubits (int): number of qubits in QSystem
        
    Returns:
        operator (np.array): resulting tensor for the single qubit
    """
    
    operator_l = np.array([gates[sparse]['I']] * num_qubits)    
    operator_l[index] = gate
    
    return tensor_operator(sparse, operator_l)

@cache
def get_double_operator(sparse: bool, gate: Union[np.array, sp.csr_matrix], c_index: int, t_index: int, t_num_qubits: int) -> Union[np.array, sp.csr_matrix]:
    
    """
    Creates the tensor operator for a controlled qubit gate
    
    Args:
        sparse (bool): indicator whether the operators are full or sparse
        gate (np.array): 2x2 gate to apply to the target qubit
        c_index (int): index of control qubit
        t_index (int): index of target index
        t_num_qubits (int): number of qubits in target QSystem
        
    Returns:
        operator (np.array): resulting tensor for the controlled qubit gate
    """
    
    key = f'{sparse}_s_m0_{1}_{1}_{t_num_qubits}_{c_index}'
    proj0 = get_single_operator(key, sparse, gates[sparse]['P0'], c_index, t_num_qubits)
    
    proj1 = np.array([gates[sparse]['I']] * t_num_qubits)
    proj1[c_index] = gates[sparse]['P1']
    proj1[t_index] = gate

    return proj0 + tensor_operator(sparse, proj1)

@cache
def get_triple_operator(sparse: bool, gate_l: Union[np.array, sp.csr_matrix], c1_index: int, c2_index: int, t_index: int, t_num_qubits: int) -> Union[np.array, sp.csr_matrix]:
    
    """
    Generates the operator for a 3 qubit gate
    
    Args:
        sparse (bool): indicator whether the operators are full or sparse
        gate_l (np.array): gate list to apply to target
        c1_index (int): index of first control qubit
        c2_index (int): index of second control qubit
        t_index (int): index of target qubit
        t_num_qubit (int): number of qubits in target QSystem
        
    Returns:
        operator (np.array): resulting tensor for the controlled controlled qubit gate
    """
    
    c1, c2 = sorted([c1_index, c2_index])
    
    gates_list = [[gates[sparse]['P0'], gates[sparse]['P0'], gate_l[0]], 
                  [gates[sparse]['P0'], gates[sparse]['P1'], gate_l[1]], 
                  [gates[sparse]['P1'], gates[sparse]['P0'], gate_l[2]], 
                  [gates[sparse]['P1'], gates[sparse]['P1'], gate_l[3]]]
    ops = np.array([gates[sparse]['I']] * t_num_qubits)
        
    CNOTijk = 0
    for gates in gates_list:
        ops[c1] = gates[0]
        ops[c2] = gates[1]
        ops[t_index] = gates[2]
        CNOTijk += tensor_operator(sparse, ops)
    return CNOTijk

@cache
def get_bell_operator(sparse: bool, bell_state: int, c_index: int, t_index: int, t_num_qubits: int) -> Union[np.array, sp.csr_matrix]:
    
    """
    Generates the bell operator
    
    Args:
        sparse (bool): indicator whether the operators are full or sparse
        c_index (int): index of control qubit
        t_index (int): index of target qubit
        t_num_qubits (int): number of qubits in target QSystem
        
    Returns:
        res (np.array): resulting tensor for the bsm operator
    """
    
    h_gate = get_single_operator(f'{sparse}_s_h_{t_num_qubits}_{c_index}', sparse, gates[sparse]['H'], c_index, t_num_qubits)
    cnot_gate = get_double_operator(f'{sparse}_d_x_{t_num_qubits}_{c_index}_{t_index}', sparse, gates[sparse]['X'], c_index, t_index, t_num_qubits)
    
    if bell_state == 0:
        return cnot_gate.dot(h_gate)
    if bell_state == 1:
        x_gate = get_single_operator(f'{sparse}_s_x_{t_num_qubits}_{t_index}', sparse, gates[sparse]['X'], t_index, t_num_qubits)
        return cnot_gate.dot(h_gate).dot(x_gate)
    if bell_state == 2:
        x_gate = get_single_operator(f'{sparse}_s_x_{t_num_qubits}_{c_index}', sparse, gates[sparse]['X'], c_index, t_num_qubits)
        return cnot_gate.dot(h_gate).dot(x_gate)
    if bell_state == 3:
        x_gate_c = get_single_operator(f'{sparse}_s_x_{t_num_qubits}_{c_index}', sparse, gates[sparse]['X'], c_index, t_num_qubits)
        x_gate_t = get_single_operator(f'{sparse}_s_x_{t_num_qubits}_{t_index}', sparse, gates[sparse]['X'], t_index, t_num_qubits)
        return cnot_gate.dot(h_gate).dot(x_gate_c).dot(x_gate_t)

@cache
def get_bsm_operator(sparse: bool, c_index: int, t_index: int, t_num_qubits: int) -> np.array:
    
    """
    Generates the bsm operator
    
    Args:
        sparse (bool): indicator whether the operators are full or sparse
        c_index (int): index of control qubit
        t_index (int): index of target qubit
        t_num_qubits (int): number of qubits in target QSystem
        
    Returns:
        res (np.array): resulting tensor for the bsm operator
    """
    
    cnot_gate = get_double_operator(f'{sparse}_d_x_{t_num_qubits}_{c_index}_{t_index}', sparse, gates[sparse]['X'], c_index, t_index, t_num_qubits)
    h_gate = get_single_operator(f'{sparse}_s_h_{t_num_qubits}_{c_index}', sparse, gates[sparse]['H'], c_index, t_num_qubits)
    
    return h_gate.dot(cnot_gate) 

def depolarization_error(sparse: bool, qubit: Qubit, fidelity: float) -> None:
    
    """
    Applys a depolarization error to the qubit given a fidelity
    
    Args:
        qubit (Qubit): qubit to apply error to
        fidelity (float): fidelity of the depolarization error
        
    Returns:
        /
    """
    
    key_x = f'{sparse}_s_x_{qubit._qsystem._num_qubits}_{qubit._index}'
    key_y = f'{sparse}_s_y_{qubit._qsystem._num_qubits}_{qubit._index}'
    key_z = f'{sparse}_s_z_{qubit._qsystem._num_qubits}_{qubit._index}'
    
    gate_x = get_single_operator(key_x, sparse, gates[sparse]['X'], qubit._index, qubit._qsystem._num_qubits)    
    gate_y = get_single_operator(key_y, sparse, gates[sparse]['y'], qubit._index, qubit._qsystem._num_qubits)
    gate_z = get_single_operator(key_z, sparse, gates[sparse]['Z'], qubit._index, qubit._qsystem._num_qubits)
        
    qubit._qsystem._state = fidelity * qubit._qsystem._state + ((1 - fidelity) / 3) * (dot(qubit._qsystem._state, gate_x) + dot(qubit._qsystem._state, gate_y) + dot(qubit._qsystem._state, gate_z))

def combine_state(_q_l: List[Qubit]) -> QSystem:
        
        """
        Combines multiple qsystems into one qsystem
        
        Args:
            _q_l (list): List of qsystems
            
        Returns:
            qsys_n (QSystem): new qsystem
        """
        
        if len(set([id(qubit._qsystem) for qubit in _q_l])) == 1:
            return q_l[0]._qsystem
        
        qsys_n = Simulation.create_qsystem(1, sparse=_q_l[0]._qsystem._sparse)
        
        qsys_l = [q._qsystem for n, q in enumerate(_q_l) if q not in _q_l[:n]]
        
        num_qubits_n = sum([qsys._num_qubits for qsys in qsys_l])
        
        qsys_n._qubits = [q for qsys in qsys_l for q in qsys._qubits]
        qsys_n._num_qubits = num_qubits_n
        qsys_n._state = tensor_operator(q_l[0]._qsystem._sparse, np.array([qsys._state for qsys in qsys_l], dtype=object))
        qsys_n._state = qsys_n._state.astype(np.complex128)
        for i in range(num_qubits_n):
            qsys_n._qubits[i]._index = i
            qsys_n._qubits[i]._qsystem = qsys_n

        return qsys_n
    
def ptrace(_q: Qubit) -> None:
    
    """
    Traces out a qubit from the state, adjusts the QSystem accordingly
    
    Args:
        _q (Qubit): qubit to trace out
        
    Returns:
        /
    """

    dims_ = np.array([2] * 2 * _q._qsystem._num_qubits, dtype=np.int8)
    state_dim = 1 << (_q._qsystem._num_qubits - 1)

    reshaped_state = _q._qsystem._state.reshape(dims_)

    reshaped_state = np.moveaxis(reshaped_state, _q._index, -1)
    reshaped_state = np.moveaxis(reshaped_state, _q._qsystem._num_qubits + _q._index - 1, -1)

    _q._qsystem._state = np.trace(reshaped_state, axis1 = -2, axis2 = -1).reshape([state_dim, state_dim])
    _q._qsystem._num_qubits -= 1
    
    del _q._qsystem._qubits[_q._index]  
    for i in range(_q._qsystem._num_qubits):
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
    
    key_1 = f'{1}_v1_{_q._qsystem._num_qubits}_{_q._index}'
    key_2 = f'{1}_v1T_{_q._qsystem._num_qubits}_{_q._index}'
    key_3 = f'{1}_v2_{_q._qsystem._num_qubits}_{_q._index}'
    key_4 = f'{1}_v2T_{_q._qsystem._num_qubits}_{_q._index}'
    
    op_1_a = get_single_operator_gate(key_1, 1, v1, _q._index, _q._qsystem._num_qubits)
    op_1_b = get_single_operator_gate(key_2, 1, v1.T, _q._index, _q._qsystem._num_qubits)
    op_2_a = get_single_operator_gate(key_3, 1, v2, _q._index, _q._qsystem._num_qubits)
    op_2_b = get_single_operator_gate(key_4, 1, v2.T, _q._index, _q._qsystem._num_qubits)

    _q._qsystem._state = sp.csr_matrix(op_1_a.dot(_q._qsystem._state.dot(op_1_b)) + op_2_a.dot(_q._qsystem._state.dot(op_2_b)))
    _q._qsystem._state.eliminate_zeros()
    _q._qsystem._num_qubits -= 1
    
    del _q._qsystem._qubits[_q._index]
    for i in range(_q._qsystem._num_qubits):
        _q._qsystem._qubits[i]._index = i

def remove_qubits(_q_l: List[Qubit]) -> None:

    """
    Traces out the given qubits from the state, adjusts the QSystem accordingly
    
    Args:
        _q_l (list): list of qubits to trace out
        
    Returns:
        /
    """
    
    for q in _q_l:
        if q._qsystem._sparse:
            ptrace_sparse(q)
        else:
            ptrace(q)

class Qubit:
    
    """
    Represents a single qubit
    """
    
    def __init__(self, _qsystem: QSystem, _index: int) -> None:
        
        """
        Instantiate a single qubit
        
        Args:
            qsystem (QSystem): reference to parent qsystem, which the qubit is part of
            index (int): index of the qubit in the parent qsystem
            
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
        
        return str(self._qsystem._state)
    
    def X(self, fidelity: float=1.) -> None:

        """
        Applys the X gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_s_x_{self._qsystem._num_qubits}_{self._index}'
        gate = get_single_operator(key, self._qsystem._sparse, gates[self._qsystem._sparse]['X'], self._index, self._qsystem._num_qubits)
        self._qsystem._state = dot(self._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
        
    def Y(self, fidelity: float=1.) -> None:
        
        """
        Applys the Y gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_s_y_{self._qsystem._num_qubits}_{self._index}'
        gate = get_single_operator(key, self._qsystem._sparse, gates[self._qsystem._sparse]['Y'], self._index, self._qsystem._num_qubits)
        self._qsystem._state = dot(self._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
        
    def Z(self, fidelity: float=1.) -> None:
        
        """
        Applys the Z gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_s_z_{self._qsystem._num_qubits}_{self._index}'
        gate = get_single_operator(key, self._qsystem._sparse, gates[self._qsystem._sparse]['Z'], self._index, self._qsystem._num_qubits)
        self._qsystem._state = dot(self._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
        
    def H(self, fidelity: float=1.) -> None:
        
        """
        Applys the Hadamard gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_s_h_{self._qsystem._num_qubits}_{self._index}'
        gate = get_single_operator(key, self._qsystem._sparse, gates[self._qsystem._sparse]['H'], self._index, self._qsystem._num_qubits)
        self._qsystem._state = dot(self._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)

    def SX(self, fidelity: float=1.) -> None:
        
        """
        Applys the square root X gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_s_sx_{self._qsystem._num_qubits}_{self._index}'
        gate = get_single_operator(key, self._qsystem._sparse, gates[self._qsystem._sparse]['SX'], self._index, self._qsystem._num_qubits)
        self._qsystem._state = dot(self._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
    
    def SY(self, fidelity: float=1.) -> None:
        
        """
        Applys the square root Y gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_s_sy_{self._qsystem._num_qubits}_{self._index}'
        gate = get_single_operator(key, self._qsystem._sparse, gates[self._qsystem._sparse]['SY'], self._index, self._qsystem._num_qubits)
        self._qsystem._state = dot(self._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
    
    def SZ(self, fidelity: float=1.) -> None:
        
        """
        Applys the square root Z gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_s_sz_{self._qsystem._num_qubits}_{self._index}'
        gate = get_single_operator(key, self._qsystem._sparse, gates[self._qsystem._sparse]['SZ'], self._index, self._qsystem._num_qubits)
        self._qsystem._state = dot(self._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
    
    def T(self, fidelity: float=1.) -> None:
        
        """
        Applys the T gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
    
        key = f'{self._qsystem._sparse}_s_t_{self._qsystem._num_qubits}_{self._index}'
        gate = get_single_operator(key, self._qsystem._sparse, gates[self._qsystem._sparse]['T'], self._index, self._qsystem._num_qubits)
        self._qsystem._state = dot(self._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
    
    def K(self, fidelity: float=1.) -> None:
        
        """
        Applys the K gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_s_k_{self._qsystem._num_qubits}_{self._index}'
        gate = get_single_operator(key, self._qsystem._sparse, gates[self._qsystem._sparse]['K'], self._index, self._qsystem._num_qubits)
        self._qsystem._state = dot(self._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
    
    def iSX(self, fidelity: float=1.) -> None:
        
        """
        Applys the inverse square root X gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_s_isx_{self._qsystem._num_qubits}_{self._index}'
        gate = get_single_operator(key, self._qsystem._sparse, gates[self._qsystem._sparse]['iSX'], self._index, self._qsystem._num_qubits)
        self._qsystem._state = dot(self._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
    
    def iSY(self, fidelity: float=1.) -> None:
        
        """
        Applys the inverse square root Y gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_s_isy_{self._qsystem._num_qubits}_{self._index}'
        gate = get_single_operator(key, self._qsystem._sparse, gates[self._qsystem._sparse]['iSY'], self._index, self._qsystem._num_qubits)
        self._qsystem._state = dot(self._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
    
    def iSZ(self, fidelity: float=1.):
        
        """
        Applys the inverse square root Z gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_s_isz_{self._qsystem._num_qubits}_{self._index}'
        gate = get_single_operator(key, self._qsystem._sparse, gates[self._qsystem._sparse]['iSZ'], self._index, self._qsystem._num_qubits)
        self._qsystem._state = dot(self._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
    
    def iT(self, fidelity: float=1.):
        
        """
        Applys the inverse of T gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_s_it_{self._qsystem._num_qubits}_{self._index}'
        gate = get_single_operator(key, self._qsystem._sparse, gates[self._qsystem._sparse]['iT'], self._index, self._qsystem._num_qubits)
        self._qsystem._state = dot(self._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
    
    def iK(self, fidelity: float=1.):
        
        """
        Applys the inverse of K gate to the qubit
        
        Args:
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_s_ik_{self._qsystem._num_qubits}_{self._index}'
        gate = get_single_operator(key, self._qsystem._sparse, gates[self._qsystem._sparse]['iK'], self._index, self._qsystem._num_qubits)
        self._qsystem._state = dot(self._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
    
    def Rx(self, theta: float, fidelity: float=1.) -> None:
        
        """
        Applys the rotation gate around the x axis to the qubit
        
        Args:
            theta (float): angle for the rotation matrix
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_s_rx_{theta}_{self._qsystem._num_qubits}_{self._index}'
        gate_s = np.array([[np.cos(theta/2), -1j * np.sin(theta/2)], [-1j * np.sin(theta/2), np.cos(theta/2)]], dtype=np.complex128)
        if self._qsystem._sparse:
            gate_s = sp.csr_matrix([[np.cos(theta/2), -1j * np.sin(theta/2)], [-1j * np.sin(theta/2), np.cos(theta/2)]], dtype=np.complex128)
        gate = get_single_operator(key, self._qsystem._sparse, gate_s, self._index, self._qsystem._num_qubits)
        self._qsystem._state = dot(self._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
    
    def Ry(self, theta: float, fidelity: float=1.) -> None:
        
        """
        Applys the rotation gate around the y axis to the qubit
        
        Args:
            theta (float): angle for the rotation matrix
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_s_ry_{theta}_{self._qsystem._num_qubits}_{self._index}'
        gate_s = np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]], dtype=np.complex128)
        if self._qsystem._sparse:
            gate_s = sp.csr_matrix([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]], dtype=np.complex128)
        gate = get_single_operator(key, self._qsystem._sparse, gate_s, self._index, self._qsystem._num_qubits)
        self._qsystem._state = dot(self._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
        
    def Rz(self, theta: float, fidelity: float=1.) -> None:
        
        """
        Applys the rotation gate around the z axis to the qubit
        
        Args:
            theta (float): angle for the rotation matrix
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_s_rz_{theta}_{self._qsystem._num_qubits}_{self._index}'
        gate_s = np.array([[np.exp(-1j * theta/2), 0], [0, np.exp(1j * theta/2)]], dtype=np.complex128)
        if self._qsystem._sparse:
            sp.csr_matrix([[np.exp(-1j * theta/2), 0], [0, np.exp(1j * theta/2)]], dtype=np.complex128)
        gate = get_single_operator(key, self._qsystem._sparse, gate_s, self._index, self._qsystem._num_qubits)
        self._qsystem._state = dot(self._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
        
    def PHASE(self, theta: float, fidelity: float=1.) -> None:
        
        """
        Applys the PHASE gate to the qubit
        
        Args:
            theta (float): angle for the rotation matrix
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_s_p_{theta}_{self._qsystem._num_qubits}_{self._index}'
        gate_s = np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=np.complex128)
        if self._qsystem._sparse:
            gate_s = sp.csr_matrix([[1, 0], [0, np.exp(1j * theta)]], dtype=np.complex128)
        gate = get_single_operator(key, self._qsystem._sparse, gate_s, self._index, self._qsystem._num_qubits)
        self._qsystem._state = dot(self._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
    
    def general_rotation(self, theta: float, phi: float, psi: float, fidelity: float=1.) -> None:
        
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
        
        key = f'{self._qsystem._sparse}_s_gr_{theta}_{phi}_{psi}_{self._qsystem._num_qubits}_{self._index}'
        gate_s = np.array([[np.cos(theta/2), -np.exp(1j*psi)*np.sin(theta/2)], [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+psi))*np.cos(theta/2)]], dtype=np.complex128)
        if self._qsystem._sparse:
            gate_s = sp.csr_matrix([[np.cos(theta/2), -np.exp(1j*psi)*np.sin(theta/2)], [np.exp(1j*phi)*np.sin(theta/2), np.exp(1j*(phi+psi))*np.cos(theta/2)]], dtype=np.complex128)
        gate = get_single_operator(key, self._qsystem._sparse, gate_s, self._index, self._qsystem._num_qubits)
        self._qsystem._state = dot(self._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
    
    def custom_gate(self, gate: np.array, fidelity: float=1.) -> None:
        
        """
        Applys a custom gate to the qubit
        
        Args:
            gate (np.array): unitary gate to apply
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        gate_s = get_single_operator('', self._qsystem._sparse, gate, self._index, self._qsystem._num_qubits)
        self._qsystem._state = dot(self._qsystem._state, gate_s)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
     
    def CNOT(self, target: Qubit, fidelity: float=1.) -> None:
        
        """
        Applys the CNOT gate to a target qubit, with this qubit as the controll qubit
        
        Args:
            target (qubit): target qubit to apply the CNOT gate to
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_d_x_{target._qsystem._num_qubits}_{self._index}_{target._index}'
        gate = get_double_operator(key, target._qsystem._sparse, gates[self._qsystem._sparse]['X'], self._index, target._index, target._qsystem._num_qubits)
        target._qsystem._state = dot(target._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
            depolarization_error(target, target._qsystem._sparse, fidelity)

    def CX(self, target: Qubit, fidelity: float=1.) -> None:
        
        """
        Applys the CNOT gate to a target qubit, with this qubit as the controll qubit
        
        Args:
            target (qubit): target qubit to apply the CNOT gate to
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_d_x_{target._qsystem._num_qubits}_{self._index}_{target._index}'
        gate = get_double_operator(key, target._qsystem._sparse, gates[self._qsystem._sparse]['X'], self._index, target._index, target._qsystem._num_qubits)
        target._qsystem._state = dot(target._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
            depolarization_error(target, target._qsystem._sparse, fidelity)

    def CY(self, target: Qubit, fidelity: float=1.) -> None:
        
        """
        Applys the CY gate to a target qubit, with this qubit as the control qubit
        
        Args:
            target (qubit): target qubit to apply the CNOT gate to
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_d_y_{target._qsystem._num_qubits}_{self._index}_{target._index}'
        gate = get_double_operator(key, target._qsystem._sparse, gates[self._qsystem._sparse]['Y'], self._index, target._index, target._qsystem._num_qubits)
        target._qsystem._state = dot(target._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
            depolarization_error(target, target._qsystem._sparse, fidelity)
    
    def CZ(self, target: Qubit, fidelity: float=1.) -> None:
        
        """
        Applys the CZ gate to a target qubit, with this qubit as the control qubit
        
        Args:
            target (qubit): target qubit to apply the CNOT gate to
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_d_z_{target._qsystem._num_qubits}_{self._index}_{target._index}'
        gate = get_double_operator(key, target._qsystem._sparse, gates[self._qsystem._sparse]['Z'], self._index, target._index, target._qsystem._num_qubits)
        target._qsystem._state = dot(target._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
            depolarization_error(target, target._qsystem._sparse, fidelity)
    
    def CH(self, target: Qubit, fidelity: float=1.) -> None:
        
        """
        Applys the CH gate to a target qubit, with this qubit as the control qubit
        
        Args:
            target (qubit): target qubit to apply the CNOT gate to
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_d_h_{target._qsystem._num_qubits}_{self._index}_{target._index}'
        gate = get_double_operator(key, target._qsystem._sparse, gates[self._qsystem._sparse]['H'], self._index, target._index, target._qsystem._num_qubits)
        target._qsystem._state = dot(target._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
            depolarization_error(target, target._qsystem._sparse, fidelity)
    
    def CPHASE(self, target: Qubit, theta: float, fidelity: float=1.) -> None:
        
        """
        Applys the CPHASE gate to a target qubit, with this qubit as the controll qubit
        
        Args:
            target (qubit): target qubit to apply the CPHASE gate to
            theta (float): angle for the rotation matrix
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_d_p_{theta}_{target._qsystem._num_qubits}_{self._index}_{target._index}'
        gate_s = np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=np.complex128)
        if self._qsystem._sparse:
            gate_s = sp.csr_matrix([[1, 0], [0, np.exp(1j * theta)]], dtype=np.complex128)
        gate = get_double_operator(key, target._qsystem._sparse, gate_s, self._index, target._index, target._qsystem._num_qubits)
        target._qsystem._state = dot(target._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
            depolarization_error(target, self._qsystem._sparse, fidelity)
    
    def CU(self, target: Qubit, gate: np.array, fidelity: float=1.) -> None:
        
        """
        Applys a custom gate to a target qubit, with this qubit as the controll qubit
        
        Args:
            target (qubit): target qubit to apply a custom gate to
            gate (np.array): 2x2 unitary array
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        gate_s = get_double_operator('', target._qsystem._sparse, gate, self._index, target._index, target._qsystem._num_qubits)
        target._qsystem._state = dot(target._qsystem._state, gate_s)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
            depolarization_error(target, self._qsystem._sparse, fidelity)
    
    def SWAP(self, target: Qubit, fidelity: float=1.) -> None:
        
        """
        Swaps the state of this qubit with the target qubit by applying CNOT gates
        
        Args:
            target (qubit): target qubit to swap state with
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        self._index, target._index = target._index, self._index
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
            depolarization_error(target, target._qsystem._sparse, fidelity)
    
    def iSWAP(self, target: Qubit, fidelity: float=1.) -> None:
        
        """
        Applys the Imaginary Swap gate to this qubit and target qubit
        
        Args:
            target (qubit): target qubit to apply the CPHASE gate to
            fidelity (float): fidelity of the depolarization error
        
        Returns:
            /
        """
        
        self.SWAP(target)
        self.SZ()
        target.SZ()
        target.H()
        self.CNOT(target)
        target.H()
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
            depolarization_error(target, self._qsystem._sparse, fidelity)
    
    def QAND(self, control: Qubit, target: Qubit, fidelity: float=1.) -> None:
        
        """
        Applys the quantum AND gate to the target qubit with this qubit and control qubit as control qubits
        
        Args:
            control (qubit): second control qubit
            target (qubit): target qubit
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_t_iiix_{target._qsystem._num_qubits}_{self._index}_{control._index}_{target._index}'
        gate = get_triple_operator(key, target._qsystem._sparse, 
                                    [gates[self._qsystem._sparse]['I'],gates[self._qsystem._sparse]['I'], gates[self._qsystem._sparse]['I'], gates[self._qsystem._sparse]['X']], 
                                    self._index, control._index, target._index, target._qsystem._num_qubits)
        target._qsystem._state = dot(target._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
            depolarization_error(control, self._qsystem._sparse, fidelity)
            depolarization_error(target, self._qsystem._sparse, fidelity)
    
    def QOR(self, control: Qubit, target: Qubit, fidelity: float=1.) -> None:
        
        """
        Applys the quantum OR gate to the target qubit with this qubit and control qubit as control qubits
        
        Args:
            control (qubit): second control qubit
            target (qubit): target qubit
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_t_ixxx_{target._qsystem._num_qubits}_{self._index}_{control._index}_{target._index}'
        gate = get_triple_operator(key, target._qsystem._sparse, 
                                    [gates[self._qsystem._sparse]['I'],gates[self._qsystem._sparse]['X'], gates[self._qsystem._sparse]['X'], gates[self._qsystem._sparse]['X']], 
                                    self._index, control._index, target._index, target._qsystem._num_qubits)
        target._qsystem._state = dot(target._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
            depolarization_error(control, self._qsystem._sparse, fidelity)
            depolarization_error(target, self._qsystem._sparse, fidelity)
    
    def QXOR(self, control: Qubit, target: Qubit, fidelity: float=1.) -> None:
        
        """
        Applys the quantum XOR gate to the target qubit with this qubit and control qubit as control qubits
        
        Args:
            control (qubit): second control qubit
            target (qubit): target qubit
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_t_ixxi_{target._qsystem._num_qubits}_{self._index}_{control._index}_{target._index}'
        gate = get_triple_operator(key, target._qsystem._sparse, 
                                    [gates[self._qsystem._sparse]['I'],gates[self._qsystem._sparse]['X'], gates[self._qsystem._sparse]['X'], gates[self._qsystem._sparse]['I']], 
                                    self._index, control._index, target._index, target._qsystem._num_qubits)
        target._qsystem._state = dot(target._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
            depolarization_error(control, self._qsystem._sparse, fidelity)
            depolarization_error(target, self._qsystem._sparse, fidelity)
    
    def QNAND(self, control: Qubit, target: Qubit, fidelity: float=1.) -> None:
        
        """
        Applys the quantum NAND gate to the target qubit with this qubit and control qubit as control qubits
        
        Args:
            control (qubit): second control qubit
            target (qubit): target qubit
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_t_xxxi_{target._qsystem._num_qubits}_{self._index}_{control._index}_{target._index}'
        gate = get_triple_operator(key, target._qsystem._sparse, 
                                    [gates[self._qsystem._sparse]['X'],gates[self._qsystem._sparse]['X'], gates[self._qsystem._sparse]['X'], gates[self._qsystem._sparse]['I']], 
                                    self._index, control._index, target._index, target._qsystem._num_qubits)
        target._qsystem._state = dot(target._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
            depolarization_error(control, self._qsystem._sparse, fidelity)
            depolarization_error(target, self._qsystem._sparse, fidelity)
    
    def QNOR(self, control: Qubit, target: Qubit, fidelity: float=1.) -> None:
        
        """
        Applys the quantum NOR gate to the target qubit with this qubit and control qubit as control qubits
        
        Args:
            control (qubit): second control qubit
            target (qubit): target qubit
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_t_xiii_{target._qsystem._num_qubits}_{self._index}_{control._index}_{target._index}'
        gate = get_triple_operator(key, target._qsystem._sparse, 
                                    [gates[self._qsystem._sparse]['X'],gates[self._qsystem._sparse]['I'], gates[self._qsystem._sparse]['I'], gates[self._qsystem._sparse]['I']], 
                                    self._index, control._index, target._index, target._qsystem._num_qubits)
        target._qsystem._state = dot(target._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
            depolarization_error(control, self._qsystem._sparse, fidelity)
            depolarization_error(target, self._qsystem._sparse, fidelity)
    
    def QXNOR(self, control: Qubit, target: Qubit, fidelity: float=1.) -> None:
        
        """
        Applys the quantum XNOR gate to the target qubit with this qubit and control qubit as control qubits
        
        Args:
            control (qubit): second control qubit
            target (qubit): target qubit
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
        
        key = f'{self._qsystem._sparse}_t_xiix_{target._qsystem._num_qubits}_{self._index}_{control._index}_{target._index}'
        gate = get_triple_operator(key, target._qsystem._sparse, 
                                    [gates[self._qsystem._sparse]['X'],gates[self._qsystem._sparse]['I'], gates[self._qsystem._sparse]['I'], gates[self._qsystem._sparse]['X']], 
                                    self._index, control._index, target._index, target._qsystem._num_qubits)
        target._qsystem._state = dot(target._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
            depolarization_error(control, self._qsystem._sparse, fidelity)
            depolarization_error(target, self._qsystem._sparse, fidelity)
    
    def CCU(self, control: Qubit, target: Qubit, gate: np.array, fidelity: float=1.) -> None:
        
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
        
        gate_s = get_triple_operator('', target._qsystem._sparse, 
                                        [gates[self._qsystem._sparse]['I'], gates[self._qsystem._sparse]['I'], gates[self._qsystem._sparse]['I'], gate],
                                        self._index, control._index, target._index, target._qsystem._num_qubits)
        target._qsystem._state = dot(target._qsystem._state, gate_s)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
            depolarization_error(control, self._qsystem._sparse, fidelity)
            depolarization_error(target, self._qsystem._sparse, fidelity)
    
    def CSWAP(self, target1: Qubit, target2: Qubit, fidelity: float=1.) -> None:
        
        """
        Applys the CSWAP gate with self as control and target1 and target2 as targets
        
        Args:
            target1 (qubit): first target qubit
            target2 (qubit): second target qubit
            fidelity (float): fidelity of the depolarization error
            
        Returns:
            /
        """
            
        self.TOFFOLI(target2, target1)
        self.TOFFOLI(target1, target2)
        self.TOFFOLI(target2, target1)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
            depolarization_error(target1, target1._qsystem._sparse, fidelity)
            depolarization_error(target2, target2._qsystem._sparse, fidelity)
    
    def bell_state(self, target: Qubit, bell_state: int=0, fidelity: float=1.) -> None:
        
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
        
        key = f'{self._qsystem._sparse}_bs_{bell_state}_{target._qsystem._num_qubits}_{self._index}_{target._index}'
        gate = get_bell_operator(key, target._qsystem._sparse, bell_state, self._index, target._index, target._qsystem._num_qubits)
        target._qsystem._state = dot(target._qsystem._state, gate)
        
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)
            depolarization_error(target, target._qsystem._sparse, fidelity)
    
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
        if self._qsystem._sparse:
            P0 = sp.csr_matrix([[np.sqrt(fid_0), 0], [0, np.sqrt(1 - fid_1)]])
            P1 = sp.csr_matrix([[np.sqrt(1 - fid_0), 0], [0, np.sqrt(fid_1)]])
        
        key = f'{self._qsystem._sparse}_s_m0_{fid_0}_{fid_1}_{self._qsystem._num_qubits}_{self._index}'
        measure_0 = get_single_operator(key, self._qsystem._sparse, P0, self._index, self._qsystem._num_qubits)
        prob = np.real(np.trace(np.dot(measure_0, self._qsystem._state)))
        
        if np.random.uniform(0, 1) <= prob:
            self._qsystem._state = dot(self._qsystem._state, measure_0) / prob
            return 0
        else:
            key = f'{self._qsystem._sparse}_s_m1_{fid_0}_{fid_1}_{self._qsystem._num_qubits}_{self._index}'
            measure_1 = get_single_operator(key, self._qsystem._sparse, P1, self._index, self._qsystem._num_qubits)
            self._qsystem._state = dot(self._qsystem._state, measure_1) / (1 - prob)
            return 1

    def state_transfer(self, source: Qubit) -> None:
        
        """
        Transfers a state from the source qubit
        
        Args:
            source (Qubit): qubit from which state is taken
            
        Returns:
            /
        """
        
        self.CNOT(source)
        res = source.measure()
        
        remove_qubits([source])
        
        if res:
            self.X()

    def bsm(self, target: Qubit, basis_0: str='z', basis_1: str='z', fid_0: float=1., fid_1: float=1.) -> int:
        
        """
        Performs a Bell state measurement on this qubit and another qubit
        
        Args:
            target (Qubit): target qubit
            fid_0 (float): fidelity of measuring 0
            fid_1 (flaot): fidelity of measuring 1
            
        Returns:
            res (int): measurement result
        """
        
        key = f'{self._qsystem._sparse}_bsm_{target._qsystem._num_qubits}_{self._index}_{target._index}'
        gate = get_bsm_operator(key, target._qsystem._sparse, self._index, target._index, target._qsystem._num_qubits)
        target._qsystem._state = dot(target._qsystem._state, gate)
        
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
        
        key = f'{q_2._qsystem._sparse}_d_{purification_gates[gate].lower()}_{q_2._qsystem._num_qubits}_{q_1._index}_{q_2._index}'
        gate_f = get_double_operator(key, q_2._qsystem._sparse, gates[q_1._qsystem._sparse][purification_gates[gate]], q_1._index, q_2._index, q_2._qsystem._num_qubits)
        q_2._qsystem._state = dot(q_2._qsystem._state, gate_f)
        
        return target.measure(basis)

    def exp_pauli(self, theta: float, gates: List[int], fidelity: float=1.) -> None:
        
        """
        Applies a exponential of a Pauli gate sequence with custom angle 
        
        Args:
            theta (float): angle to rotate
            gates (list): list of pauli gates
            fidelity (float): fidelity of depolarization error
            
        Returns:
            /
        """
        
        identity_seq = tensor_operator([gates[self._qsystem._sparse]['I']] * len(gates))
        pauli_gates = [gates[self._qsystem._sparse]['I'], gates[self._qsystem._sparse]['X'], gates[self._qsystem._sparse]['Y'], gates[self._qsystem._sparse]['Z']]
        pauli_seq = tensor_operator([pauli_gates[gate] for gate in gates])
        gate = np.cos(theta/2) * identity_seq - 1j * np.sin(theta/2) * pauli_seq
        
        self._qsystem._state = dot(self._qsystem._state, gate)
        if fidelity < 1.:
            depolarization_error(self, self._qsystem._sparse, fidelity)

    def fidelity(self, operator: np.array) -> float:

        """
        Computes the quantum fidelity of this qubit state and a operator

        Args:
            operator (np.array): operator to compare the state of this qubit to

        Returns:
            fidelity (float): fidelity of the quantum state
        """
            
        if self._qsystem._sparse:
            return float(np.real((self._qsystem._state.dot(operator).trace())) + 2 * np.real(np.sqrt(sp.linalg.det(self._qsystem._state) * sp.linalg.det(operator))))
        else:
            return float(np.real(np.einsum('ij,ji', self._qsystem._state, operator)) + 2 * np.real(np.sqrt(np.linalg.det(self._qsystem._state) * np.linalg.det(operator))))
    
class QSystem:
    
    """
    Represents a system consisting of multiple entanglable qubits
    """
    
    def __init__(self, _num_qubits: int=1, _fidelity: float=1., _sparse: bool=0) -> None:
        
        """
        Instantiates the qubit state for a qubit system, initializes to the zero state
        
        Args:
            num_qubits (int): number of qubits in the system
            fidelity (float): fidelity of the initial quantum system
            
        Returns:
            /
        """
        
        self._num_qubits = _num_qubits
        self._qubits = [Qubit(self, i) for i in range(self._num_qubits)]
        self._sparse = _sparse
        
        init_state = np.array([[_fidelity, 0], [0, 1 - _fidelity]], dtype=np.complex128)
        if _sparse:
            init_state = sp.csr_matrix([[_fidelity, 0], [0, 1 - _fidelity]], dtype=np.complex128)

        self._state = init_state
        if self._num_qubits > 1:
            self._state = tensor_operator(_sparse, np.array([init_state] * self._num_qubits))
    
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
    def qubits(self) -> List[Qubit]:
        
        """
        Makes the retrieving of qubits out of a QSystem more convienent
        
        Args:
            /
            
        Returns:
            qubits (list/Qubit): returns a Qubit or a list of Qubits, depending on the number of qubits
        """
        
        if self._num_qubits > 1:
            return self._qubits
        return self._qubits[0]
    
    def qubit(self, index: int) -> Qubit:
        
        """
        Returns the qubit at the specified index
        
        Args:
            index (int): index of qubit
            
        Returns:
            qubit (qubit): qubit at specified index
        """
        
        return self._qubits[index]
    