import numpy as np
from itertools import combinations
from typing import List, Tuple

class Qubit:
    pass

__all__ = ['GHZ_state', 'GHZ_measurement', 'W_state', 'graph_state', 'get_normalized_probability_amplitudes']

def GHZ_state(q_l: List[Qubit]) -> None:

    """
    Brings the qubits into a GHZ state

    Args:
        q_l (list): list of qubits

    Returns:
        /
    """

    q_l[0].H()
    for i in range(len(q_l) - 1):
        q_l[i].CNOT(q_l[i + 1])

def GHZ_measurement(q_l: List[Qubit]) -> int:
        
    """
    Measures in which GHZ state the qubits are, generalization of bell state measurement
    
    Args:
        q_l (list): list of qubits to measure
        
    Returns:
        res (int): measurement result
    """
    
    for i in range(len(q_l) - 1, 0, -1):
        q_l[i - 1].CNOT(q_l[i])
    q_l[0].H()
    
    meas = [q.measure() for q in q_l]
    
    _res = 0
    for m in meas:
        _res = (_res << 1) | m
        
    return _res

def get_w_operator(p: float, q: float) -> np.array:
    
    """
    Creates the w_state operator

    Args:
        p (float): first parameter
        q (float): second parameter

    Returns:
        op (np.array): operator to apply to each qubit
    """

    return 1/np.sqrt(p + q) * np.array([[np.sqrt(p), np.sqrt(q)], [-np.sqrt(q), np.sqrt(p)]])

def W_state(q_l: List[Qubit]) -> None:

    """
    Creates a W state out of the given qubits

    Args:
        q_l (list): list of qubits

    Returns:
        /
    """

    op = get_w_operator(1, len(q_l) - 1)
    q_l[0].custom_single_gate(op)

    for i in range(len(q_l) - 2):
        q_l[i].CU(q_l[i + 1], get_w_operator(1, len(q_l) - 2 - i))

    for i in reversed(range(len(q_l) - 1)):
        q_l[i].CNOT(q_l[i + 1])

    q_l[0].X()

def graph_state(q_l: List[Qubit], graph: List[Tuple[int, int]]) -> None:
    
    """
    Creates a graph state out of a given list of qubits and a graph
    
    Args:
        q_l (list): list of qubits
        graph (list): list of edges
        
    Returns:
        /
    """
    
    for q_1, q_2 in combinations(q_l, 2):
        if not q_1.qsystem == q_2.qsystem:
            raise ValueError('Qubits need to be in the same qsystem')
        
    for q in q_l:
        q.H()
        
    for edge in graph:
        q_l[edge[0]].CZ(q_l[edge[1]])

def get_normalized_probability_amplitudes(num_samples: int) -> np.array:
    
    """
    Generates n normalized probability amplitudes
    
    Args:
        num_samples (int): number of samples
        
    Returns:
        rand (np.array): normalized probability amplitudes
    """
    
    rand = np.sqrt(np.random.uniform(0, 1, num_samples)) * np.exp(1.j * np.random.uniform(0, 2 * np.pi, num_samples))
    
    norm = np.sqrt(np.sum(np.abs(rand) ** 2))
    
    return rand / norm
