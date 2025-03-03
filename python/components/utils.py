import numpy as np
from numpy import pi, e
import scipy.sparse as sp
import itertools as it
from re import sub, findall
from typing import List, Tuple, Dict

class Qubit:
    pass

__all__ = ['GHZ_state', 'W_state', 'graph_state', 'ghz_m', 'get_measure_dict', 'get_normalized_probability_amplitudes', 'load_qasm_2_0_file', 'load_qasm_3_0_file', 'apply_circuit', 'dqc_apply_circuit']

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

def W_state(q_l: List[Qubit]) -> None:

    """
    Creates a W state out of the given qubits

    Args:
        q_l (list): list of qubits

    Returns:
        /
    """

    op = get_w_operator(1, len(q_l) - 1)
    q_l[0].custom_gate(op)

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
    
    for q_1, q_2 in it.combinations(q_l, 2):
        if not q_1.qsystem == q_2.qsystem:
            raise ValueError('Qubits need to be in the same qsystem')
        
    for q in q_l:
        q.H()
        
    for edge in graph:
        q_l[edge[0]].CZ(q_l[edge[1]])
    
def ghz_m(q_l: List[Qubit]) -> int:
        
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
   
def get_measure_dict(num_qubits: int) -> Dict[str, int]:
    
    """
    Creates a measurement dictionary with the binary states as keys and 0 as values
    
    Args:
        num_qubits (int): number of qubits to measure
        
    Returns:
        meas_stats (dict): dictonary with str as keys and 0s as values
    """
    
    return {np.binary_repr(i, width=num_qubits): 0 for i in range(2 ** num_qubits)}

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

def load_qasm_2_0_file(path, parse_comments=False):
    
    circuit = []
    num_qubits = 1
    circuit_str = []
    circuit_start = -1
    comments = []
    
    with open(path, 'r') as f:
        for line_num, line in enumerate(f):
            if line.startswith('qreg'):
                num_qubits = list(map(int, findall(r'\d+', line)))[0]
                circuit_start = line_num
                continue
            if line.startswith('creg'):
                circuit_start = line_num
                continue
            if circuit_start == -1:
                continue
        
            line, comment = line.split(';')
            circuit_str.append(line)
            
            if parse_comments:
                comments.append(comment.replace('\n', ''))
    
    for gate in circuit_str:
        if gate.startswith('u3'):
            gate = gate.replace('u3', '')
            thetas, qubit = gate.split(' ')
            qubit = int(qubit.replace(f'q[', '').replace(']', '').strip())
            thetas = thetas.replace(f'(', '').replace(')', '').strip().split(',')
            thetas_new = []
            for theta in thetas:
                thetas_new.append(eval(theta))
            circuit.append(['U', qubit] + thetas_new)
            continue
        if gate.startswith('u1'):
            gate = gate.replace('u1', '')
            thetas, qubit = gate.split(' ')
            qubit = int(qubit.replace(f'q[', '').replace(']', '').strip())
            thetas = thetas.replace(f'(', '').replace(')', '').strip().split(',')
            thetas_new = []
            for theta in thetas:
                thetas_new.append(eval(theta))
            circuit.append(['Rz', qubit] + thetas_new)
            continue
        if gate.startswith('u'):
            gate = gate.replace('u', '')
            thetas, qubit = gate.split(' ')
            qubit = int(qubit.replace(f'q[', '').replace(']', '').strip())
            thetas = thetas.replace(f'(', '').replace(')', '').strip().split(',')
            thetas_new = []
            for theta in thetas:
                thetas_new.append(eval(theta))
            circuit.append(['U', qubit] + thetas_new)
            continue
        if gate.startswith('rx'):
            gate = gate.replace('rx', '')
            theta, qubit = gate.split(' ')
            qubit = int(qubit.replace(f'q[', '').replace(']', '').strip())
            theta = eval(theta.replace(f'(', '').replace(')', '').strip())
            circuit.append(['Rx', qubit, theta])
            continue
        if gate.startswith('ry'):
            gate = gate.replace('ry', '')
            theta, qubit = gate.split(' ')
            qubit = int(qubit.replace(f'q[', '').replace(']', '').strip())
            theta = eval(theta.replace(f'(', '').replace(')', '').strip())
            circuit.append(['Ry', qubit, theta])
            continue
        if gate.startswith('rz'):
            gate = gate.replace('rz', '')
            theta, qubit = gate.split(' ')
            qubit = int(qubit.replace(f'q[', '').replace(']', '').strip())
            theta = eval(theta.replace(f'(', '').replace(')', '').strip())
            circuit.append(['Rz', qubit, theta])
            continue
        if gate.startswith('cx'):
            gate = gate.replace('cx ', '')
            qubits = []
            gates = gate.split(f',')
            for l in gates:
                qubits.append(int(l.replace(f'q[', '').replace(']', '')))
            circuit.append(['CNOT'] + qubits)
            continue
        if gate.startswith('h'):
            gate = gate.replace('h ', '')
            qubit = int(gate.replace(f'q[', '').replace(']', '').strip())
            circuit.append(['h', qubit])
            continue
    
    if parse_comments:
        
        for gate, comment in zip(circuit, comments):
            if not comment:
                gate.append(None)
                continue
            comment = comment.replace(' // ', '')
            
            if not (comment.startswith('Encoding') or comment.startswith('Variational')):
                gate.append(comment)
                continue
            
            gate[-1] = None
            gate.append(comment)
     
    return circuit, num_qubits
    
def load_qasm_3_0_file(path):
    
    circuit = []
    circuit_str = []
    classical_bits = []
    circuit_name = None
    num_qubits = 1
    with open(path, 'r') as f:
        for x in f:
            if x.startswith('qubit'):
                num_qubits = list(map(int, findall(r'\d+', x)))[0]
                circuit_name = x.replace(f'qubit[{num_qubits}]', '').replace(';', '').strip()
            if x.startswith('bit'):
                classical_bits.append(x.replace('bit ', '').replace(';\n', ''))
            if not x.startswith('//'):
                line = sub(r'// .*', '', x)
                circuit_str.append(line.replace('\n', '').replace(';', ''))
    
    for i, line in enumerate(circuit_str):
        if 'qubit' in line:
            circuit_str = circuit_str[i + 1:]
            break
    
    if classical_bits:
        for i, line in enumerate(circuit_str):
            if line == f'bit {classical_bits[-1]}':
                circuit_str = circuit_str[i + 1:]
    
    for line in circuit_str:
        if 'measure' in line:
            line = line.replace(' = measure ', '')
            for bit in classical_bits:
                if line.startswith(bit):
                    line = line.replace(bit, '')
            qubit = int(line.replace(f'{circuit_name}[', '').replace(']', ''))
            circuit.append(['measure', qubit])
            continue
        if line.startswith('U'):
            line = line.replace('U(', '')
            lines = line.split(')')
            qubit = int(lines[1].replace(f' {circuit_name}[', '').replace(']', ''))
            angles = lines[0].split(',')
            for i, angle in enumerate(angles):
                angles[i] = float(angle)
            circuit.append(['U', qubit] + angles)
            continue
        if line.startswith('cx'):
            line = line.replace('cx ', '')
            qubits = []
            lines = line.split(f',')
            for l in lines:
                qubits.append(int(l.replace(f'{circuit_name}[', '').replace(']', '')))
            circuit.append(['cx'] + qubits)
            continue
        if line.startswith('x'):
            line = line.replace('x ', '')
            qubit = int(line.replace(f'{circuit_name}[', '').replace(']', '').strip())
            circuit.append(['x', qubit])
            continue
        if line.startswith('y'):
            line = line.replace('y ', '')
            qubit = int(line.replace(f'{circuit_name}[', '').replace(']', '').strip())
            circuit.append(['y', qubit])
            continue
        if line.startswith('z'):
            line = line.replace('z ', '')
            qubit = int(line.replace(f'{circuit_name}[', '').replace(']', '').strip())
            circuit.append(['z', qubit])
            continue
        if line.startswith('h'):
            line = line.replace('h ', '')
            qubit = int(line.replace(f'{circuit_name}[', '').replace(']', '').strip())
            circuit.append(['h', qubit])
            continue
        if line.startswith('rx'):
            line = line.replace('rx ', '')
    
    classical_bits = {b: None for b in classical_bits}
    
    return circuit, num_qubits, classical_bits

def apply_circuit(circuit, qubits, classical_bits=None):
    
    for circuit_p in circuit:
        
        if circuit_p[0] == 'U':
            qubits[circuit_p[1]].general_rotation(circuit_p[2], circuit_p[3], circuit_p[4])
            continue
        if circuit_p[0] == 'CNOT':
            qubits[circuit_p[1]].CNOT(qubits[circuit_p[2]])
            continue
        if circuit_p[0] == 'Rx':
            qubits[circuit_p[1]].Rx(circuit_p[2])
            continue
        if circuit_p[0] == 'Ry':
            qubits[circuit_p[1]].Ry(circuit_p[2])
            continue
        if circuit_p[0] == 'Rz':
            qubits[circuit_p[1]].Rz(circuit_p[2])
            continue
        if circuit_p[0] == 'measure':
            classical_bits[circuit_p[2]] = qubits[circuit_p[1]].measure()
            continue
        if circuit_p[0] == 'x':
            qubits[circuit_p[1]].X()
            continue
        if circuit_p[0] == 'y':
            qubits[circuit_p[1]].Y()
            continue
        if circuit_p[0] == 'z':
            qubits[circuit_p[1]].Z()
            continue
        if circuit_p[0] == 'h':
            qubits[circuit_p[1]].H()
            continue

async def dqc_apply_circuit(host, circuit, qubits):
    
    for circuit_p in circuit:
        
        if circuit_p[0] == 'U':
            await host.apply_gate('general_rotation', qubits[circuit_p[1]], circuit_p[2], circuit_p[3], circuit_p[4])
            continue
        if circuit_p[0] == 'CNOT':
            await host.apply_gate('CNOT', qubits[circuit_p[1]], qubits[circuit_p[2]])
            continue
        if circuit_p[0] == 'Rx':
            await host.apply_gate('Rx', qubits[circuit_p[1]], circuit_p[2])
            continue
        if circuit_p[0] == 'Ry':
            await host.apply_gate('Ry', qubits[circuit_p[1]], circuit_p[2])
            continue
        if circuit_p[0] == 'Rz':
            await host.apply_gate('Rz', qubits[circuit_p[1]], circuit_p[2])
            continue
        if circuit_p[0] == 'measure':
            host.classical_bits[circuit_p[2]] = await host.apply_gate('measure', qubits[circuit_p[1]])
            continue
        if circuit_p[0] == 'x':
            await host.apply_gate('X', qubits[circuit_p[1]])
            continue
        if circuit_p[0] == 'y':
            await host.apply_gate('Y', qubits[circuit_p[1]])
            continue
        if circuit_p[0] == 'z':
            await host.apply_gate('Z', qubits[circuit_p[1]])
            continue
        if circuit_p[0] == 'h':
            await host.apply_gate('H', qubits[circuit_p[1]])
            continue