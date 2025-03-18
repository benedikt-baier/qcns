from re import sub, findall
import itertools as it
import json as js

import numpy as np
from numpy import pi
import networkx as nx
import scipy.sparse as sp

from typing import List, Dict

from qcns.python.components.qubit import combine_gates

class Qubit:
    pass

__all__ = ['get_measure_dict', 'load_qasm_2_0_file', 'save_config', 'load_config', 'load_qasm_3_0_file', 'apply_circuit', 'dqc_apply_circuit']
  
def get_measure_dict(num_qubits: int) -> Dict[str, int]:
    
    """
    Creates a measurement dictionary with the binary states as keys and 0 as values
    
    Args:
        num_qubits (int): number of qubits to measure
        
    Returns:
        meas_stats (dict): dictonary with str as keys and 0s as values
    """
    
    return {np.binary_repr(i, width=num_qubits): 0 for i in range(2 ** num_qubits)}

def load_qasm_2_0_file(path: str, parse_comments: bool=False):
    
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
    
def load_qasm_3_0_file(path: str):
    
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

def save_config(path: str, graph: nx.Graph, num_routers: int, num_clients: int=0, routing_table: Dict[int, Dict[int, int]]=None, traffic_type: str=None, traffic_matrix: np.array=None, connection_type: str='l3') -> None:
    
    output = {}
    output['graph'] = {'nodes': dict(graph.nodes(data=True)), 'edges': {str((u, v)): data for u, v, data in graph.edges(data=True)}}
    output['num_routers'] = num_routers
    output['num_clients'] = num_clients
    output['routing_table'] = routing_table
    output['traffic_type'] = traffic_type
    output['traffic_matrix'] = traffic_matrix.tolist() if isinstance(traffic_matrix, np.ndarray) else traffic_matrix
    output['connection_type'] = connection_type
    
    with open(path, 'w') as f:
        js.dump(output, f)

def load_config(path: str) -> None:
    
    with open(path, 'r') as f:
        data = js.load(f)
        
    graph = nx.Graph()
    
    graph.add_nodes_from([(eval(k), v) for k, v in data['graph']['nodes'].items()])
    graph.add_edges_from([(*eval(k), v) for k, v in data['graph']['edges'].items()])
    
    return graph, data['num_routers'], data['num_clients'], data['routing_table'], data['traffic_type'], np.array(data['traffic_matrix']), data['connection_type']

def apply_circuit(circuit, qubits, classical_bits=None):
    
    gates = []
    
    for circuit_p in circuit:
        
        if circuit_p[0] == 'U':
            gates.append(qubits[circuit_p[1]].general_rotation(circuit_p[2], circuit_p[3], circuit_p[4], apply=False))
            continue
        if circuit_p[0] == 'CNOT':
            gates.append(qubits[circuit_p[1]].CNOT(qubits[circuit_p[2]], apply=False))
            continue
        if circuit_p[0] == 'Rx':
            gates.append(qubits[circuit_p[1]].Rx(circuit_p[2], apply=False))
            continue
        if circuit_p[0] == 'Ry':
            gates.append(qubits[circuit_p[1]].Ry(circuit_p[2], apply=False))
            continue
        if circuit_p[0] == 'Rz':
            gates.append(qubits[circuit_p[1]].Rz(circuit_p[2], apply=False))
            continue
        if circuit_p[0] == 'measure':
            classical_bits[circuit_p[2]] = qubits[circuit_p[1]].measure()
            continue
        if circuit_p[0] == 'x':
            gates.append(qubits[circuit_p[1]].X(apply=False))
            continue
        if circuit_p[0] == 'y':
            gates.append(qubits[circuit_p[1]].Y(apply=False))
            continue
        if circuit_p[0] == 'z':
            gates.append(qubits[circuit_p[1]].Z(apply=False))
            continue
        if circuit_p[0] == 'h':
            gates.append(qubits[circuit_p[1]].H(apply=False))
            continue
    
    if not gates:
        return
    
    gate = combine_gates(0, gates)
    
    next(iter(qubits.values())).custom_gate(gate)

async def dqc_apply_circuit(host, circuit, qubits):
    
    gates = []
    
    for circuit_p in circuit:
        
        if circuit_p[0] == 'U':
            gates.append(qubits[circuit_p[1]].general_rotation(circuit_p[2], circuit_p[3], circuit_p[4], apply=False))
            continue
        if circuit_p[0] == 'CNOT':
            gates.append(qubits[circuit_p[1]].CNOT(qubits[circuit_p[2]], apply=False))
            continue
        if circuit_p[0] == 'Rx':
            gates.append(qubits[circuit_p[1]].Rx(circuit_p[2], apply=False))
            continue
        if circuit_p[0] == 'Ry':
            gates.append(qubits[circuit_p[1]].Ry(circuit_p[2], apply=False))
            continue
        if circuit_p[0] == 'Rz':
            gates.append(qubits[circuit_p[1]].Rz(circuit_p[2], apply=False))
            continue
        if circuit_p[0] == 'measure':
            host.classical_bits[circuit_p[2]] = await host.apply_gate('measure', qubits[circuit_p[1]])
            continue
        if circuit_p[0] == 'x':
            gates.append(qubits[circuit_p[1]].X(apply=False))
            continue
        if circuit_p[0] == 'y':
            gates.append(qubits[circuit_p[1]].Y(apply=False))
            continue
        if circuit_p[0] == 'z':
            gates.append(qubits[circuit_p[1]].Z(apply=False))
            continue
        if circuit_p[0] == 'h':
            gates.append(qubits[circuit_p[1]].H(apply=False))
            continue
    
    if not gates:
        return
    
    gate = combine_gates(0, gates)
    
    await host.apply_gate('custom_gate', next(iter(qubits.values())), gate)