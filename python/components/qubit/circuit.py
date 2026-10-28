"""Small flat circuits using QCNS qubits and the existing node event scheduler.

Offline gates are ideal. Scheduled execution uses the supplied node's gate model,
durations and RNG. Measurements retain qubits and return their latest outcomes.
QASM support is intentionally a primitive-gate subset, not a full language parser.
"""
from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import math
import operator
import os
from pathlib import Path
import re
from typing import Any, Protocol

import numpy as np

from .qubit import Qubit, combine_state

__all__ = ['Circuit', 'CircuitExecutor', 'apply_circuit', 'dqc_apply_circuit',
           'execute_circuit', 'update_circuit', 'load_qasm_2_0_file', 'load_qasm_3_0_file']

PathLike = str | os.PathLike[str]
CircuitRows = Sequence[Sequence[Any]]
MeasurementResults = dict[str | int, int]


class CircuitExecutor(Protocol):
    """Gate interface implemented by Node; avoids a dependency on network internals."""

    def apply_gate(self, gate: str, *args: Any, remove: bool = True) -> int | None:
        ...

    async def execute_gate(self, gate: str, *args: Any, remove: bool = True) -> int | None:
        ...

@dataclass(frozen=True)
class Circuit:
    """Reusable gate rows, independent of qubit instances and simulation state.

    Rows have the same form as the QASM loaders: ('H', 0), ('CNOT', 0, 1),
    ('Rx', 0, angle), or ('measure', 0, 'result'). Row containers are copied
    to tuples. Execution validates all rows before applying the first gate.
    """

    operations: CircuitRows = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, 'operations', tuple(tuple(row) for row in self.operations))

    def apply(self, qubits: Mapping[int, Qubit], *,
              rng: np.random.Generator | None = None) -> MeasurementResults | None:
        """Apply ideal gates offline; pass a generator for repeatable measurements."""
        return apply_circuit(self.operations, qubits, rng=rng)

    def compile(self, qubits: Mapping[int, Qubit], *,
                max_matrix_qubits: int = 10) -> np.ndarray | None:
        """Return a dense unitary without applying it; measurements are unsupported."""
        return apply_circuit(self.operations, qubits, apply=False,
                             max_matrix_qubits=max_matrix_qubits)

    def apply_on_node(self, node: CircuitExecutor,
                      qubits: Mapping[int, Qubit]) -> MeasurementResults | None:
        """Use synchronous node gates for offline calculations."""
        return dqc_apply_circuit(node, self.operations, qubits)

    async def execute(self, node: CircuitExecutor,
                      qubits: Mapping[int, Qubit]) -> MeasurementResults | None:
        """Yield through the node scheduler for every gate duration."""
        return await execute_circuit(node, self.operations, qubits)

    def with_parameters(self, parameters: Sequence[float]) -> Circuit:
        """Return an independent circuit replacing every angle slot in order."""
        return Circuit(update_circuit(self.operations, parameters))

def _count(value: int, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f'{name} must be a non-negative integer.')

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_GATE = {'u': ('general_rotation', 1, 3), 'u3': ('general_rotation', 1, 3),
         'u1': ('PHASE', 1, 1), 'p': ('PHASE', 1, 1), 'rx': ('Rx', 1, 1),
         'ry': ('Ry', 1, 1), 'rz': ('Rz', 1, 1), 'cx': ('CNOT', 2, 0),
         'cnot': ('CNOT', 2, 0), 'x': ('X', 1, 0), 'y': ('Y', 1, 0),
         'z': ('Z', 1, 0), 'h': ('H', 1, 0), 'measure': ('measure', 1, 0),
         'phase': ('PHASE', 1, 1), 'general_rotation': ('general_rotation', 1, 3)}

def _angle_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id == 'pi':
        return math.pi
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _angle_node(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_angle_node(node.left), _angle_node(node.right))
    raise ValueError('QASM angles support numeric literals, pi, parentheses and + - * / only.')

def _angle(expression: str) -> float:
    result = _angle_node(ast.parse(expression.strip(), mode='eval').body)
    if not math.isfinite(result):
        raise ValueError('QASM angles must be finite.')
    return result

def _operand(text: str, registers: dict[str, list[int]]) -> int:
    match = re.fullmatch(r'(\w+)(?:\[\s*(\d+)\s*\])?', text.strip())
    if not match or match[1] not in registers:
        raise ValueError(f'Unknown QASM qubit: {text}')
    indices = registers[match[1]]
    if match[2] is None and len(indices) != 1:
        raise ValueError('Register broadcasting is unsupported; index each qubit explicitly.')
    offset = int(match[2] or 0)
    if offset >= len(indices):
        raise ValueError(f'QASM qubit index out of range: {text}')
    return indices[offset]

def _qasm_gate(statement: str, version: int, registers: dict[str, list[int]], bits: dict[str, None]) -> list[Any]:
    measurement = re.fullmatch(r'measure\s+(.+?)\s*->\s*(.+)', statement) if version == 2 else re.fullmatch(r'(.+?)\s*=\s*measure\s+(.+)', statement)
    if measurement:
        source, target = measurement.groups() if version == 2 else (measurement[2], measurement[1])
        target = re.sub(r'\s+', '', target)
        if target not in bits:
            raise ValueError(f'Unknown classical bit: {target}')
        return ['measure', _operand(source, registers), target]
    match = re.fullmatch(r'(\w+)(?:\((.*)\))?\s+(.+)', statement)
    if not match or match[1].lower() not in _GATE or match[1].lower() == 'measure':
        raise ValueError(f'Unsupported QASM statement: {statement}')
    method, arity, angles = _GATE[match[1].lower()]
    args = match[3].split(',')
    params = [] if match[2] is None else [_angle(v) for v in match[2].split(',')]
    if len(args) != arity or len(params) != angles:
        raise ValueError(f'Wrong operand/angle count: {statement}')
    return [('U' if method == 'general_rotation' else method),
            *(_operand(arg, registers) for arg in args), *params]

def _statements(path: PathLike) -> Iterator[tuple[str, str | None]]:
    pending = ''
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        code, _, comment = line.partition('//')
        pending += ' ' + code.strip()
        statements = pending.split(';')
        pending = statements.pop()
        for index, statement in enumerate(statements):
            if statement.strip():
                yield statement.strip(), (comment.strip() or None) if index == len(statements) - 1 else None
    if pending.strip():
        raise ValueError(f'Unterminated QASM statement: {pending.strip()}')

def _load_qasm(path: PathLike, version: int, comments: bool = False) -> tuple[list[list[Any]], int, dict[str, None]]:
    """Flat primitive circuits only; reject control flow, custom gates and broadcasting."""
    circuit: list[list[Any]] = []
    registers: dict[str, list[int]] = {}
    bits: dict[str, None] = {}
    names: set[str] = set()
    count = 0
    for statement, comment in _statements(path):
        if statement == f'OPENQASM {version}.0' or (version == 3 and statement == 'OPENQASM 3'):
            continue
        if statement in ('include "qelib1.inc"', 'include "stdgates.inc"'):
            continue
        declaration = re.fullmatch(r'(qreg|creg)\s+(\w+)\[(\d+)\]', statement) if version == 2 else re.fullmatch(r'(qubit|bit)(?:\[(\d+)\])?\s+(\w+)', statement)
        if declaration:
            kind, name, size = declaration.groups() if version == 2 else (declaration[1], declaration[3], declaration[2] or '1')
            if name in names:
                raise ValueError(f'Duplicate QASM register: {name}')
            names.add(name)
            size = int(size)
            if size <= 0:
                raise ValueError('QASM registers must have positive size.')
            if kind in ('qreg', 'qubit'):
                registers[name] = list(range(count, count + size))
                count += size
            elif version == 3 and declaration[2] is None:
                bits[name] = None
            else:
                bits.update({f'{name}[{i}]': None for i in range(size)})
            continue
        row = _qasm_gate(statement, version, registers, bits)
        if comments:
            row.append(comment)
        circuit.append(row)
    if not registers:
        raise ValueError('QASM file declares no qubits.')
    return circuit, count, bits

def load_qasm_2_0_file(path: PathLike, parse_comments: bool = False) -> tuple[list[list[Any]], int]:
    """Load flat QASM 2 primitive gates; comments never replace angle values."""
    circuit, count, _ = _load_qasm(path, 2, parse_comments)
    return circuit, count

def load_qasm_3_0_file(path: PathLike) -> tuple[list[list[Any]], int, dict[str, None]]:
    """Load the flat QASM 3 subset. Measurements retain their classical destination."""
    return _load_qasm(path, 3)

def _operations(circuit: CircuitRows, qubits: Mapping[int, Qubit]) -> list[tuple[str, list[Any], str | int | None]]:
    operations = []
    for row in circuit:
        if not row or str(row[0]).lower() not in _GATE:
            raise ValueError(f'Unsupported circuit gate: {row!r}')
        method, arity, angles = _GATE[str(row[0]).lower()]
        length = 1 + arity + angles
        maximum = length + (2 if method == 'measure' else 1)
        if not length <= len(row) <= maximum:
            raise ValueError(f'Wrong gate operand count: {row!r}')
        ids = row[1:1 + arity]
        if any(i not in qubits for i in ids) or len(set(ids)) != arity:
            raise ValueError(f'Unknown or repeated qubit operand: {row!r}')
        params = [float(p) for p in row[1 + arity:length]]
        if not all(math.isfinite(p) for p in params):
            raise ValueError('Gate angles must be finite.')
        target = row[2] if method == 'measure' and len(row) > 2 else ids[0]
        if method == 'measure' and not isinstance(target, (str, int)):
            raise ValueError('Measurement destination must be a classical bit name or integer.')
        operations.append((method, [*(qubits[i] for i in ids), *params], target))
    return operations

def _compile(operations: list[tuple[str, list[Any], Any]], limit: int) -> np.ndarray | None:
    _count(limit, 'max_matrix_qubits')
    if not operations:
        return None
    if any(method == 'measure' for method, _, _ in operations):
        raise ValueError('Measurements cannot be compiled into a unitary.')
    first = operations[0][1][0]
    operands = [arg for _, args, _ in operations for arg in args if isinstance(arg, Qubit)]
    if any(q.num_qubits > limit for q in operands):
        raise ValueError('Dense unitary needs 16 * 4**num_qubits bytes plus temporaries; exceeds max_matrix_qubits.')
    if any(q.state is not first.state for q in operands):
        raise ValueError('Compilation requires qubits already belonging to one joint state.')
    combined = None
    for method, args, _ in operations:
        gate = getattr(args[0], method)(*args[1:], apply=False)
        combined = gate.copy() if combined is None else gate @ combined
    return combined

def apply_circuit(circuit: CircuitRows, qubits: Mapping[int, Qubit], apply: bool = True,
                  *, rng: np.random.Generator | None = None, max_matrix_qubits: int = 10) -> np.ndarray | dict | None:
    """Apply ideal gates offline, returning outcomes when measurements occur.

    apply=False returns a unitary without applying it; all operands must already
    share one state. Dense compilation is capped by max_matrix_qubits.
    """
    operations = _operations(circuit, qubits)
    if not apply:
        return _compile(operations, max_matrix_qubits)
    outcomes = {}
    rng = np.random.default_rng() if rng is None else rng
    for method, args, target in operations:
        if method == 'CNOT':
            combine_state(args[:2])
        if method == 'measure':
            outcomes[target] = args[0].measure(rng=rng)
        else:
            getattr(args[0], method)(*args[1:])
    return outcomes or None

def dqc_apply_circuit(node: CircuitExecutor, circuit: CircuitRows, qubits: Mapping[int, Qubit], apply: bool = True) -> np.ndarray | dict | None:
    """Synchronous offline node execution; use execute_circuit during simulation."""
    operations = _operations(circuit, qubits)
    if not apply:
        return _compile(operations, 10)
    outcomes = {}
    for method, args, target in operations:
        result = node.apply_gate(method, *args, remove=False)
        if method == 'measure':
            outcomes[target] = result
    return outcomes or None

async def execute_circuit(node: CircuitExecutor, circuit: CircuitRows, qubits: Mapping[int, Qubit]) -> dict | None:
    """Execute scheduled gates with node durations; measured qubits remain available."""
    outcomes = {}
    for method, args, target in _operations(circuit, qubits):
        result = await node.execute_gate(method, *args, remove=False)
        if method == 'measure':
            outcomes[target] = result
    return outcomes or None

def update_circuit(circuit: CircuitRows, parameter: Sequence[float]) -> list[list[Any]]:
    """Copy and replace every angle in gate order (U has three), preserving comments.

    Requires exactly one finite value per angle slot. Never changes operands.
    """
    slots = []
    result = [list(row) for row in circuit]
    for index, row in enumerate(result):
        if not row or str(row[0]).lower() not in _GATE:
            raise ValueError(f'Unsupported circuit gate: {row!r}')
        _, arity, angles = _GATE[str(row[0]).lower()]
        if len(row) < 1 + arity + angles:
            raise ValueError(f'Missing circuit operands: {row!r}')
        slots.extend((index, slot) for slot in range(1 + arity, 1 + arity + angles))
    values = [float(v) for v in parameter]
    if len(values) != len(slots) or not all(math.isfinite(v) for v in values):
        raise ValueError(f'Expected exactly {len(slots)} finite angle parameters.')
    for (index, slot), value in zip(slots, values):
        result[index][slot] = value
    return result
