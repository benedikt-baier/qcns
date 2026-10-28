"""Array-level quantum kernels; NumPy owns every public density matrix.

CppBackend delegates supported local operations to QMath through a checked C ABI.
Dense custom/multi-control operators and tensor products remain exact NumPy
operations. Measurements and random draws remain in the Python simulator.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from . import qubit as _quantum

__all__ = ['PythonBackend', 'CppBackend', 'get_backend']


class PythonBackend:
    name = 'python'
    max_qubits: int | None = None
    native_operations: tuple[str, ...] = ()

    def validate_size(self, qubits: int) -> None:
        if self.max_qubits is not None and qubits > self.max_qubits:
            raise ValueError(f'{self.name} backend supports at most {self.max_qubits} qubits per joint state')

    def single(self, state: np.ndarray, gate: np.ndarray, target: int, *, key: str = '') -> np.ndarray:
        full = _quantum.get_single_operator(key, gate, target, state.shape[0].bit_length() - 1)
        return _quantum.dot(state, full)

    def controlled(self, state: np.ndarray, gate: np.ndarray, control: int, target: int,
                   *, key: str = '') -> np.ndarray:
        full = _quantum.get_double_operator(key, gate, control, target, state.shape[0].bit_length() - 1)
        return _quantum.dot(state, full)

    def cnot(self, state: np.ndarray, control: int, target: int, *, key: str = '') -> np.ndarray:
        return self.controlled(state, _quantum.full_gates['X'], control, target, key=key)

    def partial_trace(self, state: np.ndarray, target: int) -> np.ndarray:
        n = state.shape[0].bit_length() - 1
        dim = state.shape[0] // 2
        return np.trace(state.reshape([2] * (2*n)), axis1=target, axis2=target+n).reshape(dim, dim)

    def dephasing(self, state: np.ndarray, target: int, probability: float) -> np.ndarray:
        n = state.shape[0].bit_length() - 1
        return (1-probability)*state + probability*self.single(state, _quantum.full_gates['Z'], target, key=f's_z_{n}_{target}')

    def amplitude_damping(self, state: np.ndarray, target: int, probability: float) -> np.ndarray:
        first = np.diag([1., np.sqrt(1-probability)])
        second = np.array([[0., np.sqrt(probability)], [0., 0.]])
        return self.single(state, first, target) + self.single(state, second, target)

    def depolarizing(self, state: np.ndarray, target: int, probability: float) -> np.ndarray:
        n = state.shape[0].bit_length() - 1
        errors = [self.single(state, _quantum.full_gates[name], target, key=f's_{name.lower()}_{n}_{target}')
                  for name in ('X', 'Y', 'Z')]
        return (1-probability)*state + probability/3*sum(errors)


class CppBackend(PythonBackend):
    """QMath kernels, 1..15 qubits, native complex128 row-major buffers.

    Inputs are never mutated. Strided/real inputs are converted explicitly;
    QMath uses aligned owned storage internally and copies its output into NumPy.
    threads bounds OpenMP work for each call (default 1). No global RNG or backend
    selection is changed. An unavailable native library raises an actionable error.
    """
    name = 'cpp'
    max_qubits = 15
    native_operations = ('single', 'controlled', 'cnot', 'dephasing',
                         'amplitude_damping', 'depolarizing', 'partial_trace')

    def __init__(self, library: str | os.PathLike[str] | None = None, *, threads: int = 1) -> None:
        if isinstance(threads, bool) or not isinstance(threads, int) or not 1 <= threads <= 1024:
            raise ValueError('backend threads must be an integer in [1,1024]')
        self.threads = threads
        suffix = '.dll' if sys.platform == 'win32' else '.dylib' if sys.platform == 'darwin' else '.so'
        default = Path(__file__).resolve().parents[3] / 'native' / 'qmath' / 'build' / ('qcns_qmath' + suffix)
        self.library_path = Path(library or os.environ.get('QCNS_QMATH_LIBRARY', '') or default).resolve()
        if not self.library_path.is_file():
            raise RuntimeError(f'QMath library not found: {self.library_path}. Run python scripts/build_qmath_backend.py or supply CppBackend(library=...).')
        self._dll_directories: list[Any] = []
        if sys.platform == 'win32':
            self._dll_directories = [os.add_dll_directory(str(self.library_path.parent))]
        try:
            self._library = ctypes.CDLL(str(self.library_path))
        except OSError as error:
            raise RuntimeError(f'Cannot load {self.library_path}; ensure its OpenMP runtime is available: {error}') from error
        self._library.qcns_qmath_abi.restype = ctypes.c_int
        if self._library.qcns_qmath_abi() != 1:
            raise RuntimeError('Incompatible QMath bridge ABI; rebuild the native library')
        self._library.qcns_qmath_error.restype = ctypes.c_char_p
        self._transform = self._library.qcns_qmath_transform
        self._transform.restype = ctypes.c_int
        self._transform.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t,
                                    ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                                    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_void_p,
                                    ctypes.c_size_t, ctypes.c_double, ctypes.c_int]

    def _apply(self, operation: int, state: np.ndarray, target: int, *, control: int = 0,
               gate: np.ndarray | None = None, probability: float = 0.) -> np.ndarray:
        array = np.asarray(state)
        if array.dtype.kind not in 'biufc' or array.ndim != 2 or array.shape[0] != array.shape[1]:
            raise ValueError('State must be a square numeric density matrix')
        dim = array.shape[0]
        if dim < 2 or dim & (dim-1):
            raise ValueError('Density matrix dimension must be 2**n with n >= 1')
        n = dim.bit_length()-1
        self.validate_size(n)
        for name, index in (('target', target), ('control', control)):
            if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < n:
                raise ValueError(f'{name} is outside this quantum state')
        if operation in (1, 2) and control == target:
            raise ValueError('Control and target must differ')
        array = np.ascontiguousarray(array, dtype=np.complex128)
        local = None if gate is None else np.ascontiguousarray(gate, dtype=np.complex128)
        if local is not None and (local.shape != (2, 2) or not np.all(np.isfinite(local))):
            raise ValueError('Local gate must be a finite 2x2 numeric matrix')
        out_dim = dim//2 if operation == 6 else dim
        output = np.empty((out_dim, out_dim), dtype=np.complex128)
        status = self._transform(operation, array.ctypes.data, array.nbytes,
                                 output.ctypes.data, output.nbytes, dim, int(target), int(control),
                                 None if local is None else local.ctypes.data,
                                 0 if local is None else local.nbytes, probability, self.threads)
        if status:
            raise ValueError(self._library.qcns_qmath_error().decode('utf-8'))
        return output

    def single(self, state: np.ndarray, gate: np.ndarray, target: int, *, key: str = '') -> np.ndarray:
        return self._apply(0, state, target, gate=gate)

    def controlled(self, state: np.ndarray, gate: np.ndarray, control: int, target: int,
                   *, key: str = '') -> np.ndarray:
        return self._apply(1, state, target, control=control, gate=gate)

    def cnot(self, state: np.ndarray, control: int, target: int, *, key: str = '') -> np.ndarray:
        return self._apply(2, state, target, control=control)

    def dephasing(self, state: np.ndarray, target: int, probability: float) -> np.ndarray:
        return self._apply(3, state, target, probability=probability)

    def amplitude_damping(self, state: np.ndarray, target: int, probability: float) -> np.ndarray:
        return self._apply(4, state, target, probability=probability)

    def depolarizing(self, state: np.ndarray, target: int, probability: float) -> np.ndarray:
        return self._apply(5, state, target, probability=probability)

    def partial_trace(self, state: np.ndarray, target: int) -> np.ndarray:
        return self._apply(6, state, target)


def get_backend(backend: str | PythonBackend = 'python') -> PythonBackend:
    """Resolve per-simulation/per-state configuration without modifying global state."""
    if isinstance(backend, PythonBackend):
        return backend
    if backend == 'python':
        return PythonBackend()
    if backend == 'cpp':
        return CppBackend()
    raise ValueError('backend must be python, cpp, or a PythonBackend implementation')
