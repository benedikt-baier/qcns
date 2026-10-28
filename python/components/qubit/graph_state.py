"""Compact measurement-based stabilizer resources and repeater error models.

Resource generators and Pauli frames replace the 4n-qubit Hilbert space.
Recovery follows Eqs. (3)-(5) and resources follow Eq. (6) in
https://arxiv.org/html/2502.11174v2 . Noise is independent Pauli noise;
this module does not simulate coherent errors or noisy preparation circuits.
"""
from collections import Counter, deque
from dataclasses import dataclass
from itertools import combinations
from numbers import Integral

import numpy as np

from .qec import StabilizerPauliCode, _masks, _commutes_bit

__all__ = ['MeasurementBasedCode', 'StabilizerResource', 'BellMeasurementRecord',
           'GraphStateRecovery', 'LogicalPauliSamples', 'MeasurementBasedRepeaterModel',
           'compose_pauli_channels', 'depolarizing_channel']


def _pauli_masks(string, length):
    if not isinstance(string, str) or len(string) != length or set(string) - set('IXYZ'):
        raise ValueError('Pauli strings must have code length and contain only I/X/Y/Z')
    return _masks(['IXYZ'.index(value) for value in string])


def _independent(rows):
    pivots = {}
    for row in rows:
        while row and row.bit_length() in pivots:
            row ^= pivots[row.bit_length()]
        if not row:
            return False
        pivots[row.bit_length()] = row
    return True


def _bits(values, length, name):
    array = np.asarray(values)
    if array.shape != (length,) or not np.all((array == 0) | (array == 1)):
        raise ValueError(f'{name} must contain exactly {length} binary values')
    return np.array(array, dtype=bool, copy=True)


def _channel(values):
    values = np.asarray(values, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 4 or not np.all(np.isfinite(values)):
        raise ValueError('Supply finite I/X/Y/Z probabilities')
    if np.any(values < 0) or not np.allclose(values.sum(axis=-1), 1, atol=1e-10, rtol=0):
        raise ValueError('Pauli probabilities must be nonnegative and normalized')
    return values


def depolarizing_channel(error_rate):
    """I/X/Y/Z probabilities; error_rate is the total nonidentity probability."""
    rate = float(error_rate)
    if not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError('Depolarizing error probability must be between zero and one')
    return np.array([1-rate, rate/3, rate/3, rate/3])


def compose_pauli_channels(first, second):
    """Compose independent Pauli errors by XOR convolution, ignoring phase."""
    a, b = np.broadcast_arrays(_channel(first), _channel(second))
    return np.stack([sum(a[..., value] * b[..., value ^ output] for value in range(4))
                     for output in range(4)], axis=-1)


class MeasurementBasedCode:
    """A validated [[n,k]] stabilizer code retaining all k logical frames."""
    MAX_AUTOMATIC_CHECKS = 12

    def __init__(self, name, stabilizers, logical_x, logical_z, *, decoder=None):
        if not isinstance(name, str) or not name:
            raise ValueError('Supply a nonempty code name')
        self.name = name
        self.logical_x = (logical_x,) if isinstance(logical_x, str) else tuple(logical_x)
        self.logical_z = (logical_z,) if isinstance(logical_z, str) else tuple(logical_z)
        if not self.logical_x or len(self.logical_x) != len(self.logical_z):
            raise ValueError('Supply paired logical X/Z operators for every output')
        self.length = len(self.logical_x[0])
        self.num_logical = len(self.logical_x)
        self.stabilizers = tuple(stabilizers)
        self._checks = tuple(_pauli_masks(row, self.length) for row in self.stabilizers)
        self._xs = tuple(_pauli_masks(row, self.length) for row in self.logical_x)
        self._zs = tuple(_pauli_masks(row, self.length) for row in self.logical_z)
        self._validate()
        if decoder is not None and not callable(decoder):
            raise ValueError('Decoder must be callable')
        if decoder is None and len(self._checks) > self.MAX_AUTOMATIC_CHECKS:
            raise ValueError('Large codes require a supplied syndrome decoder')
        self._decoder = decoder
        self._corrections = self._make_decoder() if decoder is None else None

    @classmethod
    def from_qec_code(cls, code):
        if not isinstance(code, StabilizerPauliCode):
            raise TypeError('Expected a StabilizerPauliCode')
        # Preserve the chosen decoder, including preset tie-breaking.
        def decode(bits):
            syndrome = sum(int(bit) << index for index, bit in enumerate(bits))
            x, z = code._correction(syndrome)
            return ''.join('IXZY'[((x >> i) & 1) | (((z >> i) & 1) << 1)]
                           for i in range(code.length))
        return cls(code.name, code.stabilizers, code.logical_x, code.logical_z, decoder=decode)

    def _validate(self):
        if self.length < 1 or len(self._checks) != self.length - self.num_logical:
            raise ValueError('A [[n,k]] code needs n-k independent stabilizer generators')
        if not _independent([x | (z << self.length) for x, z in self._checks]):
            raise ValueError('Stabilizer generators must be independent')
        if any(_commutes_bit(a, b) for a, b in combinations(self._checks, 2)):
            raise ValueError('Stabilizer generators must commute')
        if any(_commutes_bit(logical, check) for logical in self._xs + self._zs for check in self._checks):
            raise ValueError('Logical operators must commute with stabilizers')
        if any(_commutes_bit(a, b) for a, b in combinations(self._xs, 2)) or any(
                _commutes_bit(a, b) for a, b in combinations(self._zs, 2)):
            raise ValueError('Logical operators of the same kind must commute')
        if any(_commutes_bit(x, z) != int(i == j)
               for i, x in enumerate(self._xs) for j, z in enumerate(self._zs)):
            raise ValueError('Logical X/Z pairs must have canonical commutation relations')

    def syndrome(self, error):
        return sum(_commutes_bit(error, check) << index for index, check in enumerate(self._checks))

    def logical_frame(self, error):
        return (np.array([_commutes_bit(error, z) for z in self._zs], dtype=bool),
                np.array([_commutes_bit(error, x) for x in self._xs], dtype=bool))

    def _make_decoder(self):
        corrections, queue = {0: (0, 0)}, deque([0])
        errors = [(1 << i, 0) for i in range(self.length)]
        errors += [(1 << i, 1 << i) for i in range(self.length)]
        errors += [(0, 1 << i) for i in range(self.length)]
        edges = [(self.syndrome(error), error) for error in errors]
        while queue:
            current = queue.popleft()
            x, z = corrections[current]
            for shift, (ex, ez) in edges:
                neighbor = current ^ shift
                if neighbor not in corrections:
                    corrections[neighbor] = (x ^ ex, z ^ ez)
                    queue.append(neighbor)
        return corrections

    def correction(self, syndrome):
        if self._decoder is None:
            return self._corrections[syndrome]
        bits = tuple((syndrome >> i) & 1 for i in range(len(self._checks)))
        correction = _pauli_masks(self._decoder(bits), self.length)
        if self.syndrome(correction) != syndrome:
            raise ValueError('Decoder correction does not match the supplied syndrome')
        return correction


@dataclass(frozen=True)
class StabilizerResource:
    """Pure stabilizer resource: packed (X-mask, Z-mask) positive generators.

    This is a stabilizer representation of the resource, rather than a graph
    adjacency matrix. Local Clifford conversion is unnecessary for decoding.
    """
    code: MeasurementBasedCode
    num_qubits: int
    generators: tuple
    kind: str

    @classmethod
    def repeater(cls, code):
        n = code.length
        rows = list(code._checks) + [(x << n, z << n) for x, z in code._checks]
        rows += [(x | (x << n), z | (z << n)) for x, z in code._xs + code._zs]
        return cls(code, 2*n, tuple(rows), 'logical_logical')

    @classmethod
    def distillation(cls, code):
        n = code.length
        rows = list(code._checks)
        rows += [(x | (1 << (n+j)), z) for j, (x, z) in enumerate(code._xs)]
        rows += [(x, z | (1 << (n+j))) for j, (x, z) in enumerate(code._zs)]
        return cls(code, n+code.num_logical, tuple(rows), 'logical_physical')


@dataclass(frozen=True)
class BellMeasurementRecord:
    """s contains XX parity bits; t contains ZZ parity bits."""
    s: object
    t: object


@dataclass(frozen=True)
class GraphStateRecovery:
    syndrome: np.ndarray
    bit_flips: np.ndarray
    phase_flips: np.ndarray
    correction: tuple


@dataclass(frozen=True)
class LogicalPauliSamples:
    """Joint trajectories preserve correlations between logical outputs."""
    bit_flips: np.ndarray
    phase_flips: np.ndarray

    @property
    def joint_fidelity(self):
        return float(np.mean(~np.any(self.bit_flips | self.phase_flips, axis=1)))

    def joint_counts(self):
        return Counter(tuple(np.concatenate((bits, phases)).astype(int))
                       for bits, phases in zip(self.bit_flips, self.phase_flips))

    def bell_marginals(self):
        labels = np.array([0, 1, 3, 2])[self.bit_flips.astype(int) + 2*self.phase_flips]
        return np.array([np.bincount(column, minlength=4) / len(labels) for column in labels.T])

    def compose(self, other):
        if self.bit_flips.shape != other.bit_flips.shape:
            raise ValueError('Joint frame samples must have matching dimensions')
        return LogicalPauliSamples(self.bit_flips ^ other.bit_flips, self.phase_flips ^ other.phase_flips)


class MeasurementBasedRepeaterModel:
    """Classical recovery and independent-Pauli segment sampling; no matrices."""

    def __init__(self, code):
        self._exact_code = code if isinstance(code, StabilizerPauliCode) else None
        if isinstance(code, StabilizerPauliCode):
            code = MeasurementBasedCode.from_qec_code(code)
        if not isinstance(code, MeasurementBasedCode):
            raise TypeError('Supply a validated stabilizer code definition')
        self.code = code

    def exact_segment_channel(self, bell_probabilities: np.ndarray) -> np.ndarray:
        """Exact decoded interface channel, averaged over communicated records.

        Restricted to independent Bell-diagonal pairs and ideal graph resources
        and measurements. Uses the same syndrome decoder as simulate_segment;
        logical errors remain in the output ensemble and are not heralds.
        Allocation is O(2**(n-k) * 4), bounded by the QEC backend's exact limit.
        """
        if self._exact_code is None or self._exact_code.method != 'exact':
            raise NotImplementedError('Exact graph interfaces require an exact StabilizerPauliCode')
        probabilities = self.effective_channel(bell_probabilities=bell_probabilities)
        return self._exact_code.logical_channel(probabilities)

    def prepare_resource(self):
        return StabilizerResource.repeater(self.code)

    def recover(self, first, second):
        """Recover from both endpoints' Bell records, including Pauli-Y signs."""
        n = self.code.length
        s = _bits(first.s, n, 'XX parities') ^ _bits(second.s, n, 'XX parities')
        t = _bits(first.t, n, 'ZZ parities') ^ _bits(second.t, n, 'ZZ parities')
        s_mask = sum(int(bit) << i for i, bit in enumerate(s))
        t_mask = sum(int(bit) << i for i, bit in enumerate(t))
        measured = (t_mask, s_mask)
        syndrome_bits = self._measured_parities(self.code._checks, measured)
        syndrome = sum(int(bit) << i for i, bit in enumerate(syndrome_bits))
        correction = self.code.correction(syndrome)
        bits, phases = self.code.logical_frame(correction)
        bits ^= self._measured_parities(self.code._zs, measured)
        phases ^= self._measured_parities(self.code._xs, measured)
        return GraphStateRecovery(syndrome_bits, bits, phases, correction)

    @staticmethod
    def _measured_parities(operators, measured):
        return np.array([_commutes_bit(operator, measured) ^ ((operator[0] & operator[1]).bit_count() & 1)
                         for operator in operators], dtype=bool)

    def effective_channel(self, *, bell_probabilities=None, bell_error_rate=0.,
                          resource_error_rate=0., measurement_error_rate=0.):
        """Both-end noise; supplied Bell probabilities already include both halves.

        bell_error_rate instead denotes per-half depolarizing input noise.
        Resource and measurement rates denote per-end contributions.
        """
        bell = depolarizing_channel(bell_error_rate)
        if bell_probabilities is not None:
            if bell_error_rate != 0:
                raise ValueError('Supply Bell probabilities or a per-half rate, not both')
            bell = _channel(bell_probabilities)
        else:
            bell = compose_pauli_channels(bell, bell)
        resource = depolarizing_channel(resource_error_rate)
        measurement = depolarizing_channel(measurement_error_rate)
        endpoint = compose_pauli_channels(resource, measurement)
        channel = compose_pauli_channels(bell, compose_pauli_channels(endpoint, endpoint))
        if channel.shape not in ((4,), (self.code.length, 4)):
            raise ValueError('Supply a common channel or one channel per physical input pair')
        return np.broadcast_to(channel, (self.code.length, 4)).copy()

    def simulate_segment(self, *, shots=10000, seed=None, rng=None, **noise):
        if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 1:
            raise ValueError('Shots must be a positive integer')
        if rng is not None and seed is not None:
            raise ValueError('Supply an RNG or seed, not both')
        rng = np.random.default_rng(seed) if rng is None else rng
        probabilities = self.effective_channel(**noise)
        cumulative = np.cumsum(probabilities, axis=1)
        bits = np.zeros((shots, self.code.num_logical), dtype=bool)
        phases = np.zeros_like(bits)
        for start in range(0, shots, 1024):
            size = min(1024, shots-start)
            draws = rng.random((size, self.code.length))
            errors = (draws[:, :, None] >= cumulative[None, :, :3]).sum(axis=2)
            for offset, error in enumerate(errors):
                masks = _masks(error.tolist())
                correction = self.code.correction(self.code.syndrome(masks))
                residual = (masks[0] ^ correction[0], masks[1] ^ correction[1])
                bits[start+offset], phases[start+offset] = self.code.logical_frame(residual)
        return LogicalPauliSamples(bits, phases)

    def simulate_chain(self, segments, *, shots=10000, seed=None, **noise):
        if isinstance(segments, bool) or not isinstance(segments, Integral) or segments < 1:
            raise ValueError('Segments must be a positive integer')
        rng = np.random.default_rng(seed)
        result = self.simulate_segment(shots=shots, rng=rng, **noise)
        for _ in range(segments-1):
            result = result.compose(self.simulate_segment(shots=shots, rng=rng, **noise))
        return result
