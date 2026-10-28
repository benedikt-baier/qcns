"""Ideal stabilizer decoding for independent Bell-diagonal pairs.

Track (syndrome, logical X, logical Z) probabilities, rather than a block
Hilbert space. Small codes use exact syndrome/frame probabilities; larger
codes can sample Pauli trajectories with a supplied decoder. Definitions
accept any [[n,k]] stabilizer code with explicit logical operators. Five-qubit
and Steane presets use 64 and 256 probabilities. This is an ideal effective model;
coherent noise and noisy syndrome circuits are unsupported. The channel API
requires independent inputs; decode_errors also accepts caller-supplied
correlated Pauli trajectories without factoring their joint error patterns.

Packed binary syndromes use Gottesman's thesis section 3.4, equations
3.23-3.24: symplectic products determine commutation, and XOR composes Pauli
errors modulo phase. This is a compiled ideal channel, not a circuit simulator.
Sampling has polynomial bookkeeping plus the supplied decoder's cost; exact
ensemble propagation uses 2**(n+k) syndrome/frame states. Joint logical
channels use 4**k entries; streaming logical trajectories avoids that table.

References: https://arxiv.org/abs/quant-ph/9705052
            https://arxiv.org/abs/quant-ph/9604024
"""
from collections import deque, OrderedDict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from itertools import combinations, product
from numbers import Integral

import numpy as np
from qcns.python.components.qubit.qubit import Qubit

__all__ = ['StabilizerPauliCode', 'PauliChannelEstimate', 'get_qec_code', 'register_qec_code']

# Public probability order: I, X, Y, Z. Internally X/Z frame bits use 0,1,3,2.
_X = (0, 1, 1, 0)
_Z = (0, 0, 1, 1)
_X_BINARY = str.maketrans('IXYZ', '0110')
_Z_BINARY = str.maketrans('IXYZ', '0011')
_FRAME_TO_PAULI = np.array([0, 1, 3, 2])
_BELL = np.array([[1, 0, 0, 1], [0, 1, 1, 0],
                  [0, 1, -1, 0], [1, 0, 0, -1]], dtype=complex).T / np.sqrt(2)

_CODE_SPECS = {
    'five_qubit': ('XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ'),
    'steane': ('IIIXXXX', 'IXXIIXX', 'XIXIXIX',
               'IIIZZZZ', 'IZZIIZZ', 'ZIZIZIZ'),
}

def _masks(paulis: Sequence[int]) -> tuple[int, int]:
    
    return (sum(_X[value] << position for position, value in enumerate(paulis)),
            sum(_Z[value] << position for position, value in enumerate(paulis)))

def _commutes_bit(first: tuple[int, int], second: tuple[int, int]) -> int:
    return ((first[0] & second[1]) ^ (first[1] & second[0])).bit_count() & 1


@dataclass(frozen=True)
class PauliChannelEstimate:
    """Monte Carlo joint counts, index=sum(P_i * 4**i), P_i=I/X/Y/Z=0/1/2/3."""

    counts: np.ndarray

    @property
    def shots(self) -> int:
        return int(self.counts.sum())

    @property
    def probabilities(self) -> np.ndarray:
        return self.counts / self.shots

    @property
    def standard_errors(self) -> np.ndarray:
        """Marginal plug-in errors; zero observed errors do not prove zero risk."""
        probabilities = self.probabilities
        return np.sqrt(probabilities * (1 - probabilities) / self.shots)

class StabilizerPauliCode:
    """Stabilizer code with explicit logical operators and a syndrome decoder."""

    MAX_EXACT_CHECKS = 12

    def __init__(self, name: str, stabilizers: Sequence[str] | None = None,
                 logical_x: str | Sequence[str] | None = None,
                 logical_z: str | Sequence[str] | None = None, *,
                 distance: int | None = None,
                 decoder: Callable[[tuple[int, ...]], str] | None = None,
                 method: str = 'exact', shots: int = 10000, seed: int | None = None,
                 batch_size: int = 1024, decoder_cache_size: int = 4096,
                 max_working_bytes: int = 64 * 1024 * 1024) -> None:
        """Accept an arbitrary [[n,k]] stabilizer code.

        Supply one X/Z string per logical qubit (strings remain valid for k=1).
        distance is declared metadata, not a verified correction guarantee.

        Generators and logical operators are phase-free I/X/Y/Z strings.
        An optional decoder receives syndrome bits in generator order and
        returns a correction Pauli string. Without it, small codes use a
        minimum-weight decoder. Sampling large codes requires a decoder.
        Custom decoders must be deterministic functions of syndrome bits;
        correction frames are cached in a bounded LRU. Working-memory estimates
        cover numerical tables and batch temporaries, not Python/decoder memory.
        """

        if stabilizers is None:
            if name not in _CODE_SPECS:
                raise ValueError('Unknown preset; supply stabilizers and logical operators')
            stabilizers = _CODE_SPECS[name]
            logical_x = 'X' * len(stabilizers[0])
            logical_z = 'Z' * len(stabilizers[0])
    
        xs = (logical_x,) if isinstance(logical_x, str) else tuple(logical_x or ())
        zs = (logical_z,) if isinstance(logical_z, str) else tuple(logical_z or ())
        if not xs or len(xs) != len(zs) or not isinstance(xs[0], str) or not xs[0]:
            raise ValueError('Supply matching nonempty logical_x and logical_z operators')
        self.name, self.length, self.num_logical = name, len(xs[0]), len(xs)
        if distance is not None and (isinstance(distance, bool) or not isinstance(distance, Integral)
                                     or not 1 <= distance <= self.length):
            raise ValueError('distance must be an integer between 1 and n, or None')
        self.distance = distance
        self._frame_bits = 2 * self.num_logical
        self._frame_mask = (1 << self._frame_bits) - 1
        self.stabilizers = tuple(stabilizers)
        self.logical_x = xs[0] if self.num_logical == 1 else xs
        self.logical_z = zs[0] if self.num_logical == 1 else zs
        self._stabilizers = tuple(self._parse_pauli(row) for row in self.stabilizers)
        self._xs = tuple(self._parse_pauli(row) for row in xs)
        self._zs = tuple(self._parse_pauli(row) for row in zs)
        self._logical_x, self._logical_z = self._xs[0], self._zs[0]
        self._validate_code()
        if method not in ('exact', 'sample'):
            raise ValueError('QEC method must be exact or sample')
        if isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 1:
            raise ValueError('QEC shots must be a positive integer')
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0):
            raise ValueError('QEC seed must be a nonnegative integer or None')
        if decoder is not None and not callable(decoder):
            raise ValueError('A QEC decoder must be callable')
        self.method, self.shots, self.seed = method, int(shots), seed
        for field, value, minimum in (('batch_size', batch_size, 1), ('decoder_cache_size', decoder_cache_size, 0),
                                      ('max_working_bytes', max_working_bytes, 1)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f'{field} must be an integer >= {minimum}')
        self.batch_size, self.decoder_cache_size = int(batch_size), int(decoder_cache_size)
        self._decoder_frames: OrderedDict[int, int] = OrderedDict()
        self._rng = np.random.default_rng(seed)
        self._decoder = decoder
        checks = len(self._stabilizers)
        if checks > self.MAX_EXACT_CHECKS and (method == 'exact' or decoder is None):
            raise ValueError('Large codes require method="sample" and a supplied decoder')
        words = (checks + self._frame_bits + 63) // 64
        tables = self.length * 4 * words * 8
        batch = self.batch_size * ((32 + 8 * words) * self.length + 32 * words)
        exact = (4 * self.length + 12) * (1 << (checks + self._frame_bits)) * 8 if method == 'exact' else 0
        self.estimated_working_bytes = tables + max(batch, exact)
        if self.estimated_working_bytes > max_working_bytes:
            raise ValueError(f'QEC numerical working memory needs approximately {self.estimated_working_bytes} bytes; '
                             'reduce batch_size or increase max_working_bytes explicitly')
        self.max_working_bytes = int(max_working_bytes)
        self._corrections = self._make_decoder() if decoder is None else None
        self._indices = None
        self._labels = None
        self._contributions = None
        self._wide_contributions = None
        self._correction_frames = None
        if self._corrections is not None:
            self._correction_frames = np.array([self._frame(self._corrections[i]) for i in range(2**checks)], dtype=object if self.num_logical > 31 else np.uint64)
        if checks + self._frame_bits <= 64:
            self._contributions = np.array([[self._state_index(self._single_error(position, value))
                                            for value in range(4)] for position in range(self.length)], dtype=np.uint64)
        else:
            self._wide_contributions = np.empty((self.length, 4, words), dtype=np.uint64)
            for position, value in product(range(self.length), range(4)):
                state = self._state_index(self._single_error(position, value))
                self._wide_contributions[position, value] = [(state >> (64 * word)) & ((1 << 64) - 1)
                                                             for word in range(words)]
        if method == 'exact':
            self._indices = np.arange(1 << (checks + self._frame_bits))
            frames = np.array([self._decoder_frame(i) for i in range(2**checks)], dtype=np.intp)
            residual = (self._indices & self._frame_mask) ^ frames[self._indices >> self._frame_bits]
            self._labels = self._joint_labels(residual)
            self._transitions = self._indices ^ self._contributions.astype(np.intp)[:, :, None]

    def _parse_pauli(self, string: str) -> tuple[int, int]:
        if not isinstance(string, str) or len(string) != self.length or set(string) - set('IXYZ'):
            raise ValueError('Pauli strings must have code length and contain only I/X/Y/Z')
        return int(string.translate(_X_BINARY)[::-1], 2), int(string.translate(_Z_BINARY)[::-1], 2)

    def _validate_code(self) -> None:
        if len(self._stabilizers) != self.length - self.num_logical:
            raise ValueError('An [[n,k]] code requires n-k independent generators')
        if any(_commutes_bit(a, b) for a, b in combinations(self._stabilizers, 2)):
            raise ValueError('Stabilizer generators must commute')
        pivots = {}
        for x, z in self._stabilizers:
            row = x | (z << self.length)
            while row and row.bit_length() in pivots:
                row ^= pivots[row.bit_length()]
            if not row:
                raise ValueError('Stabilizer generators must be independent')
            pivots[row.bit_length()] = row
        if any(_commutes_bit(logical, generator)
               for logical in self._xs + self._zs for generator in self._stabilizers):
            raise ValueError('Logical operators must commute with every stabilizer')
        for i, j in product(range(self.num_logical), repeat=2):
            if _commutes_bit(self._xs[i], self._zs[j]) != (i == j):
                raise ValueError('Logical X_i and Z_j must anticommute exactly when i=j')
            if _commutes_bit(self._xs[i], self._xs[j]) or _commutes_bit(self._zs[i], self._zs[j]):
                raise ValueError('Different logical X operators and logical Z operators must commute')

    def _single_error(self, position: int, value: int) -> tuple[int, int]:
        return _X[value] << position, _Z[value] << position

    def _syndrome(self, error: tuple[int, int]) -> int:
        return sum(_commutes_bit(error, generator) << index
                   for index, generator in enumerate(self._stabilizers))

    def _frame(self, error: tuple[int, int]) -> int:
        return sum((_commutes_bit(error, z) | (_commutes_bit(error, x) << 1)) << (2*i)
                   for i, (x, z) in enumerate(zip(self._xs, self._zs)))

    def _state_index(self, error: tuple[int, int]) -> int:
        return self._syndrome(error) << self._frame_bits | self._frame(error)

    def _make_decoder(self) -> dict[int, tuple[int, int]]:
        # Breadth-first search in syndrome space. Shortest paths correspond
        # to minimum-weight Pauli representatives; no weight-two restriction.
        corrections = {0: (0, 0)}
        queue = deque([0])
        edges = [(self._syndrome(self._single_error(position, value)),
                  self._single_error(position, value))
                 for position in range(self.length) for value in range(1, 4)]
        while queue:
            syndrome = queue.popleft()
            x, z = corrections[syndrome]
            for shift, (ex, ez) in edges:
                neighbor = syndrome ^ shift
                if neighbor not in corrections:
                    corrections[neighbor] = (x ^ ex, z ^ ez)
                    queue.append(neighbor)
        return corrections

    def _correction(self, syndrome: int) -> tuple[int, int]:
        if self._decoder is None:
            return self._corrections[syndrome]
        bits = tuple((syndrome >> index) & 1 for index in range(len(self._stabilizers)))
        correction = self._parse_pauli(self._decoder(bits))
        if self._syndrome(correction) != syndrome:
            raise ValueError('Decoder correction does not match the supplied syndrome')
        return correction

    def _residual_label(self, index: int) -> int:
        correction = self._correction(int(index) >> self._frame_bits)
        frame = (int(index) & self._frame_mask) ^ self._frame(correction)
        return sum(int(_FRAME_TO_PAULI[(frame >> (2*i)) & 3]) << (2*i) for i in range(self.num_logical))

    def decode_error(self, paulis: Sequence[int] | np.ndarray) -> int:
        """Return joint label sum(P_i * 4**i), with P_i = I/X/Y/Z = 0/1/2/3."""
        values = np.asarray(paulis)
        if values.shape != (self.length,) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError('Supply one integer Pauli label per code qubit')
        if np.any(values < 0) or np.any(values > 3):
            raise ValueError('Pauli labels are I=0, X=1, Y=2, Z=3')
        return int(self._residual_label(self._state_index(_masks(values.tolist()))))

    def _decoder_frame(self, syndrome: int) -> int:
        if self._correction_frames is not None:
            return int(self._correction_frames[syndrome])
        if syndrome in self._decoder_frames:
            self._decoder_frames.move_to_end(syndrome)
            return self._decoder_frames[syndrome]
        frame = self._frame(self._correction(syndrome))
        if self.decoder_cache_size:
            self._decoder_frames[syndrome] = frame
            if len(self._decoder_frames) > self.decoder_cache_size:
                self._decoder_frames.popitem(last=False)
        return frame

    def decode_errors(self, paulis: np.ndarray) -> np.ndarray:
        """Decode a batch of Pauli trajectories, preserving correlations per row.

        Input shape is (shots, n); output is (shots, k), except k=1
        retains the legacy (shots,) shape. Labels use I/X/Y/Z = 0/1/2/3.
        This also accepts correlated input trajectories supplied by a caller.
        It does not infer correlations from independent input probabilities.
        """
        values = np.asarray(paulis)
        if values.ndim != 2 or values.shape[1] != self.length or not np.issubdtype(values.dtype, np.integer):
            raise ValueError('Supply a (shots, code length) array of integer Pauli labels')
        if np.any(values < 0) or np.any(values > 3):
            raise ValueError('Pauli labels are I=0, X=1, Y=2, Z=3')
        output = np.empty(len(values) if self.num_logical == 1 else (len(values), self.num_logical), dtype=np.uint8)
        for start in range(0, len(values), self.batch_size):
            output[start:start+self.batch_size] = self._decode_batch(values[start:start+self.batch_size])
        return output

    def _decode_batch(self, errors: np.ndarray) -> np.ndarray:
        if self.num_logical > 1:
            return self._decode_multiple(errors)
        if self._contributions is None:
            return self._decode_packed_batch(errors)
        indices = np.bitwise_xor.reduce(self._contributions[np.arange(self.length), errors], axis=1)
        if self._labels is not None:
            return self._labels[indices.astype(np.intp)]
        syndromes, frames = indices >> 2, (indices & 3).astype(np.uint8)
        if self._correction_frames is not None:
            return _FRAME_TO_PAULI[frames ^ self._correction_frames[syndromes.astype(np.intp)]]
        unique, inverse = np.unique(syndromes, return_inverse=True)
        corrections = np.array([self._decoder_frame(int(value)) for value in unique], dtype=np.uint8)
        return _FRAME_TO_PAULI[frames ^ corrections[inverse]]

    def _decode_packed_batch(self, errors: np.ndarray) -> np.ndarray:
        # Gottesman thesis section 3.4: H_z*x + H_x*z modulo two.
        # Compile each physical Pauli's syndrome/frame once; XOR packed words.
        packed = np.bitwise_xor.reduce(self._wide_contributions[np.arange(self.length), errors], axis=1)
        syndromes = packed >> 2
        syndromes[:, :-1] |= packed[:, 1:] << 62
        unique, inverse = np.unique(syndromes, axis=0, return_inverse=True)
        corrections = np.array([self._decoder_frame(int.from_bytes(row.astype('<u8', copy=False).tobytes(), 'little'))
                                for row in unique], dtype=np.uint8)
        frames = (packed[:, 0] & 3).astype(np.uint8)
        return _FRAME_TO_PAULI[frames ^ corrections[inverse]]

    def _joint_labels(self, frames: np.ndarray) -> np.ndarray:
        labels = np.zeros(frames.shape, dtype=np.intp)
        for i in range(self.num_logical):
            labels |= _FRAME_TO_PAULI[(frames >> (2*i)) & 3] << (2*i)
        return labels

    def _decode_multiple(self, errors: np.ndarray) -> np.ndarray:
        """Return joint trajectories without constructing a 4**k distribution."""
        if self._contributions is not None:
            packed = np.bitwise_xor.reduce(self._contributions[np.arange(self.length), errors], axis=1)[:, None]
        else:
            packed = np.bitwise_xor.reduce(self._wide_contributions[np.arange(self.length), errors], axis=1)
        offset, shift = divmod(self._frame_bits, 64)
        syndromes = packed[:, offset:].copy()
        if shift:
            syndromes >>= shift
            syndromes[:, :-1] |= packed[:, offset+1:] << (64-shift)
        unique, inverse = np.unique(syndromes, axis=0, return_inverse=True)
        frame_words = (self._frame_bits + 63) // 64
        corrections = np.empty((len(unique), frame_words), dtype=np.uint64)
        for index, row in enumerate(unique):
            syndrome = int.from_bytes(row.astype('<u8', copy=False).tobytes(), 'little')
            frame = self._decoder_frame(syndrome)
            corrections[index] = [(frame >> (64*word)) & ((1 << 64)-1) for word in range(frame_words)]
        residuals = packed[:, :frame_words] ^ corrections[inverse]
        output = np.empty((len(errors), self.num_logical), dtype=np.uint8)
        for i in range(self.num_logical):
            word, shift = divmod(2*i, 64)
            output[:, i] = _FRAME_TO_PAULI[(residuals[:, word] >> shift) & 3]
        return output

    def _check_channel_size(self) -> None:
        required = 3 * 8 * 4**self.num_logical
        if required > self.max_working_bytes:
            raise ValueError(f'Joint logical channel needs approximately {required} bytes; '
                             'use iter_logical_errors for bounded-memory trajectories')

    def iter_logical_errors(self, probabilities: np.ndarray, *,
                            rng: np.random.Generator | None = None) -> Iterator[np.ndarray]:
        """Yield (batch, k) correlated logical Pauli samples; never allocate 4**k.

        Requires explicit method="sample". Each row is one joint realization;
        splitting rows into independent marginal ensembles loses correlations.
        """
        if self.method != 'sample':
            raise ValueError('Logical trajectories require method="sample"')
        noise = self._validate_noise(probabilities)
        rng = self._rng if rng is None else rng
        cumulative = np.cumsum(noise, axis=1)
        for start in range(0, self.shots, self.batch_size):
            draws = rng.random((min(self.batch_size, self.shots-start), self.length))
            errors = (draws[:, :, None] >= cumulative[None, :, :3]).sum(axis=2)
            yield self._decode_batch(errors).reshape(-1, self.num_logical)

    def logical_channel(self, probabilities: np.ndarray, *, rng: np.random.Generator | None = None) -> np.ndarray:
        """Joint residual channel for independent physical noise.

        Entry sum(P_i * 4**i) corresponds to logical Pauli labels P_i.
        Exact or sampled according to method; allocation is guarded."""
        noise = self._validate_noise(probabilities)
        self._check_channel_size()
        if self.method == 'sample':
            return self._sample_channel(noise, rng=rng)
        return self._exact_channel(noise)

    def _validate_noise(self, probabilities: np.ndarray) -> np.ndarray:
        noise = np.asarray(probabilities, dtype=float)
        if noise.shape != (self.length, 4) or not np.all(np.isfinite(noise)):
            raise ValueError('Expected finite I/X/Y/Z probabilities for every code qubit')
        if np.any(noise < 0) or not np.allclose(noise.sum(axis=1), 1, atol=1e-10, rtol=0):
            raise ValueError('Every Pauli distribution must be normalized and nonnegative')
        return noise

    def _exact_channel(self, noise: np.ndarray) -> np.ndarray:
        distribution = np.zeros(len(self._indices))
        distribution[0] = 1.
        scratch = np.empty((4, len(distribution)))
        following = np.empty_like(distribution)
        for row, transitions in zip(noise, self._transitions):
            np.take(distribution, transitions, out=scratch)
            np.einsum('i,ij->j', row, scratch, out=following)
            distribution, following = following, distribution
        return np.bincount(self._labels, weights=distribution, minlength=4**self.num_logical)

    def _sample_channel(self, noise: np.ndarray, *, rng: np.random.Generator | None = None) -> np.ndarray:
        return self._sample_counts(noise, rng=rng) / self.shots

    def estimate_channel(self, probabilities: np.ndarray, *, rng: np.random.Generator | None = None) -> PauliChannelEstimate:
        """Expose sample counts and marginal uncertainty for a sampled backend."""
        if self.method != 'sample':
            raise ValueError('estimate_channel requires method="sample"; logical_channel is exact for this code')
        self._check_channel_size()
        return PauliChannelEstimate(self._sample_counts(self._validate_noise(probabilities), rng=rng))

    def _sample_counts(self, noise: np.ndarray, *, rng: np.random.Generator | None = None) -> np.ndarray:
        """Estimate the decoded channel with bounded-memory Pauli trajectories."""
        rng = self._rng if rng is None else rng
        cumulative = np.cumsum(noise, axis=1)
        counts = np.zeros(4**self.num_logical, dtype=int)
        remaining = self.shots
        while remaining:
            count = min(remaining, self.batch_size)
            draws = rng.random((count, self.length))
            errors = (draws[:, :, None] >= cumulative[None, :, :3]).sum(axis=2)
            labels = self._decode_batch(errors)
            if self.num_logical > 1:
                labels = (labels.astype(np.intp) * (4**np.arange(self.num_logical))).sum(axis=1)
            counts += np.bincount(labels, minlength=len(counts))
            remaining -= count
        return counts

    @staticmethod
    def bell_state(probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
        """A single decoded pair remains a 4-by-4 density matrix."""
        return (_BELL * np.asarray(probabilities)) @ _BELL.conj().T

    @staticmethod
    def bell_probabilities(qubit: Qubit) -> np.ndarray:
        if qubit.num_qubits != 2 or qubit.state.shape != (4, 4):
            raise ValueError('QEC requires independent two-qubit Bell pairs')
        state = _BELL.conj().T @ qubit.state @ _BELL
        probabilities = np.real(np.diag(state))
        if not np.allclose(state, np.diag(probabilities), atol=1e-9, rtol=0):
            raise ValueError('QEC requires Bell-diagonal noise; coherent noise is unsupported')
        if np.any(probabilities < -1e-9) or not np.isclose(probabilities.sum(), 1, atol=1e-9, rtol=0):
            raise ValueError('Input pair is not a normalized positive density matrix')
        probabilities = np.maximum(probabilities, 0)
        return probabilities / probabilities.sum()

def werner_purification(first: float, second: float) -> tuple[float, float]:
    if not all(np.isfinite(f) and 0 <= f <= 1 for f in (first, second)):
        raise ValueError('Werner fidelities must be finite probabilities')
    q1, q2 = (1 - first) / 3, (1 - second) / 3
    probability = (first + q1) * (second + q2) + 4 * q1 * q2
    return (first * second + q1 * q2) / probability, probability

def werner_swap(first: float, second: float) -> float:
    return first * second + (1 - first) * (1 - second) / 3

def purification_rounds(fidelity: float, target: float, limit: int = 8) -> int:
    """Estimate binary recurrence depth; 2**depth inputs exclude failed retries."""
    for depth in range(limit + 1):
        if fidelity >= target:
            return depth
        fidelity, _ = werner_purification(fidelity, fidelity)
    raise ValueError('Target fidelity is unreachable within the purification round limit')

# Packet headers select a shared simulation code configuration by name.
# A custom decoder is Python configuration, not a serialized wire payload.
_CODE_REGISTRY = {}

def register_qec_code(code: StabilizerPauliCode) -> str:
    """Register a validated code for selection by name in L2 packets."""
    if not isinstance(code, StabilizerPauliCode):
        raise TypeError('Register a StabilizerPauliCode instance')
    if code.name in _CODE_SPECS or code.name in _CODE_REGISTRY:
        raise ValueError(f'QEC code name {code.name!r} is already registered or reserved')
    if sum(name not in _CODE_SPECS for name in _CODE_REGISTRY) >= 253:
        raise ValueError('The modeled 8-bit QEC code field has no free identifiers')
    _CODE_REGISTRY[code.name] = code
    return code.name

def get_qec_code(code: str | StabilizerPauliCode) -> StabilizerPauliCode:
    """Resolve a registered name, a preset, or an explicit model for Node calls."""
    if isinstance(code, StabilizerPauliCode):
        return code
    if not isinstance(code, str):
        raise ValueError('Supply a registered QEC code name or a StabilizerPauliCode')
    if code not in _CODE_REGISTRY:
        if code not in _CODE_SPECS:
            raise ValueError(f'Unknown QEC code {code!r}; register its definition first')
        _CODE_REGISTRY[code] = StabilizerPauliCode(code)
    return _CODE_REGISTRY[code]
