from __future__ import annotations

import numpy as np
from typing import List, Tuple, Union, Any
import json
from qcns.python.components.qubit.qec import get_qec_code
from numbers import Integral

__all__ = ['encode_fidelity', 'decode_fidelity', 'L1_Protocol', 'L1_SRP', 'L1_TPSP', 'L1_BSMP', 'L1_FSP', 
           'L2_Protocol', 'L2_EPP', 'L2_QECP', 
           'L3_PRP', 'L3_ECP', 'L3_Protocol', 'L3_CRP', 'L3_FRP', 'L3_ERP',
           'L4_Protocol', 'L4_EPP', 'L4_QECP', 
           'L7_Protocol', 'L7_TPP', 'L7_RGP']

def _json_wire_value(value: Any) -> Any:
    """Convert supported numerical payload values for JSON wire-size accounting."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return list(value)
    raise TypeError(f'Payload type {type(value).__name__} has no wire-size encoding')


def _count(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError('Qubit counts must be nonnegative integers')
    return int(value)

def _bits(values, size):
    result = np.asarray(values, dtype=np.bool_)
    if result.shape != (size,):
        raise ValueError(f'Expected {size} outcome bits, got shape {result.shape}')
    return result.copy()

def _input_ratio(requested: int, needed: int):
    """Infer a whole number of input qubits per output; defer empty headers."""
    if requested > needed:
        raise ValueError('requested must be <= needed')
    if requested == needed == 0:
        return None
    if requested == 0 or needed % requested:
        raise ValueError('Initial needed must be a multiple of positive requested')
    return needed // requested

def _resize(array, size):
    result = np.zeros(size, dtype=np.bool_)
    result[:min(size, len(array))] = array[:min(size, len(array))]
    return result

# Original logarithmic wire encoding: ceil(31.875 * log2(511 - 510*F)).
# Valid on [0.5, 1]; byte 0 means 1 and byte 255 means 0.5.
_FIDELITY_LEVELS = (511.0 - np.exp2(np.arange(256, dtype=np.float64) / 31.875)) / 510.0
_FIDELITY_LEVELS.setflags(write=False)

def encode_fidelity(fidelity: float | np.ndarray) -> np.uint8 | np.ndarray:
    """Encode [0.5, 1] conservatively, with finer resolution near fidelity 1.

    Searching decoded levels implements the original ceiling rule without
    off-by-one round-trip errors at floating-point bin boundaries.
    Scalars return np.uint8; arrays return uint8 arrays with the original shape.
    """
    values = np.asarray(fidelity)
    if values.dtype.kind not in 'iuf' or not np.all(np.isfinite(values)):
        raise ValueError('fidelity must contain finite real numbers in [0.5, 1]')
    if np.any((values < 0.5) | (values > 1.0)):
        raise ValueError('fidelity must be in [0.5, 1] for the logarithmic 8-bit encoding')
    codes = np.searchsorted(-_FIDELITY_LEVELS, -values.astype(np.float64), side='left').astype(np.uint8)
    return np.uint8(codes) if codes.ndim == 0 else codes

def decode_fidelity(code: int | np.ndarray) -> float | np.ndarray:
    """Decode integer bytes in [0, 255] to float64 fidelity values."""
    values = np.asarray(code)
    if values.dtype.kind not in 'iu' or np.any((values < 0) | (values > 255)):
        raise ValueError('fidelity_ir must contain integers in [0, 255]')
    fidelities = _FIDELITY_LEVELS[values.astype(np.int64)]
    return float(fidelities) if fidelities.ndim == 0 else fidelities

class L1_Protocol:
    
    """
    Represents a L1 Protocol Header
    
    Attrs:
        _requested (int): number of requested qubits
        _needed (int): number of needed qubits
        _success (np.ndarray): success array
        _ack (int): ack flag
        _ps (int): photon source flag
        _protocol (int): id of the protocol
        _next_protocol (int): id of the protocol one layer above
        _header_length (int): header length in bits
    """
    
    def __init__(self, _requested: int, _needed: int, _protocol: int) -> None:
        
        """
        Initializes a L1 Protocol Header
        
        Args:
            _requested (int): number of requested qubits
            _needed (int): number of needed qubits
            
        Returns:
            /
        """
        
        self._requested: int = _count(_requested) # 1byte
        self._needed: int = _count(_needed) # 1byte
        
        if not self._requested and self._needed > 0:
            self._requested = self._needed
            
        if self._requested > 0 and not self._needed:
            self._needed = self._requested
        
        self._success: np.ndarray = np.zeros(self._needed, dtype=np.bool_) # 2^8 bits = 32 byte
        self._ack: int = 0 # 1 bit
        
        self._protocol: int = _protocol # 1byte
        # requested + needed + success + ack + protocol (bits).
        self._header_length: int = 8 + 8 + 256 + 1 + 8
    
    def __len__(self) -> int:
        
        """
        TODO add docstring
        """
        
        return self._header_length
    
    def __repr__(self) -> str:
        
        """
        Custom print function
        
        Args:
            /
            
        Returns:
            layer1 (str): str repr of layer 1
        """
        
        return f'L1: Req {self._requested} Need {self._needed} Ack {self._ack} Proto {self._protocol} Len {self._header_length} Success {self._success}'
    
    @property
    def requested(self) -> int:
        
        """
        Returns the number of requested qubits
        
        Args:
            /
            
        Returns:
            _requested (int): number of requested qubits
        """
        
        return self._requested
    
    @requested.setter
    def requested(self, requested: int) -> None:
        
        """
        TODO add docstring
        """
        
        self._requested = _count(requested)
    
    @property
    def needed(self) -> int:
        
        """
        Returns the number of needed qubits
        
        Args:
            /
            
        Returns:
            needed (int): number of needed qubits
        """
        
        return self._needed
    
    @needed.setter
    def needed(self, needed: int) -> None:
        
        """
        TODO add docstring
        """
        
        self._needed = _count(needed)
        self._success = _resize(self._success, self._needed)
    
    @property
    def ack(self) -> int:
        
        """
        Checks whether the Ack flag is set
        
        Args:
            /
            
        Returns:
            /
        """
        
        return self._ack
    
    def set_ack(self) -> None:
        
        """
        Sets the Ack flag
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._ack = 1
        
    def reset_ack(self) -> None:
        
        """
        Resets the Ack flag
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._ack = 0
           
    @property
    def success(self) -> np.ndarray:
        
        """
        Returns the success array
        
        Args:
            /
            
        Returns:
            success (np.ndarray): success array
        """
        
        return self._success
    
    @success.setter
    def success(self, success: np.ndarray) -> None:

        """
        TODO add docstring
        """

        self._success = _bits(success, len(self._success))
    
    def set_success(self, idx: int) -> None:
        
        """
        Sets the success array at the index
        
        Args:
            _idx (int): index to set success array at
            
        Returns:
            /
        """
        
        self._success[idx] = 1
    
    def reset_success(self, idx: int) -> None:
        
        """
        Resets the success array at the index
        
        Args:
            idx (int): index to set success array at
            
        Returns:
            /
        """
        
        self._success[idx] = 0
    
    @property
    def protocol(self) -> int:
        
        """
        Returns the protocol of this layer
        
        Args:
            /
            
        Returns:
            protocol (int): protocol of this layer
        """
        
        return self._protocol
    
    @protocol.setter
    def protocol(self, protocol: int) -> None:
        
        """
        Sets the protocol for this layer
        
        Args:
            protocol (int): protocol of this layer
            
        Returns:
            /
        """
        
        self._protocol = protocol

class L1_SRP(L1_Protocol):
    
    def __init__(self, requested: int, needed: int) -> None:
        super().__init__(requested, needed, 0)

class L1_TPSP(L1_Protocol):
    
    def __init__(self, requested: int, needed: int) -> None:
        super().__init__(requested, needed, 1)
        
        self._ps: int = 0 # 1 bit
        self._header_length += 1
    
    def __repr__(self) -> str:
        
        """
        Custom print function
        
        Args:
            /
            
        Returns:
            layer1 (str): str repr of layer 1
        """
        
        return f'L1: Req {self._requested} Need {self._needed} Ack {self._ack} PS {self._ps} Proto {self._protocol} Len {self._header_length} Success {self._success}'
     
    @property
    def ps(self) -> int:
            
        """
        Checks whether the Photon Source flag is set
        
        Args:
            /
            
        Returns:
            ps (int): Photon source flag
        """
        
        return self._ps
        
    def set_ps(self) -> None:
        
        """
        Sets the Photon Source flag
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._ps = 1
        
    def reset_ps(self) -> None:
        
        """
        Resets the Photon Source flag
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._ps = 0

class L1_BSMP(L1_Protocol):
    
    def __init__(self, requested: int, needed: int) -> None:
        super().__init__(requested, needed, 2)

class L1_FSP(L1_Protocol):
    
    def __init__(self, requested: int, needed: int) -> None:
        super().__init__(requested, needed, 3)     
    
class L2_Protocol:
    
    """
    Represents a L2 Protocol Header
    
    Attrs:
        _src (int): source address
        _dst (int): destination address
        _requested (int): number of requested qubits
        _needed (int): number of needed qubits
        _success (np.ndarray): success array
        _ack (int): Ack flag
        _protocol (int): id of the protocol header
        _next_protocol (int): id of the protocol one layer above
        _header_length (int): length of the header
    """
    
    def __init__(self, _src: int, _dst: int, _requested: int, _needed: int, _protocol: int) -> None:
        
        """
        Initializes a L2 protocol header
        
        Args:
            _src (int): source address
            _dst (int): destination address
            _requested (int): number of requested qubits
            _needed (int): number of needed qubits
            
        Returns:
            /
        """
        
        self._src: int = _src # 6byte
        self._dst: int = _dst # 6byte
        
        self._requested: int = _count(_requested) # 1byte
        self._needed: int = _count(_needed) # 1byte
        
        if self._requested < 1 and self._needed > 0:
            self._requested = self._needed
            
        if self._requested > 0 and self._needed < 1:
            self._needed = self._requested
        
        self._inputs_per_output = _input_ratio(self._requested, self._needed)
        self._success: np.ndarray = np.zeros(self._output_count(), dtype=np.bool_) # 32byte
        self._ack: int = 0 # 1 bit
        
        self._protocol: int = _protocol # 1byte
        # src + dst + requested + needed + success + ack + protocol (bits).
        # _inputs_per_output is derived bookkeeping, not a transmitted field.
        self._header_length: int = 48 + 48 + 8 + 8 + 256 + 1 + 8
    
    def _output_count(self) -> int:
        if self._inputs_per_output is None:
            return 0
        return self._needed // self._inputs_per_output

    def set_counts(self, requested: int, needed: int) -> None:
        """Start a new count configuration and clear outcomes from the old one.

        Establishes the input/output ratio, including for an empty header.
        Subsequent needed changes retain this ratio and preserve outcome bits.
        """
        requested, needed = _count(requested), _count(needed)
        ratio = _input_ratio(requested, needed)
        success = np.zeros(0 if ratio is None else needed // ratio, dtype=np.bool_)
        self._requested, self._needed = requested, needed
        self._inputs_per_output, self._success = ratio, success

    def __len__(self) -> int:
        
        return self._header_length
    
    @property
    def src(self) -> int:
        
        """
        Returns the source address
        
        Args:
            /
            
        Returns:
            src (int): source address
        """
        
        return self._src
    
    @src.setter
    def src(self, _src: int) -> None:
        
        """
        Sets the source address
        
        Args:
            _src (int): source address
            
        Returns:
            /
        """
        
        self._src = _src
    
    @property
    def dst(self) -> int:
        
        """
        Returns the destination address
        
        Args:
            /
            
        Returns:
            dst (int): destination address
        """
        
        return self._dst
    
    @dst.setter
    def dst(self, _dst: int) -> None:
        
        """
        Sets the destination address
        
        Args:
            _dst (int): destination address
            
        Returns:
            /
        """
        
        self._dst = _dst
    
    def switch_src_dst(self) -> None:
        
        """
        Switches the source and destination address
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._src, self._dst = self._dst, self._src
    
    @property
    def requested(self) -> int:
        
        """
        Returns the number of requested qubits
        
        Args:
            /
            
        Returns:
            requested (int): number of requested qubits
        """
        
        return self._requested
    
    @requested.setter
    def requested(self, requested: int) -> None:
        
        """
        TODO add docstring
        """
        
        requested = _count(requested)
        if requested > self._needed:
            raise ValueError('requested must be <= needed; use set_counts to reconfigure')
        self._requested = requested
    
    @property
    def needed(self) -> int:
        
        """
        Returns the number of needed qubits
        
        Args:
            /
            
        Returns:
            needed (int): number of needed qubits
        """
        
        return self._needed
    
    @needed.setter
    def needed(self, needed: int):
        
        """
        TODO add docstring
        """
        
        needed = _count(needed)
        if needed < self._requested:
            raise ValueError('needed must be >= requested')
        if needed and self._inputs_per_output is None:
            raise ValueError('Use set_counts(requested, needed) to initialize the input/output ratio')
        size = 0 if self._inputs_per_output is None else needed // self._inputs_per_output
        success = _resize(self._success, size)
        self._needed, self._success = needed, success
    
    @property
    def ack(self) -> int:
        
        """
        Checks if the Ack flag is set
        
        Args:
            /
            
        Returns:
            ack (int): Ack flag
        """
        
        return self._ack
    
    def set_ack(self) -> None:
        
        """
        Sets the Ack flag
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._ack = 1
        
    def reset_ack(self) -> None:
        
        """
        Resets the Ack flag
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._ack = 0
    
    @property
    def success(self) -> np.ndarray:
        
        """
        Returns the success array
        
        Args:
            /
            
        Returns:
            success (np.ndarray): success array
        """
        
        return self._success

    @success.setter
    def success(self, success: np.ndarray):
        
        """
        TODO add docstring
        """
        
        self._success = _bits(success, len(self._success))

    def set_success(self, idx: int) -> None:
        
        """
        Sets the success array at the index
        
        Args:
            idx (int): index to set success array at
            
        Returns:
            /
        """
        
        self._success[idx] = 1
        
    def reset_success(self, idx: int) -> None:
        
        """
        Resets the success array at the index
        
        Args:
            idx (int): index to set success array at
            
        Returns:
            /
        """
        
        self._success[idx] = 0
    
    @property
    def protocol(self) -> int:
        
        """
        Returns the protocol of this layer
        
        Args:
            /
            
        Returns:
            protocol (int): protocol of this layer
        """
        
        return self._protocol
    
    @protocol.setter
    def protocol(self, protocol: int) -> None:
        
        """
        Sets the protocol for this layer
        
        Args:
            protocol (int): protocol of this layer
            
        Returns:
            /
        """
        
        self._protocol = protocol

class L2_EPP(L2_Protocol):
    
    def __init__(self, src: int, dst: int, requested: int, needed: int, code: str=None) -> None:
        
        """
        TODO add docstring
        """
        
        super().__init__(src, dst, requested, needed, 0)
    
    def __repr__(self) -> str:
            
        """
        Custom print function
        
        Args:
            /
            
        Returns:
            layer2 (str): str repr of layer 1
        """
        
        return f' | L2: Src {self._src} Dst {self._dst} Req {self._requested} Need {self._needed} Ack {self._ack} Proto {self._protocol} Len {self._header_length} Success {self._success}'
       
class L2_QECP(L2_Protocol):
    
    def __init__(self, src: int, dst: int, requested: int, needed: int, code: str) -> None:
        
        """
        TODO add docstring
        """
        
        get_qec_code(code)
        self._code = code
        super().__init__(src, dst,requested, needed, 1)
        
        self._code = code #1 byte
        self._complete = False # 1 bit
        self._header_length += 8 + 1  # code ID + completion flag
    
    def __repr__(self) -> str:
            
        """
        Custom print function
        
        Args:
            /
            
        Returns:
            layer2 (str): str repr of layer 1
        """
        
        return f' | L2: Src {self._src} Dst {self._dst} Req {self._requested} Need {self._needed} Code {self._code} Complete {self._complete} Ack {self._ack} Proto {self._protocol} Len {self._header_length} Success {self._success}'
    
    @property
    def code(self):
        
        """
        TODO add docstring
        """
        
        return self._code

    @code.setter
    def code(self, name: str):
        
        """
        TODO add docstring
        """
        
        get_qec_code(name)
        self._code = name

    @property
    def complete(self):
        
        """
        TODO add docstring
        """
        
        return self._complete

    def set_complete(self):
        
        """
        TODO add docstring
        """
        
        self._complete = True
        
    def reset_complete(self):
        
        """
        TODO add docstring
        """
        
        self._complete = False

class L3_Protocol:
    
    """
    Represents a L3 Protocol Header
    
    Attrs:
        _src (int): source address
        _dst (int): destination address
        _requested (int): number of requested qubits
        _needed (int): number of needed qubits
        _success (np.ndarray): success array
        _x_count (np.ndarray): amplitude flip array
        _z_count (np.ndarray): phase flip array
        _fmode (int): mode how to access quantum data plane or simply forward
        _protocol (int): id of protocol header
        _next_protocol (int): id of protocol header one layer above
        _header_length (int): header length
    """
    
    def __init__(self, _src: int, _dst: int, _curr: int, _next: int, _requested: int, _needed: int, _protocol: int, *, fidelity: float=0.5) -> None:
        
        """
        Initializes the L3 Protocol Header
        
        Args:
            _src (int): source address
            _dst (int): destination address
            _requested (int): number of requested qubits
            _needed (int): number of needed qubits
            
        Returns:
            /
        """
        
        self._src: int = _src # 16 byte
        self._dst: int = _dst # 16 byte
        self._curr: int = _curr # 16 byte
        self._next: int = _next # 16 byte
        
        self._requested: int = _count(_requested) # 1 byte
        self._needed: int = _count(_needed) # 1 byte
        self._fidelity: np.uint8 = encode_fidelity(fidelity) # 1 byte
        
        if self._requested < 1 and self._needed > 0:
            self._requested = self._needed
            
        if self._requested > 0 and self._needed < 1:
            self._needed = self._requested        
    
        self._x_count: np.ndarray = np.zeros(self._needed, dtype=np.bool_) # 32 byte
        self._z_count: np.ndarray = np.zeros(self._needed, dtype=np.bool_) # 32 byte

        self._fmode: int = 1 # 2 bit    
        self._hop_count: int = 0 # 1 byte
    
        self._protocol: int = _protocol # 1 byte
        # Four addresses + two counts + fidelity + two outcome arrays
        # + forwarding mode + hop count + protocol (bits).
        self._header_length: int = 1066
    
    @property
    def fidelity(self) -> float:
        """Quantized request metadata; not an estimate of hidden quantum state."""
        return decode_fidelity(self._fidelity)

    @fidelity.setter
    def fidelity(self, fidelity: float) -> None:
        if np.ndim(fidelity) != 0:
            raise ValueError('The L3 fidelity field must be a scalar')
        if fidelity < 0.5 or fidelity > 1.0:
            raise ValueError('fidelity must be in [0.5, 1] for the logarithmic 8-bit encoding')
        self._fidelity = encode_fidelity(fidelity)

    @property
    def fidelity_ir(self) -> np.uint8:
        """Raw 8-bit integer representation (0 = 1.0, 255 = 0.5)."""
        return self._fidelity

    @fidelity_ir.setter
    def fidelity_ir(self, fidelity: np.uint8) -> None:
        if np.ndim(fidelity) != 0:
            raise ValueError('The L3 fidelity_ir field must be a scalar')
        if fidelity < 0 or fidelity > 255:
            raise ValueError('fidelity_ir must be in [0, 255] for the logarithmic 8-bit encoding')
        self._fidelity = fidelity

    def __len__(self) -> int:
        
        """
        TODO add docstring
        """
        
        return self._header_length

    def __repr__(self) -> str:
        
        """
        Custom print function
        
        Args:
            /
            
        Returns:
            layer3 (str): str repr of layer 1
        """
        
        return f' | L3: Src {self._src} Dst {self._dst} Req {self._requested} Need {self._needed} Fidelity {self.fidelity} FMode {self._fmode} Proto {self._protocol}Len {self._header_length} X {self._x_count} Z {self._z_count}'
     
    @property
    def src(self) -> int:
        
        """
        Returns the source address
        
        Args:
            /
            
        Returns:
            src (int): source address
        """
        
        return self._src
    
    @src.setter
    def src(self, src: int) -> None:
        
        """
        Sets the source address
        
        Args:
            src (int): source address
            
        Returns:
            /
        """
        
        self._src = src
        
    @property
    def dst(self) -> int:
        
        """
        Returns the destination address
        
        Args:
            /
            
        Returns:
            dst (int): destination address
        """
        
        return self._dst
    
    @dst.setter
    def dst(self, dst: int) -> None:
        
        """
        Sets the destination address
        
        Args:
            dst (int): destination address
            
        Returns:
            /
        """
        
        self._dst = dst
    
    def switch_src_dst(self) -> None:
        
        """
        Switches the source and destination address
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._src, self._dst = self._dst, self._src
    
    @property
    def curr(self) -> int:
        """
        Returns the current address
        
        Args:
            /
            
        Returns:
            curr (int): current address
        """
        
        return self._curr
    
    @curr.setter
    def curr(self, curr: int) -> None:
        
        """
        Sets the current address
        
        Args:
            curr (int): current address
            
        Returns:
            /
        """
        
        self._curr = curr
    
    @property
    def next(self) -> int:
        """
        Returns the next address
        
        Args:
            /
            
        Returns:
            next (int): next address
        """
        
        return self._next
    
    @next.setter
    def next(self, next: int) -> None:
        
        """
        Sets the next address
        
        Args:
            next (int): next address
            
        Returns:
            /
        """
        
        self._next = next
    
    def update_curr_next(self, next: int) -> None:
        
        """
        Updates the current and next address
        
        Args:
            next (int): next address
            
        Returns:
            /
        """
        
        self._curr = self._next
        self._next = next
    
    @property
    def requested(self) -> int:
        
        """
        Returns the number of requested qubits
        
        Args:
            /
            
        Returns:
            requested (int): number of requested qubits
        """
        
        return self._requested
    
    @requested.setter
    def requested(self, requested: int) -> None:
        
        """
        TODO add docstring
        """
        
        self._requested = _count(requested)
    
    @property
    def needed(self) -> int:
        
        """
        Returns the number of needed qubits
        
        Args:
            /
            
        Returns:
            needed (int): number of needed qubits
        """
        
        return self._needed
    
    @needed.setter
    def needed(self, needed: int) -> None:
        
        """
        TODO add docstring
        """
        
        self._needed = _count(needed)
        self._x_count = _resize(self._x_count, self._needed)
        self._z_count = _resize(self._z_count, self._needed)
     
    @property
    def fmode(self) -> int:
        
        """
        Returns the forwarding mode of the packet
        
        00: classical forwarding
        01: no reject mode
        10: partial reject mode
        11: complete reject mode
        
        Args:
            /
            
        Returns:
            mode (int): quantum data plane mode of the header
        """
        
        return self._fmode
    
    def set_cf(self) -> None:
        
        """
        Sets the mode of the header to classical forwarding
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._fmode = 0
    
    def set_nr(self) -> None:
        
        """
        Sets the mode of the header to no reject mode
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._fmode = 1
    
    def set_pr(self) -> None:
        
        """
        Sets the mode of the header to partial reject mode
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._fmode = 2
        
    def set_cr(self) -> None:
        
        """
        Sets the mode of the header to complete reject mode
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._fmode = 3
    
    @property
    def is_cf(self) -> bool:
        
        """
        Returns whether the packet is in cf mode
        
        Args:
            /
            
        Returns:
            is_cf (bool): whether the packet is in cf mode
        """
        
        return self._fmode == 0
    
    @property
    def is_nr(self) -> bool:
        
        """
        Returns whether the packet is in nr mode
        
        Args:
            /
            
        Returns:
            is_nr (bool): whether the packet is in nr mode
        """
        
        return self._fmode == 1
    
    @property
    def is_pr(self) -> bool:
        
        """
        Returns whether the packet is in pr mode
        
        Args:
            /
            
        Returns:
            is_pr (bool): whether the packet is in pr mode
        """
        
        return self._fmode == 2
    
    @property
    def is_cr(self) -> bool:
        
        """
        Returns whether the packet is in cr mode
        
        Args:
            /
            
        Returns:
            is_cr (bool): whether the packet is in cr mode
        """
        
        return self._fmode == 3
    
    @property
    def hop_count(self) -> int:
        
        """
        Returns the hop count 
        
        Args:
            /
            
        Returns:
            hop_count (int): hop count of the packet
        """
        
        return self._hop_count
    
    @hop_count.setter
    def hop_count(self, hop_count: int) -> None:
        
        """
        Sets the hop count of a packet
        
        Args:
            hop_count (int): new hop count
            
        Returns:
            /
        """
        
        self._hop_count = hop_count
    
    @property
    def es_result(self) -> Tuple[np.ndarray, np.ndarray]:
        
        """
        Returns both the X and Z array
        
        Args:
            /
            
        Returns:
            es_result (tuple): tuple containing the X and Z array
        """
        
        return self._x_count, self._z_count
    
    @es_result.setter
    def es_result(self, results: Tuple[np.ndarray, np.ndarray]) -> None:
    
        """
        TODO add docstring
        """
    
        self._x_count, self._z_count = results
    
    def update_es(self, res: int, idx: int) -> None:
        
        """
        TODO add docstring
        """
        
        self._z_count[idx] ^= bool(res & 2)
        self._x_count[idx] ^= bool(res & 1)
    
    def reset_es(self, idx: int) -> None:
        
        """
        Resets the X and Z array
        
        Args:
            idx (int): index to reset
            
        Returns:
            /
        """
        
        self._x_count[idx] = 0
        self._z_count[idx] = 0
    
    @property
    def x_count(self) -> np.ndarray:
        
        """
        Returns the X count array
        
        Args:
            /
            
        Returns:
            x_count (np.ndarray): X count array
        """
        
        return self._x_count
    
    @x_count.setter
    def x_count(self, results: np.ndarray) -> None:
        
        """
        TODO add docstring
        """
        
        self._x_count = results
    
    @property
    def z_count(self) -> np.ndarray:
        
        """
        Returns the Z count array
        
        Args:
            /
            
        Returns:
            z_count (np.ndarray): Z count array
        """
        
        return self._z_count
    
    @z_count.setter
    def z_count(self, results: np.ndarray) -> None:
        
        """
        TODO add docstring
        """
        
        self._z_count = results
    
    @property
    def protocol(self) -> int:
        
        """
        Returns the protocol of this layer
        
        Args:
            /
            
        Returns:
            protocol (int): protocol of this layer
        """
        
        return self._protocol
    
    @protocol.setter
    def protocol(self, protocol: int) -> None:
        
        """
        Sets the protocol for this layer
        
        Args:
            protocol (int): protocol of this layer
            
        Returns:
            /
        """
        
        self._protocol = protocol

    def update_x_count(self, result: bool, index: int=0):
        
        """
        TODO add docstring
        """
        
        self._x_count[index] ^= bool(result)

    def update_z_count(self, result: bool, index: int=0):
        
        """
        TODO add docstring
        """
        
        self._z_count[index] ^= bool(result)

class L3_CRP(L3_Protocol):
    
    def __init__(self, src: int, dst: int, curr: int, next: int, requested: int, needed: int, *, fidelity: float=0.5) -> None:
        
        """
        TODO add docstring
        """
        
        super().__init__(src, dst, curr, next, requested, needed, 0, fidelity=fidelity)
        
class L3_FRP(L3_Protocol):
    
    def __init__(self, src: int, dst: int, curr: int, next: int, requested: int, needed: int, *, fidelity: float=0.5) -> None:
        
        """
        TODO add docstring
        """
        
        super().__init__(src, dst, curr, next, requested, needed, 1, fidelity=fidelity)
        
class L3_ERP(L3_Protocol):
    
    def __init__(self, src: int, dst: int, curr: int, next: int, requested: int, needed: int, *, fidelity: float=0.5) -> None:
        
        """
        TODO add docstring
        """
        
        super().__init__(src, dst, curr, next, requested, needed, 2, fidelity=fidelity)

class L3_PRP(L3_Protocol):
    """PR control envelope; request path/configuration is exchanged only at setup.

    Transaction IDs correlate replies; pair IDs name local resources, never
    quantum states. The routing addresses are the current control endpoints.
    """
    def __init__(self, src: int, dst: int, kind: str, key: tuple[int, int],
                 transaction: tuple[int, int], data: dict | None = None, *, reply: bool = False) -> None:
        super().__init__(src, dst, src, dst, 0, 0, 3)
        self.set_cf()
        self.kind = kind
        self.key = tuple(key)
        self.transaction = tuple(transaction)
        self.data = {} if data is None else dict(data)
        self.reply = bool(reply)

    def __len__(self) -> int:
        record = (self.kind, self.key, self.transaction, self.data, self.reply)
        return super().__len__() + 8 * len(json.dumps(record, default=_json_wire_value, separators=(',', ':')).encode('utf8'))

class L3_ECP(L3_PRP):
    """Encoded Control Protocol; code selection travels with each transaction."""

    def __init__(self, src: int, dst: int, kind: str, key: tuple[int, int],
                 transaction: tuple[int, int], data: dict | None = None, *,
                 code: str = 'five_qubit', reply: bool = False) -> None:
        super().__init__(src, dst, kind, key, transaction, data, reply=reply)
        self._protocol = 2
        self.code = code

    def __len__(self) -> int:
        return super().__len__() + 8 * len(self.code.encode('utf8'))

class L4_Protocol:
    
    """
    Represents a L4 Protocol Header
    
    Attrs:
        _src (int): source port
        _dst (int): destination port
        _requested (int): number of requested qubits
        _needed (int): number of needed qubits
        _success (np.ndarray): success array
        _ack (int): Ack flag
        _protocol (int): id of the current header
        _next_protocol (int): id of the header one layer above
        _header_length (int): header length
    """
    
    def __init__(self, _src: int, _dst: int, _requested: int, _needed: int, _protocol: int) -> None:
        
        """
        Initializes a L4 Protocol Header
        
        Args:
            _src (int): source port
            _dst (int): destination port
            _requested (int): number of requested qubit
            _needed (int): number of needed qubits
        """
        
        self._src: int = _src # 2byte
        self._dst: int = _dst # 2byte

        self._requested: int = _count(_requested) # 1byte
        self._needed: int = _count(_needed) # 1byte
        
        if self._requested < 1 and self._needed > 0:
            self._requested = self._needed
            
        if self._requested > 0 and self._needed < 1:
            self._needed = self._requested
            
        self._inputs_per_output = _input_ratio(self._requested, self._needed)
        self._success: np.ndarray = np.zeros(self._output_count(), dtype=np.bool_) # 32byte
        self._ack: int = 0 # 1 bit
        
        self._protocol: int = _protocol # 1byte
        # src + dst + requested + needed + success + ack + protocol (bits).
        # _inputs_per_output is derived bookkeeping, not a transmitted field.
        self._header_length: int = 16 + 16 + 8 + 8 + 256 + 1 + 8
    
    def _output_count(self) -> int:
        if self._inputs_per_output is None:
            return 0
        return self._needed // self._inputs_per_output

    def set_counts(self, requested: int, needed: int) -> None:
        """Start a new count configuration and clear outcomes from the old one.

        Establishes the input/output ratio, including for an empty header.
        Subsequent needed changes retain this ratio and preserve outcome bits.
        """
        requested, needed = _count(requested), _count(needed)
        ratio = _input_ratio(requested, needed)
        success = np.zeros(0 if ratio is None else needed // ratio, dtype=np.bool_)
        self._requested, self._needed = requested, needed
        self._inputs_per_output, self._success = ratio, success

    def __len__(self) -> int:
        
        """
        TODO add docstring
        """
        
        return self._header_length
       
    @property
    def src(self) -> int:
        
        """
        Returns the source port
        
        Args:
            /
            
        Returns:
            src (int): source port
        """
        
        return self._src
    
    @src.setter
    def src(self, src: int) -> None:
        
        """
        Sets the source port
        
        Args:
            src (int): source port
            
        Returns:
            /
        """
        
        self._src = src
        
    @property
    def dst(self) -> int:
        
        """
        Returns the destination port
        
        Args:
            /
            
        Returns:
            dst (int): destination port
        """
        
        return self._dst
    
    @dst.setter
    def dst(self, dst: int) -> None:
        
        """
        Sets the destination port
        
        Args:
            dst (int): destination port
            
        Returns:
            /
        """
        
        self._dst = dst    
     
    def switch_src_dst(self) -> None:
        
        """
        Switches the source and destination port
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._src, self._dst = self._dst, self._src 
      
    @property
    def requested(self) -> int:
        
        """
        Returns the number of requested qubits
        
        Args:
            /
            
        Returns:
            requested (int): number of requested qubits
        """
        
        return self._requested
    
    @requested.setter
    def requested(self, requested: int) -> None:
        
        """
        TODO add docstring
        """
        
        requested = _count(requested)
        if requested > self._needed:
            raise ValueError('requested must be <= needed; use set_counts to reconfigure')
        self._requested = requested
    
    @property
    def needed(self) -> int:
        """
        Returns the number of needed qubits
        
        Args:
            /
            
        Returns:
            needed (int): number of needed qubits
        """
        
        return self._needed

    @needed.setter
    def needed(self, needed: int):
        
        """
        TODO add docstring
        """
        
        needed = _count(needed)
        if needed < self._requested:
            raise ValueError('needed must be >= requested')
        if needed and self._inputs_per_output is None:
            raise ValueError('Use set_counts(requested, needed) to initialize the input/output ratio')
        size = 0 if self._inputs_per_output is None else needed // self._inputs_per_output
        success = _resize(self._success, size)
        self._needed, self._success = needed, success
    
    @property
    def ack(self) -> int:
        
        """
        Checks whether the Ack flag is set
        
        Args:
            /
            
        Returns:
            ack (int): Ack flag
        """
        
        return self._ack
    
    def set_ack(self) -> None:
       
        """
        Sets the Ack flag
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._ack = 1
        
    def reset_ack(self) -> None:
        
        """
        Resets the Ack flag
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._ack = 0
    
    @property
    def success(self) -> np.ndarray:
        
        """
        Returns the success array
        
        Args:
            /
            
        Returns:
            success (np.ndarray): success array
        """
        
        return self._success
    
    @success.setter
    def success(self, success: np.ndarray) -> None:
        
        """
        TODO add docstring
        """
        
        self._success = _bits(success, len(self._success))
    
    def set_success(self, idx: int) -> None:
        
        """
        Sets the success array at the index
        
        Args:
            idx (int): index to set success array at
            
        Returns:
            /
        """
        
        self._success[idx] = 1
        
    def reset_success(self, idx: int) -> None:
        
        """
        Resets the success array at the index
        
        Args:
            idx (int): index to reset success array at
            
        Returns:
            /
        """
        
        self._success[idx] = 0
    
    @property
    def protocol(self) -> int:
        
        """
        Returns the protocol of this layer
        
        Args:
            /
            
        Returns:
            protocol (int): protocol of this layer
        """
        
        return self._protocol
    
    @protocol.setter
    def protocol(self, protocol: int) -> None:
        
        """
        Sets the protocol for this layer
        
        Args:
            protocol (int): protocol of this layer
            
        Returns:
            /
        """
        
        self._protocol = protocol

class L4_EPP(L4_Protocol):
    
    def __init__(self, src: int, dst: int, requested: int, needed: int, code: str=None) -> None:
        
        """
        TODO add docstring
        """
        
        super().__init__(src, dst, requested, needed, 0)

    def __repr__(self) -> str:
                
        """
        Custom print function
        
        Args:
            /
            
        Returns:
            layer2 (str): str repr of layer 1
        """
        
        return f' | L2: Src {self._src} Dst {self._dst} Req {self._requested} Need {self._needed} Ack {self._ack} Proto {self._protocol} Len {self._header_length} Success {self._success}'

class L4_QECP(L4_Protocol):
    
    def __init__(self, src: int, dst: int, requested: int, needed: int, code: str) -> None:
        
        """
        TODO add docstring
        """
        
        get_qec_code(code)
        self._code = code
        super().__init__(src, dst, requested, needed, 1)
        
        self._code = code #1 byte
        self._complete = False # 1 bit
        self._header_length += 8 + 1  # code ID + completion flag
    
    def __repr__(self) -> str:
                
        """
        Custom print function
        
        Args:
            /
            
        Returns:
            layer2 (str): str repr of layer 1
        """
        
        return f' | L2: Src {self._src} Dst {self._dst} Req {self._requested} Need {self._needed} Code {self._code} Complete {self._complete} Ack {self._ack} Proto {self._protocol} Len {self._header_length} Success {self._success}'
    
    @property
    def code(self):
        
        """
        TODO add docstring
        """
        
        return self._code

    @code.setter
    def code(self, name: str):
        
        """
        TODO add docstring
        """
        
        get_qec_code(name)
        self._code = name

    @property
    def complete(self):
        
        """
        TODO add docstring
        """
        
        return self._complete

    def set_complete(self):
        
        """
        TODO add docstring
        """
        
        self._complete = True
        
    def reset_complete(self):
        
        """
        TODO add docstring
        """
        
        self._complete = False

class L7_Protocol:
    
    """
    Represents a L7 Protocol Header
    
    Attrs:
        _requested (int): number of requested qubits
        _needed (int): number of needed qubits
        _success (int): success array
        _protocol (int): id of current header
        _header_length (int): length of the header
        _payload (list): payload of the packet
    """
    
    def __init__(self, _requested: int, _protocol: int, _payload: Any) -> None:
        
        self._requested = _count(_requested)
        self._protocol = _protocol
        self._header_length = 8 + 8 + 256  # requested + protocol + success (bits)
        self._payload = [] if _payload is None else _payload
        self._success = np.zeros(self._requested, dtype=bool)
    
    def __len__(self) -> int:
        
        # Length in bits of the UTF-8 JSON representation used for timing. No
        # transport serialization is otherwise imposed on simulator packets.
        size = len(json.dumps(self._payload, default=_json_wire_value, separators=(',', ':'),
                              ensure_ascii=False).encode('utf-8')) * 8
        return self._header_length + size
    
    def __repr__(self) -> str:
        
        """
        Custom print function
        
        Args:
            /
            
        Returns:
            layer7 (str): str repr of layer 1
        """
        
        return f' | L7: Req {self._requested} Proto {self._protocol} Len {self._header_length} Payload {self._payload}'
    
    @property
    def requested(self) -> int:
        
        """
        Returns the number of requested qubits
        
        Args:
            /
            
        Returns:
            requested (int): number of requested qubits
        """
        
        return self._requested
    
    @requested.setter
    def requested(self, requested: int) -> None:
        
        """
        TODO add docstring
        """
        
        self._requested = _count(requested)
        self._success = _resize(self._success, self._requested)
    
    @property
    def protocol(self) -> int:
        
        """
        Returns the protocol of this layer
        
        Args:
            /
            
        Returns:
            protocol (int): protocol of this layer
        """
        
        return self._protocol
    
    @protocol.setter
    def protocol(self, protocol: int) -> None:
        
        """
        Sets the protocol for this layer
        
        Args:
            protocol (int): protocol of this layer
            
        Returns:
            /
        """
        
        self._protocol = protocol
    
    @property
    def payload(self) -> List[Any]:
        
        """
        Returns the payload of the packet
        
        Args:
            /
            
        Returns:
            payload (list): payload
        """
        
        return self._payload
    
    @payload.setter
    def payload(self, payload: Any) -> None:
   
        """
        TODO add docstring
        """
   
        self._payload = payload
        
class L7_TPP(L7_Protocol):
    
    def __init__(self, requested: int, payload: Any) -> None:
        
        """
        TODO add docstring
        """
        
        super().__init__(requested, 0, payload)
        
class L7_RGP(L7_Protocol):
    
    def __init__(self, requested: int, payload: Any) -> None:
        
        """
        TODO add docstring
        """
        
        super().__init__(requested, 1, payload)
