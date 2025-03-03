
import numpy as np
from typing import List, Tuple, Any

__all__ = ['L1_Protocol', 'L2_Protocol', 'L3_Protocol', 'L4_Protocol', 'L7_Protocol']

class L1_Protocol:
    
    """
    Represents a L1 Protocol Header
    
    Attrs:
        _num_requested (int): number of requested qubits
        _num_needed (int): number of needed qubits
        _success (np.array): success array
        _ack (int): ack flag
        _ps (int): photon source flag
        _id (int): id of the protocol
        _next_id (int): id of the protocol one layer above
        _header_length (int): header length in bits
    """
    
    def __init__(self, requested: int, needed: int) -> None:
        
        """
        Initializes a L1 Protocol Header
        
        Args:
            num_requested (int): number of requested qubits
            num_needed (int): number of needed qubits
        """
        
        self._requested: int = requested # 1byte
        self._needed: int = needed # 1byte
        
        if not self._requested and self._needed > 0:
            self._requested = self._needed
            
        if self._requested > 0 and not self._needed:
            self._needed = self._requested
        
        self._success: np.array = np.zeros(self._needed, dtype=np.bool_) # 2^8 bits = 32 byte
        self._ack: int = 0 # 1 bit
        self._ps: int = 0 # 1 bit
        
        self._protocol: int = 0 # 1byte
        self._next_protocol: int = 0 # 1byte
        self._header_length: int = 290
    
    @property
    def requested(self) -> int:
        
        """
        Returns the number of requested qubits
        
        Args:
            /
            
        Returns:
            num_requested (int): number of requested qubits
        """
        
        return self._requested
    
    @requested.setter
    def requested(self, _requested: int) -> None:
        
        """
        Sets the number of requested qubits
        
        Args:
            _num_requested (int): number of requested qubits
            
        Returns:
            /
        """
        
        self._requested = _requested
    
    @property
    def needed(self) -> int:
        
        """
        Returns the number of needed qubits
        
        Args:
            /
            
        Returns:
            num_needed (int): number of needed qubits
        """
        
        return self._needed
    
    @needed.setter
    def needed(self, _needed: int) -> None:
        
        """
        Sets the number of needed qubits, WARNING resets the success array
        
        Args:
            _num_needed (int): number of needed qubits
            
        Returns:
            /
        """
        
        self._needed = _needed
        self._success = np.zeros(self._needed, dtype=np.bool_)
    
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
       
    @property
    def success(self) -> np.array:
        
        """
        Returns the success array
        
        Args:
            /
            
        Returns:
            success (np.array): success array
        """
        
        return self._success
    
    @success.setter
    def success(self, success: np.array) -> None:
        
        """
        Sets the success array
        
        Args:
            success (np.array): success array
            
        Returns:
            /
        """
        
        self._success = success
        self._needed = len(success)
    
    def set_success(self, _idx: int) -> None:
        
        """
        Sets the success array at the index
        
        Args:
            _idx (int): index to set success array at
            
        Returns:
            /
        """
        
        self._success[_idx] = 1
    
    def reset_success(self, _idx) -> None:
        
        """
        Resets the success array at the index
        
        Args:
            _idx (int): index to set success array at
            
        Returns:
            /
        """
        
        self._success[_idx] = 0
    
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
    def protocol(self, _protocol: int) -> None:
        
        """
        Sets the protocol for this layer
        
        Args:
            _protocol (int): protocol of this layer
            
        Returns:
            /
        """
        
        self._protocol = _protocol
    
    @property
    def next_protocol(self) -> int:
        
        """
        Returns the next protocol of this layer
        
        Args:
            /
            
        Returns:
            next_protocol (int): next protocol of this layer
        """
        
        return self._next_protocol
    
    @next_protocol.setter
    def next_protocol(self, _next_protocol: int) -> None:
        
        """
        Set the next protocol for this layer
        
        Args:
            _next_protocol (int): next protocol of this layer
            
        Returns:
            /
        """
        
        self._next_protocol = _next_protocol
    
    def __len__(self) -> int:
        
        """
        Custom length function for the L1 Protocol
        
        Args:
            /
            
        Returns:
            len (int): length of the L1 Protocol Header
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
        
        return f'L1: Req {self._requested} Need {self._needed} Ack {self._ack} Ps {self._ps} Proto {self._protocol} Next Proto {self._next_protocol} Len {self._header_length} Success {self._success}'
    
class L2_Protocol:
    
    """
    Represents a L2 Protocol Header
    
    Attrs:
        _src (int): source address
        _dst (int): destination address
        _num_requested (int): number of requested qubits
        _num_needed (int): number of needed qubits
        _success (np.array): success array
        _ack (int): Ack flag
        _id (int): id of the protocol header
        _next_id (int): id of the protocol one layer above
        _header_length (int): length of the header
    """
    
    def __init__(self, src: int, dst: int, requested: int, needed: int) -> None:
        
        """
        Initializes a L2 protocol header
        
        Args:
            src (int): source address
            dst (int): destination address
            num_requested (int): number of requested qubits
            num_needed (int): number of needed qubits
            
        Returns:
            /
        """
        
        self._src: int = src # 6byte
        self._dst: int = dst # 6byte
        
        self._requested: int = requested # 1byte
        self._needed: int = needed # 1byte
        
        if not self._requested and self._needed > 0:
            self._requested = self._needed
        
        if self._requested > 0 and not self._needed:
            self._needed = self._requested
        
        self._success: np.array = np.zeros(int(np.floor(self._needed/2)), dtype=np.bool_) # 32 byte
        self._ack: int = 0 # 1 bit
        
        self._protocol: int = 0 # 1byte
        self._next_protocol: int = 0 # 1byte
        self._header_length: int = 385
     
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
            num_requested (int): number of requested qubits
        """
        
        return self._requested
    
    @requested.setter
    def requested(self, _requested: int) -> None:
        
        """
        Sets the number of requested qubits
        
        Args:
            _num_requested (int): number of requested qubits
            
        Returns:
            /
        """
        
        self._requested = _requested
    
    @property
    def needed(self) -> int:
        
        """
        Returns the number of needed qubits
        
        Args:
            /
            
        Returns:
            num_needed (int): number of needed qubits
        """
        
        return self._needed
    
    @needed.setter
    def needed(self, _needed: int) -> None:
        
        """
        Sets the number of needed qubits, WARNING resets the success array
        
        Args:
            _num_needed (int): number of needed qubits
            
        Returns:
            /
        """
        
        self._needed = _needed
        self._success = np.zeros(int(np.floor(self._needed/2)), dtype=np.bool_)
    
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
    def success(self) -> np.array:
        
        """
        Returns the success array
        
        Args:
            /
            
        Returns:
            success (np.array): success array
        """
        
        return self._success

    @success.setter
    def success(self, success: np.array) -> None:
        
        """
        Sets the success array
        
        Args:
            success (np.array): success array
            
        Returns:
            /
        """
        
        self._success = success

    def set_success(self, _idx: int) -> None:
        
        """
        Sets the success array at the index
        
        Args:
            _idx (int): index to set success array at
            
        Returns:
            /
        """
        
        self._success[_idx] = 1
        
    def reset_success(self, _idx: int) -> None:
        
        """
        Resets the success array at the index
        
        Args:
            _idx (int): index to set success array at
            
        Returns:
            /
        """
        
        self._success[_idx] = 0
    
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
    def protocol(self, _protocol: int) -> None:
        
        """
        Sets the protocol for this layer
        
        Args:
            _protocol (int): protocol of this layer
            
        Returns:
            /
        """
        
        self._protocol = _protocol
    
    @property
    def next_protocol(self) -> int:
        
        """
        Returns the next protocol of this layer
        
        Args:
            /
            
        Returns:
            next_protocol (int): next protocol of this layer
        """
        
        return self._next_protocol
    
    @next_protocol.setter
    def next_protocol(self, _next_protocol: int) -> None:
        
        """
        Set the next protocol for this layer
        
        Args:
            _next_protocol (int): next protocol of this layer
            
        Returns:
            /
        """
        
        self._next_protocol = _next_protocol
    
    def __len__(self) -> int:
        
        """
        Custom length function for the L2 Protocol Header
        
        Args:
            /
            
        Returns:
            len (int): length of the Protocol Header
        """
        
        return self._header_length
    
    def __repr__(self) -> str:
        
        """
        Custom print function
        
        Args:
            /
            
        Returns:
            layer2 (str): str repr of layer 1
        """
        
        return f' | L2: Src {self._src} Dst {self._dst} Req {self._requested} Need {self._needed} Ack {self._ack} Proto {self._protocol} Next Proto {self._next_protocol} Len {self._header_length} Success {self._success}'
    
class L3_Protocol:
    
    """
    Represents a L3 Protocol Header
    
    Attrs:
        _src (int): source address
        _dst (int): destination address
        _num_requested (int): number of requested qubits
        _num_needed (int): number of needed qubits
        _success (np.array): success array
        _x_count (np.array): amplitude flip array
        _z_count (np.array): phase flip array
        _mode (int): mode whether to access quantum data plane or simply forward
        _id (int): id of protocol header
        _next_id (int): id of protocol header one layer above
        _header_length (int): header length
    """
    
    def __init__(self, src: int, dst: int, requested: int, needed: int) -> None:
        
        """
        Initializes the L3 Protocol Header
        
        Args:
            src (int): source address
            dst (int): destination address
            num_requested (int): number of requested qubits
            num_needed (int): number of needed qubits
            mode (int): mode whether to access quantum data plane or simply forward
        """
        
        self._src: int = src # 16 byte
        self._dst: int = dst # 16 byte
        
        self._requested: int = requested # 1byte
        self._needed: int = needed # 1byte
        
        if not self._requested and self._needed > 0:
            self._requested =  self._needed
        
        if self._requested > 0 and not self._needed:
            self._needed = self._requested
         
        self._success: np.array = np.zeros(self._needed, dtype=np.bool_) # 32 byte
        self._x_count: np.array = np.zeros(self._needed, dtype=np.bool_) # 32 byte
        self._z_count: np.array = np.zeros(self._needed, dtype=np.bool_) # 32 byte
        
        self._mode: int = 1 # 2 bit
        
        self._protocol: int = 0 # 1byte
        self._next_protocol: int = 0 # 1byte
        self._header_length: int = 1058
        
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
            num_requested (int): number of requested qubits
        """
        
        return self._requested
    
    @requested.setter
    def requested(self, _requested: int) -> None:
        
        """
        Sets the number of requested qubits
        
        Args:
            _num_requested (int): number of requested qubits
            
        Returns:
            /
        """
        
        self._requested = _requested
    
    @property
    def needed(self) -> int:
        
        """
        Returns the number of needed qubits
        
        Args:
            /
            
        Returns:
            num_needed (int): number of needed qubits
        """
        
        return self._needed
    
    @needed.setter
    def needed(self, _needed: int) -> None:
        
        """
        Sets the number of needed qubits, WARNING resets the succes, X and Z array
        
        Args:
            _num_needed (int): number of needed qubits
            
        Returns:
            /
        """
        
        self._needed = _needed
        self._success = np.zeros(self._needed, dtype=np.bool_)
        self._x_count = np.zeros(self._needed, dtype=np.bool_)
        self._z_count = np.zeros(self._needed, dtype=np.bool_)
     
    @property
    def mode(self) -> int:
        
        """
        Returns the mode of the packet
        
        00: classical forwarding
        01: no reject mode
        10: partial reject mode
        11: complete reject mode
        
        Args:
            /
            
        Returns:
            mode (int): quantum data plane mode of the header
        """
        
        return self._mode
    
    def set_cf(self) -> None:
        
        """
        Sets the mode of the header to classical forwarding
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._mode: int = 0
    
    def set_nr(self) -> None:
        
        """
        Sets the mode of the header to no reject mode
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._mode: int = 1
    
    def set_pr(self) -> None:
        
        """
        Sets the mode of the header to partial reject mode
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._mode: int = 2
        
    def set_cr(self) -> None:
        
        """
        Sets the mode of the header to complete reject mode
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._mode: int = 3
    
    @property
    def es_result(self) -> Tuple[np.array, np.array]:
        
        """
        Returns both the X and Z array
        
        Args:
            /
            
        Returns:
            es_result (tuple): tuple containing the X and Z array
        """
        
        return self._x_count, self._z_count
    
    @es_result.setter
    def es_result(self, es_result: Tuple[np.array, np.array]) -> None:
        
        """
        Sets the X and Z array
        
        Args:
            x_count (tuple): X and Z array
            
        Returns:
            /
        """
        
        self._x_count, self._z_count = es_result
    
    def update_es(self, _idx: int, _res: int) -> None:
        
        """
        Updates the X and Z array at the index with the result
        
        Args:
            _idx (int): index to update X and Z array
            _res (int): result to update X and Z array
        """
        
        if _res >> 1:
            self._z_count[_idx] ^= 1
        if _res & 1:
            self._x_count[_idx] ^= 1
    
    def reset_es(self, idx: int) -> None:
        
        """
        Resets the X and Z array
        
        Args:
            /
            
        Returns:
            /
        """
        
        if idx is None:
            self._x_count, self._z_count = np.zeros(self._needed, dtype=np.bool_), np.zeros(self._needed, dtype=np.bool_)
            return
        
        self._x_count[idx] = 0
        self._z_count[idx] = 0
    
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
    def protocol(self, _protocol: int) -> None:
        
        """
        Sets the protocol for this layer
        
        Args:
            _protocol (int): protocol of this layer
            
        Returns:
            /
        """
        
        self._protocol = _protocol
    
    @property
    def next_protocol(self) -> int:
        
        """
        Returns the next protocol of this layer
        
        Args:
            /
            
        Returns:
            next_protocol (int): next protocol of this layer
        """
        
        return self._next_protocol
    
    @next_protocol.setter
    def next_protocol(self, _next_protocol: int) -> None:
        
        """
        Set the next protocol for this layer
        
        Args:
            _next_protocol (int): next protocol of this layer
            
        Returns:
            /
        """
        
        self._next_protocol = _next_protocol
    
    def __len__(self) -> int:
        
        """
        Custom length function
        
        Args:
            /
            
        Returns:
            len (int): header length
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
        
        return f' | L3: Src {self._src} Dst {self._dst} Req {self._requested} Need {self._needed} Mode {self._mode} Proto {self._protocol} Next Proto {self._next_protocol} Len {self._header_length} X {self._x_count} Z {self._z_count}'
    
class L4_Protocol:
    
    """
    Represents a L4 Protocol Header
    
    Attrs:
        _src (int): source port
        _dst (int): destination port
        _num_requested (int): number of requested qubits
        _num_needed (int): number of needed qubits
        _success (np.array): success array
        _ack (int): Ack flag
        _id (int): id of the current header
        _next_id (int): id of the header one layer above
        _header_length (int): header length
    """
    
    def __init__(self, src: int, dst: int, requested: int, needed: int) -> None:
        
        """
        Initializes a L4 Protocol Header
        
        Args:
            src (int): source port
            dst (int): destination port
            num_requested (int): number of requested qubit
            num_needed (int): number of needed qubits
        """
        
        self._src: int = src # 2byte
        self._dst: int = dst # 2byte

        self._requested: int = requested # 1byte
        self._needed: int = needed # 1byte
        
        if not self._requested and self._needed > 0:
            self._requested = self._needed
            
        if self._requested > 0 and not self._needed:
            self._needed = self._requested
        
        self._success: np.array = np.zeros(int(np.floor(self._needed/2)), dtype=np.bool_) # 32byte
        self._ack: int = 0
        
        self._protocol: int = 0 # 1byte
        self._next_protocol: int = 0 # 1byte
        self._header_length: int = 321
        
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
    def src(self, _src: int) -> None:
        
        """
        Sets the source port
        
        Args:
            _src (int): source port
            
        Returns:
            /
        """
        
        self._src = _src
        
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
    def dst(self, _dst: int) -> None:
        
        """
        Sets the destination port
        
        Args:
            _dst (int): destination port
            
        Returns:
            /
        """
        
        self._dst = _dst    
     
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
            num_requested (int): number of requested qubits
        """
        
        return self._requested
    
    @requested.setter
    def requested(self, _requested: int) -> None:
        
        """
        Sets the number of requested qubits:
        
        Args:
            _num_requested (int): number of requested qubits
            
        Returns:
            /
        """
        
        self._requested = _requested
    
    @property
    def needed(self) -> int:
        
        """
        Returns the number of needed qubits
        
        Args:
            /
            
        Returns:
            num_needed (int): number of needed qubits
        """
        
        return self._needed
    
    @needed.setter
    def needed(self, _needed: int) -> None:
        
        """
        Sets the number of needed qubits, WARNING resets the success array
        
        Args:
            _num_needed (int): number of needed qubits
            
        Returns:
            /
        """
        
        self._needed = _needed
        self._success = np.zeros(int(np.floor(self._needed/2)), dtype=np.bool_)  
      
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
    def success(self) -> np.array:
        
        """
        Returns the success array
        
        Args:
            /
            
        Returns:
            success (np.array): success array
        """
        
        return self._success
    
    def set_success(self, _idx: int) -> None:
        
        """
        Sets the success array at the index
        
        Args:
            _idx (int): index to set success array at
            
        Returns:
            /
        """
        
        self._success[_idx] = 1
        
    def reset_success(self, _idx: int) -> None:
        
        """
        Resets the success array at the index
        
        Args:
            _idx (int): index to reset success array at
            
        Returns:
            /
        """
        
        self._success[_idx] = 0
    
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
    def protocol(self, _protocol: int) -> None:
        
        """
        Sets the protocol for this layer
        
        Args:
            _protocol (int): protocol of this layer
            
        Returns:
            /
        """
        
        self._protocol = _protocol
    
    @property
    def next_protocol(self) -> int:
        
        """
        Returns the next protocol of this layer
        
        Args:
            /
            
        Returns:
            next_protocol (int): next protocol of this layer
        """
        
        return self._next_protocol
    
    @next_protocol.setter
    def next_protocol(self, _next_protocol: int) -> None:
        
        """
        Set the next protocol for this layer
        
        Args:
            _next_protocol (int): next protocol of this layer
            
        Returns:
            /
        """
        
        self._next_protocol = _next_protocol
    
    def __len__(self) -> int:
        
        """
        Custom length function
        
        Args:
            /
            
        Returns:
            len (int): length of the header
        """
        
        return self._header_length
    
    def __repr__(self) -> str:
        
        """
        Custom print function
        
        Args:
            /
            
        Returns:
            layer4 (str): str repr of layer 1
        """
        
        return f' | L4: Src {self._src} Dst {self._dst} Req {self._requested} Need {self._needed} Ack {self._ack} Proto {self._protocol} Next Proto {self._next_protocol} Len {self._header_length} Success {self._success}'
     
class L7_Protocol:
    
    """
    Represents a L7 Protocol Header
    
    Attrs:
        _num_requested (int): number of requested qubits
        _num_needed (int): number of needed qubits
        _success (int): success array
        _id (int): id of current header
        _header_length (int): length of the header
        _payload (list): payload of the packet
    """
    
    def __init__(self, requested: int, payload: List[Any]) -> None:
        
        """
        Initializes a L7 Protocol Header
        
        Args:
            num_requested (int): number of requested qubits
            num_needed (int): number of needed qubits
            
        Returns:
            /
        """
        
        self._requested: int = requested # 1 byte
        
        self._success: np.array = np.zeros(self._requested, dtype=np.bool_) # 32byte
        
        self._protocol: int = 0 # 1byte
        self._header_length: int = 280
        
        self._payload: List[Any] = payload
        if payload is None:
            self._payload: List[Any] = []
        if not isinstance(payload, list):
            self._payload: List[Any] = [payload]
    
    @property
    def requested(self) -> int:
        
        """
        Returns the number of requested qubits
        
        Args:
            /
            
        Returns:
            num_requested (int): number of requested qubits
        """
        
        return self._requested
    
    @requested.setter
    def requested(self, _requested: int) -> None:
        
        """
        Sets the number of requested qubits
        
        Args:
            num_requested (int): number of requested qubits
            
        Returns:
            /
        """
        
        self._requested = _requested
    
    @property
    def success(self) -> np.array:
        
        """
        Returns the success array
        
        Args:
            /
            
        Returns:
            success (np.array): success array
        """
        
        return self._success
    
    def set_success(self, _idx: int) -> None:
        
        """
        Sets the success array at the index
        
        Args:
            _idx (int): index to set array at
            
        Returns:
            /
        """
        
        self._success[_idx] = 1
        
    def reset_success(self, _idx: int) -> None:
        
        """
        Resets the success array at the index
        
        Args:
            _idx (int): index to reset array at
        """
        
        self._success[_idx] = 0
    
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
    def protocol(self, _protocol: int) -> None:
        
        """
        Sets the protocol for this layer
        
        Args:
            _protocol (int): protocol of this layer
            
        Returns:
            /
        """
        
        self._protocol = _protocol
    
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
    
    def __len__(self) -> int:
        
        """
        Custom len function
        
        Returns:
            len (int): header length + length of payload
        """
        
        return self._header_length + len(self._payload)
    
    def __repr__(self) -> str:
        
        """
        Custom print function
        
        Args:
            /
            
        Returns:
            layer7 (str): str repr of layer 1
        """
        
        return f' | L7: Req {self._requested} Proto {self._protocol} Len {self._header_length} Success {self._success}'