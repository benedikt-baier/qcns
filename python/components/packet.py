import sys
import numpy as np
from typing import Any, List

__all__ = ['Packet']

class Packet:
    
    """
    Represents a classical packet
    """
    
    def __init__(self, _l2_src: int, _l2_dst: int, l1_requested: int=0, l1_needed: int=0, l2_requested: int=0, l2_needed: int=0, l3_src: int=None, l3_dst: int=None, mode: int=0, num_channels: int=1, l4_src: int=None, l4_dst: int=None, l4_requested: int=0, l4_needed: int=0, time_stamp: float=0., payload: Any=None, upayload: Any='') -> None:
        
        """
        Instantiates a classical packet tracked and untracked payload with respect to timing update
        
        Args:
            _l2_src (int): L2 source address
            _l2_dst (int): L2 destination address
            l1_requested (int): number of requested qubits for L1 transmission
            l1_needed (int): number of needed qubits for L1 transmission
            l2_requested (int): number of requested qubit pairs for L2 purification
            l2_needed (int): number of needed qubit pairs for L2 purification
            l3_src (int): L3 source address
            l3_dst (int): L3 destination address
            mode (bool): L3 mode of packet, whether to just apply quantum operations or just forward packet
            num_channels (int): number of channels for L3 protocol
            l4_src (int): L4 source port
            l4_dst (int): L4 destination port
            l4_requested (int): number of requested qubits for L4 purification
            l4_needed (int): number of needed qubits for L4 purification
            time_stamp (float): time stamp of packet creation
            payload (Any): tracked payload of packet, has influence on sending time of packet
            upayload (Any): untracked payload of packet, has no influence on sending time of packet
            
        Returns:
            /
        """
        
        # layer 1
        self._is_l1 = 0
        self._l1 = Layer1(l1_requested, l1_needed)
        
        if not l2_needed:
            self._is_l1 = 1
        
        # layer 2
        self._is_l2 = 0
        self._l2 = Layer2(_l2_src, _l2_dst, l2_requested, l2_needed)
        
        if l2_needed:
            self._is_l2 = 1
        
        # layer 3
        self._is_l3 = 0
        self._l3 = ''
        if l3_src is not None or l3_dst is not None:
            self._l3 = Layer3(l3_src, l3_dst, mode, num_channels)
            self._is_l3 = 1

        # layer 4
        self._is_l4 = 0
        self._l4 = ''
        if l4_src is not None or l4_dst is not None:
            self._l4 = Layer4(l4_src, l4_dst, l4_requested, l4_needed)
            self._is_l4 = 1
        
        # layer 7
        self._l7 = Layer7(payload)
        
        # misc
        self._time_stamp = time_stamp   
        self._upayload = upayload
    
    def __len__(self) -> int:
        
        """
        Returns the tracked length of the packet
        
        Args:
            /
            
        Returns:
            length (int): length of packet in bits
        """
        
        return len(self._l1) + len(self._l2) + len(self._l3) + len(self._l4) + len(self._l7)
    
    def __repr__(self) -> str:
        
        """
        Prints the packet
        
        Args:
            /
            
        Returns:
            /
        """
    
        return f'Time Stamp: {self._time_stamp} {str(self._l1)}{str(self._l2)}{str(self._l3)}{str(self._l4)}{str(self._l7)}'
    
    @property
    def is_l1(self) -> bool:
        
        """
        Returns if this packet is purely L1
        
        Args:
            /
            
        Returns:
            l1 (bool): flag if l1 is implemented
        """
        
        return self._is_l1
    
    @property
    def is_l2(self) -> bool:
        
        """
        Returns if this packet is L2
        
        Args:
            /
            
        Returns:
            l2 (bool): flag if l2 is implemented
        """
        
        return self._is_l2
    
    @property
    def is_l3(self) -> bool:
        
        """
        Returns if L3 is implemented in the packet
        
        Args:
            /
            
        Returns:
            l3 (bool): flag if l3 is implemented
        """
        
        return self._is_l3
    
    @property
    def is_l4(self) -> bool:
        
        """
        Returns if L4 is implemented in the packet
        
        Args:
            /
            
        Returns:
            l4 (bool): flag if l4 is implemented
        """
        
        return self._is_l4
    
    @property
    def l1_requested(self) -> int:
        
        """
        Returns the L1 number of requested entanglement
        
        Args:
            /
            
        Returns:
            l1_requested (int): L1 number of requested entanglements 
        """
        
        return self._l1._num_requested
    
    @property
    def l1_needed(self) -> int:
        
        """
        Returns the L1 number of needed entanglement
        
        Args:
            /
            
        Returns:
            l1_needed (int): L1 number of needed entanglements 
        """
        
        return self._l1._num_needed
    
    @property
    def l1_ack(self) -> bool:
        
        """
        Returns the L1 ACK flag
        
        Args:
            /
            
        Returns:
            l1_ack (bool): l1 ack flag
        """
        
        return self._l1._ack
    
    @property
    def l1_ps(self) -> bool:
        
        """
        Whether L1 Photon Source flag is set
        
        Args:
            /
            
        Returns:
            l1_ps (bool): l1 ps flag
        """
        
        return self._l1._ps
    
    @property
    def l1_entanglement_success(self) -> np.array:
        
        """
        Returns the L1 entanglement success
        
        Args:
            /
            
        Returns:
            l1_entanglement_success (np.array): L1 entanglement success
        """
        
        return self._l1._entanglement_success
    
    def set_l1_ack(self) -> None:
        
        """
        Sets the L1 ACK flag
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._l1._ack = 1
        
    def reset_l1_ack(self) -> None:
        
        """
        Resets the L1 ACK flag
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._l1._ack = 0
    
    def set_l1_ps(self) -> None:
        
        """
        Sets the L1 PS flag
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._l1._ps = 1
        
    def reset_l1_ps(self) -> None:
        
        """
        Resets the L1 PS flag
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._l1._ps = 0
    
    def l1_success(self) -> bool:
        
        """
        Checks wether all entanglement attempts are successfull
        
        Args:
            /
            
        Returns:
            l1_success (bool): wether all attempts were successfull
        """
        
        return self._l1._num_requested <= self.num_l1_success()
    
    def num_l1_success(self) -> int:
        
        """
        Counts the number of successfull entanglement attempts
        
        Args:
            /
            
        Returns:
            num_l1_success (int): number of successfull entanglement attempts
        """
        
        return np.count_nonzero(self._l1._entanglement_success)
    
    def num_l1_failures(self) -> int:
        
        """
        Counts the number of failed entanglement attempts
        
        Args:
            /
            
        Returns:
            num_l1_failures (int): number of failed entanglement attempts
        """
        
        return np.count_nonzero(self._l1._entanglement_success==0)
    
    def update_l1_success(self, index: int) -> None:
        
        """
        Updates the value of L1 success array at index
        
        Args:
            index (int): index set to True
            
        Returns:
            /
        """
        
        self._l1._entanglement_success[index] = True
        
    @property
    def l2_src(self) -> int:
        
        """
        Returns the L2 source address
        
        Args:
            /
            
        Returns:
            l2_src (int): l2 source address
        """
        
        return self._l2._src
    
    @property
    def l2_dst(self) -> int:
        
        """
        Returns the L2 destination address
        
        Args:
            /
            
        Returns:
            l2_dst (int): l2 destination address
        """
        
        return self._l2._dst
    
    @property
    def l2_ack(self) -> bool:
        
        """
        Returns wether L2 ack flag is set
        
        Args:
            /
            
        Returns:
            l2_ack (bool): L2 ack flag
        """
        
        return self._l2._ack
    
    @property
    def l2_purification_success(self) -> np.array:
        
        """
        Returns the L2 purification success
        
        Args:
            /
            
        Returns:
            l2_purification_success (np.array): L2 purification success
        """
        
        return self._l2._purification_success
    
    def switch_l2_src_dst(self) -> None:
        
        """
        Switches the L2 source and destination address
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._l2._src, self._l2._dst = self._l2._dst, self._l2._src
    
    def set_l2_ack(self) -> None:
        
        """
        Sets the L2 ACK flag
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._l2._ack = 1
        
    def reset_l2_ack(self) -> None:
        
        """
        Resets the L2 ACK flag
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._l2._ack = 0
    
    def update_l2_success(self, index: int) -> None:
        
        """
        Updates the value of L2 success array at index
        
        Args:
            index (int): index set to True
            
        Returns:
            /
        """
        
        self._l2._purification_success[index] = 1
    
    def compare_l2_purification(self, l2_purification: np.array) -> np.array:
    
        """
        Compares the L2 purification results of this packet with another packet
        
        Args:
            l2_purification (np.array): l2 purification results of other packet
            
        Returns:
            l2_results (np.array): comparison result
        """
        
        return np.logical_not(np.logical_xor(self._l2._purification_success, l2_purification))
    
    @property
    def l3_src(self) -> int:
        
        """
        Returns the L3 source address
        
        Args:
            /
            
        Returns:
            l3_src (int): L3 source address
        """
        
        if not self._is_l3:
            raise ValueError('No Layer 3 in this packet')
        return self._l3._src
    
    @property
    def l3_dst(self) -> int:
        
        """
        Returns the L3 destination address
        
        Args:
            /
            
        Returns:
            l3_dst (int): L3 destination address
        """
        
        if not self._is_l3:
            raise ValueError('No Layer 3 in this packet')
        return self._l3._dst
    
    @property
    def l3_mode(self) -> bool:
        
        """
        Returns the L3 mode
        
        Args:
            /
            
        Returns:
            l3_mode (bool): l3 mode
        """
        
        if not self._is_l3:
            raise ValueError('No Layer 3 in this packet')
        return self._l3._mode
    
    @property
    def l3_num_channels(self) -> int:
        
        """
        Returns the L3 number of channels
        
        Args:
            /
            
        Returns:
            l3_num_channels (int): l3 number of channels
        """
        
        if not self._is_l3:
            raise ValueError('No Layer 3 in this packet')
        return self._l3._num_channels
    
    @property
    def l3_x_count(self) -> np.array:
        
        """
        Returns the L3 X count
        
        Args:
            /
            
        Returns:
            l3_x_count (np.array): L3 X count
        """
        
        return self._l3._x_count
    
    @property
    def l3_z_count(self) -> np.array:
        
        """
        Returns the L3 Z count
        
        Args:
            /
            
        Returns:
            l3_z_count (np.array): L3 Z count
        """
        
        return self._l3._z_count
    
    def switch_l3_src_dst(self) -> None:
        
        """
        Switches L3 source and destination address
        
        Args:
            /
            
        Returns:
            /
        """
        
        if not self._is_l3:
            raise ValueError('No Layer 3 in this packet')
        
        self._l3._src, self._l3._dst = self._l3._dst, self._l3._src
    
    def reset_l3_swapping(self) -> None:
        
        """
        Resets the L3 x count and z count array
        
        Args:
            /
            
        Returns:
            /
        """
        
        if not self._is_l3:
            raise ValueError('No Layer 3 in this packet')
        self._l3.reset()
        
    def update_l3_swapping(self, res: int, channel: int) -> None:
        
        """
        Updates the L3 x count and z count
        
        Args:
            /
        
        Returns:
            /
        """
        
        if not self._is_l3:
            raise ValueError('No Layer 3 in this packet')
        self._l3.update(res, channel)
    
    @property
    def l4_src(self) -> int:
        
        """
        Returns the L4 source port
        
        Args:
            /
            
        Returns:
            l4_src (int): L4 source port
        """
        
        if not self._is_l4:
            raise ValueError('No Layer 4 in this packet')
        return self._l4._src
        
    @property
    def l4_dst(self) -> int:
        
        """
        Returns the L4 destination port
        
        Args:
            /
            
        Returns:
            l4_src (int): L4 destination port
        """
        
        if not self._is_l4:
            raise ValueError('No Layer 4 in this packet')
        return self._l4._dst
    
    @property
    def l4_num_purification(self) -> int:
        
        """
        Returns the number of L2 purification attempts
        
        Args:
            /
            
        Returns:
            l2_num_purification (int): number of L2 purification
        """

        if not self._is_l4:
            raise ValueError('No Layer 4 in this packet')

        return self._l4._num_purification
    
    @property
    def l4_ack(self) -> bool:
        
        """
        Returns wether L4 ACK flag is set
        
        Args:
            /
            
        Returns:
            l4_ack (bool): L4 ACK flag
        """
        
        if not self._is_l4:
            raise ValueError('No Layer 4 in this packet')
        
        return self._l4._ack
    
    def switch_l4_src_dst(self) -> None:
        
        """
        Switches L4 source and destination address
        
        Args:
            /
            
        Returns:
            /
        """
        
        if not self._is_l4:
            raise ValueError('No Layer 4 in this packet')
        
        self._l4._src, self._l4._dst = self._l4._dst, self._l4._src
    
    def set_l4_ack(self) -> None:
        
        """
        Sets the L4 ACK flag
        
        Args:
            /
            
        Returns:
            /
        """
        
        if not self._is_l4:
            raise ValueError('No Layer 4 in this packet')
        
        self._l4._ack = 1
        
    def reset_l4_ack(self) -> None:
        
        """
        Resets the L4 ACK flag
        
        Args:
            /
            
        Returns:
            /
        """
        
        if not self._is_l4:
            raise ValueError('No Layer 4 in this packet')
        
        self._l4._ack = 0
    
    @property
    def payload(self) -> List[Any]:
        
        """
        Return the L7 payload
        
        Args:
            /
            
        Returns:
            l7_payload (list): payload
        """
        
        return self._l7._payload
    
    def reset_payload(self):
        
        """
        Resets the payload of the packet
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._l7._payload = []
    
    def __iter__(self) -> Any:
    
        """
        Iterates over the payload
        
        Args:
            /
            
        Returns:
            item (any): item in payload
        """
        
        for _ in range(len(self.payload)):
            yield self.payload.pop(0)
    
class Layer1:
    
    """
    Represents the Physical Layer (L1) of a packet
    """
    
    def __init__(self, _num_requested: int, _num_needed: int) -> None:
        
        """
        Instantiates a L1 object
        
        Args:
            _num_requested (int): number of requested qubits for L1
            _num_needed (int): number of needed qubits for L1
            
        Retunrs:
            /
        """
        
        self._num_requested = _num_requested
        self._num_needed = _num_needed
        self._entanglement_success = np.zeros(_num_needed, dtype=np.bool_)
        self._ack = 0
        self._ps = 0

    def __len__(self) -> int:
        
        """
        Returns the length of Layer 1
        
        Args:
            /
            
        Returns:
            _len (int): length of Layer 1
        """
        
        return 98

    def __repr__(self) -> str:
        
        """
        Custom print function
        
        Args:
            /
            
        Returns:
            _repr (str): stringified Layer 1
        """
        
        return f'L1: Num Requested: {self._num_requested}, Num: Needed {self._num_needed}, ACK: {self._ack}, PS: {self._ps}, Success: {self._entanglement_success} | '

class Layer2:
    
    """
    Represents the MAC Layer (L2) of a packet
    """
    
    def __init__(self, _src: int, _dst: int, _num_requested: int, _num_needed: int) -> None:
        
        """
        Instantiates a L2 object
        
        Args:
            _src (int): L2 source address
            _dst (int): L2 destination address
            _num_requested (int): number of requested qubit pairs for purification
            _num_needed (int): number of needed qubit pairs for purification
            
        Returns:
            /
        """
        
        self._src = _src
        self._dst = _dst
        self._num_requested = _num_requested
        self._num_needed = _num_needed
        self._purification_success = np.zeros(_num_requested, dtype=np.bool_)
        self._ack = 0
    
    def __len__(self):
        
        """
        Returns the length of Layer 2
        
        Args:
            /
            
        Returns:
            _len (int): length of Layer 2
        """
        
        return 193
        
    def __repr__(self):
        
        """
        Custom print function
        
        Args:
            /
            
        Returns:
            _repr (str): stringified Layer 2
        """
        
        return f'L2: Src: {self._src}, Dst: {self._dst}, Num Requested: {self._num_requested}, Num Needed: {self._num_needed}, Success: {self._purification_success}, ACK: {self._ack} | '
    
class Layer3:
    
    """
    Represents the Network Layer (L3) of a packet
    """
    
    def __init__(self, _src: int, _dst: int, _mode: bool, _num_channels: int=1) -> None:
        
        """
        Instantiates a Layer 3 object
        
        Args:
            _src (int): L3 source address
            _dst (int): L3 destintaion address
            _mode (bool): whether to perform quantum operations or not
            _num_channels (int): number of channels to swap
            
        Returns:
            /
        """
        
        self._src = _src
        self._dst = _dst
        self._mode = _mode
        self._num_channels = _num_channels
        self._swap_success = np.zeros(_num_channels, dtype=np.bool_)
        self._x_count = np.zeros(_num_channels, dtype=np.bool_)
        self._z_count = np.zeros(_num_channels, dtype=np.bool_)

    def __len__(self):
        
        """
        Returns the length of Layer 3
        
        Args:
            /
            
        Returns:
            _len (int): length of Layer 3
        """
        
        return 385
    
    def __repr__(self) -> str:
        
        """
        Custom print function
        
        Args:
            /
            
        Returns:
            _repr (str): stringified Layer 3
        """
        
        return f'L3: Src: {self._src}, Dst: {self._dst}, Mode: {self._mode}, Num: {self._num_channels}, Success: {self._swap_success}, X: {self._x_count}, Z: {self._z_count} | '
    
    def reset(self) -> None:
        
        """
        Resets the x_count and z_count arrays
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._x_count = np.zeros(self._num_channels, dtype=np.bool_)
        self._z_count = np.zeros(self._num_channels, dtype=np.bool_)
        
    def update(self, res: int, channel: int) -> None:
        
        """
        Updates the x_count and z_count of layer 3 given the result and the channel
        
        Args:
            res (int): result of the entanglement swapping
            channel (int): channel of the entanglement swapping
            
        Returns:
            /
        """
        
        if res == 1:
            self._x_count[channel] ^= 1
        if res == 2:
            self._z_count[channel] ^= 1
        if res == 3:
            self._x_count[channel] ^= 1
            self._z_count[channel] ^= 1
    
class Layer4:
    
    """
    Represents the Transport Layer (L4) of a packet
    """
    
    def __init__(self, _src: int, _dst: int, _num_requested: int, _num_needed: int) -> None:
        
        """
        Instantiates a L4 object
        
        Args:
            _src (int): L4 source port
            _dst (int): L4 destination port
            _num_requested (int): number of requested qubits for E2E purification
            _num_needed (int): number of needed qubits for E2E purification
            
        Returns:
            /
        """
        
        self._src = _src
        self._dst = _dst
        self._num_requested = _num_requested
        self._num_needed = _num_needed
        self._purification_success = np.zeros(_num_needed, dtype=np.bool_)
        self._ack = 0

    def __len__(self):
        
        """
        Returns the length of Layer 4
        
        Args:
            /
            
        Returns:
            _len (int): length of Layer 4
        """
        
        return 129
    
    def __repr__(self):
        
        """
        Custom print function
        
        Args:
            /
            
        Returns:
            _repr (str): stringified Layer 4
        """
        
        return f'L4: Src: {self._src}, Dst: {self._dst}, Num Requested: {self._num_requested}, Num Needed: {self._num_needed}, Success: {self._purification_success} | '
    
class Layer7:
    
    """
    Represents the Application Layer (L7) of a packet
    """
    
    def __init__(self, _payload: List[Any]):
        
        """
        Instantiates a L7 object
        
        Args:
            _payload (list): payload of packet
            
        Returns:
            /
        """
        
        self._payload = _payload
        if _payload is None:
            self._payload = []
        
    def __len__(self):
        
        """
        Returns the length of Layer 7
        
        Args:
            /
            
        Returns:
            _len (int): length of Layer 7
        """
        
        return len(self._payload)
    
    def __repr__(self):
        
        """
        Custom print function
        
        Args:
            /
            
        Returns:
            _repr (str): stringified Layer 7
        """
        
        return f'L7: Payload: {self._payload}'
