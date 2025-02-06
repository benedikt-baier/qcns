import numpy as np
from typing import List, Tuple, Union, Any

from python.components.protocol import *

__all__ = ['Packet']

class Protocol:
    
    pass

class Packet:
    
    """
    Represents a packet
    
    Attrs:

        _layer_counter (int): count which of the layers are present in the packet
        _layer1 (L1_Protocol): Physical Layer should always be present, as its needed for forwarding the packet
        _layer2 (L2_Protocol): Link Layer used for resource allocation/access control, only the MAC adresses are always needed
        _layer3 (L3_Protocol): Network/IP Layer used to forward packets through a network
        _layer4 (L4_Protocol): Transport Layer used for End-to-End control of packets
        _layer7 (L7_Protocol): Application Layer used for application specific data, contains the payload of the packet
    """
    
    def __base_init(self, l2_src: int, l2_dst: int,
                      l1_num_requested: int=1, l1_num_needed: int=1,
                      l2_num_requested: int=None, l2_num_needed: int=None,
                      l3_src: int=None, l3_dst: int=None, l3_num_requested: int=None, l3_num_needed: int=None,
                      l4_src: int=None, l4_dst: int=None, l4_num_requested: int=None, l4_num_needed: int=None,
                      l7_num_requested: int=None, l7_num_needed: int=None, payload: List[Any]=None) -> None:
        
        """
        Initializes the packet with standard inputs such as L2 src and dst
        
        Args:
            l2_src (int): L2 source address
            l2_dst (int): L2 destination address
            l1_num_requested (int): L1 number of requested qubits
            l1_num_needed (int): L1 number of needed qubits
            l2_num_requested (int): L2 number of requested qubits
            l2_num_needed (int): L2 number of needed qubits
            l3_src (int): L3 source address
            l3_dst (int): L3 destination address
            l3_num_requested (int): L3 number of requested qubits
            l3_num_needed (int): L3 number of needed qubits
            l4_src (int): L4 source port
            l4_dst (int): L4 destination port
            l4_num_requested (int): L4 number of requested qubits
            l4_num_needed (int): L4 number of needed qubits
            l7_num_requested (int): L7 number of requested qubits
            l7_num_needed (int): L7 number of needed qubits
            
        Returns:
            /
        """
        
        self._layer1: L1_Protocol = L1_Protocol(l1_num_requested, l1_num_needed)
        self._layer2: L2_Protocol = L2_Protocol(l2_src, l2_dst, l2_num_requested, l2_num_needed)
        self._layer3: L3_Protocol = ''
        self._layer4: L4_Protocol = ''
        self._layer7: L7_Protocol = ''
        
        if l2_num_requested is not None:
            self._layer_counter = 1
        
        if l3_src is not None or l3_dst is not None:
            self._layer_counter = 2
        
        if l4_src is not None or l4_dst is not None:
            self._layer_counter = 3
        
        if l7_num_requested is not None or payload is not None:
            self._layer_counter = 4
        
        if self._layer_counter > 1:
            self._layer3 = L3_Protocol(l3_src, l3_dst, l3_num_requested, l3_num_needed)
            self._layer2.next_protocol = self._layer3.protocol
            
        if self._layer_counter > 2:
            self._layer4: L4_Protocol = L4_Protocol(l4_src, l4_dst, l4_num_requested, l4_num_needed)
            self._layer3.next_protocol = self._layer4.protocol
        
        if self._layer_counter > 3:
            self._layer7: L7_Protocol = L7_Protocol(l7_num_requested, l7_num_needed, payload)
            self._layer4.next_protocol = self._layer7.protocol
    
    def __derived_init(self, layer1: L1_Protocol, layer2: L2_Protocol, layer3: L3_Protocol='', 
                     layer4: L4_Protocol='', layer7: L7_Protocol='') -> None:
        
        """
        Initializes the packet with each layer
        
        Args:
            layer1 (L1_Protocol): base or derived L1_Protocol class, represents the first layer
            layer2 (L2_Protocol): base or derived L2_Protocol class, represents the second layer
            layer3 (L3_Protocol): base or derived L3_Protocol class, represents the third layer
            layer4 (L4_Protocol): base or derived L4_Protocol class, represents the fourth layer
            layer7 (L7_Protocol): base or derived L7_Protocol class, represents the seventh layer
            
        Returns:
            /
        """
        
        self._layer1: L1_Protocol = layer1
        
        self._layer2: L2_Protocol = layer2
        if self._layer2.num_requested is not None:
            self._layer1.next_protocol = self._layer2.protocol
            self._layer_counter = 1
            
        self._layer3: L3_Protocol = layer3
        if self._layer3:
            self._layer2.next_protocol = self._layer3.protocol
            self._layer_counter = 2
        
        self._layer4: L4_Protocol = layer4
        if self._layer4:
            self._layer3.next_protocol = self._layer4.protocol
            self._layer_counter = 3
        
        self._layer7: L7_Protocol = layer7
        if self._layer7:
            self._layer4.next_protocol = self._layer7.protocol
            self._layer_counter = 4
    
    def __init__(self, *args: Union[int, Protocol], **kwargs: Union[int, Protocol]) -> None:
        
        """
        Initializes a packet
        
        Args:
            *args (int/Protocol): args matching __base_init or __derived_init
            **kwargs (int/Protocol): kwargs matching __base_init or __derived_init
            
        Returns:
            /
        """
        
        self._layer_counter: int = 0
        
        if args and isinstance(args[0], L1_Protocol):
            self.__derived_init(*args, **kwargs)
            return
        
        if kwargs and list(kwargs.keys())[0] == 'layer1':
            self.__derived_init(*args, **kwargs)
            return
        
        self.__base_init(*args, **kwargs)
        
    def __len__(self) -> int:
        
        """
        Returns the tracked length of the packet
        
        Args:
            /
            
        Returns:
            length (int): length of packet in bits
        """
        
        return len(self._layer1) + len(self._layer2) + len(self._layer3) + len(self._layer4) + len(self._layer7)
    
    def __repr__(self) -> str:
        
        """
        Custom print function
        
        Args:
            /
            
        Returns:
            packet (str): packet in str repr
        """
        
        return f'{self._layer1}{self._layer2}{self._layer3}{self._layer4}{self._layer7}'
     
    @property
    def is_l1(self) -> bool:
        
        """
        Checks whether the packet is purely L1
        
        Args:
            /
            
        Returns:
            is_l1 (bool): whether the packet is purely L1
        """
        
        return self._layer_counter == 0
    
    @property
    def is_l2(self) -> bool:
        
        """
        Checks whether the packet is purely L2
        
        Args:
            /
            
        Returns:
            is_l2 (bool): whether the packet is purely L2
        """
        
        return self._layer_counter == 1
    
    @property
    def is_l3(self) -> bool:
        
        """
        Checks whether the packet is purely L3
        
        Args:
            /
            
        Returns:
            is_l3 (bool): whether the packet is purely L3
        """
        
        return self._layer_counter == 2
    
    @property
    def is_l4(self) -> bool:
        
        """
        Checks whether the packet is purely L4
        
        Args:
            /
            
        Returns:
            is_l4 (bool): whether the packet is purely L4
        """
        
        return self._layer_counter == 3
    
    @property
    def is_l7(self) -> bool:
        
        """
        Checks whether the packet is purely L7
        
        Args:
            /
            
        Returns:
            is_l7 (bool): whether the packet is purely L7
        """
        
        return self._layer_counter == 4
    
    @property
    def has_l1(self) -> bool:
        
        """
        Checks whether the packet has L1
        
        Args:
            /
            
        Returns:
            has_l1 (bool): whether the packet L1
        """
        
        return True
    
    @property
    def has_l2(self) -> bool:
        
        """
        Checks whether the packet has L2
        
        Args:
            /
            
        Returns:
            has_l2 (bool): whether the packet L2
        """
        
        return self._layer_counter > 0
        
    @property
    def has_l3(self) -> bool:
        
        """
        Checks whether the packet has L3
        
        Args:
            /
            
        Returns:
            has_l3 (bool): whether the packet L3
        """
        
        return self._layer_counter > 1
    
    @property
    def has_l4(self) -> bool:
        
        """
        Checks whether the packet has L4
        
        Args:
            /
            
        Returns:
            has_l4 (bool): whether the packet L4
        """
        
        return self._layer_counter > 2
    
    @property
    def has_l7(self) -> bool:
        
        """
        Checks whether the packet has L7
        
        Args:
            /
            
        Returns:
            has_l7 (bool): whether the packet L7
        """
        
        return self._layer_counter > 3
    
    # Layer 1
    
    @property
    def l1_num_requested(self) -> int:
        
        """
        Returns the number requested qubits on L1
        
        Args:
            /
            
        Returns:
            l1_num_requested (int): number of requested qubits on L1
        """
        
        return self._layer1.num_requested
    
    @l1_num_requested.setter
    def l1_num_requested(self, _num_requested: int) -> None:
        
        """
        Sets the number of requested qubits on L1
        
        Args:
            _num_requested (int): number of requested qubits to set
            
        Returns:
            /
        """
        
        self._layer1.num_requested = _num_requested
        
    @property
    def l1_num_needed(self) -> int:
        
        """
        Returns the number needed qubits on L1
        
        Args:
            /
            
        Returns:
            l1_num_needed (int): number of needed qubits on L1
        """
        
        return self._layer1.num_needed
    
    @l1_num_needed.setter
    def l1_num_needed(self, _num_needed: int) -> None:
        
        """
        Sets the number of needed qubits on L1, WARNING resets the L1 success array
        
        Args:
            _num_needed (int): number of needed qubits to set
            
        Returns:
            /
        """
        
        self._layer1.num_needed = _num_needed
        
    @property
    def l1_ack(self) -> int:
        
        """
        Checks whether the L1 Ack flag is set
        
        Args:
            /
            
        Returns:
            l1_ack (int): L1 Ack flag
        """
        
        return self._layer1.ack
    
    def l1_set_ack(self) -> None:
        
        """
        Sets the L1 Ack flag
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._layer1.set_ack()
        
    def l1_reset_ack(self) -> None:
        
        """
        Resets the L1 Ack flag
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._layer1.reset_ack()
        
    @property
    def l1_ps(self) -> int:
        
        """
        Checks whether the L1 Photon Source flag is set
        
        Args:
            /
            
        Returns:
            /
        """
        
        return self._layer1.ps
    
    def l1_set_ps(self) -> None:
        
        """
        Sets the Photon Source flag
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._layer1.set_ps()
        
    def l1_reset_ps(self) -> None:
        
        """
        Resets the Photon Source flag
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._layer1.reset_ps()
        
    @property
    def l1_success(self) -> np.array:
        
        """
        Returns the L1 success array
        
        Args:
            /
            
        Returns:
            l1_success (np.array): L1 success array
        """
        
        return self._layer1.success

    @l1_success.setter
    def l1_success(self, l1_success: np.array) -> None:
        
        """
        Sets the L1 success array
        
        Args:
            l1_success (np.array): L1 success array
            
        Returns:
            /
        """
        
        self._layer1.success = l1_success

    def l1_set_success(self, index: int) -> None:
        
        """
        Sets the L1 success array the specified index
        
        Args:
            index (int): index to set L1 success array at
            
        Returns:
            /
        """
        
        self._layer1.set_success(index)
        
    def l1_reset_success(self, index: int) -> None:
        
        """
        Resets the L1 success array at the index
        
        Args:
            index (int): index to reset L1 success array at
            
        Returns:
            /
        """
        
        self._layer1.reset_success(index)
    
    @property
    def l1_protocol(self) -> int:
        
        """
        Returns the L1 protocol
        
        Args:
            /
            
        Returns:
            l1_protocol (int): L1 protocol
        """
        
        return self._layer1.protocol
    
    @l1_protocol.setter
    def l1_protocol(self, l1_protocol: int) -> None:
        
        """
        Sets the L1 protocol
        
        Args:
            l1_protocol (int): L1 protocol
            
        Returns:
            /
        """
        
        self._layer1.protocol = l1_protocol
        
    @property
    def l1_next_protocol(self) -> int:
        
        """
        Returns the L1 next protocol
        
        Args:
            /
            
        Returns:
            l1_next_protocol (int): L1 next protocol
        """
        
        return self._layer1.next_protocol
    
    @l1_next_protocol.setter
    def l1_next_protocol(self, l1_next_protocol: int) -> None:
        
        """
        Sets the L1 next protocol
        
        Args:
            l1_next_protocol (int): L1 next protocol
            
        Returns:
            /
        """
        
        self._layer1.next_protocol = l1_next_protocol
    
    # Layer 2
    
    @property
    def l2_src(self) -> int:
        
        """
        Returns the L2 source address
        
        Args:
            /
            
        Returns:
            l2_src (int): L2 source address
        """
        
        return self._layer2.src
    
    @l2_src.setter
    def l2_src(self, l2_src: int) -> None:
        
        """
        Sets the L2 source address
        
        Args:
            l2_src (int): L2 source address to set
            
        Returns:
            /
        """
        
        self._layer2.src = l2_src
        
    @property
    def l2_dst(self) -> int:
        
        """
        Returns the L2 destination address
        
        Args:
            /
            
        Returns:
            l2_dst (int) L2 destination address
        """
        
        return self._layer2.dst
    
    @l2_dst.setter
    def l2_dst(self, l2_dst: int) -> None:
        
        """
        Sets the L2 destination address
        
        Args:
            l2_dst (int): L2 destination address
            
        Returns:
            /
        """
        
        self._layer2.dst = l2_dst
    
    def l2_switch_src_dst(self) -> None:
        
        """
        Switches the L2 source and destination address
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._layer2.switch_src_dst()
      
    @property
    def l2_num_requested(self) -> int:
        
        """
        Returns the L2 number of requested qubits
        
        Args:
            /
            
        Returns:
            l2_num_requested (int): number of L2 requested qubits
        """
        
        return self._layer2.num_requested
    
    @l2_num_requested.setter
    def l2_num_requested(self, _num_requested: int) -> None:
        
        """
        Sets the L2 number of requested qubits
        
        Args:
            _num_requested (int): number of L2 requested qubits to set
            
        Returns:
            /
        """
        
        self._layer2.num_requested = _num_requested
        
    @property
    def l2_num_needed(self) -> int:
        
        """
        Returns the L2 number of needed qubits
        
        Args:
            /
            
        Returns:
            l2_num_needed (int): L2 number of needed qubits
        """
        
        return self._layer2.num_needed
    
    @l2_num_needed.setter
    def l2_num_needed(self, _num_needed: int) -> None:
        
        """
        Sets the L2 number of needed qubits, WARNING resets the L2 success array
        
        Args:
            _num_needed (int): L2 number of needed qubits to set
            
        Returns:
            /
        """
        
        self._layer2.num_needed = _num_needed
        
    @property
    def l2_ack(self) -> int:
        
        """
        Returns whether the L2 Ack is set
        
        Args:
            /
            
        Returns:
            l2_ack (int): L2 ack
        """
        
        return self._layer2.ack
    
    def l2_set_ack(self) -> None:
        
        """
        Sets the L2 Ack
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._layer2.set_ack()
        
    def l2_reset_ack(self) -> None:
        
        """
        Resets the L2 Ack
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._layer2.reset_ack()
          
    @property
    def l2_success(self) -> np.array:
        
        """
        Returns the L2 success array
        
        Args:
            /
            
        Returns:
            l2_success (np.array): L2 success array
        """
        
        return self._layer2.success

    @l2_success.setter
    def l2_success(self, l2_success: np.array) -> None:
        
        """
        Sets the L2 success array
        
        Args:
            l2_success (np.array): L2 success array

        Returns:
            /
        """
        
        self._layer2.success = l2_success

    def l2_set_success(self, index: int) -> None:
        
        """
        Sets the L2 success array the index
        
        Args:
            index (int): index to set L2 array
            
        Returns:
            /
        """
        
        self._layer2.set_success(index)
        
    def l2_reset_success(self, index: int) -> None:
        
        """
        Resets the L2 success array at the index
        
        Args:
            index (int): index to reset L2 success array at
            
        Returns:
            /
        """
        
        self._layer2.reset_success(index)
    
    @property
    def l2_protocol(self) -> int:
        
        """
        Returns the L2 protocol
        
        Args:
            /
            
        Returns:
            l2_protocol (int): L2 protocol
        """
        
        return self._layer2.protocol
    
    @l2_protocol.setter
    def l2_protocol(self, l2_protocol: int) -> None:
        
        """
        Sets the L2 protocol
        
        Args:
            l2_protocol (int): L2 protocol
            
        Returns:
            /
        """
        
        self._layer2.protocol = l2_protocol
        
    @property
    def l2_next_protocol(self) -> int:
        
        """
        Returns the L2 next protocol
        
        Args:
            /
            
        Returns:
            l2_next_protocol (int): L2 next protocol
        """
        
        return self._layer2.next_protocol
    
    @l2_next_protocol.setter
    def l2_next_protocol(self, l2_next_protocol: int) -> None:
        
        """
        Sets the L2 next protocol
        
        Args:
            l2_next_protocol (int): L2 next protocol
            
        Returns:
            /
        """
        
        self._layer2.next_protocol = l2_next_protocol
        
    # Layer 3
    
    @property
    def l3_src(self) -> int:
        
        """
        Returns the L3 source address
        
        Args:
            /
            
        Returns:
            l3_src (int): L3 source address
        """
        
        return self._layer3.src
    
    @l3_src.setter
    def l3_src(self, l3_src: int) -> None:
        
        """
        Sets the L3 source address
        
        Args:
            l3_src (int): L3 source address
            
        Returns:
            /
        """
        
        self._layer3.src = l3_src
        
    @property
    def l3_dst(self) -> int:
        
        """
        Returns the L3 destination address
        
        Args:
            /
            
        Returns:
            l3_dst (int): L3 destination address
        """
        
        return self._layer3.dst
    
    @l3_dst.setter
    def l3_dst(self, l3_dst) -> None:
        
        """
        Sets the L3 destination address
        
        Args:
            l3_dst (int): L3 destination address
            
        Returns:
            /
        """
        
        self._layer3.dst = l3_dst
    
    def l3_switch_src_dst(self) -> None:
        
        """
        Switches the L3 source and destination address
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._layer3.switch_src_dst()
    
    @property
    def l3_num_requested(self) -> int:
        
        """
        Returns the L3 number of requested qubits
        
        Args:
            /
            
        Returns:
            l3_num_requested (int): L3 number of requested qubits
        """
        
        return self._layer3.num_requested
    
    @l3_num_requested.setter
    def l3_num_requested(self, _num_requested: int) -> None:
        
        """
        Sets the L3 number of requested qubits
        
        Args:
            l3_num_requested (int): L3 number of requested qubits
            
        Returns:
            /
        """
        
        self._layer3.num_requested = _num_requested
        
    @property
    def l3_num_needed(self) -> int:
        
        """
        Returns the L3 number of needed qubits
        
        Args:
            /
            
        Returns:
            l3_num_needed (int): L3 number of needed qubits
        """
        
        return self._layer3.num_needed
    
    @l3_num_needed.setter
    def l3_num_needed(self, _num_needed: int) -> None:
        
        """
        Sets the L3 number of needed qubits, WARNING resets the L3 success, X and Z array
        
        Args:
            _num_needed (int): L3 number of needed qubits
            
        Returns:
            /
        """
        
        self._layer3.num_needed = _num_needed
        
    @property
    def l3_mode(self) -> int:
        
        """
        Checks whether the L3 mode flag is set
        
        Args:
            /
            
        Returns:
            l3_mode (int): L3 mode
        """
        
        return self._layer3.mode
    
    def l3_set_mode(self) -> None:
        
        """
        Sets the L3 mode flag
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._layer3.set_mode()
        
    def l3_reset_mode(self) -> None:
        
        """
        Resets the L3 mode flag
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._layer3.reset_mode()
    
    @property
    def l3_es_result(self) -> Tuple[np.array, np.array]:
        
        """
        Returns both the L3 X and Z array
        
        Args:
            /
            
        Returns:
            es_result (tuple): tuple containing L3 X and Z array
        """
        
        return self._layer3.es_result
    
    @l3_es_result.setter
    def l3_es_result(self, l3_es_result: np.array) -> None:
        
        """
        Sets the X and Z array
        
        Args:
            l3_es_result (np.array): X and Z array
            
        Returns:
            /
        """
        
        self._layer3.es_result = l3_es_result
    
    def l3_update_es(self, index: int, result: int) -> None:
        
        """
        Updates the L3 X and Z array at the index with the result
        
        Args:
            index (int): index to update X and Z array
            result (int): result to update X and Z array with
            
        Returns:
            /
        """
        
        self._layer3.update_es(index, result)
    
    def l3_reset_es(self) -> None:
        
        """
        Resets the L3 X and Z array
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._layer3.reset_es()
    
    @property
    def l3_protocol(self) -> int:
        
        """
        Returns the L3 protocol
        
        Args:
            /
            
        Returns:
            l3_protocol (int): L3 protocol
        """
        
        return self._layer3.protocol
    
    @l3_protocol.setter
    def l3_protocol(self, l3_protocol: int) -> None:
        
        """
        Sets the L3 protocol
        
        Args:
            l3_protocol (int): L3 protocol
            
        Returns:
            /
        """
        
        self._layer3.protocol = l3_protocol
        
    @property
    def l3_next_protocol(self) -> int:
        
        """
        Returns the L3 next protocol
        
        Args:
            /
            
        Returns:
            l3_next_protocol (int): L3 next protocol
        """
        
        return self._layer3.next_protocol
    
    @l3_next_protocol.setter
    def l3_next_protocol(self, l3_next_protocol: int) -> None:
        
        """
        Sets the L3 next protocol
        
        Args:
            l3_next_protocol (int): L3 next protocol
            
        Returns:
            /
        """
        
        self._layer3.next_protocol = l3_next_protocol
    
    # Layer 4
    
    @property
    def l4_src(self) -> int:
        
        """
        Returns the L4 source port
        
        Args:
            /
            
        Returns:
            l4_src (int): L4 source port
        """
        
        return self._layer4.src
    
    @l4_src.setter
    def l4_src(self, l4_src: int) -> None:
        
        """
        Sets the L4 source port
        
        Args:
            l4_src (int): L4 source port
            
        Returns:
            /
        """
        
        self._layer4.src = l4_src
        
    @property
    def l4_dst(self) -> int:
        
        """
        Returns the L4 destination port
        
        Args:
            /
            
        Returns:
            l4_dst (int): L4 destination port
        """
        
        return self._layer4.dst
    
    @l4_dst.setter
    def l4_dst(self, l4_dst: int) -> None:
        
        """
        Sets the L4 destination port
        
        Args:
            l4_dst (int): L4 destination port
            
        Returns:
            /
        """
        
        self._layer4.dst = l4_dst
        
    def l4_switch_src_dst(self) -> None:
        
        """
        Switches the L4 source and destination port
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._layer4.switch_src_dst()
        
    @property
    def l4_num_requested(self) -> int:
        
        """
        Returns the L4 number of requested qubits
        
        Args:
            /
            
        Returns:
            l4_num_requested (int): L4 number of requested qubits
        """
        
        return self._layer4.num_requested
    
    @l4_num_requested.setter
    def l4_num_requested(self, _num_requested: int) -> None:
        
        """
        Sets the L4 number of requested qubits
        
        Args:
            _num_requested (int): L4 number of requested qubits
            
        Returns:
            /
        """
        
        self._layer4.num_requested = _num_requested
        
    @property
    def l4_num_needed(self) -> int:
        
        """
        Returns the L4 number of needed qubits
        
        Args:
            /
            
        Returns:
            l4_num_needed (int): number of needed qubits
        """
        
        return self._layer4.num_needed
    
    @l4_num_needed.setter
    def l4_num_needed(self, _num_needed: int) -> None:
        
        """
        Sets the L4 number of needed qubits, WARNING resets the L4 success array
        
        Args:
            _num_needed (int): L4 number of needed qubits
            
        Returns:
            /
        """
        
        self._layer4.num_needed = _num_needed
        
    @property
    def l4_ack(self) -> int:
        
        """
        Checks whether the L4 Ack flag is set
        
        Args:
            /
            
        Returns:
            l4_ack (int): L4 Ack flag
        """
        
        return self._layer4.ack
    
    def l4_set_ack(self) -> None:
        
        """
        Sets the L4 Ack flag
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._layer4.set_ack()
        
    def l4_reset_ack(self) -> None:
        
        """
        Resets the L4 Ack flag
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._layer4.reset_ack()
    
    @property
    def l4_success(self) -> np.array:
        
        """
        Returns the L4 success array
        
        Args:
            /
            
        Returns:
            l4_success (np.array): L4 success array
        """
        
        return self._layer4.success

    def l4_set_success(self, index: int) -> None:
        
        """
        Sets the L4 sccess array at the index
        
        Args:
            index (int): index to set L4 success array at
            
        Returns:
            /
        """
        
        self._layer4.set_success(index)
        
    def l4_reset_success(self, index: int) -> None:
        
        """
        Resets the L4 sccess array at the index
        
        Args:
            index (int): index to set L4 success array at
            
        Returns:
            /
        """
        
        self._layer4.reset_success(index)
     
    @property
    def l4_protocol(self) -> int:
        
        """
        Returns the L4 protocol
        
        Args:
            /
            
        Returns:
            l4_protocol (int): L4 protocol
        """
        
        return self._layer4.protocol
    
    @l4_protocol.setter
    def l4_protocol(self, l4_protocol: int) -> None:
        
        """
        Sets the L4 protocol
        
        Args:
            l4_protocol (int): L4 protocol
            
        Returns:
            /
        """
        
        self._layer4.protocol = l4_protocol
        
    @property
    def l4_next_protocol(self) -> int:
        
        """
        Returns the L4 next protocol
        
        Args:
            /
            
        Returns:
            l4_next_protocol (int): L4 next protocol
        """
        
        return self._layer4.next_protocol
    
    @l4_next_protocol.setter
    def l4_next_protocol(self, l4_next_protocol: int) -> None:
        
        """
        Sets the L4 next protocol
        
        Args:
            l4_next_protocol (int): L4 next protocol
            
        Returns:
            /
        """
        
        self._layer4.next_protocol = l4_next_protocol 
     
    # Layer 7
    
    @property
    def l7_num_requested(self) -> int:
        
        """
        Returns the L7 number of requested qubits
        
        Args:
            /
            
        Returns:
            l7_num_requested (int): L7 number of requested qubits
        """
        
        return self._layer7.num_requested
    
    @l7_num_requested.setter
    def l7_num_requested(self, _num_requested: int) -> None:
        
        """
        Sets the L7 number of requested qubits
        
        Args:
            _num_requested (int): L7 number of requested qubits
            
        Returns:
            /
        """
        
        self._layer7.num_requested = _num_requested
        
    @property
    def l7_num_needed(self) -> int:
        
        """
        Returns the L7 number of needed qubits
        
        Args:
            /
            
        Returns:
            l7_num_needed (int): L7 number of needed qubits
        """
        
        return self._layer7.num_needed
    
    @l7_num_needed.setter
    def l7_num_needed(self, _num_needed: int) -> None:
        
        """
        Setst the L7 number of needed qubits, WARNING resets the L7 success array
        
        Args:
            _num_needed (int): L7 number of needed qubits
            
        Returns:
            /
        """
        
        self._layer7.num_needed = _num_needed
        
    @property
    def l7_success(self) -> np.array:
        
        """
        Returns the L7 success arrayk
        
        Args:
            /
            
        Returns:
            l7_success (np.array): L7 success array
        """
        
        return self._layer7.success

    def l7_set_success(self, index: int) -> None:
        
        """
        Sets the L7 success array at the index
        
        Args:
            index (int): index to set L7 success array at
            
        Returns:
            /
        """
        
        self._layer7.set_success(index)
        
    def l7_reset_success(self, index: int) -> None:
        
        """
        Resets the L7 success array at the index
        
        Args:
            index (int): index to set L7 success array at
            
        Returns:
            /
        """
        
        self._layer7.reset_success(index)
    
    @property
    def l7_protocol(self) -> int:
        
        """
        Returns the L7 protocol
        
        Args:
            /
            
        Returns:
            l7_protocol (int): L7 protocol
        """
        
        return self._layer7.protocol
    
    @l7_protocol.setter
    def l7_protocol(self, l7_protocol: int) -> None:
        
        """
        Sets the L7 protocol
        
        Args:
            l7_protocol (int): L7 protocol
            
        Returns:
            /
        """
        
        self._layer7.protocol = l7_protocol
    
    @property
    def payload(self) -> List[Any]:
        
        """
        Returns the L7 payload
        
        Args:
            /
            
        Returns:
            payload (list): L7 payload
        """
        
        return self._layer7.payload
    
    def __iter__(self) -> Any:
        
        """
        Iterator for the L7 payload
        
        Args:
            /
            
        Returns:
            item (any): current item that is in the payload
        """
        
        for item in self.payload:
            yield item
