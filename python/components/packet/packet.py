import numpy as np
from typing import List, Tuple, Union, Any
from copy import deepcopy
from itertools import count

from qcns.python.components.packet.protocol import *

__all__ = ['Packet']

Protocol = Union[L1_Protocol, L2_Protocol, L3_Protocol, L4_Protocol, L7_Protocol]

_L1_PROTOCOLS = {0: L1_SRP, 1: L1_TPSP, 2: L1_BSMP, 3: L1_FSP}
_L2_PROTOCOLS = {0: L2_EPP, 1: L2_QECP}
_L3_PROTOCOLS = {0: L3_CRP, 1: L3_FRP, 2: L3_ERP}
_L4_PROTOCOLS = {0: L4_EPP, 1: L4_QECP}
_L7_PROTOCOLS = {0: L7_TPP, 1: L7_RGP}

class Packet:
    
    """Independent protocol headers with classical hop addresses.

    ``src``/``dst`` identify the current hop. Layer-specific addresses belong
    to their protocols and are not changed when a packet is forwarded.
    Custom headers may be supplied directly or through per-packet
    ``header_factories={layer: {protocol_id: callable}}`` overrides.
    """

    def __base_init(self, l1_requested: int=0, l1_needed: int=0, l1_protocol: int=0,
                      l2_src: int=None, l2_dst: int=None, l2_requested: int=0, l2_needed: int=0, l2_code: str=None, l2_protocol: int=0,
                      l3_src: int=None, l3_dst: int=None, l3_curr: int=None, l3_next: int=None, l3_requested: int=0, l3_needed: int=0, l3_protocol: int=0,
                      l4_src: int=None, l4_dst: int=None, l4_requested: int=0, l4_needed: int=0, l4_code: str=None, l4_protocol: int=0,
                      l7_requested: int=0, l7_protocol: int=0, payload: Any=None, *, l3_fidelity: float=None) -> None:
        
        """
        Initializes the packet with standard inputs such as L2 src and dst
        
        Args:
            src (int): Source address
            dst (int): Destination address
            l1_requested (int): L1 number of requested qubits
            l1_needed (int): L1 number of needed qubits
            l2_src (int): L2 source address
            l2_dst (int): L2 destination address
            l2_requested (int): L2 number of requested qubits
            l2_needed (int): L2 number of needed qubits
            l3_src (int): L3 source address
            l3_dst (int): L3 destination address
            l3_requested (int): L3 number of requested qubits
            l3_needed (int): L3 number of needed qubits
            l4_src (int): L4 source port
            l4_dst (int): L4 destination port
            l4_requested (int): L4 number of requested qubits
            l4_needed (int): L4 number of needed qubits
            l7_requested (int): L7 number of requested qubits
            payload (list): payload of packet
            
        Returns:
            /
        """
        
        self._layer1: L1_Protocol = None
        self._layer2: L2_Protocol = None
        self._layer3: L3_Protocol = None
        self._layer4: L4_Protocol = None
        self._layer7: L7_Protocol = None
        
        if l1_requested > 0 or l1_needed > 0:
            self._layer1 = self._factory(1, l1_protocol, _L1_PROTOCOLS)(l1_requested, l1_needed)
        
        if l2_code is not None or l2_src is not None or l2_dst is not None or l2_requested > 0 or l2_needed > 0:
            l2_src = self.src if l2_src is None else l2_src
            l2_dst = self.dst if l2_dst is None else l2_dst
            if l2_code is not None and l2_protocol == 0:
                l2_protocol = 1
            self._layer2 = self._factory(2, l2_protocol, _L2_PROTOCOLS)(l2_src, l2_dst, l2_requested, l2_needed, l2_code)
        
        if l3_fidelity is not None or l3_src is not None or l3_dst is not None or l3_requested > 0 or l3_needed > 0:
            l3_src = self.src if l3_src is None else l3_src
            l3_dst = self.dst if l3_dst is None else l3_dst
            self._layer3 = self._factory(3, l3_protocol, _L3_PROTOCOLS)(l3_src, l3_dst, l3_curr, l3_next, l3_requested, l3_needed)
            if l3_fidelity is not None:
                self._layer3.fidelity = l3_fidelity
        
        if l4_code is not None or l4_src is not None or l4_dst is not None or l4_requested > 0 or l4_needed > 0:
            l4_src = self.src if l4_src is None else l4_src
            l4_dst = self.dst if l4_dst is None else l4_dst
            if l4_code is not None and l4_protocol == 0:
                l4_protocol = 1
            self._layer4 = self._factory(4, l4_protocol, _L4_PROTOCOLS)(l4_src, l4_dst, l4_requested, l4_needed, l4_code)
        
        if l7_requested > 0 or payload is not None:
            self._layer7 = self._factory(7, l7_protocol, _L7_PROTOCOLS)(l7_requested, payload)

    def __derived_init(self, layer1: L1_Protocol=None, layer2: L2_Protocol=None, layer3: L3_Protocol=None, 
                     layer4: L4_Protocol=None, layer7: L7_Protocol=None) -> None:
        
        self._layer1: L1_Protocol = layer1
        self._layer2: L2_Protocol = layer2
        self._layer3: L3_Protocol = layer3
        self._layer4: L4_Protocol = layer4
        self._layer7: L7_Protocol = layer7
    
    _request_ids = count(1)

    def __init__(self, src: int, dst: int, *args, request_id=None,
                 header_factories=None, **kwargs) -> None:
        self._src, self._dst = src, dst
        self.rejected = False
        self.request_id = next(self._request_ids) if request_id is None else request_id
        self._header_factories = {layer: dict(factories) for layer, factories in
                                 (header_factories or {}).items()}
        header_keys = {'layer1', 'layer2', 'layer3', 'layer4', 'layer7'}
        derived = bool(header_keys.intersection(kwargs)) or bool(args and isinstance(args[0], Protocol))
        if derived:
            self.__derived_init(*args, **kwargs)
        else:
            self.__base_init(*args, **kwargs)
        for layer, base in ((1, L1_Protocol), (2, L2_Protocol), (3, L3_Protocol),
                            (4, L4_Protocol), (7, L7_Protocol)):
            header = getattr(self, f'_layer{layer}')
            if header is not None and not isinstance(header, base):
                raise TypeError(f'layer{layer} must derive from {base.__name__}')

    def _factory(self, layer, identifier, builtins):
        factory = self._header_factories.get(layer, {}).get(identifier, builtins.get(identifier))
        if factory is None:
            raise ValueError(f'Unknown L{layer} protocol {identifier}; pass layer{layer}=header '
                             'or provide header_factories={layer: {protocol_id: factory}}')
        return factory

    @property
    def src(self) -> int:
        """Sender on the current classical hop."""
        return self._src

    @src.setter
    def src(self, value: int) -> None:
        self._src = value

    @property
    def dst(self) -> int:
        """Recipient on the current classical hop."""
        return self._dst

    @dst.setter
    def dst(self, value: int) -> None:
        self._dst = value

    def set_hop(self, src: int, dst: int) -> None:
        """Update hop addresses together, preserving headers and request identity.

        Call after consuming incoming-hop resources and before transmitting.
        """
        self._src, self._dst = src, dst

    def reverse_hop(self) -> None:
        """Address a reply to the previous hop."""
        self.set_hop(self.dst, self.src)

    def copy(self):
        """Snapshot packet contents while retaining request identity."""
        return deepcopy(self)

    @property
    def layer(self) -> int:
        """Highest present layer in simulator numbering (application layer = 5)."""
        return next((index for index, header in
                     ((5, self._layer7), (4, self._layer4), (3, self._layer3),
                      (2, self._layer2), (1, self._layer1)) if header is not None), 0)

    @property
    def is_l1(self) -> bool:
        return self.layer == 1

    @property
    def is_l2(self) -> bool:
        return self.layer == 2

    @property
    def is_l3(self) -> bool:
        return self.layer == 3

    @property
    def is_l4(self) -> bool:
        return self.layer == 4

    @property
    def is_l7(self) -> bool:
        return self.layer == 5

    def __len__(self) -> int:
        
        """
        TODO add docstring
        """
        
        return sum(len(header) for header in (self._layer1, self._layer2, self._layer3, self._layer4, self._layer7) if header is not None)
    
    def __repr__(self) -> str:
        
        """
        Custom print function
        
        Args:
            /
            
        Returns:
            packet (str): packet in str repr
        """
        
        return ''.join(str(header) for header in (self._layer1, self._layer2, self._layer3, self._layer4, self._layer7) if header is not None)
    
    @property
    def has_l1(self) -> bool:
        
        """
        Checks whether the packet has a Layer 1 header
        
        Args:
            /
            
        Returns:
            has_l1 (bool): True if the packet has a Layer 1 header, False otherwise
        """
        
        return self._layer1 is not None
    
    @property
    def has_l2(self) -> bool:
        
        """
        Checks whether the packet has a Layer 2 header
        
        Args:
            /
            
        Returns:
            has_l2 (bool): True if the packet has a Layer 2 header, False otherwise
        """
        
        return self._layer2 is not None
    
    @property
    def has_l3(self) -> bool:
        
        """
        Checks whether the packet has a Layer 3 header
        
        Args:
            /
            
        Returns:
            has_l3 (bool): True if the packet has a Layer 3 header, False otherwise
        """
        
        return self._layer3 is not None
    
    @property
    def has_l4(self) -> bool:
        
        """
        Checks whether the packet has a Layer 4 header
        
        Args:
            /
            
        Returns:
            has_l4 (bool): True if the packet has a Layer 4 header, False otherwise
        """
        
        return self._layer4 is not None
    
    @property
    def has_l7(self) -> bool:
        
        """
        Checks whether the packet has a Layer 7 header
        
        Args:
            /
            
        Returns:
            has_l7 (bool): True if the packet has a Layer 7 header, False otherwise
        """
        
        return self._layer7 is not None
    
    # Layer 1
    
    @property
    def layer1(self) -> L1_Protocol:
        
        """
        Returns the Layer 1 Protocol
        
        Args:
            /
            
        Returns:
            layer1 (L1_Protocol): Layer 1 Protocol
        """
        
        return self._layer1
    
    @layer1.setter
    def layer1(self, l1_protocol: L1_Protocol) -> None:
        
        """
        TODO add docstring
        """
        
        self._layer1 = l1_protocol
    
    @property
    def l1_requested(self) -> int:
        
        """
        Returns the number requested qubits on L1
        
        Args:
            /
            
        Returns:
            l1_requested (int): number of requested qubits on L1
        """
        
        return self._layer1.requested
    
    @l1_requested.setter
    def l1_requested(self, l1_requested: int) -> None:
        
        """
        Sets the number of requested qubits on L1
        
        Args:
            l1_requested (int): number of requested qubits to set
            
        Returns:
            /
        """
        
        self._layer1.requested = l1_requested
        
    @property
    def l1_needed(self) -> int:
        
        """
        Returns the number needed qubits on L1
        
        Args:
            /
            
        Returns:
            l1_needed (int): number of needed qubits on L1
        """
        
        return self._layer1.needed
    
    @l1_needed.setter
    def l1_needed(self, l1_needed: int) -> None:
        
        """
        Sets the number of needed qubits on L1, WARNING resets the L1 success array
        
        Args:
            l1_needed (int): number of needed qubits to set
            
        Returns:
            /
        """
        
        self._layer1.needed = l1_needed
        
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
        
        if self._layer1.protocol != 1:
            raise ValueError('Photon Source flag is only available for TPSP protocol')
        
        return self._layer1.ps
    
    def l1_set_ps(self) -> None:
        
        """
        Sets the Photon Source flag
        
        Args:
            /
            
        Returns:
            /
        """
        
        if self._layer1.protocol != 1:
            raise ValueError('Photon Source flag is only available for TPSP protocol')
        
        self._layer1.set_ps()
        
    def l1_reset_ps(self) -> None:
        
        """
        Resets the Photon Source flag
        
        Args:
            /
            
        Returns:
            /
        """
        
        if self._layer1.protocol != 1:
            raise ValueError('Photon Source flag is only available for TPSP protocol')
        
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
    def l1_success(self, l1_success: np.ndarray) -> None:
        
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
        
        self._layer1.protocol = l1_protocol
        
    # Layer 2
    
    @property
    def layer2(self) -> L2_Protocol:
        
        """
        Returns the Layer 2 Protocol
        
        Args:
            /
            
        Returns:
            layer2 (L2_Protocol): Layer 2 Protocol
        """
        
        return self._layer2
    
    @layer2.setter
    def layer2(self, l2_protocol: L2_Protocol) -> None:
        
        """
        
        """
        
        self._layer2 = l2_protocol
    
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
    def l2_requested(self) -> int:
        
        """
        Returns the L2 number of requested qubits
        
        Args:
            /
            
        Returns:
            l2_requested (int): number of L2 requested qubits
        """
        
        return self._layer2.requested
    
    @l2_requested.setter
    def l2_requested(self, l2_requested: int) -> None:
        
        """
        Sets the L2 number of requested qubits
        
        Args:
            l2_requested (int): number of L2 requested qubits to set
            
        Returns:
            /
        """
        
        self._layer2.requested = l2_requested
        
    @property
    def l2_needed(self) -> int:
        
        """
        Returns the L2 number of needed qubits
        
        Args:
            /
            
        Returns:
            l2_needed (int): L2 number of needed qubits
        """
        
        return self._layer2.needed
    
    @l2_needed.setter
    def l2_needed(self, l2_needed: int) -> None:
        
        """
        Sets the L2 number of needed qubits, WARNING resets the L2 success array
        
        Args:
            l2_needed (int): L2 number of needed qubits to set
            
        Returns:
            /
        """
        
        self._layer2.needed = l2_needed
        
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
    def l2_success(self, l2_success: np.ndarray) -> None:
        
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
    def l2_code(self) -> str:
        
        """
        Returns the L2 code
        
        Args:
            /
            
        Returns:
            code (str): L2 code
        """
        
        return getattr(self._layer2, 'code', None)
    
    @l2_code.setter
    def l2_code(self, name: str) -> None:
        
        """
        TODO add docstring
        """
        
        if self._layer2._protocol != 1:
            raise ValueError('Code is only available for QECP protocol')
        
        self._layer2.code = name
    
    @property
    def l2_complete(self) -> bool:
        
        """
        Returns whether the L2 header is complete
        
        Args:
            /
            
        Returns:
            complete (bool): whether the L2 header is complete
        """
        
        return getattr(self._layer2, 'complete', False)
    
    def l2_set_complete(self) -> None:
        
        """
        Sets the L2 header to complete
        
        Args:
            /
            
        Returns:
            /
        """
        
        if self._layer2._protocol != 1:
            raise ValueError('Complete Flag is only available for QECP protocol')
        
        self._layer2.set_complete()
    
    def l2_reset_complete(self) -> None:
        
        """
        Resets the L2 header to incomplete
        
        Args:
            /
            
        Returns:
            /
        """
        
        if self._layer2._protocol != 1:
            raise ValueError('Complete Flag is only available for QECP protocol')
        
        self._layer2.reset_complete()
    
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
        TODO add docstring
        """
        
        self._layer2.protocol = l2_protocol
        
    # Layer 3
    
    @property
    def layer3(self) -> L3_Protocol:
        
        """
        Returns the Layer 3 Protocol
        
        Args:
            /
            
        Returns:
            layer3 (L3_Protocol): Layer 3 Protocol
        """
        
        return self._layer3
    
    @layer3.setter
    def layer3(self, l3_protocol: L3_Protocol) -> None:
        
        self._layer3 = l3_protocol
    
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
    def l3_dst(self, l3_dst: int) -> None:
        
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
    def l3_curr(self) -> int:
        
        """
        Returns the L3 current address
        
        Args:
            /
            
        Returns:
            l3_curr (int): L3 current address
        """
        
        return self._layer3.curr
    
    @l3_curr.setter
    def l3_curr(self, l3_curr: int) -> None:
        
        """
        Sets the L3 current address
        
        Args:
            l3_curr (int): L3 current address
            
        Returns:
            /
        """
        
        self._layer3.curr = l3_curr
    
    @property
    def l3_next(self) -> int:
        
        """
        Returns the L3 next address
        
        Args:
            /
            
        Returns:
            l3_next (int): L3 next address
        """
        
        return self._layer3.next
    
    @l3_next.setter
    def l3_next(self, l3_next: int) -> None:
        
        """
        Sets the L3 next address
        
        Args:
            l3_next (int): L3 next address
            
        Returns:
            /
        """
        
        self._layer3.next = l3_next
    
    def l3_update_next(self, l3_next: int) -> None:
        
        """
        Updates the L3 next address
        
        Args:
            l3_next (int): L3 next address
            
        Returns:
            /
        """
        
        self._layer3.update_next(l3_next)
    
    @property
    def l3_requested(self) -> int:
        
        """
        Returns the L3 number of requested qubits
        
        Args:
            /
            
        Returns:
            l3_requested (int): L3 number of requested qubits
        """
        
        return self._layer3.requested
    
    @l3_requested.setter
    def l3_requested(self, l3_requested: int) -> None:
        
        """
        Sets the L3 number of requested qubits
        
        Args:
            l3_requested (int): L3 number of requested qubits
            
        Returns:
            /
        """
        
        self._layer3.requested = l3_requested
        
    @property
    def l3_needed(self) -> int:
        
        """
        Returns the L3 number of needed qubits
        
        Args:
            /
            
        Returns:
            l3_needed (int): L3 number of needed qubits
        """
        
        return self._layer3.needed
    
    @l3_needed.setter
    def l3_needed(self, l3_needed: int) -> None:
        
        """
        Sets the L3 number of needed qubits, WARNING resets the L3 success, X and Z array
        
        Args:
            l3_needed (int): L3 number of needed qubits
            
        Returns:
            /
        """
        
        self._layer3.needed = l3_needed
        
    @property
    def l3_fidelity(self) -> float:
        if self._layer3 is None:
            raise ValueError('Packet has no L3 header')
        return self._layer3.fidelity

    @l3_fidelity.setter
    def l3_fidelity(self, value: float) -> None:
        if self._layer3 is None:
            raise ValueError('Packet has no L3 header')
        self._layer3.fidelity = value

    @property
    def l3_fidelity_ir(self) -> int:
        if self._layer3 is None:
            raise ValueError('Packet has no L3 header')
        return self._layer3.fidelity_ir

    @l3_fidelity_ir.setter
    def l3_fidelity_ir(self, value: int) -> None:
        if self._layer3 is None:
            raise ValueError('Packet has no L3 header')
        self._layer3.fidelity_ir = value

    @property
    def l3_fmode(self) -> int:
        
        """
        Checks whether the L3 mode flag is set
        
        Args:
            /
            
        Returns:
            l3_mode (int): L3 mode
        """
        
        return self._layer3.fmode
    
    def l3_set_cf(self) -> None:
        
        """
        Sets the L3 flag to classical forwarding mode
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._layer3.set_cf()
        
    def l3_set_nr(self) -> None:
        
        """
        Sets the L3 flag to no reject mode
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._layer3.set_nr()
    
    def l3_set_pr(self) -> None:
        
        """
        Sets the L3 flag to partial reject mode
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._layer3.set_pr()
    
    def l3_set_cr(self) -> None:
        
        """
        Sets the L3 flag to complete reject mode
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._layer3.set_cr()
    
    @property
    def l3_is_cf(self) -> bool:
        
        """
        Returns whether the cf flag is set
        
        Args:
            /
            
        Returns:
            is_cf (bool): whether the cf flag is set
        """
        
        return self._layer3.is_cf
    
    @property
    def l3_is_nr(self) -> bool:
        
        """
        Returns whether the nr flag is set
        
        Args:
            /
            
        Returns:
            is_nr (bool): whether the nr flag is set
        """
        
        return self._layer3.is_nr
    
    @property
    def l3_is_pr(self) -> bool:
        
        """
        Returns whether the pr flag is set
        
        Args:
            /
            
        Returns:
            is_pr (bool): whether the pr flag is set
        """
        
        return self._layer3.is_pr
    
    @property
    def l3_is_cr(self) -> bool:
        
        """
        Returns whether the cr flag is set
        
        Args:
            /
            
        Returns:
            is_cr (bool): whether the cr flag is set
        """
        
        return self._layer3.is_cr

    @property
    def l3_hop_count(self) -> int:
        
        """
        Returns the L3 hop count
        
        Args:
            /
            
        Returns:
            hop_count (int): hop count of L3
        """
        
        return self._layer3.hop_count
    
    @l3_hop_count.setter
    def l3_hop_count(self, hop_count: int) -> None:
        
        """
        Sets the new L3 hop count
        
        Args:
            hop_count (int): new hop count of L3
            
        Returns:
            /
        """
        
        self._layer3.hop_count = hop_count
    
    @property
    def l3_es_result(self) -> Tuple[np.array, np.array]:
        
        """
        Returns both the L3 X and Z array
        
        Args:
            /
            
        Returns:
            l3_es_result (tuple): tuple containing L3 X and Z array
        """
        
        return self._layer3.es_result
    
    @l3_es_result.setter
    def l3_es_result(self, l3_es_result: Tuple[np.ndarray, np.ndarray]) -> None:
        
        """
        Sets the X and Z array
        
        Args:
            l3_es_result (np.array): X and Z array
            
        Returns:
            /
        """
        
        self._layer3.es_result = l3_es_result
    
    def l3_update_es(self, l3_es_result: int, index: int=0) -> None:
        
        """
        Updates the L3 X and Z array at the index with the result
        
        Args:
            index (int): index to update X and Z array
            result (int): result to update X and Z array with
            
        Returns:
            /
        """
        
        self._layer3.update_es(l3_es_result, index)
    
    def l3_reset_es(self, index: int=None) -> None:
        
        """
        Resets the L3 X and Z array
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._layer3.reset_es(index)
    
    @property
    def l3_x_count(self) -> np.array:
        
        """
        Returns the L3 X count array
        
        Args:
            /
        
        Returns:
            l3_x_count (np.array): L3 X count array
        """
        
        return self._layer3.x_count
    
    @l3_x_count.setter
    def l3_x_count(self, results: np.ndarray):
        self._layer3.x_count = results
    
    @property
    def l3_z_count(self) -> np.array:
        
        """
        Returns the L3 Z count array
        
        Args:
            /
        
        Returns:
            l3_z_count (np.array): L3 Z count array
        """
        
        return self._layer3.z_count
    
    @l3_z_count.setter
    def l3_z_count(self, results: np.ndarray):
        self._layer3.z_count = results
    
    def l3_update_x_count(self, result: bool, index: int=0):
        
        """
        TODO add docstring
        """
        
        self._layer3.update_x_count(result, index)

    def l3_update_z_count(self, result: bool, index: int=0):
        
        """
        TODO add docstring
        """
        
        self._layer3.update_z_count(result, index)
    
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
        
        self._layer3.protocol = l3_protocol
        
    # Layer 4
    
    @property
    def layer4(self) -> L4_Protocol:
        
        """
        Returns the Layer 4 Protocol
        
        Args:
            /
            
        Returns:
            layer4 (L4_Protocol): Layer 4 Protocol
        """
        
        return self._layer4
    
    @layer4.setter
    def layer4(self, l4_protocol: L4_Protocol) -> None:
        
        self._layer4 = l4_protocol
    
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
    def l4_requested(self) -> int:
        
        """
        Returns the L4 number of requested qubits
        
        Args:
            /
            
        Returns:
            l4_requested (int): L4 number of requested qubits
        """
        
        return self._layer4.requested
    
    @l4_requested.setter
    def l4_requested(self, l4_requested: int) -> None:
        
        """
        Sets the L4 number of requested qubits
        
        Args:
            l4_requested (int): L4 number of requested qubits
            
        Returns:
            /
        """
        
        self._layer4.requested = l4_requested
        
    @property
    def l4_needed(self) -> int:
        
        """
        Returns the L4 number of needed qubits
        
        Args:
            /
            
        Returns:
            l4_needed (int): number of needed qubits
        """
        
        return self._layer4.needed
    
    @l4_needed.setter
    def l4_needed(self, l4_needed: int) -> None:
        
        """
        Sets the L4 number of needed qubits, WARNING resets the L4 success array
        
        Args:
            l4_needed (int): L4 number of needed qubits
            
        Returns:
            /
        """
        
        self._layer4.needed = l4_needed
        
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

    @l4_success.setter
    def l4_success(self, l4_success: np.ndarray) -> None:
        
        """
        Sets the L4 success array
        
        Args:
            l4_success (np.array): L4 success array
            
        Returns:
            /
        """
             
        self._layer4.success = l4_success

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
    def l4_code(self) -> str:
        """
        Returns the L4 code
        
        Args:
            /
            
        Returns:
            l4_code (str): L4 code
        """
        
        return getattr(self._layer4, 'code', None)
     
    @l4_code.setter
    def l4_code(self, name: str) -> None:
        """
        Sets the L4 code
        
        Args:
            name (str): L4 code
            
        Returns:
            /
        """
        
        if self._layer4.protocol != 1:
            raise ValueError('Code is only available for QECP protocol')
        
        self._layer4.code = name 
    
    @property
    def l4_complete(self) -> bool:
        """
        Returns whether the L4 header is complete
        
        Args:
            /
            
        Returns:
            complete (bool): whether the L4 header is complete
        """
        
        return getattr(self._layer4, 'complete', False)
    
    def l4_set_complete(self) -> None:
        
        """
        Sets the L4 header to complete
        
        Args:
            /
            
        Returns:
            /
        """
        
        if self._layer4.protocol != 1:
            raise ValueError('Complete Flag is only available for QECP protocol')
        
        self._layer4.set_complete()
        
    def l4_reset_complete(self) -> None:
        
        """
        Resets the L4 header to incomplete
        
        Args:
            /
            
        Returns:
            /
        """
        
        if self._layer4.protocol != 1:
            raise ValueError('Complete Flag is only available for QECP protocol')
        
        self._layer4.reset_complete()
    
    @property
    def l4_protocol(self) -> int:
        
        """
        Returns the L4 protocol
        
        Args:
            /
            
        Returns:
            l4_protocol (int): L4 protocol
        """
        
        if not self._layer4:
            raise ValueError('Layer 4 is not present in the packet')
        
        return self._layer4.protocol
    
    @l4_protocol.setter
    def l4_protocol(self, l4_protocol: int) -> None:
        
        """
        TODO add docstring
        """
        
        self._layer4.protocol = l4_protocol
        
    # Layer 7
    
    @property
    def layer7(self) -> L7_Protocol:
        
        """
        Returns the Layer 7 Protocol
        
        Args:
            /
            
        Returns:
            layer7 (L7_Protocol): Layer 7 Protocol
        """
        
        return self._layer7
    
    @layer7.setter
    def layer7(self, l7_protocol: L7_Protocol) -> None:
        
        self._layer7 = l7_protocol
    
    @property
    def l7_requested(self) -> int:
        
        """
        Returns the L7 number of requested qubits
        
        Args:
            /
            
        Returns:
            l7_requested (int): L7 number of requested qubits
        """
        
        return self._layer7.requested
    
    @l7_requested.setter
    def l7_requested(self, l7_requested: int) -> None:
        
        """
        Sets the L7 number of requested qubits
        
        Args:
            l7_requested (int): L7 number of requested qubits
            
        Returns:
            /
        """
        
        self._layer7.requested = l7_requested
        
    @property
    def l7_protocol(self) -> int:
        
        """
        Returns the L7 protocol
        
        Args:
            /
            
        Returns:
            l7_protocol (int): L7 protocol
        """
        
        if not self._layer7:
            raise ValueError('Layer 7 is not present in the packet')
        
        return self._layer7.protocol
    
    @l7_protocol.setter
    def l7_protocol(self, l7_protocol: int) -> None:
        
        """
        TODO add docstring
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
        
        if not self._layer7:
            raise ValueError('Layer 7 is not present in the packet')
        
        return self._layer7.payload
    
    @payload.setter
    def payload(self, payload: Any) -> None:
        
        """
        Sets the payload
        
        Args:
            payload (list): payload of packet
            
        Returns:
            /
        """
        
        self._layer7.payload = payload
    
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
