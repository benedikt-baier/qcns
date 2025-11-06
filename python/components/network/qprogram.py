
from functools import partial
from typing import List, Dict, Union, Awaitable

import numpy as np
import copy

from qcns.python.components.qubit.qubit import Qubit
from qcns.python.components.packet.packet import Packet

__all__ = ['QProgram', 'L1_EGP', 'L2_FIP', 'L3_QFP', 'L7_TPP', 'L7_DQC', 'QProgram_Model']

L0 = 0
L1 = 1
L2 = 2
L3 = 3
L4 = 4
L7 = 5

class Node:
    
    pass

class QProgram:
    
    pass

class QProgram:
    
    """
    Represents a generic Quantum Program to run
    
    Attrs:
        _host (Node): host on which the program runs
        _layer (int): identifier on which layer the qprogram runs
        _prev_protocol (QProgram): protocol of the previous layer
        _next_protocol (QProgram): protocol of the next layer
    """
    
    def __init__(self) -> None:
        
        """
        Initializes a generic Quantum Program
        
        Args:
            host (Node): host on which the program runs
            
        Returns:
            /
        """
        
        self._host: Node = None
        self._layer: int = 0
        
        self._prev_protocol: QProgram = None
        self._next_protocol: QProgram = None
        
    @property
    def host(self) -> Node:
        
        """
        Property to access host more easily
        
        Args:
            /
            
        Returns:
            host (Node): host on which the program runs
        """
        
        return self._host
    
    @host.setter
    def host(self, host: Node) -> None:
        
        """
        Sets the host for the qprogram
        
        Args:
            host (Node): host the qprogram runs on
            
        Returns:
            /
        """
        
        self._host = host
    
    @property
    def prev_protocol(self) -> QProgram:
        
        """
        Property to access the previous protocol
        
        Args:
            /
            
        Returns:
            prev_protocol (QProgram): program of the lower layer
        """
        
        return self._prev_protocol
    
    @prev_protocol.setter
    def prev_protocol(self, prev_protocol: QProgram) -> None:
        
        """
        Sets the previous protocol
        
        Args:
            prev_protocol (QProgram): previous protocol to run
            
        Returns:
            /
        """
        
        self._prev_protocol: QProgram = prev_protocol
    
    @property
    def next_protocol(self) -> QProgram:
        
        """
        Property to access the next protocol
        
        Args:
            /
            
        Returns:
            next_protocol (QProgram): program of the higher layer
        """
        
        return self._next_protocol
    
    @next_protocol.setter
    def next_protocol(self, next_protocol: QProgram) -> None:
        
        """
        Sets the next protocol
        
        Args:
            next_protocol (QProgram): next protocol to run
            
        Returns:
            /
        """
        
        self._next_protocol: QProgram = next_protocol
    
    @property
    def layer(self) -> int:
        
        """
        Returns the layer of the qprogram
        
        Args:
            /
            
        Returns:
            layer (int): layer of the qprogram
        """
        
        return self._layer
    
    @layer.setter
    def layer(self, layer: int) -> None:
        
        """
        Sets the layer of the qprogram
        
        Args:
            layer (int): layer to set
            
        Returns:
            /
        """
        
        self._layer = layer
    
class L1_EGP(QProgram):
    
    """
    Program for the Entanglement Generation Protocol (EGP) in Layer 1
    
    Attrs:
        _eg_mode (str): mode of the entanglement generation
        _rap_mode (str): mode of the reallocation
        _qdp_modes (dict): dictionary of functions of the _rap_mode
    """
    
    def __init__(self, eg_mode: str='l3cp', rap_mode: str='attempt', reattempt: bool=True, qubits_requested: int=1) -> None:
        
        """
        Initializes the L1 Entanglement Generation Protocol (EGP)
        
        Args:
            host (Node): host the program runs on
            eg_mode (str): mode of the entanglement generation
            rap_mode (str): mode of allocation of qubits
            
        Returns:
            /
        """

        self.qubits_requested = qubits_requested
        
        super(L1_EGP, self).__init__()
        self.layer = 1
        
        _rap_modes = {'attempt': 3, 'create': 2}
        _eg_modes = {'l3cp': self._l3cp,'srp': self._srp, 'tpsp': self._tpsp, 'bsmp': self._bsmp, 'fsp': self._fsp}
        
        self._eg_mode: str = eg_mode
        self.classical_data_plane = _eg_modes[self._eg_mode]
        self.classical_data_plane = partial(self._protocol_n, self.classical_data_plane)
        if reattempt:
            if self._eg_mode == 'srp' or self._eg_mode == 'tpsp':
                self.classical_data_plane = partial(self._reattempt_srp_tpsp, self.classical_data_plane)
            else:
                self.classical_data_plane = partial(self._reattempt, self.classical_data_plane)
        # self.classical_data_plane = partial(self._protocol_n, self.classical_data_plane)
        
        self._rap_mode: str = _rap_modes[rap_mode]
        self._qdp_modes: Dict[str, Awaitable] = {3: self.attempt_bell_pairs, 2: self.create_bell_pairs}

    async def attempt_bell_pairs(self, receiver: int, requested: int=1, estimate: bool=False) -> None:
    
        """
        Attempts to create the number of requested bell pairs, can estimate the number of needed qubits based on success probability
        
        Args:
            receiver (int): receiver of bell pairs
            requested (int): number of requested bell pairs
            estimate (bool): whether to estimate the number of needed qubits or not
            
        Returns:
            /
        """
        
        # print(f'RECEIVER: {receiver}')
        await self.host.attempt_bell_pairs(receiver, requested, estimate)

        
    async def create_bell_pairs(self, receiver: int, requested: int=1) -> None:
        
        """
        Creates the number of requested qubits, no matter how long it takes
        
        Args:
            receiver (int): receiver of bell pairs
            requested (int): number of requested bell pairs
            
        Returns:
            /
        """
        
        await self.host.create_bell_pairs(receiver, requested)

    async def _reattempt(self, func: Awaitable, packet: Packet) -> None:
        
        """
        Wrapper function when qubits should be reattempted
        
        Args:
            func (Awaitable): function to execute before reattempting
            packet (Packet): packet needed for the reattempt
            
        Returns:
            /
        """
        
        print(f'ID: {self.host.id}; execute _reattempt')

        if self.host.l3_num_qubits(packet.l2_src, packet.l1_ack) > 0.5 * self.qubits_requested:
            return

        await func(packet)

        # if not packet.l1_ack:
        #     return

        if self.host.l3_num_qubits(packet.l2_src, packet.l1_ack) < 0.5 * self.qubits_requested:
            print(f'ID: {self.host.id}; L1_NUM_QUBITS: {self.host.l1_num_qubits(packet.l2_src, packet.l1_ack)} requested: {self.qubits_requested - self.host.l1_num_qubits(packet.l2_src, packet.l1_ack)}')
            await self.quantum_data_plane(packet.l2_src, self.qubits_requested - self.host.l1_num_qubits(packet.l2_src, packet.l1_ack), True)
       
        # if self.host.l1_num_qubits(packet.l2_src, packet.l1_ack) < self.qubits_requested:
        #     print(f'ID: {self.host.id}; L1_NUM_QUBITS: {self.host.l1_num_qubits(packet.l2_src, packet.l1_ack)} requested: {self.qubits_requested - self.host.l1_num_qubits(packet.l2_src, packet.l1_ack)}')
        #     await self.quantum_data_plane(packet.l2_src, self.qubits_requested - self.host.l1_num_qubits(packet.l2_src, packet.l1_ack), True)

        print(f'ID: {self.host.id}; finished _reattempt')


    async def _reattempt_srp_tpsp(self, func: Awaitable, packet: Packet) -> None:
        
        """
        Wrapper function when qubits should be reattempted
        
        Args:
            func (Awaitable): function to execute before reattempting
            packet (Packet): packet needed for the reattempt
            
        Returns:
            /
        """
        
        print(f'ID: {self.host.id}; execute _reattempt_srp')

        if self.host.l3_num_qubits(packet.l2_src, packet.l1_ack) > 0.5 * self.qubits_requested:
            return

        await func(packet)

        # if packet.l1_ack:
        #     return
        
        if self._eg_mode == 'tpsp':
            if packet.l2_src not in self.host._connections['memory'] or (not packet.l1_ack) not in self.host._connections['memory'][packet.l2_src]:
                print('sike')
                return

        if self.host.l1_num_qubits(packet.l2_src, not packet.l1_ack) < self.qubits_requested:
            print(f'ID: {self.host.id}; L1_NUM_QUBITS: {self.host.l1_num_qubits(packet.l2_src, not packet.l1_ack)} requested: {self.qubits_requested - self.host.l1_num_qubits(packet.l2_src, not packet.l1_ack)}')
            await self.quantum_data_plane(packet.l2_src, self.qubits_requested - self.host.l1_num_qubits(packet.l2_src, not packet.l1_ack), True)

        print(f'ID: {self.host.id}; finished _reattempt')


    async def _protocol_n(self, func: Awaitable, packet: Packet) -> None:
        
        """
        Wrapper function for executing the next protocol
        
        Args:
            func (Awaitable): function to execute
            packet (Packet): packet needed for the next protocol
            
        Returns:
            /
        """

        print(f'ID: {self.host.id}; Executing Protocol_n')

        await func(packet)

        if self._eg_mode == 'tpsp':
            if packet.l2_src not in self.host._connections['memory'] or (not packet.l1_ack) not in self.host._connections['memory'][packet.l2_src]:
                return
        
        if self.next_protocol is not None:
            print(f'NEXTTTTTT: {self.next_protocol}')
            await self.next_protocol.quantum_data_plane(packet)
            

    async def _l3cp(self, packet: Packet) -> None:
        
        """
        Protocol for the L3 connection
        
        Args:
            packet (Packet): packet needed for the removal of qubits
            
        Returns:
            /
        """
        
        self.host.l3_remove_qubits(packet.l2_src, 1, packet.l1_success)

    async def _srp(self, packet: Packet) -> None:
        
        """
        Protocol for the SR connection
        
        Args:
            packet (Packet): packet needed for the removal of qubits
            
        Returns:
            /
        """

        print(f'PRE --- ID: {self.host.id}; L1_NUM_QUBITS: {self.host.l1_num_qubits(packet.l2_src, not packet.l1_ack)}')
        
        self.host.l0_move_qubits_l1(packet.l2_src, not packet.l1_ack, packet.l1_success)

        print(f'POST --- ID: {self.host.id}; L1_NUM_QUBITS: {self.host.l1_num_qubits(packet.l2_src, not packet.l1_ack)}')

        if not packet.l1_ack:
            packet_new = copy.deepcopy(packet)
            packet_new.l2_switch_src_dst()
            packet_new.l1_set_ack()
            await self.host.send_packet(packet_new)
            return
        
        # if not packet.l1_ack:
        #     packet.l1_set_ack()
        #     packet.l2_switch_src_dst()
        #     await self.host.send_packet(packet)
        #     print(f'MIRRORED PACKET: {packet}')
        #     return

    async def _tpsp(self, packet: Packet) -> None:
        
        """
        Protocol for the TPS connection
        
        Args:
            packet (Packet): packet needed for the removal of qubits
            
        Returns:
            /
        """
        
        if packet.l1_ps:
            
            self.host.l1_store_result(packet.l1_ack, packet)
            packet.l1_reset_ps()
            packet.l2_switch_src_dst()
            await self.host.send_packet(packet)
            return
        
        comp_res = self.host.l1_compare_results(packet)
        self.host.l0_move_qubits_l1(packet.l2_src, not packet.l1_ack, comp_res)

    async def _bsmp(self, packet: Packet) -> None:
        
        """
        Protocol for the BSM connection
        
        Args:
            packet (Packet): packet needed for the removal of qubits
            
        Returns:
            /
        """

        # print(f'ID: {self.host.id}; Executing _bsmp')
        
        self.host.l0_move_qubits_l1(packet.l2_src, packet.l1_ack, packet.l1_success)

        # print(f'ID: {self.host.id}; L1_NUM_QUBITS: {self.host.l1_num_qubits(packet.l2_src, packet.l1_ack)}')

    async def _fsp(self, packet: Packet) -> None:
        
        """
        Protocol for the FS connection
        
        Args:
            packet (Packet): packet needed for the removal of qubits
            
        Returns:
            /
        """
        
        self.host.l0_move_qubits_l1(packet.l2_src, packet.l1_ack, packet.l1_success)

    async def quantum_data_plane(self, receiver: int, requested: int=1, estimate: bool=False) -> None:
        
        """
        Quantum Data Plane of Entanglement Generation
        
        Args:
            receiver (int): host receiving the entanglement
            requested (int): requested number of qubits
            estimate (bool): whether to estimate the needed qubits from the requested qubits
            
        Returns:
            /
        """

        send_mem = self.host._connections["memory"][receiver][0]
        receive_mem_0 = self.host._sim._hosts[receiver]._connections["memory"][self.host.id][0]
        receive_mem_1 = self.host._sim._hosts[receiver]._connections["memory"][self.host.id][1]

        # if not receive_mem_0.has_space() or receive_mem_1.__len__ >= self.qubits_requested:
        #     return

        # if len(receive_mem_1) >= receive_mem_0.size:
        #     return

        # print(f'ID:{self.host.id} SEND_SPACE: {send_mem.remaining_space()} REC_SPACE:{receive_mem.remaining_space()}')

        # if not send_mem.has_space() or not receive_mem.has_space():
        #     print('NOO SPACE')
        #     return

        # if not self.host.has_space(receiver, 0) or not self.host.has_space(receiver, 1):
        #     return
        
        print(f'ID: {self.host.id}; Executing quantum data plane REQUESTED {requested}')
        await self._qdp_modes[self._rap_mode](*([receiver, requested, estimate][:self._rap_mode]))

    async def classical_data_plane(self, packet: Packet) -> None:
        
        """
        Classical Data Plane of Generating Entanglement, is overwritten
        
        Args:
            packet (Packet): packet needed for the data plane
            
        Returns:
            /
        """
        
        pass

class L2_FIP(QProgram):
    
    """
    Program to improve the entanglement fidelity of qubits on L2
    
    Attr:
        _fip_mode (str): mode how to improve fidelity
    """
    
    def __init__(self, fip_mode: str='necp', reattempt: bool=True) -> None:
        
        """
        Program to improve fidelity of qubits
        
        Args:
            fip_mode (str): mode how to improve fidelity
            reattempt (bool): how to reattempt entanglement
            
        Returns:
            /
        """
        
        super(L2_FIP, self).__init__()
        self.layer = 2
        
        _fip_qdp = {'necp': self._qdp_necp, 'epp': self._qdp_epp, 'qecp': self._qdp_qecp}
        _fip_cdp = {'necp': self._cdp_necp, 'epp': self._cdp_epp, 'qecp': self._cdp_qecp}
        
        self._fip_mode: str = fip_mode
        
        self.quantum_data_plane = _fip_qdp[self._fip_mode]
        self.classical_data_plane = _fip_cdp[self._fip_mode]
        if reattempt:
            self.classical_data_plane = partial(self._p_protocol, self.classical_data_plane)
        self.classical_data_plane = partial(self._n_protocol, self.classical_data_plane)
        
    async def _qdp_necp(self, packet: Packet) -> None:
        
        """
        Quantum Data Plane of no error correction protocol
        
        Args:
            packet (Packet): packet to use
            
        Returns:
            /
        """
        
        pass    
    
    async def _cdp_necp(self, packet: Packet) -> None:
        
        """
        Quantum Data Plane of no error correction protocol
        
        Args:
            packet (Packet): packet to use
            
        Returns:
            /
        """
        
        self.host.l1_move_qubits_l3(packet.l2_src, packet.l1_ack, np.ones(packet.l1_needed, dtype=np.bool_))
    
    async def _qdp_epp(self, packet: Packet) -> None:
        
        """
        Quantum Data Plane Protocol for the purification of qubits
        
        Args:
            packet (Packet): packet to use
            
        Returns:
            /  
        """
        
        if self.prev_protocol._eg_mode == 'srp' or self.prev_protocol._eg_mode == 'tpsp':
            purifications = self.host.l2_num_purification(packet.l2_src, not packet.l1_ack)
            print(f'SRP/TPSP NUM_PURIFICATION: {purifications}')
        else:
            purifications = self.host.l2_num_purification(packet.l2_src, packet.l1_ack)
            print(f'NUM_PURIFICATION: {purifications}')

        if not purifications:
            return
        
        packet_new = Packet(packet.l2_dst, packet.l2_src, l2_requested=purifications, l2_needed=2 * purifications)

        if self.prev_protocol._eg_mode == 'srp' or self.prev_protocol._eg_mode == 'tpsp':
            if not packet.l1_ack:
                packet_new.l2_set_ack()
        else:        
            if packet.l1_ack:
                packet_new.l2_set_ack()
            
        for i in range(purifications):
            res = self.host.l2_purify(packet_new.l2_dst, packet_new.l2_ack)
            packet_new.l2_success[i] = res

        print(packet_new)

        self.host.l2_store_result(packet_new.l2_ack, packet_new)
        
        await self.host.send_packet(packet_new)
        
        if self.host.l2_check_packets(packet_new.l2_dst, packet.l2_ack):
            print(f'CHECK PACKETS: {self.host.l2_retrieve_packet(packet_new.l2_dst, packet.l2_ack)}')
            self.classical_data_plane(self.host.l2_retrieve_packet(packet_new.l2_dst, packet.l2_ack))
    
    async def _cdp_epp(self, packet: Packet) -> None:
        
        """
        Classical Data Plane Protocol for the purification of qubits
        
        Args:
            packet (Packet): packet to use
            
        Returns:
            /  
        """
        
        if not self.host.l2_check_results(packet.l2_src, not packet.l2_ack):
            self.host.l2_store_packet(packet.l2_src, not packet.l2_ack, packet)
            print(f'STORING PACKET')
            return
        
        comp_res = self.host.l2_compare_results(packet)

        print(f'COMP_RES: {comp_res}')
        
        self.host.l2_move_qubits_l3(packet.l2_src, not packet.l2_ack, comp_res)
        
    async def _qdp_qecp(self, packet: Packet) -> None:
        
        pass
    
    async def _cdp_qecp(self, packet: Packet) -> None:
        
        pass
    
    async def _p_protocol(self, func: Awaitable, packet: Packet) -> None:
        
        """
        Wrapper function to call the quantum data plane of the previous function
        
        Args:
            func (Awaitable): function to execute before
            packet (Packet): Packet to use
            
        Returns:
            /
        """
        
        await func(packet)
        # print(f'FINISHED P_PROTOCOL')
        if self.host.l3_num_qubits(packet.l2_src, not packet.l2_ack) < packet.l2_requested:
            print(f'ID: {self.host.id}; L3_NUM_QUBITS: {self.host.l3_num_qubits(packet.l2_src, not packet.l2_ack)} requested: {packet.l2_requested - self.host.l3_num_qubits(packet.l2_src, not packet.l2_ack)}')
            await self.prev_protocol.quantum_data_plane(packet.l2_src, packet.l2_requested - self.host.l3_num_qubits(packet.l2_src, not packet.l2_ack), True)
            
    async def _n_protocol(self, func: Awaitable, packet: Packet) -> None:
        
        """
        Wrapper function for the next protocol
        
        Args:
            func (Awaitable): function to execute before
            packet (Packet): Packet to use
            
        Returns:
            /
        """
        
        await func(packet)
        # print(f'FINISHED N_PROTOCOL')
        if self.host.l3_check_packets(packet.l2_src, 0) and self.next_protocol is not None:
            print(f'NEXT IS L3!')
            await self.next_protocol.handle_stored_packets(packet.l2_src, 0)
    
    async def quantum_data_plane(self, packet: Packet) -> None:
        
        """
        Function for the quantum data plane will be overwritten
        
        Args:
            packet (Packet): Packet to use
            
        Returns:
            /
        """
        
        pass
    
    async def classical_data_plane(self, packet: Packet) -> None:
        
        """
        Function for the classical data plane will be overwritten
        
        Args:
            packet (Packet): Packet to use
            
        Returns:
            /
        """
        
        pass

class L3_QFP(QProgram):
    
    """
    Program for swapping entanglement with Classical Repeater Protocol (CRP)
    
    Attrs:
        _routing_table (dict): routing table for the classical forwarding
        _qf_mode (str): quantum forwarding mode, crp or frp
        _bsm_mode (str): mode of the BSM, bsm or prob_bsm
        _cdp (dict): classical data plane functions
        _qdp (dict): quantum data plane functions
    """
    
    def __init__(self, routing_table: Dict[int, int], qf_mode: str='frp', bsm_mode: str='bsm') -> None:
        
        """
        Initializes the L3_CRP program
        
        Args:
            routing_table (dict): routing table for the classical forwarding
            qf_mode (str): mode of the quantum forwarding, crp or frp
            bsm_mode (str): mode of the BSM, bsm or prob_bsm
            
        Returns:
            /
        """
        
        super(L3_QFP, self).__init__()
        self.layer = 3
        
        self._routing_table: Dict[int, int] = routing_table
        self._qf_mode: str = qf_mode
        self._bsm_mode: str = bsm_mode
    
        self._cdp: Dict[int, Awaitable] = {0: self.classical_forwarding, 1: self.no_reject, 2: self.partial_reject, 3: self.complete_reject}
        self._qdp: Dict[str, Awaitable] = {'crp': self.crp, 'frp': self.frp}
        self.quantum_data_plane = self._qdp[self._qf_mode]
    
    async def crp(self, packet: Packet, offset_index: int=None) -> None:
        
        """
        Function for the classical repeater protocol
        
        Args:
            packet (Packet): packet to base entanglement swapping on
            offset_index (int): offset of the quantum memory
            
        Returns:
            /
        """
        
        for index in range(packet.l3_needed):
        
            qubit_src = self.host.l3_retrieve_qubit(packet.l2_src, 1, offset_index)
            qubit_dst = self.host.l3_retrieve_qubit(packet.l2_dst, 0)
            
            if packet.l3_es_result[0][index]:
                self.host.apply_gate('X', qubit_src)
            if packet.l3_es_result[1][index]:
                self.host.apply_gate('Z', qubit_src)
            
            res = self.host.apply_gate(self._bsm_mode, qubit_src, qubit_dst, combine=True, remove=True)
            
            packet.l3_reset_es(index)
            packet.l3_update_es(res, index)  
        
        packet.l2_src = self.host.id
        await self.host.send_packet(packet)
    
    async def frp(self, packet: Packet, offset_index: int=None) -> None:
        
        """
        Function for the fast repeater protocol
        
        Args:
            packet (Packet): packet to base entanglement swapping on
            offset_index (int): offset of the quantum memory
            
        Returns:
            /
        """
        
        for index in range(packet.l3_needed):
        
            qubit_src = self.host.l3_retrieve_qubit(packet.l2_src, 1, offset_index)
            qubit_dst = self.host.l3_retrieve_qubit(packet.l2_dst, 0)
            
            res = self.host.apply_gate(self._bsm_mode, qubit_src, qubit_dst, combine=True, remove=True)
            
            packet.l3_update_es(res, index)  
        
        packet.l2_src = self.host.id
        await self.host.send_packet(packet)
    
    async def reject_packet(self, packet: Packet) -> None:
        
        """
        Function to reject a packet
        
        Args:
            packet (Packet): packet to reject
            
        Returns:
            /
        """
        
        packet.l3_switch_src_dst()
        packet.l3_set_cf()
        packet.l2_src = self.host.id
        packet.l2_dst = self._routing_table[packet.l3_dst]
        await self.host.send_packet(packet)
    
    async def classical_forwarding(self, packet: Packet) -> None:
        
        """
        Function to forward a packet
        
        Args:
            packet (Packet): packet to forward
            
        Returns:
            /
        """
        
        packet.l2_src = self.host.id
        await self.host.send_packet(packet)
        
    async def no_reject(self, packet: Packet) -> None:
        
        """
        Function to execute if the packet has the no reject flag set
        
        Args:
            packet (Packet): packet to for entanglement swapping
            
        Returns:
            /
        """
        
        if packet.l3_needed > self.host.memory_size(packet.l2_dst, 0):
            await self.reject_packet(packet)
            return
        
        if packet.l3_needed > self.host.l3_num_qubits(packet.l2_dst, 0):
            offset_index = self.host.l3_add_offset(packet.l2_src, 1, packet.l3_needed)
            self.host.l3_store_packet(packet.l2_dst, 0, packet, offset_index)
            await self.prev_protocol.prev_protocol.quantum_data_plane(packet.l2_dst, self.host.remaining_space(packet.l2_dst, 0), True)
            return
            
        await self.quantum_data_plane(packet)
            
    async def partial_reject(self, packet: Packet) -> None:
        
        """
        Function to execute if the packet has the partial reject flag set
        
        Args:
            packet (Packet): packet to for entanglement swapping
            
        Returns:
            /
        """
        
        if not self.host.l3_num_qubits(packet.l2_src, 0):
            await self.reject_packet(packet)
            await self.prev_protocol.prev_protocol.quantum_data_plane(packet.l2_dst, self.host.remaining_space(packet.l2_dst, 0), True)
            return 
        
        if packet.l3_needed > self.host.l3_num_qubits(packet.l2_dst, 0):
            packet.l3_needed = self.host.l3_num_qubits(packet.l2_dst, 0)
            
        await self.quantum_data_plane(packet)
    
    async def complete_reject(self, packet: Packet) -> None:
        
        """
        Function to execute if the packet has the complete reject flag set
        
        Args:
            packet (Packet): packet to for entanglement swapping
            
        Returns:
            /
        """
        
        if packet.l3_needed > self.host.l3_num_qubits(packet.l2_dst, 0):
            await self.reject_packet(packet)
            return
        
        await self.quantum_data_plane(packet)
    
    async def quantum_data_plane(self, packet: Packet, offset_index: int=None) -> None:
        
        """
        Protocol for Quantum Data Plane
        
        Args:
            packet (Packet): packet to use for the entanglement swapping
            offset_index (int): offset for the L3 memory
            
        Returns:
            / 
        """
        
        pass
    
    async def classical_data_plane(self, packet: Packet):
        
        """
        Protocol for the classical data plane
        
        Args:
            packet (Packet): packet to forward
            
        Returns:
            /
        """
        
        packet.l2_dst = self._routing_table[packet.l3_dst]
        
        self._cdp[packet.l3_mode](packet)
      
    async def handle_store_packets(self, host: int, store: int):
        
        """
        Handles stored packets due to no reject mode
        
        Args:
            host (int): host the packet are store to
            store (int): SEND or RECEIVE store
            
        Returns:
            /
        """
        
        for packet_index in range(self.host.l3_num_packets(host, store)):
            packet, offset_index = self.host.l3_retrieve_packet(host, store)
            if self.host.l3_num_qubits(host, store) < packet.l3_needed:
                self.host.l3_store_packet(packet.l2_dst, 0, packet, offset_index)
            await self.quantum_data_plane(packet, offset_index)
            self.host.l3_remove_offset(packet.l2_src, 1, offset_index)
        
class L4_GP(QProgram):
    
    """
    Generic Layer 4 Program
    
    Attrs:
        /
    """
    
    def __init__(self) -> None:
        
        """
        Initializes a generic L4 program
        
        Args:
            /
            
        Returns:
            /
        """
        
        super(L4_GP, self).__init__()
        self.layer = 4   
        
class L7_TPP(QProgram):
    
    """
    Quantum Program for teleporting qubits
    
    Attrs:
        _bsm_mode (str): mode to perform the teleportation with
    """
    
    def __init__(self, bsm_mode: str='bsm') -> None:
        
        """
        Initializes a Teleportation Program
        
        Args:
            bsm_mode (str): mode to perform the teleportation with
            
        Returns:
            /
        """
        
        super(L7_TPP, self).__init__()
        self.layer = 5
        
        self._bsm_mode: str = bsm_mode
        
    async def send(self, data_qubit: Union[Qubit, List[Qubit]], l3_receiver: int, l2_receiver: int):
        
        """
        Teleports qubits to a receiver
        
        Args:
            data_qubit(Qubit/list): qubit or multiple qubits to teleport
            l3_receiver (int): end host to teleport to
            l2_receiver (int): neighbor to teleport to
            
        Returns:
            /
        """
        
        if isinstance(data_qubit, Qubit):
            data_qubit = [data_qubit]
        
        results = []
        for qubit in data_qubit:
        
            com_qubit = self.host.l3_retrieve_qubit(l2_receiver, 0)
            
            res = self.host.apply_gate(self._bsm_mode, qubit, com_qubit, combine=True, remove=True)
            
            results.append(res)
            
        packet = Packet(self.host.id, l2_receiver, l3_src=self.host.id, l3_dst=l3_receiver, l7_requested=len(data_qubit), payload=results) 
        await self.host.send_packet(packet)
        
    async def receive(self):
        
        """
        Receives a teleporting qubit
        
        Args:
            /
            
        Returns:
            qubits (list): received qubits
        """
        
        packet = await self.host.receive_packet()
        
        qubits = []
        for index in packet.l7_requested:
        
            qubit = self.host.l3_retrieve_qubit(packet.l2_src, 1)
            
            if packet.l3_es_result[0][index]:
                self.host.apply_gate('X', qubit)
            if packet.l3_es_result[1][index]:
                self.host.apply_gate('Z', qubit)
            qubits.append(qubit)
        return qubits
    
class L7_DQC(QProgram):
    
    """
    QProgram for executing Quantum Circuits distributed
    
    Attrs:
        /
    """
    
    def __init__(self) -> None:
        
        """
        Initializes a QProgram for executing Quantum Circuits distributed
        
        Args:
            /
            
        Returns:
            /
        """
        
        super(L7_DQC, self).__init__()
        self.layer = 5

class QProgram_Model:
    
    def __init__(self, l1_qprogram: QProgram=QProgram(), l2_qprogram: QProgram=QProgram(), l3_qprogram: QProgram=QProgram(), l4_qprogram: QProgram=QProgram(), l7_qprogram: QProgram=QProgram()):
        
        self._qprograms: List[QProgram] = {L1: l1_qprogram, L2: l2_qprogram, L3: l3_qprogram, L4: l4_qprogram, L7: l7_qprogram}
        self._qprograms = {layer: qprogram for layer, qprogram in self._qprograms.items() if qprogram.layer}
        
        if not all([layer == qprogram.layer for layer, qprogram in self._qprograms.items()]):
            raise ValueError('Cannot have a quantum program without the program of the previous layer')
        
        for layer, qprogram in self._qprograms.items():
            qprogram.host = self
            
            if (layer - 1) in self._qprograms:
                qprogram.prev_protocol = self._qprograms[layer - 1]
            if (layer + 1) in self._qprograms:
                qprogram.next_protocol = self._qprograms[layer + 1]
            
            if not (layer + 1):
                pass
           