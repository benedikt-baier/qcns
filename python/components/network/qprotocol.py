"""Quantum operations and their classical control messages.

Entry points: handle_packet(), request_entanglement(), on_qubits_ready().
Each node owns a stack. Layer handoffs are explicit readiness notifications.
"""
from __future__ import annotations
from copy import copy, deepcopy
import numpy as np
from qcns.python.components.packet.packet import Packet
from qcns.python.components.packet import protocol as headers
from qcns.python.components.qubit.qubit import Qubit
from qcns.python.components.qubit.qec import get_qec_code
# Use the module reference: node also imports QProtocolStack during initialization.
from qcns.python.components.network import node as node_state
from qcns.python.components.qubit.qec import werner_purification, werner_swap
from typing import TypedDict, Unpack
import math

__all__ = [
    'QProtocol', 'QProtocolStack', 'L3_RawEndpoint', 'L3_PR', 'PRProtocolOptions', 'L3_Encoded', 'EncodedProtocolOptions',
    'L1_QProtocol', 'L1_SR', 'L1_TPS', 'L1_BSM', 'L1_FS', 'L1_L3C',
    'L2_QProtocol', 'L2_EP', 'L2_QEC', 'L2_NEC',
    'L3_QProtocol', 'L3_CR', 'L3_FR', 'L3_ER',
    'L4_QProtocol', 'L4_EP', 'L4_QEC', 'L4_NEC',
    'L7_QProtocol', 'L7_TP', 'L7_RG',
    'L1_EGP', 'L2_FIP', 'L3_QFP', 'L4_FIP', 'L7_TPP',
]
L0, L1, L2, L3, L4, L7 = range(6)
SEND, RECEIVE = range(2)

class QProtocol:
    """Base class for a node-owned protocol at one stack layer."""
    layer = 0
    required_layers = ()

    def __init__(self):
        self.host = None
        self.prev_protocol = None
        self.next_protocol = None

    def reset(self):
        """Clear runtime state between simulation runs."""

    def handle_packet(self, packet):
        raise NotImplementedError(f'{type(self).__name__} cannot handle packets')

    def on_qubits_ready(self, neighbor, store, count, request_id=None):
        if self.next_protocol is not None:
            self.next_protocol.on_qubits_ready(neighbor, store, count, request_id)

class L1_QProtocol(QProtocol):
    """Shared generation workflow. Subclasses implement herald interpretation.

    Direct construction is a compatibility selector for built-in modes.
    Subclasses can override any hook without registering a mode globally.
    """
    layer = L1
    eg_mode = None

    def __new__(cls, *args, **kwargs):
        if cls is not L1_QProtocol:
            return object.__new__(cls)
        eg_mode = kwargs.get('eg_mode', args[0] if len(args) > 0 else 'l3cp')
        concrete = {'l3cp': L1_L3C, 'srp': L1_SR, 'tpsp': L1_TPS,
                    'bsmp': L1_BSM, 'fsp': L1_FS}.get(eg_mode or 'l3cp')
        if concrete is None:
            raise ValueError('Unknown entanglement generation mode')
        return object.__new__(concrete)

    def __init__(self, eg_mode=None, rap_mode='attempt', reattempt=True):
        super().__init__()
        if eg_mode is not None and eg_mode != self.eg_mode:
            raise ValueError('Generation mode does not match the protocol class')
        if rap_mode not in {'attempt', 'create'}:
            raise ValueError('Unknown allocation mode')
        self.rap_mode, self.reattempt = rap_mode, reattempt

    def _prepare_results(self, packet):
        """Return (packet, memory direction, success bits), or None while waiting."""
        raise NotImplementedError('Implement herald interpretation')

    def _accept_results(self, neighbor, store, success):
        self.host.l0_move_qubits_l1(neighbor, store, success)

    def _acknowledge(self, packet):
        """Override for protocols requiring a peer acknowledgment."""

    def _notify_ready(self, neighbor, store, count, request_id, packet=None):
        if self.next_protocol is not None:
            self._notify_link_protocol(neighbor, store, count, request_id, packet)
            return
        # Raw confirmed inputs remain in L1 until an improvement is confirmed.

    def request_entanglement(self, neighbor, count=1, estimate=False, *, request=None):
        if count < 1:
            return 0
        if request is not None:
            return self._request_with_header(neighbor, count, estimate, request)
        if self.rap_mode == 'attempt':
            return self.host.attempt_bell_pairs(neighbor, count, estimate)
        return self.host.create_bell_pairs(neighbor, count)

    def attempt_bell_pairs(self, receiver, requested=1, estimate=False):
        return self.host.attempt_bell_pairs(receiver, requested, estimate)

    def create_bell_pairs(self, receiver, requested=1):
        return self.host.create_bell_pairs(receiver, requested)

    def handle_packet(self, packet):
        """Herald a completed request, notify the next layer, and retry gaps."""
        prepared = self._prepare_results(packet)
        if prepared is None:
            return
        packet, store, success = prepared
        confirmed = self.host.confirm_generation(packet.src, store, packet.request_id, success)
        if confirmed is None:
            return  # Duplicate or stale herald must not consume another batch.
        if not confirmed:
            self._accept_results(packet.src, store, success)
        self._acknowledge(packet)
        count = int(np.count_nonzero(success))
        self._notify_ready(packet.src, store, count, packet.request_id, packet)
        self._retry_missing(packet, store, count)

    def _retry_missing(self, packet, store, count):
        if not self.reattempt or store != SEND:
            return
        missing = max(0, packet.l1_requested - count)
        if not missing:
            return
        if packet.has_l2:
            self.request_entanglement(packet.src, missing, estimate=True, request=packet)
            return
        self.request_entanglement(packet.src, missing, estimate=True)

    def _request_with_header(self, neighbor, count, estimate, request):
        if self.rap_mode == 'attempt':
            return self.host.attempt_bell_pairs(neighbor, count, estimate, request=request)
        return self.host.create_bell_pairs(neighbor, count, request=request)

    def _notify_link_protocol(self, neighbor, store, count, request_id, packet):
        if packet is not None and packet.has_l2:
            self.next_protocol.on_qubits_ready(neighbor, store, count, request_id, request=packet)
            return
        self.next_protocol.on_qubits_ready(neighbor, store, count, request_id)

class L1_SR(L1_QProtocol):
    """Sender-receiver generation with return acknowledgments."""
    eg_mode = 'srp'

    def _prepare_results(self, packet):
        return packet, int(not packet.l1_ack), packet.l1_success.copy()

    def _acknowledge(self, packet):
        if packet.l1_ack:
            return
        reply = packet.copy()
        reply.l1_set_ack()
        reply.reverse_hop()
        self.host.send_packet(reply)

class L1_TPS(L1_QProtocol):
    """Two-photon-source herald comparison using node-owned result buffers."""
    eg_mode = 'tpsp'

    def _prepare_results(self, packet):
        if packet.l1_ps:
            return self._handle_tps_source(packet)
        return self._handle_tps_peer(packet)

    def _handle_tps_source(self, packet):
        neighbor, store = packet.src, int(packet.l1_ack)
        self.host.l1_store_result(store, packet)
        reply = packet.copy()
        reply.l1_reset_ps()
        reply.reverse_hop()
        self.host.send_packet(reply)
        if not self.host.l1_check_packets(neighbor, store, request_id=packet.request_id):
            return None
        peer = self.host.l1_retrieve_packet(neighbor, store, request_id=packet.request_id)
        return self._compare_tps_results(peer, store)

    def _handle_tps_peer(self, packet):
        store = int(not packet.l1_ack)
        if not self.host.l1_check_results(packet.src, store, request_id=packet.request_id):
            self.host.l1_store_packet(packet.src, store, packet.copy())
            return None
        return self._compare_tps_results(packet, store)

    def _compare_tps_results(self, packet, store):
        success = self.host.l1_compare_results(packet)
        return packet, store, success

class L1_BSM(L1_QProtocol):
    """Central Bell-measurement heralds identify the local memory direction."""
    eg_mode = 'bsmp'

    def _prepare_results(self, packet):
        return packet, int(packet.l1_ack), packet.l1_success.copy()

class L1_FS(L1_BSM):
    """Fock-state generation uses the same herald envelope as central BSM."""
    eg_mode = 'fsp'

class L1_L3C(L1_SR):
    """Heralded effective Bell-pair generation, using the L0/L1 lifecycle."""
    eg_mode = 'l3cp'

    def _accept_results(self, neighbor, store, success):
        super()._accept_results(neighbor, store, success)

    def _notify_ready(self, neighbor, store, count, request_id, packet=None):
        if packet is None or not packet.has_l2:
            return self._notify_forwarding(neighbor, store, count, request_id)
        super()._notify_ready(neighbor, store, count, request_id, packet)

    def _notify_forwarding(self, neighbor, store, count, request_id):
        protocol = self.next_protocol
        while protocol is not None and protocol.layer != L3:
            protocol = protocol.next_protocol
        if protocol is None:
            return
        protocol.on_qubits_ready(neighbor, store, count, request_id)

class L2_QProtocol(QProtocol):
    """Link-layer buffering, request correlation and resource handoff.

    Override input_block_size, create_header, _prepare_inputs and
    _complete_batch for another quantum operation. State belongs to Node.
    """
    layer = L2
    required_layers = (L1,)
    fip_mode = None
    header_type = None
    completion_handshake = False

    def __new__(cls, *args, **kwargs):
        if cls is not L2_QProtocol:
            return object.__new__(cls)
        fip_mode = kwargs.get('fip_mode', args[0] if len(args) > 0 else 'necp')
        concrete = {'necp': L2_NEC, 'epp': L2_EP, 'qecp': L2_QEC}.get(fip_mode or 'necp')
        if concrete is None:
            raise ValueError('Unknown fidelity improvement mode')
        return object.__new__(concrete)

    def __init__(self, fip_mode=None, reattempt=True):
        super().__init__()
        if fip_mode is not None and fip_mode != self.fip_mode:
            raise ValueError('Fidelity mode does not match the protocol class')
        self.reattempt = reattempt

    def input_block_size(self, request=None):
        raise NotImplementedError('Define the number of input pairs per output')

    def create_header(self, src, dst, requested, needed, code=None, *, request=None):
        """Override this factory or supply a compatible header_type."""
        if self.header_type is None:
            raise NotImplementedError('Supply a header_type or override create_header')
        return self.header_type(src, dst, requested, needed, code)

    def _prepare_inputs(self, neighbor, store, rounds, packet):
        raise NotImplementedError('Implement local quantum operations')

    def _complete_batch(self, neighbor, store, batch, peer):
        raise NotImplementedError('Implement outcome comparison and confirmation')

    def _validate_mode(self, packet):
        if not packet.has_l2:
            raise ValueError('Link control requires an L2 header')

    def _retry_missing(self, neighbor, store, missing):
        """Override to define a retry policy after unsuccessful outputs."""

    def request_more_inputs(self, neighbor, outputs, *, request=None):
        """Translate a forwarding-layer deficit into a lower-layer request."""
        size = self.input_block_size(request)
        code = None if request is None else request.l2_code
        header = self.create_header(self.host.id, neighbor, outputs, size * outputs, code, request=request)
        packet = Packet(self.host.id, neighbor, layer2=header)
        return self.request_entanglement(packet, estimate=True)

    def request_entanglement(self, packet, estimate=False):
        """Generate the raw inputs for an L2 request, including its code header."""
        self._validate_mode(packet)
        size = self.input_block_size(packet)
        if packet.l2_needed < 1 or packet.l2_needed != size * packet.l2_requested:
            raise ValueError('L2 needed must equal block size times requested outputs')
        if self.prev_protocol is None:
            raise RuntimeError('L2 requests require an L1 generation protocol')
        return self.prev_protocol.request_entanglement(packet.dst, packet.l2_needed,
                                                     estimate, request=packet)

    def request_qec_multiple_outputs(self, packet, estimate=False):
        """Placeholder for coordinating QEC with multiple logical outputs."""
        raise NotImplementedError('QEC with multiple logical outputs is not modeled yet')

    def request_qec_non_stabilizer(self, packet, estimate=False):
        """Placeholder for coordinating a non-stabilizer QEC request."""
        raise NotImplementedError('Non-stabilizer QEC is not modeled yet')

    def request_qec_coherent_noise(self, packet, estimate=False):
        """Placeholder for coordinating QEC on inputs with coherent noise."""
        raise NotImplementedError('QEC with coherent noise is not modeled yet')

    def on_qubits_ready(self, neighbor, store, count, request_id=None, *, request=None):
        size = self.input_block_size(request)
        rounds = self.host.l1_num_qubits(neighbor, store) // size
        if not rounds:
            return
        code = None if request is None else request.l2_code
        packet = self._create_packet(neighbor, store, rounds, request_id, code, size, request)
        qubits = self._prepare_inputs(neighbor, store, rounds, packet)
        self.host.l2_store_result(store, packet, qubits)
        self.host.send_packet(packet)
        self._drain_completed(neighbor, store)

    def _create_packet(self, neighbor, store, rounds, request_id, code, size, request):
        header = self.create_header(self.host.id, neighbor, rounds, size * rounds, code, request=request)
        packet = Packet(self.host.id, neighbor, layer2=header, request_id=request_id)
        if store == SEND:
            packet.l2_set_ack()
        return packet

    def handle_packet(self, packet):
        self._validate_mode(packet)
        neighbor, store = packet.src, int(packet.l2_ack)
        if self.completion_handshake and store == SEND and not packet.l2_complete:
            return  # The sender waits for the receiver's decoded completion.
        self.host.l2_store_packet(neighbor, store, packet.copy())
        self._drain_completed(neighbor, store)

    def _drain_completed(self, neighbor, store):
        # Keep the two ends' output ordering consistent when messages reorder.
        while self._complete_next(neighbor, store):
            pass

    def _complete_next(self, neighbor, store):
        batch = self.host.l2_peek_batch(neighbor, store)
        if batch is None:
            return False
        if not self.host.l2_check_packets(neighbor, store, request_id=batch.request_id):
            return False
        peer = next(packet for packet in self.host.l2_packets(neighbor, store)
                    if packet.request_id == batch.request_id)
        self._validate_batch(batch, peer, store)
        self._complete_batch(neighbor, store, batch, peer)
        self.host.l2_retrieve_packet(neighbor, store, request_id=batch.request_id)
        return True

    def _validate_batch(self, batch, peer, store):
        if batch.results.shape != peer.l2_success.shape or batch.needed != peer.l2_needed:
            raise ValueError('L2 result counts do not match')
        if batch.code != peer.l2_code:
            raise ValueError('The two ends requested different QEC codes')
        expected_complete = self.completion_handshake and store == SEND
        if peer.l2_complete != expected_complete:
            raise ValueError('Unexpected L2 completion phase')

    def _confirm_batch(self, neighbor, store, batch, success):
        self.host.l2_confirm_batch(neighbor, store, batch.request_id, success)
        count = int(np.count_nonzero(success))
        super().on_qubits_ready(neighbor, store, count, batch.request_id)
        self._retry_missing(neighbor, store, len(success) - count)

class L2_EP(L2_QProtocol):
    """Bilateral parity-check purification of two input Bell pairs."""
    fip_mode = 'epp'
    header_type = headers.L2_EPP

    def input_block_size(self, request=None):
        return 2

    def _validate_mode(self, packet):
        super()._validate_mode(packet)
        if packet.l2_code is not None or packet.l2_protocol != 0:
            raise ValueError('EPP received a QEC or incompatible control packet')

    def _prepare_inputs(self, neighbor, store, rounds, packet):
        results, qubits = self.host.l2_purify_inputs(neighbor, store, rounds)
        packet.l2_success = results
        return qubits

    def _complete_batch(self, neighbor, store, batch, peer):
        success = self.host.l2_compare_results(peer, consume=False)
        self._confirm_batch(neighbor, store, batch, success)

    def _retry_missing(self, neighbor, store, missing):
        if not self.reattempt or store != SEND or not missing:
            return
        self.prev_protocol.request_entanglement(neighbor, 2 * missing, estimate=True)

class L2_QEC(L2_QProtocol):
    """Packet-selected QEC using the existing ideal effective Pauli decoder."""
    fip_mode = 'qecp'
    header_type = headers.L2_QECP
    completion_handshake = True

    def input_block_size(self, request=None):
        if request is None or request.l2_code is None:
            raise ValueError('QEC requests must specify their code in the L2 header')
        model = get_qec_code(request.l2_code)
        if model.num_logical != 1:
            raise NotImplementedError('L2 pair storage does not yet support correlated multiple logical outputs')
        return model.length

    def _validate_mode(self, packet):
        super()._validate_mode(packet)
        if packet.l2_code is None or packet.l2_protocol != 1:
            raise ValueError('QEC packets must specify their code in the L2 header')

    def _prepare_inputs(self, neighbor, store, rounds, packet):
        self.input_block_size(packet)
        qubits = self.host.l2_stage_qec_inputs(neighbor, store, packet.l2_needed)
        packet.l2_success = np.ones(packet.l2_requested, dtype=bool)
        return qubits

    def _complete_batch(self, neighbor, store, batch, peer):
        # Apply the joint effective operation once, after both endpoints have
        # staged their halves. Individual pair matrices remain 4-by-4.
        if store == RECEIVE:
            self.host.l2_apply_qec(neighbor, store, batch.request_id)
        success = np.logical_and(batch.results, peer.l2_success)
        self._confirm_batch(neighbor, store, batch, success)
        if store == RECEIVE:
            self._send_qec_completion(peer, success)

    def _send_qec_completion(self, peer, success):
        reply = peer.copy()
        reply.reverse_hop()
        reply.l2_reset_ack()
        reply.l2_set_complete()
        reply.l2_success = success
        self.host.send_packet(reply)

class L2_NEC(L2_QProtocol):
    """Forward readiness while leaving confirmed raw pairs in L1."""
    fip_mode = 'necp'

    def input_block_size(self, request=None):
        return 1

    def on_qubits_ready(self, neighbor, store, count, request_id=None, *, request=None):
        QProtocol.on_qubits_ready(self, neighbor, store, count, request_id)

    def request_more_inputs(self, neighbor, outputs, *, request=None):
        if self.prev_protocol is None:
            raise RuntimeError('L2 requests require an L1 generation protocol')
        return self.prev_protocol.request_entanglement(neighbor, outputs, estimate=True)

    def handle_packet(self, packet):
        raise ValueError('NECP does not exchange L2 control packets')

class L3_QProtocol(QProtocol):
    """Shared routing, admission, reservation and classical forwarding.

    Implement _swap_operation for a new ordinary-pair swapping strategy;
    override _forward_quantum_request for a different resource representation.
    """
    layer = L3
    required_layers = (L2,)
    qf_mode = None

    def __new__(cls, *args, **kwargs):
        if cls is not L3_QProtocol:
            return object.__new__(cls)
        qf_mode = kwargs.get('qf_mode', args[1] if len(args) > 1 else 'frp')
        concrete = {'crp': L3_CR, 'frp': L3_FR, 'erp': L3_ER}.get(qf_mode or 'frp')
        if concrete is None:
            raise ValueError('Unknown quantum forwarding mode')
        return object.__new__(concrete)

    def __init__(self, routing_table, qf_mode=None, bsm_mode='bsm'):
        super().__init__()
        if qf_mode is not None and qf_mode != self.qf_mode:
            raise ValueError('Forwarding mode does not match the protocol class')
        if bsm_mode != 'bsm':
            raise ValueError('Only deterministic bsm is supported for swapping')
        self.routing_table, self.bsm_mode = dict(routing_table), bsm_mode

    def swap_entanglement(self, packet, offset_index=None, *, correct=None):
        """Run the installed strategy, retaining the legacy explicit override."""
        if self.host._scheduled_dispatch and correct is None:
            self.host._pending_swaps.append((self, packet, offset_index))
            return
        if correct is not None:
            operation = self.swap_entanglement_crp if correct else self.swap_entanglement_frp
            return operation(packet, offset_index)
        return self._swap_operation(packet, offset_index)

    async def swap_entanglement_scheduled(self, packet: Packet, offset_index: int | None = None) -> None:
        """Extension point for completion-time swapping backends."""
        raise NotImplementedError('Implement scheduled swapping for this protocol')

    def _swap_operation(self, packet, offset_index=None):
        raise NotImplementedError('Implement the swapping operation')

    def swap_entanglement_crp(self, packet, offset_index=None):
        """Compatibility entry point; new code can install L3_CR directly."""
        return L3_CR._swap_operation(self, packet, offset_index)

    def swap_entanglement_frp(self, packet, offset_index=None):
        """Compatibility entry point; new code can install L3_FR directly."""
        return L3_FR._swap_operation(self, packet, offset_index)

    def swap_entanglement_graph_state(self, packet, offset_index=None, **parameters):
        """Compatibility entry point for explicitly supplied encoded records."""
        return L3_ER._recover_graph_state(self, packet, offset_index, **parameters)

    def prepare_graph_state_resource(self, code):
        return L3_ER.prepare_graph_state_resource(self, code)

    def simulate_graph_state_segment(self, code, **parameters):
        return L3_ER.simulate_graph_state_segment(self, code, **parameters)

    def simulate_graph_state_chain(self, code, segments, **parameters):
        return L3_ER.simulate_graph_state_chain(self, code, segments, **parameters)

    def _next_hop(self, destination):
        neighbor = self.routing_table.get(destination)
        if neighbor is None or neighbor not in self.host._channels['pc']:
            raise ValueError(f'Next hop {neighbor} has no classical connection')
        return neighbor

    def handle_packet(self, packet):
        self._validate_request(packet)
        packet.dst = self._next_hop(packet.l3_dst)
        if packet.l3_is_cf:
            return self.classical_forwarding(packet)
        return self._forward_quantum_request(packet)

    def _validate_request(self, packet):
        if not packet.has_l3:
            raise ValueError('Forwarding requires an L3 header')
        if packet.l3_dst == self.host.id:
            raise ValueError('An end-to-end request must be handled by its destination')

    def _forward_quantum_request(self, packet):
        handler = {1: self.no_reject, 2: self.partial_reject,
                   3: self.complete_reject}.get(packet.l3_fmode)
        if handler is None:
            raise ValueError('Unknown forwarding rejection mode')
        return handler(packet)

    def classical_forwarding(self, packet):
        packet.set_hop(self.host.id, packet.dst)
        packet.l3_hop_count += 1
        self.host.send_packet(packet)

    def reject_packet(self, packet):
        reply = packet.copy()
        reply.rejected = True
        reply.l3_switch_src_dst()
        reply.l3_set_cf()
        reply.set_hop(self.host.id, self._next_hop(reply.l3_dst))
        self.host.send_packet(reply)

    def _available(self, packet):
        return min(self.host.ready_pair_count(packet.src, RECEIVE),
                   self.host.ready_pair_count(packet.dst, SEND))

    def no_reject(self, packet):
        if not self._can_accept_request(packet):
            return self.reject_packet(packet)
        if self.host.ready_pair_count(packet.dst, SEND) < packet.l3_needed:
            return self._queue_request(packet)
        return self.swap_entanglement(packet)

    def _can_accept_request(self, packet):
        capacity = self.host.memory_size(packet.dst, SEND)
        if capacity >= 0 and packet.l3_needed > capacity:
            return False
        return self.host.ready_pair_count(packet.src, RECEIVE) >= packet.l3_needed

    def _queue_request(self, packet):
        reservation = self.host.l3_add_offset(packet.src, RECEIVE, packet.l3_needed)
        self.host.l3_store_packet(packet.dst, SEND, packet.copy(), reservation)
        self._request_missing(packet.dst, request=packet)

    def partial_reject(self, packet):
        count = self._available(packet)
        if not count:
            next_hop = packet.dst
            self.reject_packet(packet)
            self._request_missing(next_hop, packet.l3_needed, request=packet)
            return
        packet.l3_needed = min(packet.l3_needed, count)
        self.swap_entanglement(packet)

    def complete_reject(self, packet):
        if self._available(packet) < packet.l3_needed:
            return self.reject_packet(packet)
        self.swap_entanglement(packet)

    def on_qubits_ready(self, neighbor, store, count, request_id=None):
        if store == SEND:
            self.handle_stored_packets(neighbor, store)

    def handle_stored_packets(self, neighbor, store=SEND):
        """Drain FIFO requests whose incoming and outgoing resources are ready."""
        while self._complete_next_request(neighbor, store):
            pass

    def _complete_next_request(self, neighbor, store):
        if not self.host.l3_check_packets(neighbor, store):
            return False
        packet, reservation = self.host.l3_packets(neighbor, store)[0]
        if not self.host.l3_can_swap(packet, reservation):
            return False
        source = packet.src
        self.swap_entanglement(packet, reservation)
        self.host.l3_retrieve_packet(neighbor, store)
        if self.host._scheduled_dispatch:
            return False
        self.host.l3_remove_offset(source, RECEIVE, reservation)
        return True

    def _request_missing(self, neighbor, needed=None, *, request=None):
        if needed is None:
            needed = sum(packet.l3_needed for packet, _ in self.host.l3_packets(neighbor, SEND))
        missing = max(0, needed - self.host.ready_pair_count(neighbor, SEND)
                      - self.host.l0_num_qubits(neighbor, SEND))
        if not missing:
            return
        protocol = self.prev_protocol
        while protocol is not None and protocol.layer != L1:
            protocol = protocol.prev_protocol
        if protocol is None:
            raise RuntimeError('Waiting for Bell pairs requires an L1 generation protocol')
        if request is None and self.host.l3_check_packets(neighbor, SEND):
            request = self.host.l3_packets(neighbor, SEND)[0][0]
        link = protocol.next_protocol
        if link is not None and link.layer == L2:
            return link.request_more_inputs(neighbor, missing, request=request)
        protocol.request_entanglement(neighbor, missing, estimate=True)

class L3_CR(L3_QProtocol):
    """Apply incoming corrections locally before each raw swapping operation."""
    qf_mode = 'crp'

    async def swap_entanglement_scheduled(self, packet: Packet, offset_index: int | None = None) -> None:
        await self.host.swap_pairs_scheduled(packet, offset_index, correct=True)
        self.classical_forwarding(packet)

    def _swap_operation(self, packet, offset_index=None):
        """Apply the incoming Pauli frame locally, swap, and forward results."""
        self.host.apply_entanglement_swapping_crp(packet, offset_index)
        self.classical_forwarding(packet)

class L3_FR(L3_QProtocol):
    """Carry the accumulated Pauli frame to the destination."""
    qf_mode = 'frp'

    async def swap_entanglement_scheduled(self, packet: Packet, offset_index: int | None = None) -> None:
        await self.host.swap_pairs_scheduled(packet, offset_index, correct=False)
        self.classical_forwarding(packet)

    def _swap_operation(self, packet, offset_index=None):
        """Swap while carrying the accumulated Pauli frame to the destination."""
        self.host.apply_entanglement_swapping_frp(packet, offset_index)
        self.classical_forwarding(packet)

class L3_ER(L3_QProtocol):
    """Encoded graph-state recovery with explicit code and measurement records.

    Automatic scheduling of encoded resources is not implemented. Ordinary
    Bell-pair buffers must not be used as encoded resource blocks.
    """
    qf_mode = 'erp'

    def _forward_quantum_request(self, packet):
        raise NotImplementedError('Encoded forwarding requires an encoded-resource scheduler')

    def _swap_operation(self, packet, offset_index=None):
        raise NotImplementedError('Use swap_entanglement_graph_state with a code and both records')

    def prepare_graph_state_resource(self, code):
        return self.host.prepare_graph_state_resource(code)

    def simulate_graph_state_segment(self, code, **parameters):
        return self.host.simulate_graph_state_segment(code, **parameters)

    def simulate_graph_state_chain(self, code, segments, **parameters):
        return self.host.simulate_graph_state_chain(code, segments, **parameters)

    def _recover_graph_state(self, packet, offset_index=None, *,
                                      code=None, first_record=None, second_record=None):
        """Recover encoded measurement frames and forward their classical record."""
        recovery = self.host.apply_entanglement_swapping_graph_state(
            packet, offset_index, code=code,
            first_record=first_record, second_record=second_record)
        self.classical_forwarding(packet)
        return recovery

class L3_RawEndpoint(L3_QProtocol):
    """Raw FRP endpoint: generate, retain a source half, correct and acknowledge.

    Routers still use L3_FR. Fidelity is carried in the header but is not an
    acceptance test. No purification or encoded resource is used. Timeout bounds
    each request; disposal of a stopped experiment releases in-flight resources.
    """
    required_layers = ()

    def __init__(self, routing_table: dict[int, int], *, timeout: float = 1.,
                 max_attempts: int = 64) -> None:
        super().__init__(routing_table)
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError('Raw timeout must be positive finite seconds')
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError('Raw max_attempts must be a positive integer')
        self.timeout, self.max_attempts = timeout, max_attempts
        self.reset()

    def reset(self) -> None:
        self._generating = False
        self._waiting_id: int | None = None
        self._answer: Packet | None = None

    def on_qubits_ready(self, neighbor: int, store: int, count: int,
                        request_id: int | None = None) -> None:
        if store == SEND:
            self._generating = False

    def handle_packet(self, packet: Packet) -> None:
        raise RuntimeError('Raw endpoints use await serve() or await establish()')

    async def serve(self) -> None:
        while True:
            await self._pump(None)

    async def serve_until(self, deadline: float) -> None:
        """Service incoming packets until an absolute local simulation time."""
        while self.host._time < deadline:
            if not await self._pump(deadline):
                return

    async def _pump(self, deadline: float | None) -> bool:
        timeout = None if deadline is None else max(0., deadline - self.host._time)
        packet = await self.host.receive_packet(time_out=timeout)
        if packet is None:
            return False
        if not packet.has_l3:
            await self.host.handle_packet_scheduled(packet)
            return True
        if packet.l3_dst != self.host.id:
            raise ValueError('Misrouted raw endpoint packet')
        if packet.l3_is_cf or packet.rejected:
            if packet.request_id == self._waiting_id:
                self._answer = packet
            return True
        await self._accept(packet)
        return True

    @staticmethod
    def _tokens(packet: Packet) -> tuple[tuple[int, int], ...]:
        return tuple((packet.l3_src, packet.request_id * 256 + i) for i in range(packet.l3_needed))

    async def _accept(self, packet: Packet) -> None:
        key = (packet.l3_src, packet.request_id)
        accepted = True
        for index, token in enumerate(self._tokens(packet)):
            captured = self.host.pr_capture(packet.src, RECEIVE, token, key)
            accepted &= captured
            if captured:
                outcome = int(packet.l3_x_count[index]) | (int(packet.l3_z_count[index]) << 1)
                accepted &= await self.host.execute_pr_correction(token, outcome, token)
        if not accepted:
            self.host.pr_discard_request(key)
        packet.rejected = not accepted
        packet.l3_switch_src_dst()
        packet.l3_set_cf()
        packet.set_hop(self.host.id, self._next_hop(packet.l3_dst))
        self.host.send_packet(packet)

    async def establish(self, destination: int, *, requested: int = 1,
                        fidelity: float = .5) -> node_state.PRResult:
        if self._waiting_id is not None:
            raise RuntimeError('Raw endpoints initiate one request at a time')
        if destination == self.host.id or isinstance(requested, bool) or not isinstance(requested, int) or not 1 <= requested <= 256:
            raise ValueError('Raw requests require a remote destination and 1..256 pairs')
        neighbor = self._next_hop(destination)
        packet = Packet(self.host.id, neighbor, l3_src=self.host.id, l3_dst=destination,
                        l3_requested=requested, l3_needed=requested, l3_protocol=1,
                        l3_fidelity=fidelity)
        key, tokens = (self.host.id, packet.request_id), self._tokens(packet)
        deadline = self.host._time + self.timeout
        self._waiting_id, self._answer = packet.request_id, None
        try:
            await self._generate(neighbor, requested, deadline)
            for token in tokens:
                if not self.host.pr_capture(neighbor, SEND, token, key):
                    raise node_state.PRFailure('source extraction failed')
            self.host.send_packet(packet)
            while self._answer is None:
                if not await self._pump(deadline):
                    raise node_state.PRFailure('raw request timeout')
            if self._answer.rejected:
                raise node_state.PRFailure('raw request rejected or destination extraction failed')
            return node_state.PRResult(key, 'complete', tokens, finished_at=self.host._time)
        except node_state.PRFailure as error:
            self.host.pr_discard_request(key)
            return node_state.PRResult(key, 'failed', finished_at=self.host._time, reason=str(error))
        finally:
            self._waiting_id = None
            self._answer = None

    async def _generate(self, neighbor: int, count: int, deadline: float) -> None:
        for _ in range(self.max_attempts):
            missing = count - self.host.ready_pair_count(neighbor, SEND)
            if missing <= 0:
                return
            if self.host._time >= deadline:
                raise node_state.PRFailure('raw generation timeout')
            self._generating = True
            self.host.attempt_bell_pairs(neighbor, missing)
            while self._generating:
                if not await self._pump(deadline):
                    raise node_state.PRFailure('raw generation timeout')
        if self.host.ready_pair_count(neighbor, SEND) < count:
            raise node_state.PRFailure('raw generation attempt limit')


class PRProtocolOptions(TypedDict, total=False):
    """Optional PR limits; timeout and effective coherence time are in seconds."""
    coherence_time: float
    timeout: float
    max_rounds: int
    max_attempts: int
    max_raw_pairs: int
    max_requests: int
    generate: bool


class L3_PR(L3_QProtocol):
    """Proposed hierarchical PR protocol with lazy FIFO session production.

    BBPSSW operations use actual local qubits; Werner formulas are planning
    estimates only. All remote decisions and outcomes arrive as packets.
    One service coroutine per node serializes its resource operations.
    """
    qf_mode = 'prp'
    required_layers = ()

    def __init__(self, routing_table: dict[int, int], *, coherence_time: float = float('inf'),
                 timeout: float = 1., max_rounds: int = 8, max_attempts: int = 64,
                 max_raw_pairs: int = 4096, max_requests: int = 64, generate: bool = True) -> None:
        super().__init__(routing_table)
        if not np.isfinite(timeout) or timeout <= 0:
            raise ValueError('PR timeout must be positive and finite')
        if coherence_time <= 0 or np.isnan(coherence_time):
            raise ValueError('PR coherence_time must be positive')
        for value in (max_rounds, max_attempts, max_raw_pairs, max_requests):
            if not isinstance(value, int) or value < 1:
                raise ValueError('PR resource limits must be positive integers')
        self.coherence_time, self.timeout = coherence_time, timeout
        self.max_rounds, self.max_attempts = max_rounds, max_attempts
        self.max_raw_pairs, self.max_requests, self.generate = max_raw_pairs, max_requests, generate
        self.reset()

    def reset(self) -> None:
        self._serial = 0
        self._replies = {}
        self._pending = set()
        self._initiating = False
        self._raw_ready = {}
        self._generation_pending = set()
        self._deferred: list[Packet] = []
        self._acquiring = False

    def _command_ready(self, packet: Packet) -> bool:
        header = packet.layer3
        if not isinstance(header, headers.L3_PRP) or header.reply or header.dst != self.host.id:
            return True
        if header.kind != 'ACQUIRE':
            return True
        if self._acquiring:
            return False
        entry = self.host.pr_requests.get(header.key)
        if entry is None or entry.status != 'active' or self.host._time >= entry.deadline:
            return True
        session = self._session(header.key, header.data['left'], header.data['right'])
        return session.requests[0] == header.key

    def _identifier(self) -> tuple[int, int]:
        self._serial += 1
        return (self.host.id, self._serial)

    def handle_packet(self, packet: Packet) -> None:
        raise RuntimeError('PR control uses await serve() or await establish(), not synchronous dispatch')

    def on_qubits_ready(self, neighbor: int, store: int, count: int, request_id: int | None = None) -> None:
        identity = (neighbor, store)
        self._raw_ready[identity] = self._raw_ready.get(identity, 0) + count
        self._generation_pending.discard(identity)

    async def serve(self) -> None:
        while True:
            await self._pump(None)

    async def serve_until(self, deadline: float) -> None:
        """Service delivered messages until an absolute simulation-time arrival."""
        while self.host._time < deadline:
            try:
                await self._pump(deadline)
            except node_state.PRFailure as error:
                if str(error) != 'timeout':
                    raise
                return

    def _send(self, remote: int, kind: str, key: tuple[int, int], transaction: tuple[int, int], data: dict, *, reply: bool = False) -> None:
        header = headers.L3_PRP(self.host.id, remote, kind, key, transaction, data, reply=reply)
        packet = Packet(self.host.id, self._next_hop(remote), layer3=header, request_id=key[1])
        self.host.send_packet(packet)

    async def _pump(self, deadline: float | None) -> None:
        remaining = None if deadline is None else max(0., deadline - self.host._time)
        packet = next((value for value in self._deferred if self._command_ready(value)), None)
        if packet is not None:
            self._deferred.remove(packet)
        else:
            packet = await self.host.receive_packet(time_out=remaining)
        if packet is None:
            raise node_state.PRFailure('timeout')
        if not isinstance(packet.layer3, headers.L3_PRP):
            return self.host.handle_packet(packet)
        header = packet.layer3
        try:
            self._observe(header)
        except node_state.PRFailure as error:
            self._send(header.src, header.kind, header.key, header.transaction,
                       {'ok': False, 'reason': str(error)}, reply=True)
            return
        if header.dst != self.host.id:
            packet.set_hop(self.host.id, self._next_hop(header.dst))
            self.host.send_packet(packet)
            return
        if header.reply:
            if header.transaction in self._pending:
                self._replies[header.transaction] = header.data
            return
        if not self._command_ready(packet):
            if len(self._deferred) >= self.max_requests:
                self._send(header.src, header.kind, header.key, header.transaction,
                           {'ok': False, 'reason': 'pending command capacity exhausted'}, reply=True)
                return
            self._deferred.append(packet)
            return
        acquiring = header.kind == 'ACQUIRE'
        if acquiring:
            self._acquiring = True
        try:
            value = await self._command(header)
            data = {'ok': True, 'value': value}
        except node_state.PRFailure as error:
            data = {'ok': False, 'reason': str(error)}
        finally:
            if acquiring:
                self._acquiring = False
        kind = 'SYN_ACK' if header.kind == 'SYN' else header.kind
        self._send(header.src, kind, header.key, header.transaction, data, reply=True)

    async def _rpc(self, remote: int, kind: str, key: tuple[int, int], data: dict, *, deadline: float | None = None) -> dict | int | bool | None:
        transaction = self._identifier()
        self._pending.add(transaction)
        try:
            self._send(remote, kind, key, transaction, data)
            if deadline is None:
                deadline = self.host.pr_requests[key].deadline
            while transaction not in self._replies:
                await self._pump(deadline)
            reply = self._replies.pop(transaction)
            if not reply['ok']:
                raise node_state.PRFailure(reply['reason'])
            return reply['value']
        finally:
            self._pending.discard(transaction)

    def _observe(self, header: headers.L3_PRP) -> None:
        key, data = header.key, header.data
        if header.kind == 'SYN' and not header.reply:
            if key in self.host.pr_requests:
                raise node_state.PRFailure('duplicate request key')
            if len(self.host.pr_requests) >= self.max_requests:
                raise node_state.PRFailure('request table capacity exhausted')
            path = tuple(data['path'])
            if self.host.id in path or len(path) >= 64:
                raise node_state.PRFailure('route loop or path length limit')
            path += (self.host.id,)
            entry = node_state.PRRequest(key, header.dst, data['count'], data['fidelity_ir'],
                              data['deadline'], path, len(path)-1, len(path)-1)
            self.host.pr_requests[key] = entry
            data['path'] = path
            return
        if key not in self.host.pr_requests:
            return  # A late message cannot resurrect a completed/unknown request.
        entry = self.host.pr_requests[key]
        if entry.status in ('complete', 'failed') and header.kind != 'CANCEL':
            return
        if header.kind == 'SYN_ACK' and data['ok']:
            entry.path = tuple(data['value']['path'])
            entry.status = 'syn_ack'
        if header.kind == 'ACK' and not header.reply:
            self._activate(key)
        if header.kind == 'RESET' and not header.reply:
            self._restore(key, data['left'], data['right'], data.get('empty', False))
        if header.kind in ('COMPLETE', 'CANCEL') and not header.reply:
            self._finish(key, data.get('outputs', ()), header.kind == 'COMPLETE')

    def _activate(self, key: tuple[int, int]) -> None:
        entry = self.host.pr_requests[key]
        entry.status = 'active'
        index = entry.hop_index
        if index:
            self._session(key, index - 1, index)
        if index + 1 < len(entry.path):
            self._session(key, index, index + 1)

    def _session(self, key: tuple[int, int], left: int, right: int) -> node_state.PRSession:
        entry = self.host.pr_requests[key]
        owner, peer = entry.path[left], entry.path[right]
        identity = (owner, peer, entry.fidelity_ir)
        session = self.host.pr_sessions.get(identity)
        if session is None:
            session = node_state.PRSession(owner, peer, entry.fidelity_ir, SEND if self.host.id == owner else RECEIVE)
            self.host.pr_sessions[identity] = session
        if key not in session.requests:
            session.requests.append(key)
        session.status = 'open'
        return session

    def _restore(self, key: tuple[int, int], left: int, right: int, empty: bool = False) -> None:
        entry = self.host.pr_requests[key]
        index = entry.hop_index
        if left < index < right:
            entry.hierarchy_index = index
            entry.owner_ready = entry.peer_ready = False
            entry.resets += 1
        if left <= index < right:
            self._session(key, index, index + 1)
        if empty and index in (left, right):
            session = self._session(key, left, right)
            session.generation += 1
            session.status = 'resetting'

    def _finish(self, key: tuple[int, int], outputs: tuple[tuple[int, int], ...], success: bool) -> None:
        entry = self.host.pr_requests[key]
        tokens = tuple(tuple(token) for token in outputs)
        entry.status = 'complete' if success else 'failed'
        entry.outputs = list(tokens) if entry.hop_index in (0, len(entry.path)-1) else []
        entry.owner_ready = entry.peer_ready = False
        self.host.pr_discard_request(key, keep=entry.outputs)
        for session in self.host.pr_sessions.values():
            if key in session.requests:
                session.requests.remove(key)
            for level, pairs in session.cache.items():
                session.cache[level] = [pair for pair in pairs if self.host.pr_has(pair.token)]
            if not session.requests and not any(session.cache.values()):
                session.status = 'idle'

    async def establish(self, destination: int, *, requested: int = 1, fidelity: float = .97) -> node_state.PRResult:
        if self._initiating:
            raise RuntimeError('A client can initiate only one PR request at a time')
        if destination == self.host.id or not isinstance(requested, int) or requested < 1:
            raise ValueError('PR requests need a remote destination and positive output count')
        if len(self.host.pr_requests) >= self.max_requests:
            raise ValueError('Request table capacity exhausted; archive completed requests first')
        fidelity_ir = headers.encode_fidelity(fidelity)
        identifier = Packet(self.host.id, destination).request_id
        key = (self.host.id, identifier)
        deadline = self.host._time + self.timeout
        self.host.pr_requests[key] = node_state.PRRequest(key, destination, requested, fidelity_ir, deadline,
                                               (self.host.id,))
        self._initiating = True
        try:
            answer = await self._rpc(destination, 'SYN', key, {
                'path': (self.host.id,), 'count': requested,
                'fidelity_ir': fidelity_ir, 'deadline': deadline})
            self.host.pr_requests[key].path = tuple(answer['path'])
            self._activate(key)
            await self._rpc(destination, 'ACK', key, {})
            pairs = [await self._acquire(key, 0, len(answer['path'])-1) for _ in range(requested)]
            tokens = tuple(pair.token for pair in pairs)
            await self._rpc(destination, 'COMPLETE', key, {'outputs': tokens})
            estimate = min(self._estimate(pair) for pair in pairs)
            if estimate < headers.decode_fidelity(fidelity_ir):
                raise node_state.PRFailure('output fidelity estimate expired while awaiting completion')
            self._finish(key, tokens, True)
            return node_state.PRResult(key, 'complete', tokens, estimate, self.host._time)
        except node_state.PRFailure as error:
            reason = str(error)
            try:
                await self._rpc(destination, 'CANCEL', key, {}, deadline=self.host._time + self.timeout)
            except node_state.PRFailure:
                reason += '; cancellation acknowledgment timed out'
            self._finish(key, (), False)
            return node_state.PRResult(key, 'failed', finished_at=self.host._time, reason=reason)
        finally:
            self._initiating = False

    async def _command(self, header: headers.L3_PRP) -> dict | int | bool | None:
        key, data, kind = header.key, header.data, header.kind
        entry = self.host.pr_requests.get(key)
        if entry is None:
            raise node_state.PRFailure('unknown request')
        if kind == 'SYN':
            return {'path': entry.path}
        if kind in ('ACK', 'RESET', 'COMPLETE', 'CANCEL'):
            return None
        if entry.status != 'active' or self.host._time >= entry.deadline:
            raise node_state.PRFailure('inactive or expired request')
        if kind == 'RAW':
            token = tuple(data['token'])
            await self._ensure_raw(header.src, RECEIVE, key)
            return self.host.pr_capture(header.src, RECEIVE, token, key)
        if kind == 'OPEN':
            self._session(key, data['left'], data['right'])
            return None
        if kind == 'ACQUIRE':
            pair = await self._acquire(key, data['left'], data['right'])
            return {'token': pair.token, 'fidelity': self._estimate(pair), 'created': self.host._time}
        if kind == 'PURIFY':
            self._require_tokens(key, data['first'], data['second'])
            return await self.host.execute_pr_purification(tuple(data['first']), tuple(data['second']))
        if kind == 'TWIRL':
            self._require_tokens(key, data['token'])
            return await self.host.execute_pr_twirl(tuple(data['token']))
        if kind == 'DROP':
            self.host.pr_discard(tuple(data['token']))
            return None
        if kind == 'CORRECT':
            self._require_tokens(key, data['old'])
            return await self.host.execute_pr_correction(tuple(data['old']), data['outcome'], tuple(data['new']))
        if kind == 'SWAP':
            return await self._swap_command(key, data)
        raise node_state.PRFailure(f'Unknown PR message {kind}')

    def _require_tokens(self, key: tuple[int, int], *tokens: tuple[int, int]) -> None:
        if any(not self.host.pr_owns(tuple(token), key) for token in tokens):
            raise node_state.PRFailure('Stale or unowned PR resource token')

    def _check_active(self, key: tuple[int, int]) -> None:
        entry = self.host.pr_requests[key]
        if entry.status != 'active' or self.host._time >= entry.deadline:
            raise node_state.PRFailure('inactive or expired request')

    async def _ensure_raw(self, neighbor: int, store: int, key: tuple[int, int]) -> None:
        entry = self.host.pr_requests[key]
        if self._next_hop(neighbor) != neighbor:
            raise node_state.PRFailure('Elementary session requires an adjacent PR endpoint')
        if not self.generate:
            if not self.host.ready_pair_count(neighbor, store):
                raise node_state.PRFailure('raw pair supply exhausted')
            return
        identity = (neighbor, store)
        attempts = 0
        while self._raw_ready.get(identity, 0) < 1:
            if store == SEND and identity not in self._generation_pending:
                if attempts >= self.generation_attempt_limit:
                    raise node_state.PRFailure('generation retry limit')
                self._generation_pending.add(identity)
                self.host.attempt_bell_pairs(neighbor, 1)
                attempts += 1
            await self._pump(entry.deadline)
        self._raw_ready[identity] -= 1
        if not self.host.ready_pair_count(neighbor, store):
            raise node_state.PRFailure('Heralded raw pair is no longer available')

    @property
    def generation_attempt_limit(self) -> int:
        return self.max_attempts

    @staticmethod
    def _rank(index: int) -> int:
        level = 0
        while index & 1:
            level += 1
            index //= 2
        return level

    async def _swap_command(self, key: tuple[int, int], data: dict) -> None:
        entry = self.host.pr_requests[key]
        entry.hierarchy_index = entry.hop_index >> self._rank(entry.hop_index)
        if entry.hop_index in (0, len(entry.path)-1) or entry.hierarchy_index % 2:
            raise node_state.PRFailure('Only even-hierarchy intermediate routers can swap')
        self._require_tokens(key, data['first'], data['second'])
        first, second = tuple(data['first']), tuple(data['second'])
        entry.peer_ready = self.host.pr_has(first)
        entry.owner_ready = self.host.pr_has(second)
        if not entry.peer_ready or not entry.owner_ready:
            raise node_state.PRFailure('Both session reservations are required before swapping')
        outcome = await self.host.execute_pr_swap(first, second)
        entry.peer_ready = entry.owner_ready = False
        if outcome is None:
            raise node_state.PRFailure('swapping extraction failed')
        entry.swaps += 1
        # The measurement record travels towards dst before completion is acknowledged.
        confirmed = await self._rpc(data['destination'], 'CORRECT', key,
                                    {'old': second, 'new': first, 'outcome': outcome})
        if not confirmed:
            raise node_state.PRFailure('remote correction extraction failed')
        return None

    def _estimate(self, pair: node_state.PRPair) -> float:
        elapsed = max(0., self.host._time - pair.created)
        return .25 + (pair.fidelity - .25) * math.exp(-elapsed / self.coherence_time)

    async def _raw_segment(self, key: tuple[int, int], left: int, right: int) -> node_state.PRPair:
        entry = self.host.pr_requests[key]
        if right == left + 1:
            if entry.raw_pairs >= self.max_raw_pairs:
                raise node_state.PRFailure('raw input budget exhausted')
            peer = entry.path[right]
            await self._ensure_raw(peer, SEND, key)
            token = self._identifier()
            local = self.host.pr_capture(peer, SEND, token, key)
            remote = await self._rpc(peer, 'RAW', key, {'token': token})
            entry.raw_pairs += 1
            if not local or not remote:
                self.host.pr_discard(token)
                await self._rpc(peer, 'DROP', key, {'token': token})
                raise node_state.PairExtractionFailure('raw pair extraction failed')
            fidelity = self.host.estimate_link_fidelity(peer, SEND)
            return node_state.PRPair(token, fidelity, self.host._time)
        middle = max(range(left + 1, right), key=self._rank)
        first = await self._acquire(key, left, middle)
        reply = await self._rpc(entry.path[middle], 'ACQUIRE', key, {'left': middle, 'right': right})
        second = node_state.PRPair(tuple(reply['token']), reply['fidelity'], reply['created'])
        await self._rpc(entry.path[middle], 'SWAP', key,
                        {'first': first.token, 'second': second.token, 'destination': entry.path[right]})
        await self._rpc(entry.path[right], 'OPEN', key, {'left': left, 'right': right})
        entry.hierarchy_index = entry.hop_index >> (self._rank(middle) + 1)
        return node_state.PRPair(first.token, werner_swap(self._estimate(first), self._estimate(second)), self.host._time)

    async def _acquire(self, key: tuple[int, int], left: int, right: int) -> node_state.PRPair:
        entry = self.host.pr_requests[key]
        self._check_active(key)
        if not 0 <= left < right < len(entry.path) or entry.path[left] != self.host.id:
            raise node_state.PRFailure('Invalid session ownership or path interval')
        session = self._session(key, left, right)
        while session.requests[0] != key:
            await self._pump(entry.deadline)
        target = headers.decode_fidelity(entry.fidelity_ir)
        for level in range(self.max_rounds + 1):
            pair = await self._at_level(key, left, right, level)
            if self._estimate(pair) >= target:
                return pair
            session.cache.setdefault(level, []).append(pair)
        raise node_state.PRFailure('purification round limit reached')

    async def _at_level(self, key: tuple[int, int], left: int, right: int, level: int) -> node_state.PRPair:
        self._check_active(key)
        session = self._session(key, left, right)
        cached = session.cache.setdefault(level, [])
        if cached:
            return cached.pop(0)
        if level == 0:
            return await self._raw_segment(key, left, right)
        entry = self.host.pr_requests[key]
        peer = entry.path[right]
        while entry.purification_attempts < self.max_attempts:
            first = await self._at_level(key, left, right, level - 1)
            second = await self._at_level(key, left, right, level - 1)
            estimate, _ = werner_purification(self._estimate(first), self._estimate(second))
            local = await self.host.execute_pr_purification(first.token, second.token)
            remote = await self._rpc(peer, 'PURIFY', key, {'first': first.token, 'second': second.token})
            entry.purification_attempts += 1
            if local is not None and remote is not None and local == remote:
                await self.host.wait(self.host._gate_duration.get('twirl', 5e-6))
                if await self._rpc(peer, 'TWIRL', key, {'token': first.token}):
                    self.host.pr_confirm(first.token)
                    return node_state.PRPair(first.token, estimate, self.host._time)
            self.host.pr_discard(first.token)
            await self._rpc(peer, 'DROP', key, {'token': first.token})
            entry.failed_attempts += 1
            empty = not any(session.cache.values())
            self._restore(key, left, right, empty)
            await self._rpc(peer, 'RESET', key, {'left': left, 'right': right, 'empty': empty})
        raise node_state.PRFailure('purification attempt limit reached')


class EncodedProtocolOptions(PRProtocolOptions, total=False):
    code: str
    max_retries: int
    purification_rounds: int
    preparation_duration: float
    max_graph_qubits: int


class L3_Encoded(L3_PR):
    """Sequential encoded forwarding using ideal effective graph interfaces.

    Reuses the PR service's classical transactions, local memory reservations,
    herald handling and cancellation. Each elementary segment consumes n raw
    pairs; logical interface fusion extends the source pair. Decoding is exact
    for independent Bell-diagonal noise, without a 4n-qubit density matrix.
    """
    qf_mode = 'erp'

    def __init__(self, routing_table: dict[int, int], *, code: str = 'five_qubit',
                 max_retries: int = 3, purification_rounds: int = 0,
                 preparation_duration: float = 1e-4, max_graph_qubits: int = 4096,
                 **options: Unpack[PRProtocolOptions]) -> None:
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError('max_retries must be a nonnegative integer')
        if isinstance(purification_rounds, bool) or not isinstance(purification_rounds, int) or not 0 <= purification_rounds <= 8:
            raise ValueError('purification_rounds must be an integer from zero to eight')
        if not np.isfinite(preparation_duration) or preparation_duration < 0:
            raise ValueError('preparation_duration must be finite nonnegative seconds')
        if get_qec_code(code).method != 'exact':
            raise NotImplementedError('Encoded routing currently requires exact [[n,1]] stabilizer decoding')
        self.code = code
        self.max_retries = max_retries
        self.purification_rounds = purification_rounds
        self.preparation_duration = preparation_duration
        if isinstance(max_graph_qubits, bool) or not isinstance(max_graph_qubits, int) or max_graph_qubits < 1:
            raise ValueError('max_graph_qubits must be a positive integer')
        self.max_graph_qubits = max_graph_qubits
        super().__init__(routing_table, **options)

    @property
    def generation_attempt_limit(self) -> int:
        return self.max_retries + 1

    def _request_code(self, key: tuple[int, int]) -> str:
        entry = self.host.pr_requests.get(key)
        return self.code if entry is None or entry.code is None else entry.code

    async def establish(self, destination: int, *, requested: int = 1, fidelity: float = .97) -> node_state.EncodedResult:
        result = await super().establish(destination, requested=requested, fidelity=fidelity)
        return node_state.EncodedResult(result.key, result.status, result.outputs, result.estimated_fidelity,
                             result.finished_at, result.reason, code=self._request_code(result.key))

    def _send(self, remote: int, kind: str, key: tuple[int, int], transaction: tuple[int, int],
              data: dict, *, reply: bool = False) -> None:
        if reply and not data.get('ok', True):
            kind = 'FAILURE'
        header = headers.L3_ECP(self.host.id, remote, kind, key, transaction,
                                           data, code=self._request_code(key), reply=reply)
        self.host.send_packet(Packet(self.host.id, self._next_hop(remote), layer3=header, request_id=key[1]))

    def _observe(self, header: headers.L3_PRP) -> None:
        if not isinstance(header, headers.L3_ECP):
            raise node_state.PRFailure('Encoded routing requires encoded control headers')
        if header.kind == 'SYN' and not header.reply:
            backend = get_qec_code(header.code)
            if backend.method != 'exact':
                raise node_state.PRFailure('Encoded routing requires an exact code')
            if backend.length + 1 > self.max_graph_qubits:
                raise node_state.PRFailure('Graph resource exceeds max_graph_qubits per local interface')
        super()._observe(header)
        if header.key in self.host.pr_requests:
            self.host.pr_requests[header.key].code = header.code

    async def _command(self, header: headers.L3_ECP) -> dict | int | bool | None:
        if header.kind != 'ENCODE_REFRESH' and header.kind != 'ENCODE_FINISH':
            return await super()._command(header)
        self._check_active(header.key)
        tokens = tuple(tuple(token) for token in header.data['tokens'])
        self._require_tokens(header.key, *tokens)
        if header.kind == 'ENCODE_REFRESH':
            self.host.pr_stage_inputs(tokens)
            await self.host.wait(self.preparation_duration)
            return self.host.encoded_refresh(tokens)
        for token in tokens[1:]:
            self.host.pr_discard(token)
        self.host.pr_confirm(tokens[0])
        return True

    async def _acquire(self, key: tuple[int, int], left: int, right: int) -> node_state.PRPair:
        self._check_active(key)
        entry = self.host.pr_requests[key]
        if not 0 <= left < right < len(entry.path) or entry.path[left] != self.host.id:
            raise node_state.PRFailure('Invalid encoded path interval or ownership')
        first = await self._encoded_link(key, left)
        if right == left + 1:
            return first
        middle = entry.path[left + 1]
        # Keep this source-side interface alive while downstream resources form.
        reply = await self._rpc(middle, 'ACQUIRE', key, {'left': left + 1, 'right': right})
        await self._rpc(middle, 'SWAP', key, {'first': first.token,
                        'second': tuple(reply['token']), 'destination': entry.path[right]})
        # Product is a conservative bound; no Werner assumption after decoding.
        return node_state.PRPair(first.token, self._estimate(first) * reply['fidelity'], self.host._time)

    async def _encoded_link(self, key: tuple[int, int], left: int) -> node_state.PRPair:
        for attempt in range(self.max_retries + 1):
            pair = await self._try_encoded_link(key, left)
            if pair is not None:
                return pair
            if attempt < self.max_retries:
                self.host.pr_requests[key].encoded_retries += 1
        raise node_state.PRFailure('encoded preparation retry limit; source entanglement retained until cancellation')

    async def _try_encoded_link(self, key: tuple[int, int], left: int) -> node_state.PRPair | None:
        entry = self.host.pr_requests[key]
        peer = entry.path[left + 1]
        code = self._request_code(key)
        backend = get_qec_code(code)
        if backend.length + 1 > self.max_graph_qubits:
            raise node_state.PRFailure('Graph resource exceeds max_graph_qubits per local interface')
        pairs: list[node_state.PRPair] = []
        try:
            for _ in range(backend.length):
                pairs.append(await self._at_level(key, left, left + 1, self.purification_rounds))
            tokens = tuple(pair.token for pair in pairs)
            self.host.pr_stage_inputs(tokens)
            await self.host.wait(self.preparation_duration)
            entry.graph_resource_qubits += 2 * (backend.length + 1)
            if not await self._rpc(peer, 'ENCODE_REFRESH', key, {'tokens': tokens}):
                return None
            if not self.host.encoded_decode(tokens, code):
                return None
            await self._rpc(peer, 'ENCODE_FINISH', key, {'tokens': tokens})
            self.host.pr_confirm(tokens[0])
            entry.encoded_blocks += 1
            estimate = self.host.estimate_encoded_fidelity(code, tuple(self._estimate(pair) for pair in pairs))
            # One logical representative remains at each endpoint.
            result = node_state.PRPair(tokens[0], float(estimate), self.host._time)
            pairs.clear()
            return result
        except node_state.PairExtractionFailure:
            return None
        finally:
            for pair in pairs:
                self.host.pr_discard(pair.token)
                await self._rpc(peer, 'DROP', key, {'token': pair.token})

    async def _swap_command(self, key: tuple[int, int], data: dict) -> None:
        self._check_active(key)
        self._require_tokens(key, data['first'], data['second'])
        first, second = tuple(data['first']), tuple(data['second'])
        outcome = await self.host.execute_encoded_fusion(first, second)
        if outcome is None:
            raise node_state.PRFailure('source entanglement consumed by failed encoded fusion; cannot retry')
        self.host.pr_requests[key].swaps += 1
        confirmed = await self._rpc(data['destination'], 'CORRECT', key,
                                    {'old': second, 'new': first, 'outcome': outcome})
        if not confirmed:
            raise node_state.PRFailure('source entanglement lost during encoded output correction; cannot retry')


class L4_QProtocol(QProtocol):
    """End-to-end addressing, batch correlation and resource handoff.

    Derived classes provide header construction, input operations and batch
    completion. Intermediate nodes only forward the classical control records.
    """
    layer = L4
    fip_mode = None
    header_type = None
    completion_handshake = False

    def __new__(cls, *args, **kwargs):
        if cls is not L4_QProtocol:
            return object.__new__(cls)
        fip_mode = kwargs.get('fip_mode', args[0] if len(args) > 0 else 'epp')
        concrete = {'necp': L4_NEC, 'epp': L4_EP, 'qecp': L4_QEC}.get(fip_mode or 'epp')
        if concrete is None:
            raise ValueError('Unknown fidelity improvement mode')
        return object.__new__(concrete)

    def __init__(self, fip_mode=None, routing_table=None):
        super().__init__()
        if fip_mode is not None and fip_mode != self.fip_mode:
            raise ValueError('Fidelity mode does not match the protocol class')
        self.routing_table = dict(routing_table or {})

    def input_block_size(self, request=None):
        raise NotImplementedError('Define the number of input pairs per output')

    def create_header(self, src, dst, requested, needed, code=None, *, request=None):
        if self.header_type is None:
            raise NotImplementedError('Supply a header_type or override create_header')
        return self.header_type(src, dst, requested, needed, code)

    def _prepare_inputs(self, neighbor, store, packet, corrections):
        raise NotImplementedError('Implement local quantum operations')

    def _complete_batch(self, remote, store, batch, peer):
        raise NotImplementedError('Implement outcome comparison and confirmation')

    def _validate_mode(self, packet):
        """Override to validate the concrete header and supported configuration."""

    def _finish_preparation(self, remote, store, packet, results, qubits):
        packet.l4_success = results
        self.host.l4_store_batch(remote, store, packet, qubits)
        self.host.send_packet(packet)
        self._drain_completed(remote, store)
        return packet

    def _next_hop(self, remote, fallback=None):
        neighbor = self.routing_table.get(remote, fallback)
        if neighbor not in self.host._channels['pc']:
            raise ValueError(f'No classical next hop for endpoint {remote}')
        return neighbor

    def request_fidelity_improvement(self, packet):
        """Improve already established end-to-end pairs in the first-hop L3 store."""
        self._validate_request(packet)
        return self.on_qubits_ready(packet.l3_dst, SEND, packet.l4_needed,
                                    packet.request_id, neighbor=packet.dst, request=packet)

    def request_entanglement(self, packet):
        """Start fidelity improvement on prepared pairs; generation belongs to L3."""
        return self.request_fidelity_improvement(packet)

    def _validate_request(self, packet):
        if not packet.has_l3 or packet.l3_src != self.host.id or packet.l3_dst == self.host.id:
            raise ValueError('L4 requests need local and remote end-to-end addresses')
        size = self.input_block_size(packet)
        if packet.l4_needed < 1 or packet.l4_needed != size * packet.l4_requested:
            raise ValueError('L4 needed must equal block size times requested outputs')
        self._next_hop(packet.l3_dst, packet.dst)

    def accept_entanglement(self, packet):
        """Accept an L3 completion and correct its accumulated swapping frame."""
        if packet.l3_dst != self.host.id or packet.l3_is_cf or packet.rejected:
            raise ValueError('Expected a successful quantum L3 completion for this endpoint')
        return self.on_qubits_ready(packet.l3_src, RECEIVE, packet.l3_needed,
                                    packet.request_id, neighbor=packet.src, request=packet,
                                    corrections=packet.l3_es_result)

    def _check_duplicate(self, remote, store, request_id):
        if self.host.l4_check_results(remote, store, request_id):
            raise ValueError('Duplicate L4 batch request ID')

    @staticmethod
    def _ports(request, store):
        if request is None or request.layer4 is None:
            return 0, 0
        if store == RECEIVE:
            return request.l4_dst, request.l4_src
        return request.l4_src, request.l4_dst

    def _drain_completed(self, remote, store):
        while self._complete_next(remote, store):
            pass

    def _confirm_batch(self, remote, store, batch, success):
        count = self.host.l4_confirm_batch(remote, store, batch, success)
        super().on_qubits_ready(remote, store, count, batch.request_id)

    def on_qubits_ready(self, remote, store, count, request_id=None, *,
                        neighbor=None, request=None, corrections=None):
        size = self.input_block_size(request)
        if count < 1 or count % size:
            raise ValueError('L4 readiness must contain complete input blocks')
        next_hop = self._next_hop(remote, neighbor)
        neighbor = next_hop if neighbor is None else neighbor
        packet = self._create_packet(remote, store, count // size, next_hop, request_id, request)
        self._check_duplicate(remote, store, packet.request_id)
        results, qubits = self._prepare_inputs(neighbor, store, packet, corrections)
        return self._finish_preparation(remote, store, packet, results, qubits)

    def _create_packet(self, remote, store, rounds, neighbor, request_id, request):
        size = self.input_block_size(request)
        source_port, destination_port = self._ports(request, store)
        code = None if request is None else request.l4_code
        header = self.create_header(source_port, destination_port, rounds, size * rounds, code, request=request)
        network = headers.L3_FRP(self.host.id, remote, self.host.id, neighbor, 0, 0)
        packet = Packet(self.host.id, neighbor, layer3=network, layer4=header,
                        request_id=request_id)
        packet.l3_set_cf()
        if store == SEND:
            packet.l4_set_ack()
        return packet

    def handle_packet(self, packet):
        self._validate_control(packet)
        remote, store = packet.l3_src, int(packet.l4_ack)
        if self.completion_handshake and store == SEND and not packet.l4_complete:
            return
        self.host.l4_store_packet(remote, store, packet)
        self._drain_completed(remote, store)

    def _validate_control(self, packet):
        if not packet.is_l4 or not packet.l3_is_cf or packet.l3_dst != self.host.id or packet.rejected:
            raise ValueError('Expected a classical L4 control packet for this endpoint')
        self._validate_mode(packet)

    def _complete_next(self, remote, store):
        batch = self.host.l4_peek_batch(remote, store)
        if batch is None:
            return False
        peer = self.host.l4_peer_packet(remote, store, batch.request_id)
        if peer is None:
            return False
        self._validate_batch(batch, peer, store)
        return self._complete_batch(remote, store, batch, peer)

    def _validate_batch(self, batch, peer, store):
        if batch.needed != peer.l4_needed or batch.results.shape != peer.l4_success.shape:
            raise ValueError('L4 input or result counts differ')
        if batch.code != peer.l4_code:
            raise ValueError('L4 endpoints selected different QEC codes')
        if peer.l4_complete != (self.completion_handshake and store == SEND):
            raise ValueError('Unexpected L4 completion phase')

class L4_EP(L4_QProtocol):
    """Bilateral parity-check purification between remote endpoints."""
    fip_mode = 'epp'
    header_type = headers.L4_EPP

    def input_block_size(self, request=None):
        if request is not None and request.l4_code is not None:
            raise ValueError('L4 code selection requires QEC mode')
        return 2

    def _prepare_inputs(self, neighbor, store, packet, corrections):
        return self.host.l4_prepare_inputs(neighbor, store, packet, 'epp', corrections=corrections)

    def _validate_mode(self, packet):
        if packet.l4_code is not None or packet.l4_protocol != 0:
            raise ValueError('Incompatible L4 EPP control packet')

    def _complete_batch(self, remote, store, batch, peer):
        success = batch.results == peer.l4_success
        self._confirm_batch(remote, store, batch, success)
        return True

class L4_QEC(L4_QProtocol):
    """Packet-selected QEC with one endpoint applying the effective decoder."""
    fip_mode = 'qecp'
    header_type = headers.L4_QECP
    completion_handshake = True

    def input_block_size(self, request=None):
        if request is None or request.l4_code is None:
            raise ValueError('L4 QEC requests must specify a code in the L4 header')
        model = get_qec_code(request.l4_code)
        if model.num_logical != 1:
            raise NotImplementedError('L4 pair storage does not yet support correlated multiple logical outputs')
        return model.length

    def _prepare_inputs(self, neighbor, store, packet, corrections):
        self.input_block_size(packet)
        return self.host.l4_prepare_inputs(neighbor, store, packet, 'qecp', corrections=corrections)

    def _validate_mode(self, packet):
        if packet.l4_code is None or packet.l4_protocol != 1:
            raise ValueError('L4 QEC control packets require a code')

    def _complete_batch(self, remote, store, batch, peer):
        if store == RECEIVE:
            self.host.l4_apply_qec(remote, store, batch.request_id)
        success = np.logical_and(batch.results, peer.l4_success)
        self._confirm_batch(remote, store, batch, success)
        if store == RECEIVE:
            self._send_completion(remote, batch.next_hop, peer, success)
        return True

    def _send_completion(self, remote, neighbor, peer, success):
        reply = peer.copy()
        reply.l3_switch_src_dst()
        reply.l4_switch_src_dst()
        reply.set_hop(self.host.id, self._next_hop(remote, neighbor))
        reply.l4_reset_ack()
        reply.l4_set_complete()
        reply.l4_success = success
        self.host.send_packet(reply)

class L4_NEC(L4_QProtocol):
    """Promote existing end-to-end pairs without correction or purification."""
    fip_mode = 'necp'
    header_type = headers.L4_EPP

    def input_block_size(self, request=None):
        if request is not None and request.l4_code is not None:
            raise ValueError('L4 code selection requires QEC mode')
        return 1

    def _prepare_inputs(self, neighbor, store, packet, corrections):
        return self.host.l4_prepare_inputs(neighbor, store, packet, 'necp', corrections=corrections)

    def _finish_preparation(self, remote, store, packet, results, qubits):
        self.host.l4_store_ready(remote, store, qubits)
        return QProtocol.on_qubits_ready(self, remote, store, len(qubits), packet.request_id)

    def _validate_mode(self, packet):
        raise ValueError('NECP does not exchange L4 control packets')

class L7_QProtocol(QProtocol):
    """Shared end-to-end application envelope validation and receive filtering.

    Override _handle_payload for another application. Override handle_packet
    as well if the application uses a different classical envelope contract.
    """
    layer = L7

    def handle_packet(self, packet):
        self._validate_packet(packet)
        return self._handle_payload(packet)

    def _handle_payload(self, packet):
        raise NotImplementedError('Implement application payload handling')

    def _validate_packet(self, packet):
        if not packet.is_l7 or not packet.has_l3 or packet.l3_dst != self.host.id or not packet.l3_is_cf:
            raise ValueError('Expected a classical application packet for this node')
        if packet.rejected:
            raise RuntimeError('Application request was rejected')

    async def receive(self, sender=None, time_out=None):
        """Filter by the remote endpoint, retaining other application messages."""
        packet = await self.host.receive_packet(time_out=time_out, layer=L7, end_sender=sender)
        if packet is None:
            return None
        return self.handle_packet(packet)

class L7_TP(L7_QProtocol):
    """Teleportation orchestration; Node owns measurements and recovery."""

    def _validate_packet(self, packet):
        super()._validate_packet(packet)
        if packet.l7_protocol != 0:
            raise ValueError('Expected a teleportation header')

    def _handle_payload(self, packet):
        return self.host.apply_teleportation_receive(packet)

    def __init__(self, bsm_mode='bsm'):
        super().__init__()
        if bsm_mode != 'bsm':
            raise ValueError('Only deterministic bsm is supported for teleportation')
        self.bsm_mode = bsm_mode

    def send(self, data_qubit, l3_receiver, l2_receiver, *, corrections=None):
        """Teleport over L3 or confirmed L4 pairs with a classical forwarded record."""
        qubits = self._normalize_data(data_qubit)
        self._validate_destination(l3_receiver, l2_receiver)
        packet = self._create_packet(l3_receiver, l2_receiver, len(qubits), corrections)
        packet.payload = self.host.apply_teleportation_send(qubits, l3_receiver, l2_receiver)
        self.host.send_packet(packet)
        return packet

    @staticmethod
    def _normalize_data(data):
        if isinstance(data, Qubit):
            return [data]
        try:
            qubits = list(data)
        except TypeError as error:
            raise ValueError('Supply a qubit or an iterable of qubits') from error
        if not qubits or not all(isinstance(qubit, Qubit) for qubit in qubits):
            raise ValueError('At least one data qubit is required')
        return qubits

    def _validate_destination(self, remote, neighbor):
        if remote == self.host.id:
            raise ValueError('Teleportation requires a different destination node')
        if neighbor not in self.host._channels['pc']:
            raise ValueError(f'No classical connection to next hop {neighbor}')

    def _create_packet(self, remote, neighbor, count, corrections):
        packet = Packet(self.host.id, neighbor, l3_src=self.host.id,
                        l3_dst=remote, l3_needed=count, l7_requested=count, payload=[])
        if corrections is not None:
            packet.l3_es_result = corrections
        packet.l3_set_cf()
        return packet

class L7_RG(L7_QProtocol):
    """Reserved extension point for remote-gate modeling."""

    def _handle_payload(self, packet):
        raise NotImplementedError('Remote-gate operations are not modeled yet')

# Explicit layer-base names for user-defined implementations.
L1_EGP = L1_QProtocol
L2_FIP = L2_QProtocol
L3_QFP = L3_QProtocol
L4_FIP = L4_QProtocol
L7_TPP = L7_TP

class QProtocolStack:
    """Copy configuration per node, connect layers, and dispatch packets."""
    def __init__(self, l1_qprotocol=None, l2_qprotocol=None, l3_qprotocol=None,
                 l4_qprotocol=None, l7_qprotocol=None):
        self._qprotocols = {}
        for layer, protocol in zip((L1, L2, L3, L4, L7),
                                  (l1_qprotocol, l2_qprotocol, l3_qprotocol,
                                   l4_qprotocol, l7_qprotocol)):
            if protocol is None or protocol.layer == 0:
                continue
            if protocol.layer != layer:
                raise ValueError(f'Protocol in slot {layer} belongs to layer {protocol.layer}')
            missing = set(protocol.required_layers).difference(self._qprotocols)
            if missing:
                raise ValueError(f'Layer {layer} requires layers {sorted(missing)}')
            self._qprotocols[layer] = protocol
        for layer, protocol in self._qprotocols.items():
            protocol.prev_protocol = self._qprotocols.get(layer - 1)
            protocol.next_protocol = self._qprotocols.get(layer + 1)

    def get_protocol(self, layer: int) -> QProtocol:
        """Return an installed operation without exposing the stack dictionary."""
        try:
            return self._qprotocols[layer]
        except KeyError as error:
            raise ValueError(f'No protocol installed for layer {layer}') from error

    def clone(self):
        protocols = {}
        for layer, protocol in self._qprotocols.items():
            cloned = copy(protocol)
            for key, value in vars(protocol).items():
                if key not in {'host', 'prev_protocol', 'next_protocol'}:
                    setattr(cloned, key, deepcopy(value))
            cloned.host = cloned.prev_protocol = cloned.next_protocol = None
            cloned.reset()
            protocols[layer] = cloned
        return QProtocolStack(**{f'l{7 if layer == L7 else layer}_qprotocol': protocol
                                 for layer, protocol in protocols.items()})

    def bind(self, host):
        self.host = host
        for protocol in self._qprotocols.values():
            if protocol.host is not None and protocol.host is not host:
                raise ValueError('A protocol stack can only belong to one node')
            protocol.host = host

    def handle_packet(self, packet):
        # Generation heralds can carry an L2 code request. They still enter at L1.
        if packet.has_l3 and packet.l3_dst == self.host.id and not packet.l3_is_cf and L4 in self._qprotocols:
            return self._qprotocols[L4].accept_entanglement(packet)
        layer = L3 if packet.has_l3 and packet.l3_dst != self.host.id else packet.layer
        if packet.has_l1 and packet.l1_requested > 0:
            layer = L1
        try:
            protocol = self._qprotocols[layer]
        except KeyError as error:
            raise ValueError(f'No protocol installed for layer {layer}') from error
        return protocol.handle_packet(packet)
