# Quantum protocol API

`protocol.py` contains request headers and result arrays. `Packet` combines
headers, hop addresses, end-to-end addresses, a request ID, and a payload.
`qprotocol.py` executes operations and exchanges classical control messages.

## Constructing a stack

```python
from qcns import Node, Simulation
from qcns.python.components.network.qprotocol import (
    QProtocolStack, L1_EGP, L2_FIP, L3_QFP, L7_TPP,
)

simulation = Simulation()
configuration = QProtocolStack(
    l1_qprotocol=L1_EGP(eg_mode="srp", rap_mode="attempt"),
    l2_qprotocol=L2_FIP(fip_mode="epp"),
    l3_qprotocol=L3_QFP(routing_table={0: 0, 2: 2}, qf_mode="frp"),
)
repeater = Node(1, simulation, qprotocols=configuration)
```

Nodes copy stack configuration, bind their own protocol instances, and reset
runtime state when the simulation resets. L2 requires L1, and L3 requires L2.
An application-only stack containing `L7_TPP` can use already prepared pairs.
Generation mode must match the physical connection: `srp`, `tpsp`, `bsmp`,
`fsp`, or the abstract `l3cp` connection. L3 connections already prepare L3
pairs and normally bypass link purification. A request with an L2 QEC code
passes these raw pairs through L2 before making the decoded output available.

## Calls and layer handoffs

| Call | Purpose |
| --- | --- |
| `node.l1_qprotocol.request_entanglement(neighbor, count, estimate=True)` | Start link generation. Only the initiating endpoint retries failed attempts. |
| `node.l2_qprotocol.request_entanglement(packet, estimate=True)` | Generate the raw inputs specified by the L2 header, including its QEC code. |
| `node.handle_packet(packet)` | Dispatch an arriving message through the stack. Application messages traverse L3 at intermediate nodes. |
| `protocol.on_qubits_ready(neighbor, store, count, request_id)` | Notify the next layer of confirmed pairs. L1 calls L2, and L2 wakes pending L3 requests. |
| `node.l3_qprotocol.swap_entanglement(packet, offset_index=reservation)` | Perform CRP or FRP swapping using ready pairs. |
| `node.l7_qprotocol.send(data, destination, next_hop, corrections=frame)` | Teleport with an optional accumulated swapping frame. |
| `await node.l7_qprotocol.receive(time_out=...)` | Receive and correct teleported qubits. Installed lower-layer control handlers run while waiting. |

The former nested `partial()` wrappers and reassigned data-plane methods are
replaced by these entry points. Node subclasses can implement their control
loop with `await self.receive_packet()` followed by `self.handle_packet(packet)`.
The existing router uses this loop and the new stack constructor.

## Resource and correction conventions

- `SEND = 0`, `RECEIVE = 1`; these select the direction of a neighbor's memory.
- L0 holds unheralded pairs; L1 holds heralded inputs; L2 holds purification
  inputs or survivors awaiting peer confirmation; L3 holds usable pairs.
- L2/EPP `needed` counts input pairs. Its success array contains `needed // 2`
  measurement bits. Assigning results does not change the input count.
- L1/L3 success arrays contain one bit per needed pair. L7 success arrays contain
  one bit per requested data qubit. Incorrect shapes and negative counts fail
  explicitly. Resizing retains existing bits, including swapping corrections.
- A BSM outcome is `2 * first_measurement + second_measurement`. Its low bit
  controls X and its high bit controls Z. Repeated swapping accumulates XOR
  parity. Teleportation combines this parity with its payload's BSM outcomes.
- L1/TPS keeps local results in the node's `_layer_results` buffer and early peer
  packets in its `_packets` buffer. L1 check/retrieve methods accept an optional
  `request_id` filter; omitting it retains FIFO retrieval. Resetting the node
  clears these buffers. L1 protocol instances hold no separate result caches.
- L2 stores `L2Batch` records (request ID, result bits, input qubits, input count,
  code) in the node's `_layer_results` buffer and peer messages in `_packets`.
  It has no separate pending/result dictionaries. Node reset clears both.
- Peer control results use a shared `request_id`. EPP confirms batches in their
  local generation order even when results arrive out of order. This preserves
  the correspondence of ready pairs at both ends.
- Replies and packets queued in classical channels copy their mutable contents.
  A reply retains the request ID; modifying it cannot alter the original packet.

## Reservations and timing

`l3_add_offset(neighbor, store, count)` returns a stable reservation handle for
concrete qubits. Reserved qubits are excluded from `l3_num_qubits()`, which
counts available pairs. `l3_retrieve_qubit(..., offset_index=handle)` consumes a
reserved qubit, and `l3_remove_offset(...)` releases the handle. Ordinary qubit
retrieval consumes memory; `_mode=True` explicitly peeks for the L0-L2 helpers.
L3 queued-packet retrieval always returns `(packet, reservation)`.

An unsatisfied L3 request stays queued with its reservation intact. Incoming
requests must already have the required input pairs. No-reject mode waits for
outgoing resources; partial and complete rejection modes check both directions.
Unlimited memory is represented by size `-1`. Rejections retain request identity
and carry an explicit `rejected` flag.

Receive timeouts use **simulation seconds**, rather than wall-clock time. A
channel cannot deliver a packet or qubit before its scheduled arrival. Sender
and layer filters retain unrelated messages. The simulator cancels background
services at completion and reports a deadlock if finite nodes wait without any
scheduled event that can make progress.

Packet lengths are measured in bits. Fixed header reservations expand for large
result arrays; metadata reserves 64 bits for the request ID and one rejection
bit. Application payload size uses compact UTF-8 JSON, supporting NumPy values
and byte sequences. Unsupported payload objects need a defined encoding.

## L3 forwarding and entanglement swapping

`L3_QFP` coordinates routing, rejection policies, queued requests, and classical
messages. Its helpers use at most one level of control nesting, like `L2_FIP`.
The quantum operations live in `Node`:

| L3 protocol entry point | Node operation | Behavior |
| --- | --- | --- |
| `swap_entanglement_crp(packet, offset_index=None)` | `apply_entanglement_swapping_crp(packet, offset_index=None)` | Apply the incoming Pauli frame locally, reset it, then record the new Bell-measurement frame. |
| `swap_entanglement_frp(packet, offset_index=None)` | `apply_entanglement_swapping_frp(packet, offset_index=None)` | Accumulate Bell-measurement outcomes into the existing Pauli frame for correction at the destination. |
| `swap_entanglement_graph_state(packet, ..., code=..., first_record=..., second_record=...)` | `apply_entanglement_swapping_graph_state(...)` | Decode supplied encoded-interface Bell records and accumulate the known logical correction frame. |

`swap_entanglement()` dispatches according to `qf_mode`; its optional `correct`
override remains available. Node operations update quantum states and correction
fields. Protocol entry points additionally forward the resulting control packet.
Only CRP and FRP are selectable modes at present.

Pending requests and their incoming reservations remain in Node buffers.
`l3_can_swap()` checks both reserved incoming resources and available outgoing
pairs. FIFO draining stops at an unsatisfied request. A queued packet is removed
and its reservation released after swapping completes; a failure before quantum
operations leaves the queued request available.

### Compact graph/stabilizer-resource model

The model follows [Shi, Patil, and Guha, Measurement-Based Entanglement
Distillation and Constant-Rate Quantum Repeaters over Arbitrary
Distances](https://arxiv.org/html/2502.11174v2). It constructs logical-logical
repeater resources and logical-physical endpoint resources from a stabilizer
code. Recovery uses the combined XX/ZZ Bell parities, including the Pauli-Y
sign offsets. The paper's independent depolarizing noise analysis motivates
the effective segment channel.

`python/components/qubit/graph_state.py` contains the numerical model. Resources
store positive stabilizer generators as pairs of packed X/Z integer masks.
This represents the resource stabilizer state, not an explicit graph adjacency
matrix or preparation circuit. A 2n-qubit resource stores 2n generators; it
never allocates a 4n-qubit density matrix. Local Clifford conversion into a
graph drawing is unnecessary for this recovery model.

```python
from qcns.python.components.qubit.qec import get_qec_code
from qcns.python.components.qubit.graph_state import (
    MeasurementBasedRepeaterModel, BellMeasurementRecord,
)

model = MeasurementBasedRepeaterModel(get_qec_code("five_qubit"))
resource = node.prepare_graph_state_resource(model)
samples = node.l3_qprotocol.simulate_graph_state_segment(
    model, shots=100_000, seed=123,
    bell_error_rate=0.003,
    resource_error_rate=0.001,
    measurement_error_rate=0.001,
)
print(samples.joint_fidelity)
print(samples.bell_marginals())
```

Rates are total nonidentity depolarizing probabilities **per physical half**
and per measurement endpoint. Alternatively, `bell_probabilities` supplies
one I/X/Y/Z distribution per physical pair, or a common four-entry distribution.
Those pair probabilities already include both halves and are not doubled.
Supplying both a Bell rate and Bell probabilities fails explicitly. Resource
and measurement contributions still include both endpoints. Noise composition
uses exact Pauli-channel convolution, retaining nonidentical input channels.

`MeasurementBasedCode(name, stabilizers, logical_x, logical_z, decoder=...)`
accepts general `[[n,k]]` stabilizer codes, with tuples of k logical X/Z strings.
It validates generator independence and canonical logical commutation relations.
Small codes have a minimum-weight default decoder; codes with more than 12
checks require a supplied decoder. The decoder contract matches the QEC model:
syndrome bits in generator order in, n-character correction string out.

`LogicalPauliSamples` retains arrays of shape `(shots, k)` for bit and phase
errors. `joint_counts()` preserves cross-output correlations; `bell_marginals()`
is only a diagnostic and must not replace the joint representation. No k-output
density matrix is constructed. `joint_fidelity` estimates the probability of no
logical errors across the entire block, rather than the product of marginal
fidelities. `simulate_graph_state_chain(code, segments, ...)` XOR-composes
independently sampled segment frames using one continuing random stream.
It does not include uncoded end-output noise unless added separately. Monte
Carlo estimates have statistical uncertainty, especially for rare errors.

For supplied measurements, each `BellMeasurementRecord(s, t)` contains n XX
and n ZZ parity bits. `model.recover(first_record, second_record)` returns the
syndrome, decoder correction, and known logical bit/phase corrections.
Node's graph-state operation XORs those corrections into the L3 packet's k
correction entries; the L3 wrapper forwards that classical packet. The sampled
residual errors are hidden errors and are never included in a known correction
packet. The L3 correction count must equal k for this single-block interface.

This model supplies resource descriptions, segment/chain error trajectories,
and classical recovery from provided measurements. It **does not yet schedule
resource preparation or generate network Bell-measurement records**. It does
not replace or consume ordinary Bell-pair memory entries. Encoded resource
storage and its lifecycle need a separate representation, so passing an
ordinary memory reservation to the graph-state operation fails explicitly.
Automatic `qf_mode` dispatch remains CRP/FRP; graph-state recovery is invoked
explicitly with its records. Coherent noise, loss/erasure decoding, noisy
preparation circuits, and physical gate durations are outside this model.

The tests compare Pauli-Y recovery with a small independent state-vector
reference, check the segment noise composition and Monte Carlo output against
exact Pauli decoding, and exercise multiple correlated logical outputs. A
65-qubit code is tested without constructing any `QSystem` or combined state.

## Packet-selected quantum error correction

Configure both endpoints with `L2_FIP("qecp")`. The code belongs to each L2
request, rather than the protocol instance:

```python
from qcns.python.components.packet.packet import Packet

request = Packet(node.id, neighbor, l2_requested=2, l2_needed=10,
                 l2_code="five_qubit")
node.l2_qprotocol.request_entanglement(request)
```

`l2_requested` counts decoded output Bell pairs; `l2_needed` counts raw input
Bell pairs. It must equal the requested output count times the code length.
`five_qubit` selects [[5,1,3]], and `steane` selects [[7,1,3]]. For two Steane
outputs, request 14 input pairs. Unknown codes and missing code selections fail
explicitly. Changing `packet.l2_code` resizes the result array to the number of
code blocks and selects protocol ID 2 (EPP uses ID 1). Header length accounts
for a code identifier and a completion flag; the project models packet sizes,
not a binary wire encoding.

L1 carries the code selection in its generation/heralding messages and retries.
A generation attempt has its own shared batch ID. L2 can accumulate raw pairs
until a complete block is available. Both ends stage their halves in L2 and
exchange readiness messages. Node owns the local purification gates, QEC input
staging, and application of the effective decoding map; L2 coordinates their
requests and classical messages. The receiver applies the effective decoding map
once, promotes the first pair in each block, and sends a completion message.
The sender then promotes the corresponding halves. QEC result bits represent
block availability/completion, whereas EPP result bits are measurement outcomes.
Queued batches finish in local order even if classical messages reorder.
L3 requests with a QEC code also use that code when requesting missing outgoing
pairs.

The backend in `python/components/qubit/qec.py` tracks probabilities over a
stabilizer syndrome and a logical Pauli frame. It convolves independent I/X/Y/Z
error distributions and applies the selected syndrome decoder. The default
decoder uses minimum-weight Pauli representatives for small codes.
It needs only 64 probabilities for the five-qubit code and 256 for Steane; no
joint block density matrix or block gate operations are built. Each retained
output is a separate 4-by-4 Bell-diagonal pair state. Nonidentical input noise is
supported, and every single-qubit Pauli error is corrected by both codes.

### Supplying a code definition

`StabilizerPauliCode` accepts any phase-free `[[n,1]]` stabilizer code; preset
names are conveniences. Supply `n-1` independent commuting Pauli-string
generators and logical X/Z strings that commute with those generators and
anticommute with each other. Invalid definitions fail at construction.

```python
from qcns.python.components.qubit.qec import (
    StabilizerPauliCode, register_qec_code,
)

shor = StabilizerPauliCode(
    "shor",
    stabilizers=("ZZIIIIIII", "IZZIIIIII", "IIIZZIIII", "IIIIZZIII",
                 "IIIIIIZZI", "IIIIIIIZZ", "XXXXXXIII", "IIIXXXXXX"),
    logical_x="ZIIZIIZII",
    logical_z="XXXIIIIII",
)
register_qec_code(shor)
request = Packet(node.id, neighbor, l2_requested=1, l2_needed=9,
                 l2_code="shor")
node.l2_qprotocol.request_entanglement(request)
```

Register the code once when configuring the simulation. Packets select its
shared configuration by name; they do not transmit Python decoder functions.
Built-in names and existing registrations cannot be overwritten. The modeled
8-bit identifier field allows two presets and up to 253 custom codes. Direct
`node.apply_qec(qubits, code)` also accepts a model object without registration.
This direct operation expects both endpoints' inputs to be staged and runs
once; L2 normally handles that coordination and discards unused physical pairs.

For `method="exact"` (the default), the engine stores `4 * 2**(n-1)` syndrome
and logical-frame probabilities. This avoids joint density matrices but still
scales exponentially with the number of stabilizer checks. Exact mode and the
automatic minimum-weight decoder are limited to 12 checks to avoid accidental
large allocations. The default decoder covers all syndromes, without a
weight-two correction restriction. Minimum weight is not necessarily the
optimal decoder for a biased noise model.

For larger codes, supply a deterministic decoder and select sampling:

```python
code = StabilizerPauliCode(
    "large_code", stabilizers=generators,
    logical_x=logical_x, logical_z=logical_z,
    decoder=decode_syndrome, method="sample", shots=100_000, seed=123,
)
register_qec_code(code)
```

`decode_syndrome(bits)` receives one 0/1 bit per generator in the supplied
order and returns an n-character I/X/Y/Z correction string. The engine checks
that the correction has the requested syndrome. Sampling draws independent
physical Pauli errors, computes their syndromes, and counts the residual
logical errors after decoding. It uses batches of at most 1024 trajectories
and builds no exhaustive syndrome table when a decoder is supplied. Scaling
then depends on the code length, shot count, and the supplied decoder.

Sampling estimates the output Bell probabilities; it has statistical error and
requires more shots to resolve rare logical errors. A fixed seed reproduces
an estimate for the same input distribution, including after simulation reset.
Sampling still assumes independent Pauli noise and ideal correction; it does
not make arbitrary non-stabilizer codes, coherent noise, or multiple logical
outputs supported. Multiple outputs can have correlated logical errors and
need a richer output representation than separate independent pair matrices.

This is an **ideal effective QEC-based entanglement-distillation model** for
independent Bell-diagonal input pairs. It does not simulate encoding and syndrome
circuits, their gate noise, operation durations, or correlated/coherent errors.
Classical messages still use the simulator's channel timing. Inputs represented
as correlated systems or non-Bell-diagonal matrices are rejected. Low input
noise can improve fidelity; arbitrary noisy inputs are not guaranteed to improve.
The model consumes n raw pairs to produce one decoded pair and is not a local
correction operation on a single unencoded mixed qubit.

The connection between Bell-pair distillation and quantum error correction is
described by [Bennett et al.](https://arxiv.org/abs/quant-ph/9604024); the syndrome
and logical-frame representation follows the
[stabilizer formalism](https://arxiv.org/abs/quant-ph/9705052).

## L4 end-to-end fidelity improvement

`L4_FIP` supports the same `necp`, `epp`, and `qecp` modes as L2, operating
on already established end-to-end Bell pairs. Intermediate nodes process its
packets through L3 classical forwarding only: every L4 control record has
`l3_set_cf()` set. L3 addresses identify the two endpoints, L2 addresses identify
the current hop, and L4 fields retain the endpoint ports and resource counts.
QEC selection belongs to `l4_code`, independently of link-level `l2_code`.

```python
configuration = QProtocolStack(
    l4_qprotocol=L4_FIP("qecp", routing_table={remote: next_hop}),
    l7_qprotocol=L7_TPP(),
)
# After L3 has established five end-to-end pairs in this first-hop store:
request = Packet(node.id, next_hop, l3_src=node.id, l3_dst=remote,
                 l4_src=10, l4_dst=20, l4_requested=1, l4_needed=5,
                 l4_code="five_qubit", request_id=batch_id)
node.l4_qprotocol.request_fidelity_improvement(request)
```

`request_entanglement(request)` is an alias for starting improvement on prepared
pairs; it does not generate raw network entanglement. For EPP, request two input
pairs per output and omit `l4_code`. NECP passes pairs through without an L4
control exchange. QEC requires a registered one-logical-output code, using the
same independent Bell-diagonal noise assumptions and ideal decoder as L2.

The other endpoint can submit a matching readiness notification:
`on_qubits_ready(remote, RECEIVE, input_count, batch_id, neighbor=last_hop,
request=request, corrections=swapping_frame)`. Successful quantum L3 completion
packets addressed to a node with L4 installed automatically enter
`accept_entanglement()`, which supplies the last hop and accumulated swapping
frame. The receiving endpoint applies that frame before purification or QEC.
The initiating endpoint must signal readiness only once its pairs span the
intended endpoints. Both sides must use the same batch ID, matching input order,
counts, mode, and code.

`routing_table` selects the outbound classical next hop. If omitted, the explicit
local memory neighbor is also used as that next hop. The input memory neighbor
and the outbound route may differ, allowing an independently configured reply
route. Normal L3 routing tables forward the control packet at each intermediate
node, without allocating pairs or swapping again.

Node holds `L4Batch` records and early peer packets in `_layer_results` and
`_packets`, indexed by **remote endpoint**, direction, and request ID. No
per-protocol result dictionaries are added. Batches finish in local FIFO order.
EPP compares peer measurement bits; mismatches discard the corresponding pairs.
QEC stages both halves, applies the effective code map once at the receiving
endpoint, and sends a completed reply before the initiating endpoint releases
its outputs. Packet shape, code, and phase checks occur before consuming a
queued batch.

Confirmed outputs live in the node's end-to-end ready queues. Use
`l4_num_qubits(remote, store)` and `l4_retrieve_qubit(remote, store)` to inspect
or consume them. With L4 installed, teleportation uses these queues, rather than
raw L3 pairs. A filtered application receive also dispatches incoming L3
completions and L4 controls while waiting. Node reset discards pending and ready
L4 halves and clears their records.

Raw pair generation and replacement after unsuccessful EPP rounds remain the
network/application layer's responsibility. L4 notifies the next protocol of
the number of confirmed outputs; it does not retry link generation directly.
The tests cover EPP/QEC across routers, frame correction, early and reordered
records, failed-batch cleanup, reset, and teleportation in a running simulation.

## L7 teleportation

`L7_TPP` handles packet construction, validation, routing, and receive filtering.
`Node.apply_teleportation_send(qubits, remote, neighbor)` performs the Bell
measurements and consumes local data and shared halves.
`Node.apply_teleportation_receive(packet)` consumes the remote halves and applies
X/Z corrections. The protocol helpers use at most one level of control nesting.

```python
packet = node.l7_qprotocol.send(data, destination, next_hop)
received_qubits = await node.l7_qprotocol.receive(sender=source, time_out=1.)
```

`send()` accepts one qubit or an iterable of distinct live qubits. It checks the
classical route, correction sizes, data validity, and available pair count before
measurements. A selected Bell-pair half cannot simultaneously be a data input.
The output packet uses classical forwarding, so intermediate nodes do not swap
or teleport again. Entanglement among data qubits, or between a data qubit and an
unteleported reference, is preserved by the quantum operations.

With L4 installed, teleportation consumes confirmed pairs addressed by remote
endpoint. Otherwise it uses the next-hop L3 pair store. Receive recovery combines
the BSM low bit with accumulated X corrections and the high bit with Z
corrections, using XOR. Supply `corrections=frame` for uncorrected L3 swapping
frames; confirmed L4 pairs have already had those frames corrected.

`L7_TPP.receive(sender=...)` selects an **end-to-end source**, rather than a
neighbor. `Node.receive_packet(sender=...)` retains its neighboring-hop meaning;
its optional `end_sender=...` selects the L3 source. Unrelated application packets
stay in the node inbox. Lower-layer control processing continues during filtered
application receives, and timeouts use one simulation-time deadline.

Incoming outcome counts and all BSM values are checked before any receiving pair
is consumed. BSM outcomes must be integer values 0 through 3; booleans are not
accepted as outcomes. The tests cover all four outcomes, swapping and L4
integration, multi-qubit entangled inputs, retained reference entanglement, input
validation without resource consumption, and endpoint filtering across a router.

## Unimplemented operations

L7 distributed circuits raise `NotImplementedError`. Swapping and teleportation
support deterministic `bsm`; unsupported measurement modes fail at construction.
End-to-end L3 completion is handled by the destination application, or by L4
when installed.

## Verification

Run the behavioral suite from the repository root:

```powershell
python -m unittest discover -s tests -v
```

It checks headers, resource reservation, reordered purification results, actual
physical generation handshakes, simulation timeouts, classical routing, all four
teleportation outcomes, and quantum-state correctness across two repeaters.
QEC checks include all single-Pauli errors, probability convolution against
exhaustive enumeration, input-model validation, reordered batches, and real
packet-selected generation handshakes for both presets and every L1 mode.
They also exercise a custom Shor code, registered packet-selected codes,
corrections beyond weight two, invalid code definitions and decoders, and
sampling a 65-qubit code without exponential tables.
