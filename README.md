# qcns

## Selecting the quantum backend

The default remains `python`. Set `backend='cpp'` to apply local density-matrix
operations using the existing QMath kernels from `qcnspp/QMath`. NumPy arrays
still own the state; the Python simulator owns scheduling, memory, protocols and
random draws. This integration does not load the standalone QCNS++ simulator.

From the repository root, build once with a C++20 GNU compiler and OpenMP:

```console
python scripts/build_qmath_backend.py
python python/examples/unified_experiment.py --mode raw --backend cpp
python python/examples/unified_experiment.py --mode purification --backend cpp
python python/examples/unified_experiment.py --mode encoded --backend cpp
```

The helper accepts `--compiler` and `--output`. It builds a small shared library
from QMath's storage and kernel sources, without BLAS/LAPACK, Boost, NLopt, Python
headers, or a binding-package dependency. On Windows it copies the matching
MinGW OpenMP runtime DLLs beside the bridge. The GNU build has been validated on
Windows; the CMake alternative below and other platforms have not been tested
in this integration session:

```console
cmake -S native/qmath -B native/qmath/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/qmath/build --config Release
```

For CMake builds, make the compiler's OpenMP runtime available to the loader.
The default library location is `native/qmath/build/qcns_qmath.dll` (or `.so` /
`.dylib`). Use `QCNS_QMATH_LIBRARY`, `backend_library` in experiment configuration,
or `CppBackend(library=...)` for a different location, including a multi-config
CMake generator's `Release` subdirectory. Selecting C++ with a missing or
incompatible library raises an actionable error; it never silently selects Python.

```python
from qcns.api import CppBackend, ExperimentConfig, Simulation, run_experiment

result = run_experiment(ExperimentConfig(backend='cpp', backend_threads=1))
print(result.metadata['quantum_backend'])

simulation = Simulation(seed=7, backend=CppBackend(threads=1))
pair = simulation.create_qsystem(2)  # connections use this factory too
first, second = pair.qubits
first.bell_state(second)
```

Selection is per simulation or per `QSystem(..., backend=...)`. Independent
simulations can use different backends concurrently. The historical
`Simulation.create_qsystem(...)` class call still defaults to Python; use the
instance factory for a configured simulation. Combining states with different
backend implementations fails before changing the states. To extend the array
interface, subclass `PythonBackend`, override the required kernels and pass the
instance to `Simulation(backend=...)` or `QSystem(backend=...)`.

Native operations are single-qubit gates (including rotations and arbitrary
2x2 operators), controlled single-qubit gates, CNOT, amplitude damping,
dephasing, isotropic Pauli noise and single-qubit partial trace. Bell preparation,
BSM, SWAP and measurement collapse compose these kernels. Dense custom and
multi-control operators, tensor products, fidelity calculations, and the existing
reduced QEC/graph-state models continue using their Python implementations.
`backend.native_operations` and experiment metadata expose the native coverage.

Both implementations use index 0 as the most significant tensor factor; the
bridge translates indices to QMath's least-significant-bit convention. Channel
parameters preserve the existing Python model: dephasing uses probability of Z,
amplitude damping uses decay probability, and depolarizing uses the total
probability of X/Y/Z (one third each). No approximation or noise channel is
introduced by choosing C++. Measurement draws remain on the supplied Python RNG.
The integration also fixes noisy measurement's Born probability to
`trace(M rho M†)`; nonideal readout therefore changes from the previous incorrect
implementation, while ideal readout is unchanged.

The bridge copies inputs to QMath's aligned storage and outputs back to NumPy;
it never aliases NumPy storage as QMath complex objects. Strided, real and
non-native-endian arrays are converted to contiguous native `complex128`.
Native joint states support 1–15 qubits, with the existing conservative dense
allocation budget still enforced by `QSystem` and `combine_state`. This limit
does not cap the number of independently represented qubits in a network.
The default of one OpenMP thread per call avoids unrequested oversubscription.

Run correctness checks with `python -m unittest discover -s tests -p test_quantum_backend.py -v`.
Native tests skip explicitly when the bridge has not been built; a present but
unloadable bridge fails. Tests cover independent dense/Kraus references, array
ownership, subsystem ordering, all Bell outcomes, seeded noisy end-to-end runs
for all three router modes, and per-instance selection. No comparative performance
benchmarks are part of these checks.

## Unified experiments and public API

New applications should use explicit imports from **`qcns.api`**. This module
exports the supported experiment configuration, simulation, node, topology,
packet/header, protocol-extension, circuit and persistence interfaces. It refers
to the existing classes rather than wrapping them in a parallel simulator.
Existing import paths remain available; the earlier move of circuit helpers out
of general `utils.py` remains in place.

```python
from qcns.api import (
    ExperimentConfig, HardwareConfig, ProtocolConfig, TrafficRequest, run_experiment,
)

config = ExperimentConfig(
    seed=7,
    protocol=ProtocolConfig(mode='purification'),  # 'raw', 'purification', 'encoded'
    hardware=HardwareConfig(fidelity=.99, dephasing_time=10.),
    requests=(TrafficRequest(source=4, destination=5, qubits=1, fidelity=.97),),
)
config.save('experiments/purification.json')
result = run_experiment(ExperimentConfig.load('experiments/purification.json'))
print(result.delivered_pairs, result.finished_at)
print([(request.status, request.reason) for request in result.requests])
```

The default experiment has four routers in a path and two clients attached to
opposite ends (client IDs 4 and 5). Change `ProtocolConfig.mode` to run the same
scenario through ordinary FRP routers, hierarchical purification routers, or
encoded graph-interface routers. The new `L3_RawEndpoint` generates source pairs,
retains their halves, and handles destination Pauli corrections and classical
acknowledgments; the existing `Router` and `L3_FR` perform the intermediate swaps.
All modes generate their pairs during the run.

**Physical defaults:** symmetric effective Werner-pair generation with initial
Phi-plus fidelity 0.99, generation duration 0.0001 seconds, success probability 1,
1 km classical links at 100 Mbit/s, FIFO memories with 128 slots, ideal gates and
no memory noise. Gate durations come from the existing node defaults. Set
`depolarization_time` and/or `dephasing_time` in seconds to enable storage noise.
The depolarization setting uses the isotropic **`DepolarizingError`** channel;
it does not use the legacy `DepolarizationError` name for amplitude damping.
Encoded mode uses the existing exact single-logical-output stabilizer backend,
with independent Bell-diagonal inputs and ideal graph resources/decoding.
Purification retains the existing BBPSSW operations and rough planning estimates.

Configuration sections:

- `TopologyConfig`: core generator or NetworkX graph, router/client counts,
  generator parameters and client attachment policy. A separately constructed
  `TopologyConfig` defaults to balanced attachment; use `attachment='explicit'`
  and `client_attachments` for a prescribed placement.
- `HardwareConfig`: effective link generation, classical channel and memory
  settings. `hardware` applies to core links; optional `access_hardware` overrides
  client links. `None` disables a noise channel; `memory_slots=-1` is unbounded.
- `ProtocolConfig`: mode, timeout, resource/retry limits, and encoded code/resource
  settings. There is no implicit switch to an approximate quantum backend.
- `TimingConfig`: `default_duration` and per-event overrides such as
  `{'gate:bsm': 2e-5}`. `timings` applies globally; `node_timings={id: TimingConfig(...)}`
  replaces it for selected nodes, following `EventTimings` semantics.
- `requests`: explicit `TrafficRequest` values; `generate_requests` can produce
  them reproducibly. Arrival times are absolute simulation seconds. Each source
  starts requests sequentially in arrival order; it continues servicing incoming
  messages while waiting for the next arrival and after finishing its own work.
- `seed`, `end_time`, and optional `routing_metric` edge-attribute name. An omitted
  deadline stops when every request finishes. A finite deadline marks unfinished
  requests `incomplete`; an empty/background-only workload requires that deadline.

For example, the 100-node raw workload can be expressed without custom clients:

```python
from qcns.api import TopologyConfig, generate_requests

config = ExperimentConfig(
    topology=TopologyConfig(core='barabasi_albert_graph', num_routers=20,
                            num_clients=80, core_parameters={'m': 2}),
    requests=generate_requests(range(20, 100), range(20, 100), seed=7),
    hardware=HardwareConfig(depolarization_time=1., dephasing_time=1.),
    protocol=ProtocolConfig(mode='raw'), seed=7,
)
```

**Current workload limits:** raw requests contain at most 256 pairs. Raw and
purification modes support multiple initiators; encoded experiments currently
require one initiating source, which may issue several sequential requests.
Concurrent encoded initiators exposed timeouts in validation and are rejected
before creating hosts. PR/encoded request records and completed output pairs
remain retained until the experiment closes; workloads must fit the configured
memory and request-table limits. These checks do not silently serialize traffic
or change the requested physics.

`ExperimentConfig.save/load` use the existing safe, versioned JSON codec.
`to_dict/from_dict` support programmatic editing and plain JSON inputs. Unknown
fields/versions and invalid model parameters fail explicitly. Results include the
effective configuration and root seed, dependency versions and model assumptions.
With `seed=None`, replay using `ExperimentConfig.from_dict(result.configuration)`.
Reproducibility covers outcomes and simulated timing under compatible dependency
versions; process-global packet identifiers can differ between runs.

`RequestResult` reports completion/failure/incompleteness and the underlying
protocol result. `estimated_fidelity` is an operational estimate, **not measured
state fidelity**. Raw forwarding has no estimate or target-fidelity acceptance
test: successful delivery alone does not establish that the requested fidelity
was achieved. The encoded/purification estimates also are not arbitrary-noise
fidelity guarantees. The result's `finished_at` values are simulation seconds,
not wall-clock benchmark measurements.

`run_experiment(config)` releases quantum resources after constructing its result.
To inspect or consume delivered outputs, retain an `Experiment` instead:

```python
from qcns.api import Experiment

with Experiment(config) as experiment:
    network = experiment.build()  # Existing Network, nodes and routing tables.
    result = experiment.run()
    request = result.requests[0]
    if request.status == 'complete':
        token = request.protocol_result.outputs[0]
        half = network.clients[request.request.source].pr_retrieve_output(token)
        # Consume/inspect the counterpart through its destination node as well.
```

An `Experiment` snapshots its input config and is single-use. Its context manager
(or explicit `close()`) releases stored pairs, pending messages and protocol state,
including after a deadline. Output tokens in a result are no longer usable after
closing. Built-in configuration deliberately supports the models listed above.
For custom hardware, asymmetric links, custom protocols or workloads, use the
public `build_network`, `NodeContext`, `LinkConfig`, `QProtocolStack` and layer
base classes, reusing the same topology, scheduler and resources. For example:

```python
from qcns.api import NodeContext, Router, QProtocolStack, L1_EGP, L2_NEC, L3_FR

class MyForwarder(L3_FR):
    pass  # Override the existing packet or scheduled-swap hooks here.

def router_factory(context: NodeContext) -> Router:
    stack = QProtocolStack(l1_qprotocol=L1_EGP(), l2_qprotocol=L2_NEC(),
                           l3_qprotocol=MyForwarder(context.routing_table))
    return Router(context.node_id, context.simulation, context.routing_table,
                  qprotocols=stack)
```

Pass that factory to `build_network` as described below; there is no registration
step. Run `python python/examples/unified_experiment.py --mode raw` (or
`purification` / `encoded`) for a complete example. Add `--save path.json` to save
the effective configuration.

## Core/periphery networks

`network/utils.py` separates topology, hardware/node construction, and traffic.
Its public helpers are exported through `qcns.python.components.network`.
`create_topology` returns a `Topology` description and creates no simulation
hosts or requests. Clients currently have one access router and never forward
traffic. The core must be a connected, simple, undirected graph without self-loops.

```python
from qcns.python.components.network.utils import create_topology, summarize_topology

topology = create_topology(
    core='barabasi_albert_graph', num_routers=20,
    core_parameters={'m': 2}, num_clients=80,
    attachment='balanced', seed=7,
)
print(summarize_topology(topology, include_paths=True))
```

Core choices include the existing NetworkX presets, an existing graph
(`core=my_graph`), and a custom factory accepting `seed=...` plus
`core_parameters`. `create_core` interprets `num_routers` as the exact total
router count, including for stars. Grids require `shape=(rows, columns)` in
`core_parameters`, whose product must equal `num_routers`. Invalid or disconnected
random draws raise an error; the helper never silently repairs or resamples them.

The returned description contains `graph`, `routers`, `clients`,
`access_routers`, `original_to_id`, and generation `metadata`. `core_graph`
returns a copy of the router subgraph. Core labels are translated to integer
IDs in their original insertion order, preserving labels as node attributes.
Routers get IDs `0..R-1`; clients get `R..R+C-1`. Input graphs and their
attributes are copied. Exact replay requires the same graph insertion order,
generator parameters and compatible dependency versions. Metadata records the
effective root seed and NetworkX/NumPy versions. Core and attachment randomness
use separate local streams and do not reseed global NumPy state.

Attachment policies are `uniform`, `balanced` (client counts differ by at most
one), or `explicit`. For example, `client_attachments={4: 'west', 5: 'east'}`
with `attachment='explicit'` maps generated client IDs to **original core labels**.
`validate_topology` checks roles, graph connectivity and access assignments.
Summaries include core/access edge counts, degree distribution and per-router
client load. Client path statistics are opt-in and aggregate distances without
retaining a matrix of all pairwise distances.

`build_network` constructs nodes via explicit factories:

```python
from qcns.python.components.network.utils import build_network, NodeContext, LinkConfig
from qcns.python.components.network.router import PurificationRouter
from qcns.python.components.network.client import PurificationClient
from qcns.python.components.simulation.simulation import Simulation

def router_factory(context: NodeContext) -> PurificationRouter:
    return PurificationRouter(context.node_id, context.simulation,
                              context.routing_table, **context.options)

def client_factory(context: NodeContext) -> PurificationClient:
    return PurificationClient(context.node_id, context.simulation,
                              context.routing_table, **context.options)

sim = Simulation(seed=7)
network = build_network(
    simulation=sim, topology=topology,
    router_factory=router_factory, client_factory=client_factory,
    node_options={20: {'destination': 21, 'fidelity': .97}},
    core_links=LinkConfig(), access_links=LinkConfig(),
)
```

Factories receive the ID, simulation, role, routing table, access router,
copied graph attributes and per-node `options`. They choose the node class,
protocol stack and request execution policy. Raw or encoded implementations
use the same factory contract. Graph attributes are metadata, not automatically
passed to node constructors. `network.nodes`, `.routers` and `.clients` are
ID-indexed mappings. `Simulation.hosts` exposes registered hosts read-only.

Use separate `core_links` and `access_links` defaults, `link_overrides={(u,v):
config}`, and `node_timings={id: EventTimings(...)}`. A `LinkConfig` contains
forward/reverse entanglement-generation models and forward/reverse memory
models. Forward means lower-ID to higher-ID; memory assignment follows the
existing `Node.set_eqs_connection` sending-direction convention, not an
independent endpoint-memory convention. Models are deep-copied per link before
connecting through that public API, which also creates classical channels.
Defaults use ideal effective L3 generation and ideal unbounded PQM memory;
physical experiments should supply explicit fidelity, rates, delays, capacities
and noise models. Lengths retain the hardware API's km units; durations are seconds.
Graph positions or edge weights do not implicitly become physical link lengths.

Routing covers every node as a possible control endpoint, including routers
needed by PR sessions. One shortest-path tree per destination provides
consistent next hops even with zero-weight ties. `routing_metric` accepts a
weight attribute name or a static undirected cost callable. Costs must be finite
and nonnegative. Missing named weights default to one. Unreachable endpoints
raise errors. Route tables are snapshots; rebuild after changing the topology.
Construction validates topology, routing and override references before creating
hosts. A custom factory that raises may leave its already registered hosts in
the simulation; discard that partially built simulation before retrying.

Traffic stays independent:

```python
from qcns.python.components.network.utils import generate_requests

requests = generate_requests(
    topology.clients, topology.clients,
    requests_per_source=1, qubits=1, fidelity=.97,
    arrival_interval=0., rng=sim.rng('traffic'),
)
```

Each `TrafficRequest` has explicit source/destination IDs, qubit count, absolute
arrival time and fidelity. Destinations are uniform among eligible endpoints,
excluding self. Periodic traffic starts at zero; `arrival_rate` instead enables
independent Poisson arrivals per source, starting after the first exponential
interval. Requests are data: client factories must install a runner that uses
`await node.wait(...)` and the selected protocol to execute them. Impossible
destination sets fail immediately instead of retrying indefinitely.

The legacy single/multi traffic matrix helpers remain available with optional
`seed` or `rng`. Rows and columns follow the supplied ID lists; `max_qubits`
retains its **exclusive** upper-bound convention. Pass `dst_nodes` to
`calculate_single_packet_count` for arbitrary destination IDs. The legacy
`create_graph` retains NetworkX's size arguments (including star leaf counts),
but now safely relabels all nodes and starts clients after the actual core size.
The old broken combined `create_topology(graph_name, ..., router, ...)` interface
is replaced by `create_topology(core=...)` followed by `build_network(...)`;
traffic is no longer generated implicitly. Legacy connection helpers now use
`set_eqs_connection` and accept a `LinkConfig` in an edge's `config` attribute.

Run `python python/examples/core_periphery.py` for two clients and four PRs,
on-demand 0.99 Werner-pair generation, finite source/classical delays, and
10-second dephasing memory. The seeded example measures conditional Phi-plus
fidelity separately from the protocol's rough fidelity estimate; that estimate
is not a guarantee under arbitrary noise. Tests cover the same end-to-end
scenario along with reproducibility, routing, validation and hardware overrides.

## Simulation stopping rules

`Simulation(end_time=None)` runs without a deadline. Every nonempty unbounded
network must contain at least one completing host (`Node(..., stop=True)`).
Once all completing hosts finish, background services (`stop=False`) are
cancelled. A finite absolute `end_time`, in simulation seconds, permits a
network consisting entirely of background services. The run stops at that
deadline or when its completing hosts finish, whichever comes first. If all
background host coroutines actually return, the network is also complete.

```python
sim = Simulation(end_time=10.0)
# Add clients and/or background routers here.
sim.run()       # Retain the configured deadline.
sim.reset()
sim.run(None)   # Explicitly clear it; requires a completing host if nonempty.
```

Use `set_end_time(seconds_or_none)` to change the configuration. `sim.time`
reports the last dispatched event time, or the finite deadline when reached.
Events exactly at the deadline execute; later callbacks do not. Unfinished
host tasks are cancelled and awaited on exit, including after exceptions.
With a finite deadline, an idle network with no events ends at that deadline;
without one, unfinished hosts with no events raise a deadlock error.
Empty simulations finish immediately.

Migration: `end_time=0` now means a deadline at time zero, not an unbounded run.
Use `None` for unbounded runs. The scheduler has one `_run_events` coroutine;
the `_handle_events`, `_handle_event` and `_handle_event_end_time` indirection
has been removed. Node completion is tracked by the scheduler, without
infinite-time stop events for background services.

## Per-node event durations

`@node_event` adds configurable processing overhead in **simulation seconds**.
Node's public operational methods use it for packet handling, generation
requests, memory access, resource queries and protocol operations. Configure
each node independently, or replace its configuration before subsequent calls:

```python
from qcns.python.components.network.node import Node, EventTimings, node_event
from qcns.python.components.simulation.simulation import Simulation

sim = Simulation(seed=7)
node = Node(0, sim, event_timings=EventTimings(
    default_duration=1e-7,
    durations={'send_packet': 2e-6, 'l1_retrieve_qubit': 5e-7,
               'l1_retrieve_result': 1e-7, 'gate:X': 5e-6},
))
node.event_timings = EventTimings(durations={'send_packet': 3e-6})
# The same assignment works for existing Router and Client instances.

class CustomNode(Node):
    @node_event
    def process_request(self, request: object) -> None:
        # Use the node's 'process_request' timing override or its default.
        pass
```

Configuration copies and validates the override mapping; durations must be
finite and nonnegative. Keys are decorated method names, including custom
methods. Configurations are immutable: replace `node.event_timings` to update
them. Unspecified overhead defaults to zero, preserving previous idealized
processing times. Nonzero defaults are user assumptions, not calibrated values.

Overhead is charged **before** the event body, including calls that fail.
Nested decorated events each incur their own cost. Gates use `@gate_event`
to charge their actual operation duration once. Both `apply_gate('X', ...)`
and `await execute_gate('X', ...)` select `durations['gate:X']`, falling back
to `durations['apply_gate']`, then to the existing per-node `gate_duration`
mapping (5 microseconds for unknown gate names). These are replacements for
the hardware gate duration, not extra overhead. Scheduled gate completion
does not charge twice, and `apply=False` queries do not charge a gate duration.
The processing default does not replace hardware gate timings. The only public
time-passing primitive is **`await node.wait(duration)`**, in simulation seconds.
It is asynchronous so the scheduler can run other nodes before resuming the
caller. It does not add configured processing overhead or advance the clock
early. A zero duration still yields through a completion event. Invalid,
negative or nonfinite durations are rejected; cancellation disables completion.

Migration: replace `await node.delay(dt)` with `await node.wait(dt)` and replace
old synchronous `node.wait(dt)` calls with `await node.wait(dt)` inside async
code. Offline experiments that need storage time to elapse should run a node
coroutine through `Simulation.run()`. The `delay` method has been removed.
Packet serialization, propagation and source hardware delays remain in their
existing models, in addition to configured processing overhead. Initialization,
connection setup, reset, properties, private helpers and standalone graph-state
analysis helpers are untimed. Reset also suppresses nested processing costs.
Subclass overrides must explicitly use `@node_event` to retain timing.

Synchronous operation methods retain their current immediate execution semantics:
an internal decorator helper advances the local clock and schedules a
`WaitEvent` accounting marker, but does not yield Python execution or defer
state changes. That marker is not the runtime waiting mechanism. Async decorated
methods await `Node.wait`, which uses a cancelable `CompletionEvent`, before executing their body;
cancellation during that delay prevents execution. For causally ordered shared
state changes, use async events and the existing scheduled gate interfaces.
Nested synchronous events remain immediate even inside an async event.
This models per-call overhead, not a serialized CPU resource: concurrently
running coroutines may overlap. Receive timeout durations start after their
configured receive overhead. Zero overhead creates no extra events.

## Independent packet headers and forwarding

`Packet(src, dst, ...)` uses `src` and `dst` for the current classical hop.
Headers are optional and independent. L3 addresses identify end-to-end nodes;
L2 header addresses and L4 ports are not rewritten by forwarding.

Header lengths sum the declared wire-field widths in bits, reserving 256 bits
per outcome array regardless of its current population. L1 uses 281 bits
(282 for TPSP), L2 uses 377 (386 for QECP), L3 uses 1066, L4 uses 313
(322 for QECP), and L7 uses 272 before its JSON payload. Control envelopes
add their existing encoded metadata lengths. Derived bookkeeping is excluded.

L3 headers include an 8-bit logarithmic fidelity field. Set it with
`Packet(1, 2, l3_fidelity=0.99)`, `packet.l3_fidelity = 0.99`, or the
`fidelity=` keyword on `L3_CRP`, `L3_FRP`, and `L3_ERP`. The supported range is
`[0.5, 1.0]`, defaulting to `0.5`. `fidelity_ir` (or `packet.l3_fidelity_ir`)
exposes the byte: 0 means 1.0 and 255 means 0.5. The encoding follows
`ceil(31.875 * log2(511 - 510*F))`; decoding follows
`(511 - 2**(code/31.875)) / 510`. Public `encode_fidelity` and
`decode_fidelity` also accept NumPy arrays. Quantization rounds down in fidelity
and has finer resolution near 1. Invalid inputs are rejected without wrapping.
The field adds eight bits to header transmission timing. It is request metadata;
it does not measure quantum states, update itself after swapping, or enforce a
purification threshold. Those policies belong to the operation protocol.

```python
from qcns.python.components.packet.packet import Packet
from qcns.python.components.packet.protocol import L1_Protocol

class ExperimentHeader(L1_Protocol):
    def __init__(self, requested, needed):
        super().__init__(requested, needed, _protocol=99)

packet = Packet(1, 2, layer1=ExperimentHeader(4, 4))
# Equivalent numeric construction with a locally supplied factory:
packet = Packet(1, 2, l1_requested=4, l1_needed=4, l1_protocol=99,
                header_factories={1: {99: ExperimentHeader}})
```

Factories receive the same positional arguments as the built-in header for
their layer. Pass a header instance directly for custom constructor arguments.
Unknown numeric IDs raise `ValueError`; mappings are copied per packet and do
not register anything globally. Install the corresponding quantum operation
implementation explicitly in the node's protocol stack.

To forward, first finish using the incoming hop's resources, then call
`packet.set_hop(current_node_id, next_hop_id)` and `node.send_packet(packet)`.
To reply to the previous hop, copy the packet and call `reply.reverse_hop()`.
These methods preserve header contents and `request_id`. `send_packet` sends a
snapshot with `src` set to the sending node; it does not mutate the caller's packet.
End-to-end replies must separately reverse their protocol's endpoint addresses
and select a route to that endpoint.

Migration: use `packet.src`/`packet.dst` instead of `l2_src`/`l2_dst` for routing
and neighbor resource lookup. Construct concrete headers instead of changing
an existing header's protocol ID to select another implementation. Built-in
L1 IDs are SRP=0, TPSP=1, BSMP=2, FSP=3; L2 and L4 use EPP=0, QECP=1.
The flat constructor also selects QECP when a code is supplied with the default
protocol ID.

L2/L4 headers derive an integer input/output ratio from their initial counts:
`inputs_per_output = needed // requested`. Initial nonempty counts must satisfy
`requested <= needed` and be divisible. The existing constructor shorthand
still copies a supplied nonzero count into an omitted/zero counterpart.
Changing `needed` resizes `success` to `needed // inputs_per_output`, preserving
the retained prefix and initializing new entries to false. Incomplete input
blocks have no outcome entry. Changing the output target `requested` or the
QEC code identifier does not redefine the retained ratio.

For example, `(requested=2, needed=10)` creates two outcomes; setting
`needed=15` grows the array to three. Use `header.set_counts(2, 10)` to initialize
an empty header or explicitly configure a new ratio; this clears old outcomes.
Invalid count changes leave the header unchanged. Quantum protocols still
validate their physical input requirements (two inputs per implemented EPP
output, or the code length per implemented QEC output); generic header sizing
does not establish whether an operation is physically supported. When changing
codes, configure the corresponding counts before executing the new operation.

Run the regression suite with `python -m unittest discover -s tests -q`.

## Quantum protocol classes and user extensions

`Router(node_id, simulation, routing_table={destination: next_hop})` delegates
ordinary swapping, reservation/waiting, and classical forwarding to its L3
operation. It accepts `qprotocols=QProtocolStack(...)` for a user-defined stack;
this replaces its mode-based defaults and must contain an L3 operation.
The supplied stack and route dictionary are copied. Updating the router's
route entries also updates its installed `L3_QProtocol` implementation.
Alternatively, pass `graph` and `clients` to compute shortest-path next hops.
Missing paths or a missing L3 operation fail during construction.

Headers describe requests; quantum protocol objects implement their behavior.
Install an operation object in each desired `QProtocolStack` slot:

```python
from qcns.python.components.network.qprotocol import (
    QProtocolStack, L1_SR, L2_QEC, L3_FR, L7_TP,
)

stack = QProtocolStack(
    l1_qprotocol=L1_SR(reattempt=False),
    l2_qprotocol=L2_QEC(reattempt=False),
    l3_qprotocol=L3_FR(routing_table={3: 2}),
    l7_qprotocol=L7_TP(),
)
```

Pass `qprotocols=stack` when constructing a node. The node clones the
configuration and binds the operations to itself; mutable runtime state must
be initialized or cleared in `reset()`. Connection and memory configuration
must still support the selected operations. This refactor preserves the
existing quantum models, resource accounting and simulation-time scheduling.

| Layer base | Built-in operations | Main extension hooks |
| --- | --- | --- |
| `L1_QProtocol` | `L1_SR`, `L1_TPS`, `L1_BSM`, `L1_FS`, `L1_L3C` | `_prepare_results`, `_accept_results`, `_acknowledge`, `_retry_missing` |
| `L2_QProtocol` | `L2_EP`, `L2_QEC`, `L2_NEC` | `input_block_size`, `create_header`, `_prepare_inputs`, `_complete_batch`, `_validate_mode`, `request_more_inputs` |
| `L3_QProtocol` | `L3_CR`, `L3_FR`, `L3_ER` | `_swap_operation`, `_forward_quantum_request`, `_next_hop` |
| `L4_QProtocol` | `L4_EP`, `L4_QEC`, `L4_NEC` | `input_block_size`, `create_header`, `_prepare_inputs`, `_complete_batch`, `_validate_mode` |
| `L7_QProtocol` | `L7_TP`; placeholders `L7_RG`, `L7_DQC` | `_handle_payload`, `_validate_packet` |

Layer bases handle packet flow and resource orchestration. Subclasses can
override these hooks or replace `handle_packet` and `on_qubits_ready` entirely.
L2/L4 subclasses can supply `header_type`, or override
`create_header(src, dst, requested, needed, code=None, *, request=None)` to copy
custom configuration from the initiating packet. L1 handoffs preserve L2
request headers even when they carry no QEC code. Custom IDs require no change
to QCNS when header instances and operation instances are supplied explicitly.

The stack installs one operation per layer; it does not automatically choose an
operation from a packet's numeric ID. Each installed operation validates the
headers it accepts. A custom operation may implement its own dispatch policy.
`required_layers` declares stack dependencies: built-in L2 requires L1 and L3
requires L2. A custom implementation with no such dependency can override it
with `required_layers = ()`.

The previous class names remain compatible selectors: `L1_EGP('srp')` returns
an `L1_SR`, `L2_FIP('qecp')` an `L2_QEC`, `L3_QFP(routes, 'frp')` an `L3_FR`,
and `L4_FIP('epp')` an `L4_EP`. They are aliases of their corresponding layer
bases, so existing `isinstance` checks continue to work. `L7_TPP` aliases
`L7_TP`. Choose a different class to change behavior; mutating a mode string
does not switch an existing object's implementation.

`L3_ER` exposes the existing graph-state model using an explicit code and both
measurement records via `swap_entanglement_graph_state`. Automatic scheduling
is available separately through `L3_Encoded` and `EncodedRouter`, using the
effective model described below. Multi-output QEC is supported by the joint
Pauli backend below; network pair-storage orchestration remains single-output.
Non-stabilizer and coherent-noise placeholders remain unimplemented.

For a complete custom header/application example using the simulation event
loop, run `python python/examples/custom_protocol.py`. It prints `hello QCNS`.
This example demonstrates transport and extension interfaces, not a new
physical quantum protocol.



## Efficient stabilizer QEC

Gottesman's [thesis, section 3.4](https://arxiv.org/abs/quant-ph/9705052)
describes binary symplectic stabilizer algebra. Efficient stabilizer simulation
does **not** imply efficient simulation of every QEC code or arbitrary noise;
see also [Aaronson and Gottesman](https://arxiv.org/abs/quant-ph/0406196).
QCNS implements an ideal effective channel for arbitrary validated `[[n,k]]`
stabilizer definitions, with perfect syndrome extraction and correction.
It does not model noisy encoding/measurement circuits, non-stabilizer codes,
or coherent noise.

`StabilizerPauliCode` compiles physical Pauli contributions to packed binary
syndromes and logical frames. Batched XOR operations replace per-shot Python
decoding. Codes larger than 64 qubits use multiple words. A supplied decoder
maps a tuple of syndrome bits (generator order) to a correction Pauli string;
it must be deterministic, and each returned correction is validated.
Large codes require an explicit decoder: efficient syndrome calculation does
not supply an efficient decoder for every code. Existing registration, packet
code selection and Node interfaces remain unchanged.

- `method='exact'` retains exact ensemble probabilities, with `2**(n+k)`
  syndrome/frame states and a 12-check limit. Precomputed XOR permutations
  exchange additional memory for faster repeated evaluations.
- `method='sample'` explicitly selects Monte Carlo estimation. `batch_size`
  bounds temporary trajectories, `decoder_cache_size` bounds cached correction
  frames, and `max_working_bytes` rejects estimated excessive NumPy allocations
  before building tables. The default budget is 64 MiB; this is not a limit on
  the whole process or arbitrary user decoder allocations. There is no automatic
  switch from exact results to sampling.
- `logical_channel` takes independent I/X/Y/Z probabilities per physical qubit.
  `decode_errors` accepts `(shots, n)` integer Pauli trajectories, including
  caller-supplied correlated patterns; correlations must be supplied explicitly.

```python
import numpy as np
from qcns.python.components.qubit.qec import StabilizerPauliCode

code = StabilizerPauliCode('five_qubit', method='sample', shots=100_000)
noise = np.tile([0.97, 0.01, 0.01, 0.01], (code.length, 1))
estimate = code.estimate_channel(noise, rng=np.random.default_rng(7))
print(estimate.counts, estimate.probabilities, estimate.standard_errors)
```

Estimates report unconditional residual logical I/X/Y/Z probabilities under
ideal correction. Zero observed logical errors do not establish zero failure
probability; the reported marginal standard errors are plug-in estimates.
For Bell pairs the I component is fidelity to the target Phi-plus state.

### Multiple logical outputs

Supply `n-k` independent stabilizers and `k` logical X/Z pairs. The backend
checks generator independence, mutual commutation, and the canonical logical
commutation relations. `length` is `n`, and `num_logical` is `k`. Optional
`distance=d` records the declared distance; it is not computed or certified.
Actual correction behavior depends on the supplied operators and decoder,
not on a label or an assumed number of correctable errors.

```python
code = StabilizerPauliCode(
    'four_two_two', ('XXXX', 'ZZZZ'),
    logical_x=('XXII', 'XIXI'), logical_z=('ZIZI', 'ZZII'),
    distance=2, method='sample', shots=10_000,
)
noise = np.tile([0.97, 0.01, 0.01, 0.01], (code.length, 1))
for batch in code.iter_logical_errors(noise, rng=np.random.default_rng(7)):
    # Each row contains one JOINT realization of all k residual logical Paulis.
    print(batch.shape)  # At most (batch_size, k).
```

The `[[4,2,2]]` example detects every single-qubit Pauli error but cannot
correct every such error. This backend always applies its decoder; it does
not interpret nonzero syndromes as heralded rejection.

`logical_channel` and `estimate_channel` return a joint array of length `4**k`,
indexed by `sum(P_i * 4**i)` with `P_i=0,1,2,3` for I,X,Y,Z. Index zero is
the probability that all logical outputs are error-free. These arrays are
memory-guarded. For large `k`, use `iter_logical_errors`: its storage is
polynomial in code size and batch size, plus decoder costs, with no `4**k`
table. `decode_errors` returns `(shots,k)` labels for caller-supplied physical
error patterns; `k=1` retains its original `(shots,)` return shape.
Never replace joint samples or joint channels by independent marginals when
propagating logical outputs: that would discard correlations.

Existing L2/L4 Node operations still retain one pair per input block and
explicitly reject `k>1` before staging resources. Their multi-output hooks
remain unimplemented until shared correlated output storage and scheduling
are available. The new backend does not silently turn correlated logical
outputs into independent pair density matrices.

Run `python python/examples/qec_multiple_outputs.py` for a runnable custom
`[[5k,k,3]]` family with a supplied product decoder and bounded joint sampling.
The recorded `benchmarks/qec_multiple_outputs.json` measures 5,000 trajectories:
`[[10,2,3]]` took 3.5 ms (0.42 MiB traced peak), `[[40,8,3]]` 12.5 ms
(1.34 MiB), and `[[160,32,3]]` 372 ms (6.97 MiB) on the recorded Windows/Intel
environment with Python 3.12 and NumPy 2.2.4. Times exclude construction;
traced peaks include fresh construction and evaluation, not whole-process RSS.
These are ideal QEC kernel measurements, not complete network runtimes.

Run `python python/examples/benchmark_qec.py --output benchmarks/qec_local.json`
to reproduce the workload. Optional `--reference path/to/previous/qec.py`
compares against a saved implementation and checks numerical agreement.
The earlier single-output paired results in `benchmarks/qec_gottesman.json` include source
hashes, construction time, repeated evaluation time, traced peak memory,
hardware and Python/NumPy versions. Tests cover exhaustive five-qubit Pauli
patterns, packed-word boundaries, decoder validation/cache limits, seeded
sampling and statistical agreement with the exact channel. Repetition-code
benchmarks test scaling, not protection against general Pauli noise.

## Purification routers

Run `python python/examples/purification_routers.py` for two clients connected
through four `PurificationRouter` instances. The example generates Werner pairs
on demand with link fidelity 0.99, finite generation and classical delays, and
16-slot physical memories. With seed 7, the delivered conditional Phi+ fidelity
is about 0.9765 for a requested 0.97. The unit test also runs the same path with
ordinary routers: raw swapping delivers about 0.9513 and misses that target.
The test uses equal prepared input supplies; it excludes their preparation
latency. These single-run values are regression checks, not rate benchmarks.

`PurificationClient(..., destination=..., requested=1, fidelity=.97)` initiates
a request. Import it from `qcns.python.components.network.client`; routers live
in `qcns.python.components.network.router`. Omit `destination` for a serving endpoint. Both clients and routers
need routes to every session endpoint, including intermediate routers. A PR
constructed with `graph=...` computes these routes automatically. Connect nodes
with the existing `set_eqs_connection` API. The default L1 mode is `l3cp`;
other connection models need their corresponding `l1_eg_mode`.

The proposed hierarchy protocol is implemented by `L3_PR`, with classical
control headers in `L3_PRP` and local quantum operations on `Node`. SYN,
SYN_ACK and ACK establish the path. Node-owned `pr_requests` and `pr_sessions`
(also exposed as router `request_table` and `session_table`) hold hop/hierarchy
indices, ownership, FIFO request queues and retained pairs. Owners use SEND;
peers use RECEIVE. Long-distance halves remain in their original physical
memories and count towards capacity. Sessions reuse matching endpoints and the
same quantized fidelity requirement. Even hierarchy indices swap, with outcomes
and physical corrections delivered towards the destination. Rejected
purification attempts rebuild the original interval while retaining other
successful cached pairs. Empty failed sessions advance their generation.

Scheduling is demand-driven: recursive acquisition follows the hierarchy's
elimination order and purifies until the pending request has enough pairs.
It does not continuously fill idle sessions or synchronize global rounds.
The six-node elimination order is PR2/PR4, PR1, then PR3; independent operations
are not required to happen simultaneously. Clients are always session endpoints.

The physical operation is the bilateral CNOT recurrence of
[Bennett et al. (1996)](https://arxiv.org/abs/quant-ph/9511027), with communicated
Z-measurement parity and an **ideal bilateral Phi+-preserving twirl** after each
accepted round. Twirling has a default 5 microsecond duration. Local gates,
measurements, memory noise and extraction use the existing backend; only local
two-pair operations combine density matrices (at most four qubits). Planning
assumes independent Werner inputs and ideal operations.
`node.estimate_link_fidelity(neighbor, SEND)` (or `RECEIVE`) returns the nominal
conditional Phi+ fidelity from that direction's configured connection model.
Both directions are characterized when `set_eqs_connection` installs the link;
the estimate excludes later memory storage and sampled fluctuations. Override
this typed node method for custom/calibrated estimates. The former
`link_fidelity` protocol/client/router parameter has been removed: configure
the link itself, for example `L3C_Model(fidelity=.99)`, instead.
Prepared-pair experiments should also configure a matching link estimate.
`coherence_time` is an optional effective pair-decay estimate;
it does not configure physical memory noise. The default is infinite.
An estimated target is not an actual-fidelity guarantee under arbitrary noise.
The existing 8-bit header quantization rounds the requested target down.

Requests return `PRResult` with status, output tokens, estimated fidelity,
completion time in seconds and a failure reason. Read actual endpoint outputs
with `node.pr_retrieve_output(token)`; this consumes the reservation and applies
elapsed storage noise. After consuming outputs, call `node.pr_archive(key)` at
each participating node to release completed metadata. Counters distinguish raw
pairs, purification attempts, rejections and swaps. Defaults bound each node's
request table to 64 entries, recurrence depth to 8, purification attempts to 64,
and raw inputs to 4096 per request per owner. The default request timeout is
1 second, with at most another timeout interval for cancellation acknowledgment.
Raw supply exhaustion, extraction/swapping loss and exhausted retry budgets
fail explicitly and cancel reservations. Parity rejection retries recursively.

Use `generate=False` only for confirmed prepared input pairs. On-demand generation waits
for L1 heralds. Classical delivery is assumed reliable; each client initiates
one request at a time. Extend `L3_PR` to change acquisition or retry policy and
install it through `QProtocolStack`; use its asynchronous `serve()`/`establish()`
entry points. The example seeds the existing process-global NumPy backend;
independent random streams for concurrent simulations are not introduced here.

## Encoded routers

Run `python python/examples/encoded_routers.py` for two `EncodedClient` endpoints
and four `EncodedRouter` nodes. The classes live in `network/client.py` and
`network/router.py`; operations live in `network/qprotocol.py` as `L3_Encoded`,
and messages in `packet/protocol.py` as `L3_ECP`. All services need
routes to the intermediate endpoints. Normal L1 connections generate the raw
Bell pairs. The source selects `code='five_qubit'` or `code='steane'` (or a
registered exact [[n,1]] stabilizer code); the header carries this selection to
every router. Add `purification_rounds=1` to perform a recurrence round before
each physical input joins the code block. Zero pre-purification is the default.
Use the same preparation policy on all service nodes.

This is a proposed **ideal effective graph-interface scheduling model**, based
on the existing `MeasurementBasedRepeaterModel`. It applies the exact decoded
segment channel to n independent Bell-diagonal pairs and fuses their logical
representatives. Syndrome/frame probabilities replace dense encoded states;
uncorrectable errors remain as logical errors in the resulting density matrix.
Both peers confirm their physical inputs before decoding. Their elapsed memory
noise and extraction losses are applied through the existing memory backend.
The final two-qubit state represents a decoded logical pair. Intermediate
logical representatives experience the configured memory noise; continuous
encoded memory correction is not part of this model.

The interface factorization uses an ideal (n+1)-qubit graph distillation
resource at each end of every elementary segment, then ideal logical BSMs.
It averages the graph Bell records and their corrections; it does not generate
a physical graph-preparation circuit or individual syndrome transcripts.
This differs in resource preparation and timing from a fused 2n-qubit repeater
resource. Costs are therefore reported for the implemented factorization,
not claimed as the minimal-resource implementation of the paper. Graph resource
and measurement noise, coherent/correlated inputs, erasure decoding, sampled
decoders and multiple logical outputs are unsupported in this router path.
Existing explicit-record graph recovery and trajectory APIs remain available.

`preparation_duration` charges ideal local interface preparation (default
100 microseconds); `max_graph_qubits` bounds each local symbolic resource
(default 4096). These resources are a separate ideal preparation pool from the
physical link memories. `graph_resource_qubits` records their cumulative cost;
`raw_pairs`, `encoded_blocks`, `encoded_retries` and `swaps` record the other
operations. Graph preparation gates and their physical memory occupancy are
not simulated. No 4n-qubit density matrix is allocated.

`max_retries=3` permits three retries after an initial preparation attempt.
Incoming source entanglement remains reserved during outgoing preparation.
If preparation/extraction loses only the new block, its remaining halves are
discarded before retrying. Consuming fusion or correction loss sends a FAILURE
reply backwards immediately. Failure replies propagate to the source and
trigger cancellation and cleanup along the path. Generation retries are bounded
separately by the same configured limit; the shared request deadline remains
the final bound. Timeouts, exhausted input supply and retry exhaustion are
explicit failures. A residual logical error is not a heralded failure: the
decoder cannot inspect simulator ground truth to decide whether to retry.

The result is an `EncodedResult` containing status, code, backend assumptions,
output tokens, completion time and a conservative estimated fidelity. Retrieve
halves with `node.encoded_retrieve_output(token)`. Shared request/reservation
storage and cleanup reuse the PR service; `node.pr_archive(key)` releases
terminal metadata after output consumption. To customize scheduling, subclass
`L3_Encoded` and install it through `QProtocolStack` with an asynchronous
`serve()` or `establish()` node loop.

With seed 7, perfect storage, 0.99 Werner links and finite classical delays,
the six-node example delivers conditional Phi+ fidelity 0.995123 using 25 raw
pairs and 60 symbolic graph resource qubits. Ordinary raw swapping would give
about 0.9513. The tests cover both code selections, single-error correction,
retained uncorrectable logical errors, optional purification and failure cleanup.
These are correctness checks under the declared model, not rate benchmarks.

Run `python python/examples/benchmark_encoded_routers.py` to measure wall-clock
runtime separately from simulated network time. It reports the first run,
20 subsequent runs of the actual example, five fresh Python processes, and a
separate allocation-traced run. Each scenario checks its output fidelity.
On 2026-10-08, Windows 10, Intel family 6/model 154, Python 3.12.0 and NumPy
2.2.4, the simulation median was 0.048 seconds (range 0.033–0.065), while the
fresh-process median including startup was 0.756 seconds (range 0.724–0.838).
Skipping zero-variance fidelity sampling reduced the fresh-process median from
1.136 seconds in the preceding measurement; it avoids loading `scipy.stats`
for deterministic links. Warm simulation times did not improve measurably.
The separate run peaked at 0.205 MiB of traced allocations, excluding imports,
pre-existing caches and untraced native memory; this is not total process RAM.
These measurements describe one delivered pair on this six-node scenario,
not scaling with network size, traffic, or larger codes.

## Scaling evidence and regression probes

Run the noisy raw-forwarding baseline with:

```powershell
python python/examples/benchmark_scalability.py --output benchmarks/scalability_baseline.json
python -m unittest discover -s tests -p test_scalability.py -v
```

The workload follows the reported 100-node/80-client experiment: a 20-router
Barabasi-Albert core and 80 attached clients, each sending one request to a
seeded randomly chosen other client. Each source can have a different number
of incoming requests. Defaults are attachment parameter m=2, seed 7, and
1-second noise constants. Those parameters and exact link delays were not all
available for the original experiment, so this is a comparable baseline, not
an exact reproduction. Both existing memory error classes run on the physical
halves. `DepolarizationError` currently implements amplitude damping despite
its name; a new characterization test makes this explicit. Dephasing is also
enabled. Source halves stay reserved in memory until their output is ready.

Default `--supply eager` generates all required raw pairs during simulation,
using the known batch traffic demand, then performs actual forwarding and
swapping. Counts follow the routers' exact next-hop tables, including shortest
path tie-breaking. It uses perfect generation and gates, 100-microsecond pair
generation duration, FIFO physical memories, 1-km core / 0.1-km access classical
links, and 100-Mbit/s classical bandwidth. There is no fidelity acceptance
threshold. Validation checks all intended endpoint pair identities, density
matrix validity, and leftover memory. `--supply prepared` excludes generation;
`--supply ondemand` exercises reactive resource acquisition instead. The latter
uses unlimited memory to distinguish protocol failures from capacity failures.

Measured on Windows 10, Intel family 6/model 154, Python 3.12.0, NumPy 2.2.4:

| Nodes | Requests | Median run + output validation | Scheduled events | Task.done calls | Peak process MiB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 20 | 16 | 0.024 s | 340 | 6,863 | 71.5 |
| 50 | 40 | 0.064 s | 968 | 48,553 | 74.7 |
| 100 | 80 | 0.158 s | 2,059 | 206,203 | 80.0 |
| 200 | 160 | 0.359 s | 4,410 | 882,603 | 86.3 |
| 400 | 320 | 0.859 s | 9,214 | 3,686,803 | 94.5 |
| 800 | 640 | 1.983 s | 19,192 | 15,356,003 | 108.9 |

The stored baseline report includes the additional 800-node case, produced
with `--sizes 20,50,100,200,400,800`. Each size runs in a fresh worker, with one
warm-up and three timed repetitions. Imports and topology/resource construction
are excluded from the table and setup is reported separately. The timed work
includes generation, swapping, and output extraction/validation with memory
noise. Lightweight event counters are enabled. Allocation tracing and cProfile
run separately. Process memory is the Windows peak working set before profiling
and tracing, including interpreter/imports; traced scenario peaks exclude warm
caches and native allocations not visible to tracemalloc. Repeated seeds measure
timing variation, not statistical uncertainty over network/noise realizations.

Findings supported by the original baseline (before the runtime fixes below):

- Raw forwarding at 100 nodes is comfortably below one second here. The
  scheduler's task inspections nevertheless grow approximately as nodes times
  events. The 800-node profile spends substantially more time in scheduler/task
  inspection functions. cProfile exaggerates frequently called Python/C boundary
  costs; use it to locate work, not to promise a particular optimization speedup.
- At 100 nodes, recursive `deepcopy` calls are also prominent in the profile.
  Packet snapshot optimization should preserve independent mutable headers.
- Synthetic repetition-code probes show exact QEC syndrome/frame state counts
  of 16, 64, 256, 1024, 4096 and 16384 for code lengths 3 through 13. This
  demonstrates exponential state growth, not the error-correction quality of
  these codes. The report separates code construction from channel evaluation.
- The cache test confirms every distinct operator-cache entry remains retained
  until its cache is destroyed. This is an eviction-policy concern, not evidence
  that repeated identical network runs leak. Completed scenarios in the
  regression test are collectable after output release and garbage collection.
- Sampled QEC recreates its seeded stream on each channel evaluation. Separate
  simulator instances also share the global quantum RNG. These are distinct
  reproducibility/independence concerns.

The original on-demand failure is retained in
`benchmarks/scalability_ondemand.json` as historical evidence. Current regression
tests require on-demand delivery and overlapping PR requests to complete.

## Runtime fixes and migration

There are exactly four physical memory stages per SEND/RECEIVE memory:

| Stage | Meaning |
| --- | --- |
| L0 | Generated inputs awaiting their generation herald |
| L1 | Confirmed raw pairs, usable for raw forwarding or fidelity improvement |
| L2 | Purification/QEC inputs or survivors awaiting completion/peer acceptance |
| L3 | Confirmed fidelity-improved outputs |

Session ownership is metadata on these entries, not a fifth store. Reserved
entries count against physical capacity and cannot be extracted by unrelated
requests. PR parity circuits keep their inputs in L2 while executing. Successful
bilateral confirmation promotes the retained half to L3. Raw forwarding and
NECP do not claim fidelity improvement by moving raw pairs to L3.

Use `node.memory_stage_count(neighbor, SEND, layer)` to inspect a physical stage,
`node.ready_pair_count(neighbor, SEND)` to count unreserved forwarding inputs,
and `node.retrieve_ready_pair(neighbor, SEND)` to consume one. When an L2
improvement protocol is installed, its raw L1 inputs are excluded from forwarding.
The historical `l3_num_qubits` / `l3_retrieve_qubit` names remain readiness aliases
for compatibility. `L3Connection.success_creation(time)` is an explicit prepared,
confirmed L1 fixture; normal generation methods insert into L0 and require heralds.
Code that previously used generated pairs immediately must receive/process L1
control before consuming them.

Generation batches identify their concrete L0 inputs. Reordered heralds cannot
consume another batch, and duplicate/stale heralds are ignored. The raw router
subtracts unconfirmed inputs when requesting more supply. L3C herald timestamps
include generation duration. PR services bound and queue acquisition commands,
while continuing to dispatch their classical replies; overlapping requests no
longer recursively block one another's session FIFO.

`Simulation(seed=7)` owns stable component RNG streams for node operations,
generation, memory extraction/noise, channels, and sampled runtime QEC. Stream
identities do not use Python's randomized hash or construction order. Use
`simulation.rng('my_protocol', node.id)` in custom protocols. Reset rewinds existing
stream objects; a new unseeded simulation uses fresh entropy exposed as `sim.seed`.
`np.random.seed(...)` no longer seeds runtime simulation operations. Standalone
quantum measurements still accept `rng=`, as do standalone sampled QEC calls.
QEC sampling advances a persistent stream instead of restarting its seed on every
block. A supplied RNG overrides the standalone code's stream. Fixed systematic
noise parameters sampled during configuration should be constructed with an
explicit `rng=` too. Reproducibility is scoped to the same effective configuration
and dependency versions, not identical floating-point output across platforms.

The scheduler receives task-failure callbacks instead of inspecting every host
after every event. Raw routers await their BSM/correction completion; PR and encoded
routers await local parity circuits, twirling, logical fusion, corrections, and
resource-preparation delays. New custom asynchronous protocols should use
`await node.execute_gate(...)` and `await node.wait(seconds)`. Inputs must remain
owned until completion. These operations model ideal gates with their configured
durations: storage noise is applied on access, and active-gate noise is not added
implicitly. PR operations retain occupied slots during their circuits.

The synchronous `apply_gate`, `wait`, and direct protocol helpers remain immediate
operations for compatibility and offline reference calculations. They are **not**
completion-time APIs. Direct L2/L4/L7 synchronous protocol entry points have not
all been migrated to awaited execution; use of these paths with interleaved quantum
operations still needs care. Tests separately assert the legacy synchronous
contract and the awaited operation's causal visibility. No wall-clock assertion
is used as a correctness test.

Dense QSystem construction and combination reject working-set estimates above
`max_bytes` (default 256 MiB, estimated as four complex128 matrices). Combination
checks before changing either input. This is a per-operation estimate, not an
OS process-memory cap. Operator caches use LRU eviction with an 8 MiB limit per
cached operator function. `cache_info()`, `cache_clear()` and
`cache_configure(max_bytes=...)` expose that limit. Large exact QEC still has its
explicit syndrome-space limit; no automatic approximation is introduced.

For isotropic depolarization use `DepolarizingError(T)`, which contracts the Bloch
vector by `exp(-t/T)` and acts correctly on one half of an entangled pair.
`AmplitudeDampingError(T)` names the old channel correctly. The legacy
`DepolarizationError(T)` keeps amplitude-damping behavior to avoid silently changing
existing experiments. The scaling comparison retains that original noise model
plus dephasing; changing the physical channel is a separate experiment.

Updated reports are `benchmarks/scalability_after.json` and
`benchmarks/scalability_ondemand_after.json`. They include source hashes, seeds,
dependency/platform information, timing, validation, and memory measurements.
Generation timing, random streams, and gate scheduling changed, so before/after
wall-clock differences are not an isolated scheduler microbenchmark.

Final measurements on the same Windows/Python/NumPy setup, three repetitions:

| Supply | Nodes / requests | Median run + validation | Peak process MiB |
| --- | ---: | ---: | ---: |
| Eager | 20 / 16 | 0.0295 s | 72.4 |
| Eager | 100 / 80 | 0.1611 s | 81.6 |
| Eager | 400 / 320 | 0.7738 s | 101.0 |
| Eager | 800 / 640 | 1.6687 s | 123.2 |
| On demand | 100 / 80 | 0.2109 s | 82.1 |

Every request delivered, with no leftover memory after output collection. Task
inspections are 43, 203, 803 and 1603 for the four eager sizes. Memory is higher
than the original baseline, partly reflecting owned RNG streams and explicit
resource bookkeeping; bounded caches do not imply lower total process memory.
The 100-node workload remains below one second; the 800-node case does not.

```powershell
python -m unittest discover -s tests -q
python python/examples/benchmark_scalability.py --sizes 20,100,400,800 --output benchmarks/scalability_after.json
python python/examples/benchmark_scalability.py --sizes 100 --supply ondemand --output benchmarks/scalability_ondemand_after.json
```

## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.lrz.de/benedikt_baier_research/qcns.git
git branch -M main
git push -uf origin main
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.lrz.de/benedikt_baier_research/qcns/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Set auto-merge](https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing (SAST)](https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!). Thanks to [makeareadme.com](https://www.makeareadme.com/) for this template.

## Suggestions for a good README

Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.

## General experiment utilities

`python/components/utils.py` provides portable configuration persistence:

```python
from qcns.python.components.utils import (
    save_config, load_config, parameter_sweep, experiment_metadata,
)

config = {
    'topology': {'core': 'barabasi_albert', 'routers': 20, 'clients': 80, 'm': 2},
    'noise': {'coherence_time_seconds': 1.0},
    'metadata': experiment_metadata(seed=7),
}
save_config('experiments/baseline.json', config)
restored = load_config('experiments/baseline.json')

for parameters in parameter_sweep({'seed': [7, 8], 'fidelity': [0.9, 0.95]}):
    # Construct and run a fresh simulation using these parameters.
    print(parameters)
```

The versioned JSON format preserves nested mappings (including integer keys),
tuples, paths, NumPy numeric/string arrays with their shapes and dtypes, and
NetworkX graph types, labels, attributes and multigraph edge keys. NumPy scalars
load as Python scalars. Files are replaced atomically and parent folders are
created. Unsupported live objects, callables and non-finite numbers are rejected
on save. Describe hardware, noise and decoders using explicit values and construct
them through their public APIs after loading; this is not a simulation checkpoint
or an automatic object factory. The same mapping format can store small result
summaries; large traces belong in a separate data format.

The existing `save_config(path, graph, num_routers, num_clients, ...)` interface
remains available. `load_config` returns its original seven-item tuple for those
files, including `None` when there is no traffic matrix. Legacy unversioned network
files and plain JSON config dictionaries are readable without executing code.
Old files cannot reliably distinguish numeric-looking string node labels from
integer labels; save new files to preserve types exactly. Metadata records Python,
NumPy, NetworkX and platform versions, plus the supplied seed and optional source
revision. It does not seed generators or promise reproducibility across platforms.
Parameter sweeps yield combinations lazily; their values are shared, so treat them
as immutable and inject a separate seeded generator into each simulation.

`get_measure_dict(n, max_states=...)` initializes computational-basis counts with
an explicit allocation cap (default `2**20` entries); zero qubits has the single
empty bitstring outcome. Use sparse counts for large registers.

### Small circuit helpers

The QASM 2/3 loaders support flat circuits with indexed registers, `U`/`u3`,
`u1`/`p`, `rx`, `ry`, `rz`, `x`, `y`, `z`, `h`, `cx`, and measurements into
classical bits. Angles accept numbers, `pi`, parentheses and `+ - * /`.
Custom gates, control flow, broadcasting, reset and other unsupported statements
raise errors rather than silently disappearing. These helpers are not full QASM
implementations. QASM 2 comments are optional trailing metadata and never replace
rotation angles. Measurements retain their classical destination in each row.

- `apply_circuit(circuit, qubits, rng=...)` executes ideal gates offline and returns
  a dictionary of measurement outcomes, or `None` without measurements. Pass an
  explicit NumPy generator for repeatable measurements. Measured qubits remain in
  the state; separate states are combined only when required by a two-qubit gate.
- `await execute_circuit(node, circuit, qubits)` executes through the simulator's
  scheduler, using the node's gate timings and RNG. `dqc_apply_circuit` uses the
  synchronous node gate API and is intended for offline calculations.
- **Migration:** `apply=False` returns a unitary without applying it. The old
  compiled path was broken. Compilation requires one existing joint state and is
  capped at ten qubits by default (`max_matrix_qubits` on `apply_circuit`). A dense
  unitary alone needs `16 * 4**n` bytes, with additional multiplication temporaries
  and backend gate caches. Measurements cannot be compiled into a unitary.
- **Migration:** `update_circuit(circuit, parameters)` returns a copy and replaces
  every angle slot in gate order (three for `U`). It requires exactly the correct
  number of finite parameters, preserving operands and comments. It no longer
  overwrites the last field of each row or silently truncates inputs.

### Node-owned purification state

Import `PRRequest`, `PRSession`, `PRPair`, `PRResult`, `EncodedResult`,
`PRFailure`, and `PairExtractionFailure` from
`qcns.python.components.network.node`. These records describe local request/session
state and establishment outcomes; they are not transmitted header fields.
The former `network.pr_state` module has been removed, and these records are no
longer exported by `packet.protocol`. Update imports to `network.node`.

### Reusable circuits

Circuit construction, QASM loading, parameter updates and execution now live in
`qcns.python.components.qubit.circuit`. Import these functions from the circuit module; general `components.utils`
contains configuration and experiment utilities.

```python
from qcns.python.components.qubit.circuit import Circuit
from qcns.python.components.simulation.simulation import Simulation

circuit = Circuit([('H', 0), ('CNOT', 0, 1)])
qubits = dict(enumerate(Simulation.create_qsystem(2).qubits))
circuit.apply(qubits)  # Ideal offline Bell-state preparation.
```

Inside a node's asynchronous `run`, use `await circuit.execute(self, qubits)`
instead to apply gates through the event scheduler with that node's timing and
random generator. The caller must own/reserve the qubits until execution finishes.
`circuit.apply_on_node(node, qubits)` uses the synchronous offline node API.
`circuit.compile(qubits, max_matrix_qubits=10)` returns a dense unitary without
applying it, under the same allocation and joint-state restrictions as the
functional API. `circuit.with_parameters(values)` returns a new circuit.

`Circuit` copies gate rows to tuples and holds no qubit resources, so it can be
reused across independent runs. Measurements return a dictionary of the latest
outcome per classical destination; they retain the measured qubits. Without
measurements, execution returns `None`. Existing row-based functions
`apply_circuit`, `dqc_apply_circuit`, `execute_circuit`, and `update_circuit`
remain available in the circuit module. To use QASM, pass the rows returned by
`load_qasm_2_0_file` or `load_qasm_3_0_file` to `Circuit(rows)`.
