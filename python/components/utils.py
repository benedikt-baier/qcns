"""Safe experiment persistence and general simulator utilities."""
from __future__ import annotations

import ast
import itertools
import json
import os
from pathlib import Path
import platform
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import networkx as nx
import numpy as np

__all__ = ['get_measure_dict', 'save_config', 'load_config', 'parameter_sweep',
           'experiment_metadata']
PathLike = str | os.PathLike[str]
NetworkConfig = tuple[nx.Graph, int, int, dict | None, str | None, np.ndarray | None, str]

def _encode(value: Any) -> Any:
    """Tag mappings as well as special values to avoid reserved-key collisions."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'biufcUS':
            raise TypeError(f'Unsupported array dtype: {value.dtype}; use numeric or string arrays.')
        return {'type': 'array', 'dtype': value.dtype.str, 'shape': list(value.shape),
                'data': _encode(value.tolist())}
    if isinstance(value, np.generic):
        return _encode(value.item())
    if isinstance(value, complex):
        return {'type': 'complex', 'real': value.real, 'imag': value.imag}
    if isinstance(value, Path):
        return {'type': 'path', 'value': str(value)}
    if isinstance(value, nx.Graph):
        edges = list(value.edges(keys=True, data=True)) if value.is_multigraph() else list(value.edges(data=True))
        return {'type': 'graph', 'directed': value.is_directed(), 'multi': value.is_multigraph(),
                'attributes': _encode(value.graph), 'nodes': _encode(list(value.nodes(data=True))),
                'edges': _encode(edges)}
    if isinstance(value, Mapping):
        return {'type': 'mapping', 'items': [[_encode(k), _encode(v)] for k, v in value.items()]}
    if isinstance(value, tuple):
        return {'type': 'tuple', 'items': [_encode(v) for v in value]}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f'Cannot save {type(value).__name__}; describe it using configuration values.')

def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return [_decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    kind = value.get('type')
    if kind == 'mapping':
        return {_decode(k): _decode(v) for k, v in value['items']}
    if kind == 'tuple':
        return tuple(_decode(v) for v in value['items'])
    if kind == 'path':
        return Path(value['value'])
    if kind == 'complex':
        return complex(value['real'], value['imag'])
    if kind == 'array':
        dtype = np.dtype(value['dtype'])
        if dtype.kind not in 'biufcUS':
            raise ValueError(f'Unsupported array dtype: {dtype}')
        return np.asarray(_decode(value['data']), dtype=dtype).reshape(value['shape'])
    if kind == 'graph':
        classes = {(False, False): nx.Graph, (True, False): nx.DiGraph,
                   (False, True): nx.MultiGraph, (True, True): nx.MultiDiGraph}
        graph = classes[value['directed'], value['multi']]()
        graph.graph.update(_decode(value['attributes']))
        graph.add_nodes_from(_decode(value['nodes']))
        graph.add_edges_from(_decode(value['edges']))
        return graph
    raise ValueError(f'Unknown configuration value type: {kind!r}')

def _count(value: int, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f'{name} must be a non-negative integer.')

def save_config(path: PathLike, graph: nx.Graph | Mapping[str, Any],
                num_routers: int | None = None, num_clients: int = 0,
                routing_table: dict | None = None, traffic_type: str | None = None,
                traffic_matrix: np.ndarray | None = None, connection_type: str = 'l3') -> None:
    """Atomically save a config mapping or the legacy network-config arguments.

    Supports mappings, tuples, Paths, NumPy numeric/string arrays and NetworkX
    graphs. Creates parent folders. Unsupported objects and non-finite values fail
    before replacing an existing file. Live models/callables must be represented
    by explicit configuration values; loading never imports or executes code.
    """
    if isinstance(graph, nx.Graph):
        if num_routers is None:
            raise ValueError('num_routers is required when saving a graph.')
        _count(num_routers, 'num_routers')
        _count(num_clients, 'num_clients')
        payload = (graph, num_routers, num_clients, routing_table, traffic_type,
                   None if traffic_matrix is None else np.asarray(traffic_matrix), connection_type)
        kind = 'network'
    elif isinstance(graph, Mapping):
        if num_routers is not None or num_clients != 0 or routing_table is not None or traffic_type is not None or traffic_matrix is not None or connection_type != 'l3':
            raise ValueError('A config mapping cannot be combined with network-config arguments.')
        payload, kind = dict(graph), 'config'
    else:
        raise TypeError('graph must be a NetworkX graph or a configuration mapping.')
    text = json.dumps({'qcns_config_version': 1, 'kind': kind, 'data': _encode(payload)},
                      indent=2, allow_nan=False)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target.parent,
                                         delete=False, suffix='.tmp') as stream:
            temporary = stream.name
            stream.write(text + '\n')
        os.replace(temporary, target)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)

def _legacy_label(value: str | int) -> str | int:
    """Recover integer IDs from JSON keys while retaining named nodes."""
    if not isinstance(value, str):
        return value
    try:
        return int(value)
    except ValueError:
        return value


def _load_legacy(data: Mapping[str, Any]) -> NetworkConfig:
    """Read the original unversioned network format without evaluating code."""
    graph = nx.Graph()
    graph.add_nodes_from((_legacy_label(node), attributes)
                         for node, attributes in data['graph']['nodes'].items())
    for key, attributes in data['graph']['edges'].items():
        try:
            edge = ast.literal_eval(key)
        except (ValueError, SyntaxError) as error:
            raise ValueError(f'Invalid legacy edge key: {key!r}') from error
        if not isinstance(edge, tuple) or len(edge) != 2:
            raise ValueError(f'Legacy edge key must contain two endpoints: {key!r}')
        graph.add_edge(_legacy_label(edge[0]), _legacy_label(edge[1]), **attributes)
    routes = data['routing_table']
    if routes is not None:
        routes = {_legacy_label(source): {_legacy_label(destination): _legacy_label(hop)
                                         for destination, hop in destinations.items()}
                  for source, destinations in routes.items()}
    traffic = data['traffic_matrix']
    return (graph, data['num_routers'], data['num_clients'], routes, data['traffic_type'],
            None if traffic is None else np.asarray(traffic), data['connection_type'])


def load_config(path: PathLike) -> dict[str, Any] | NetworkConfig:
    """Load a mapping or seven-item network tuple without executing code.

    Accepts hand-written plain JSON mappings and old network files. Numeric-looking
    string labels in old files are ambiguous; new files preserve label types.
    """
    with Path(path).open(encoding='utf-8') as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError('Configuration must contain a JSON object.')
    if 'qcns_config_version' not in data:
        fields = {'graph', 'num_routers', 'num_clients', 'routing_table',
                  'traffic_type', 'traffic_matrix', 'connection_type'}
        return _load_legacy(data) if fields <= data.keys() else data
    if data['qcns_config_version'] != 1 or data.get('kind') not in ('config', 'network'):
        raise ValueError('Unsupported QCNS configuration version or kind.')
    result = _decode(data['data'])
    if data['kind'] == 'config' and not isinstance(result, dict):
        raise ValueError('Configuration payload must be a mapping.')
    if data['kind'] == 'network' and (not isinstance(result, tuple) or len(result) != 7 or not isinstance(result[0], nx.Graph)):
        raise ValueError('Network configuration must contain a graph and six settings.')
    return result

def parameter_sweep(parameters: Mapping[str, Sequence[Any]]) -> Iterator[dict[str, Any]]:
    """Yield combinations in insertion order without storing the Cartesian product.

    Values are shared with the input; treat them as immutable.
    """
    for values in itertools.product(*parameters.values()):
        yield dict(zip(parameters, values))

def experiment_metadata(*, seed: int | None = None, revision: str | None = None) -> dict[str, Any]:
    """Record environment and caller-supplied provenance; does not seed any RNG."""
    return {'seed': seed, 'revision': revision, 'python': platform.python_version(),
            'platform': platform.platform(), 'numpy': np.__version__, 'networkx': nx.__version__}

def get_measure_dict(num_qubits: int, *, max_states: int = 1 << 20) -> dict[str, int]:
    """Zero computational-basis counts; use sparse counts for large registers."""
    _count(num_qubits, 'num_qubits')
    _count(max_states, 'max_states')
    if max_states == 0 or int(num_qubits) >= int(max_states).bit_length():
        raise ValueError('Measurement dictionary exceeds max_states; use sparse counts or raise the limit.')
    return {format(i, f'0{num_qubits}b') if num_qubits else '': 0 for i in range(1 << int(num_qubits))}

