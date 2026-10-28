"""Router resources and event processing; operations belong to quantum protocols."""
from typing import Dict, List, Unpack

import networkx as nx

from qcns.python.components.network.node import Node, PRRequest, PRSession
from qcns.python.components.network.qprotocol import (
    L1_EGP, L2_FIP, L3_QFP, L3_QProtocol, QProtocolStack, L2_NEC, L3_PR, PRProtocolOptions,
    L3_Encoded, EncodedProtocolOptions,
)
from qcns.python.components.simulation import Simulation

__all__ = ['Router', 'PurificationRouter', 'EncodedRouter']


class Router(Node):
    """Delegate quantum requests to L3 using the simulation event loop.

    The installed operation owns admission, waiting reservations, swapping,
    and frame forwarding. A supplied stack replaces the mode-based defaults
    and is cloned per router. Routes are shared with its L3 base implementation.
    """

    def __init__(self, node_id: int, sim: Simulation,
                 routing_table: Dict[int, int] = None, graph: nx.Graph = None,
                 clients: List[int] = None,
                 l1_eg_mode: str = 'l3cp', l1_rap_mode: str = 'attempt',
                 l1_reattempt: bool = True, l2_fip_mode: str = 'necp',
                 l2_reattempt: bool = True, l3_qf_mode: str = 'frp',
                 l3_bsm_mode: str = 'bsm', *,
                 qprotocols: QProtocolStack = None) -> None:
        self._node_id = node_id
        self.routing_table = self._resolve_routes(routing_table, graph, clients)
        stack = qprotocols if qprotocols is not None else QProtocolStack(
            l1_qprotocol=L1_EGP(l1_eg_mode, l1_rap_mode, l1_reattempt),
            l2_qprotocol=L2_FIP(l2_fip_mode, l2_reattempt),
            l3_qprotocol=L3_QFP(self.routing_table, l3_qf_mode, l3_bsm_mode))
        stack.get_protocol(3)
        super().__init__(node_id, sim, stop=False, qprotocols=stack)
        if isinstance(self.l3_qprotocol, L3_QProtocol):
            self.l3_qprotocol.routing_table = self.routing_table

    def _resolve_routes(self, routing_table: Dict[int, int] | None, graph: nx.Graph | None,
                        clients: List[int] | None) -> Dict[int, int]:
        if routing_table is not None:
            return dict(routing_table)
        if graph is None or clients is None:
            raise ValueError('Supply routing_table or both graph and clients')
        return self.calculate_routing_table(graph, clients)

    def calculate_routing_table(self, graph: nx.Graph, clients: List[int]) -> Dict[int, int]:
        """Compute shortest-path next hops, reporting unreachable endpoints."""
        if self.id not in graph:
            raise ValueError(f'Router {self.id} is not in the routing graph')
        return {client: self._route_to(graph, client) for client in clients if client != self.id}

    def _route_to(self, graph: nx.Graph, destination: int) -> int:
        try:
            return nx.shortest_path(graph, self.id, destination)[1]
        except (nx.NodeNotFound, nx.NetworkXNoPath) as error:
            raise ValueError(f'No route from router {self.id} to endpoint {destination}') from error

    async def run(self) -> None:
        while True:
            packet = await self.receive_packet()
            if packet is not None:
                await self.handle_packet_scheduled(packet)


class PurificationRouter(Router):
    """Hierarchical PR service with node-owned request and purification sessions.

    Pair production is demand-driven and bounded. The default generates raw
    pairs through existing connections; generate=False consumes confirmed prepared pairs.
    Clients must run L3_PR as session endpoints (see PurificationClient).
    """

    def __init__(self, node_id: int, sim: Simulation, routing_table: dict[int, int] | None = None, *,
                 graph: nx.Graph | None = None, clients: list[int] | None = None,
                 l1_eg_mode: str = 'l3cp', **protocol_options: Unpack[PRProtocolOptions]) -> None:
        self._node_id = node_id
        # Sessions address intermediate routers as well as application clients.
        endpoints = list(graph.nodes) if graph is not None else clients
        routes = self._resolve_routes(routing_table, graph, endpoints)
        stack = QProtocolStack(
            l1_qprotocol=L1_EGP(l1_eg_mode, reattempt=False), l2_qprotocol=L2_NEC(),
            l3_qprotocol=L3_PR(routes, **protocol_options))
        super().__init__(node_id, sim, routing_table=routes, qprotocols=stack)

    @property
    def request_table(self) -> dict[tuple[int, int], PRRequest]:
        return self.pr_requests

    @property
    def session_table(self) -> dict[tuple[int, int, int], PRSession]:
        return self.pr_sessions

    async def run(self) -> None:
        await self.l3_qprotocol.serve()


class EncodedRouter(Router):
    """Graph-interface QEC router; code selection comes from request headers."""

    def __init__(self, node_id: int, sim: Simulation, routing_table: dict[int, int] | None = None, *,
                 graph: nx.Graph | None = None, clients: list[int] | None = None,
                 l1_eg_mode: str = 'l3cp', **options: Unpack[EncodedProtocolOptions]) -> None:
        self._node_id = node_id
        routes = self._resolve_routes(routing_table, graph, list(graph.nodes) if graph is not None else clients)
        stack = QProtocolStack(l1_qprotocol=L1_EGP(l1_eg_mode, reattempt=False),
                               l2_qprotocol=L2_NEC(), l3_qprotocol=L3_Encoded(routes, **options))
        super().__init__(node_id, sim, routing_table=routes, qprotocols=stack)

    @property
    def request_table(self) -> dict[tuple[int, int], PRRequest]:
        return self.pr_requests

    async def run(self) -> None:
        await self.l3_qprotocol.serve()
