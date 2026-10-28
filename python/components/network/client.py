from typing import Unpack

from qcns.python.components.network.node import Node, PRResult
from qcns.python.components.simulation import Simulation
from qcns.python.components.network.qprotocol import L1_EGP, L2_NEC, L3_PR, PRProtocolOptions, QProtocolStack
from qcns.python.components.network.qprotocol import L3_Encoded, EncodedProtocolOptions

__all__ = ['Client', 'PurificationClient', 'EncodedClient']

class Client(Node):
    
    """
    Class for a Client in a Network
    
    Attrs:
        /
    """
    
    def __init__(self, node_id: int, sim: Simulation) -> None:
        
        """
        Instantiates a Client in a Network
        
        Args:
            node_id (int): ID of the Client
            sim (Simulation): simulation the client is in
            
        Returns:
            /
        """
        
        super(Client, self).__init__(node_id, sim)
        
    async def run(self) -> None:
        
        """
        Run function of the Client
        
        Args:
            /
            
        Returns:
            /
        """
        
        pass
    
class PurificationClient(Node):
    """Finite initiating client, or a serving endpoint when destination is None."""

    def __init__(self, node_id: int, sim: Simulation, routing_table: dict[int, int], *,
                 destination: int | None = None, requested: int = 1, fidelity: float = .97,
                 l1_eg_mode: str = 'l3cp', **protocol_options: Unpack[PRProtocolOptions]) -> None:
        stack = QProtocolStack(
            l1_qprotocol=L1_EGP(l1_eg_mode, reattempt=False), l2_qprotocol=L2_NEC(),
            l3_qprotocol=L3_PR(routing_table, **protocol_options))
        super().__init__(node_id, sim, stop=destination is not None, qprotocols=stack)
        self.destination, self.requested, self.fidelity = destination, requested, fidelity
        self.result: PRResult | None = None

    async def run(self) -> None:
        if self.destination is None:
            await self.l3_qprotocol.serve()
            return
        self.result = await self.l3_qprotocol.establish(
            self.destination, requested=self.requested, fidelity=self.fidelity)

class EncodedClient(PurificationClient):
    """Finite encoded request initiator or serving graph-interface endpoint."""

    def __init__(self, node_id: int, sim: Simulation, routing_table: dict[int, int], *,
                 destination: int | None = None, requested: int = 1, fidelity: float = .97,
                 l1_eg_mode: str = 'l3cp', **options: Unpack[EncodedProtocolOptions]) -> None:
        stack = QProtocolStack(l1_qprotocol=L1_EGP(l1_eg_mode, reattempt=False),
                               l2_qprotocol=L2_NEC(), l3_qprotocol=L3_Encoded(routing_table, **options))
        Node.__init__(self, node_id, sim, stop=destination is not None, qprotocols=stack)
        self.destination, self.requested, self.fidelity = destination, requested, fidelity
        self.result: PRResult | None = None
