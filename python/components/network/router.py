
import networkx as nx

from typing import List, Dict

from qcns.python.components.network import Host
from qcns.python.components.simulation import Simulation
from qcns.python.components.network.qprogram import *

__all__ = ['Router']

L0 = 0
L1 = 1
L2 = 2
L3 = 3
L4 = 4
L7 = 5

class Router(Host):
    
    """
    Class for a Router in a Network
    
    Attrs:
        routing_table (dict): routing table for the forwarding packets
    """
    
    def __init__(self, node_id: int, sim: Simulation, routing_table: Dict[int, int]=None, graph: nx.Graph=None, clients: List[int]=None, 
                 l1_eg_mode: str='l3cp', l1_rap_mode: str='attempt', l1_reattempt: bool=True,
                 l2_fip_mode: str='necp', l2_reattempt: bool=True,
                 l3_qf_mode: str='frp', l3_bsm_mode: str='bsm') -> None:
        
        """
        Instaniates a Router in a Network
        
        Args:
            node_id (int): ID of the Router
            sim (Simulation): simulation the router is in
            routing_table (dict): routing_table to forward packets
            graph (nx.Graph): graph for calculating routing table
            clients (list): list of client ids to calculate routing table for
            l1_eg_mode (str): method to generate entanglement on L1
            l1_rap_mode (str): mode to reallocate entanglement on L1
            l1_reattempt (str): whether to reattempt entanglement on L1
            l2_fip_mode (str): method to improve fidelity on L2
            l2_reattempt (str): whether to reattempt entanglement on L2
            l3_qf_mode (str): method to forward entanglement on L3
            l3_bsm_mode (str): mode for entanglement swapping on L3
            
        Returns:
            /
        """
        
        self.routing_table = routing_table
        if self.routing_table is None:
            assert graph is not None and clients is not None
            self.routing_table = self.calculate_routing_table(graph, clients)
    
        super(Router, self).__init__(node_id, sim, False, l1_qprogram=L1_EGP(l1_eg_mode, l1_rap_mode, l1_reattempt), l2_qprogram=L2_FIP(l2_fip_mode, l2_reattempt), l3_qprogram=L3_QFP(self.routing_table, l3_qf_mode, l3_bsm_mode))
    
    def calculate_routing_table(self, graph: nx.Graph, clients: List[int]) -> None:
        
        """
        Calculates the routing table for a router
        
        Args:
            graph (nx.Graph): graoh to calculate routing table for
            clients (list): list of clients for the routing table
            
        Returns:
            /
        """
        
        self.routing_table = {}
        
        for client in clients:
            self.routing_table[client] = nx.shortest_path(graph, self.id, client)[1]
      
    async def run(self) -> None:
        
        """
        Run function of the Router
        
        Args:
            /
            
        Returns:
            /
        """
        
        while 1:
            
            packet = await self.receive_packet()
            
            if packet is None:
                continue
            
            if packet.is_l1:
                self._qprograms[L1].classical_data_plane(packet)
            if packet.is_l2:
                self._qprograms[L2].classical_data_plane(packet)
            if packet.is_l3:
                self._qprograms[L3].classical_data_plane(packet)
                