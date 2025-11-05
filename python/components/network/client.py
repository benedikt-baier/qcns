from qcns.python.components.network.node import Node
from qcns.python.components.simulation import Simulation
from qcns.python.components.network.qprogram import *

__all__ = ['Client']

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