from qcns.python.components.network.node import Node
from qcns.python.components.simulation import Simulation
from qcns.python.components.network.qprogram import *

__all__ = ['Server']

class Server(Node):
    
    """
    Class for a Server in a Network
    
    Attrs:
        /
    """
    
    def __init__(self, node_id: int, sim: Simulation) -> None:
        
        """
        Instantiates a Server in a Network
        
        Args:
            node_id (int): ID of the Server
            sim (Simulation): simulation the Server is in
            
        Returns:
            /
        """
        
        super(Server, self).__init__(node_id, sim)
        
    async def run(self) -> None:
        
        """
        Run function of the Server
        
        Args:
            /
            
        Returns:
            /
        """
        
        pass