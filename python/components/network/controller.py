
from qcns.python.components.network.host import Host
from qcns.python.components.simulation.simulation import Simulation
from qcns.python.components.network.qprogram import *

__all__ = ['Controller']

class Controller(Host):
    
    """
    Class for a Network controller
    
    Attrs:
        /
    """
    
    def __init__(self, node_id: int, sim: Simulation) -> None:
        
        """
        Instantiates a Network Controller
        
        Attrs:
            node_id (int): ID of the controller
            sim (Simulation): simulation the controller is in
            
        Returns:
            /
        """
        
        super(Controller, self).__init__(node_id, sim)
        
    async def run(self) -> None:
        
        """
        Run function of the Controller
        
        Args:
            /
            
        Returns:
            /
        """
        
        pass
        
        