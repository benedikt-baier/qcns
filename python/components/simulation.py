import asyncio as asc
from heapq import heappop, heappush

from typing import List

from python.components.event import Event
from python.components.qubit import QSystem


__all__ = ['Simulation']

class Host:
    pass

class Simulation:
    
    """
    Represents a Simulation consisting of host running different protocols in parallel
    """
    
    def __init__(self) -> None:
        
        """
        Initializes a Simulation object
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._event_queue: List = []
        self._hosts: List = []
        self._sim_time: float = 0.
    
    def add_hosts(self, _hosts: List[Host]) -> None:
        
        """
        Adds hosts to the simulation
        
        Args:
            _hosts (list): List of Hosts
            
        Returns:
            /
        """
        
        self._hosts.extend(_hosts)
    
    def schedule_event(self, _event: Event) -> None:
    
        """
        Schedules an Event
        
        Args:
            _event (Event): Event to schedule
            
        Returns:
            /
        """
    
        heappush(self._event_queue, _event)
    
    @staticmethod
    def create_qsystem(_num_qubits: int, _fidelity: float=1., _sparse: bool=False) -> QSystem:
        
        """
        Creates qsystem
        
        Args:
            _num_qubits (int): number of qubits in the system
            _fidelity (float): fidelity of qsystem
            _sparse (bool): sparsity of qsystem
            
        Returns:
            qsys (QSystem): created Qsystem
        """
        
        return QSystem(_num_qubits, _fidelity, _sparse)
    
    @staticmethod
    def delete_qsystem(_qsys: QSystem) -> None:
        
        """
        Deletes a qsystem
        
        Args:
            qsys (QSystem): qsystem to delete
        
        Returns:
            /
        """
        
        del _qsys
    
    async def handle_event(self) -> None:
        
        """
        Handles Events in the event queue
        
        Args:
            /
            
        Returns:
            / 
        """
        
        tasks = {idx: asc.create_task(host.run()) for idx, host in self._hosts.items()}
        
        num_hosts = len(tasks)
        
        while num_hosts:
            
            await asc.sleep(0)
            
            if not self._event_queue:
                continue
            
            event = heappop(self._event_queue)
            
            if not event._id:
                num_hosts -= 1
                continue
            
            self._sim_time = event._end_time
            self._hosts[event._node_id]._resume.set()
    
    def run(self) -> None:
        
        """
        Runs the simulation by handling all Events in the event queue
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._hosts = {host._node_id: host for host in self._hosts}
        
        asc.run(self.handle_event())