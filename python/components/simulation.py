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
        
        self._event_queue: List[Event] = []
        self._hosts: List[Host] = []
        self._sim_time: float = 0.
    
    def add_host(self, _host: Host) -> None:
        
        """
        Adds a host to the simulation
        
        Args:
            _host (Host): host to add to simulation
            
        Returns:
            /
        """
        
        self._hosts.append(_host)
    
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
    def create_qsystem(_num_qubits: int, fidelity: float=1., sparse: bool=False) -> QSystem:
        
        """
        Creates qsystem
        
        Args:
            _num_qubits (int): number of qubits in the system
            _fidelity (float): fidelity of qsystem
            _sparse (bool): sparsity of qsystem
            
        Returns:
            qsys (QSystem): created Qsystem
        """
        
        return QSystem(_num_qubits, fidelity, sparse)
    
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
    
    async def handle_event(self, _num_hosts: int) -> None:
        
        """
        Handles Events in the event queue
        
        Args:
            _num_hosts (int): number of non terminating hosts
            
        Returns:
            / 
        """
        
        tasks = [asc.create_task(host.run()) for host in self._hosts]
        
        num_hosts = len(tasks) - _num_hosts
        
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
    
    def run(self, num_hosts: int=0) -> None:
        
        """
        Runs the simulation by handling all Events in the event queue
        
        Args:
            num_hosts (int): number of hosts that are not terminating
            
        Returns:
            /
        """
        
        asc.run(self.handle_event(num_hosts))