import asyncio as asc
from heapq import heappop, heappush
from typing import List

from python.components.event import Event

__all__ = ['Simulation']

class Host:
    pass

class Simulation:
    
    """
    Represents a Simulation consisting of host running different protocols in parallel
    
    Attr:
        _event_queue (list): event queue of simulation
        _hosts (list): list of hosts in the simulation
        _sim_time (float): current simulation time
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
    
    def add_host(self, host: Host) -> None:
        
        """
        Adds a host to the simulation
        
        Args:
            host (Host): host to add to simulation
            
        Returns:
            /
        """
        
        self._hosts.append(host)
    
    def schedule_event(self, _event: Event) -> None:
    
        """
        Schedules an Event
        
        Args:
            _event (Event): Event to schedule
            
        Returns:
            /
        """
    
        heappush(self._event_queue, _event)
    
    def stop_simulation(self):
        
        """
        Stops the simulation
        
        Args:
            /
            
        Returns:
            /
        """
        
        for host in self._hosts.values():
            host.stop = True
    
    async def handle_event(self, _num_hosts: int) -> None:
        
        """
        Handles Events in the event queue
        
        Args:
            _num_hosts (int): number of non terminating hosts
            
        Returns:
            / 
        """
        
        tasks = [asc.create_task(host.run()) for host in self._hosts.values()]
        
        num_hosts = len(tasks) - _num_hosts
        
        while 1:
            
            if not num_hosts:
                self.stop_simulation()
                return
            
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
        
        self._hosts = {host._node_id: host for host in self._hosts}
        
        asc.run(self.handle_event(num_hosts))