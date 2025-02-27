import os
import logging
import asyncio as asc
from heapq import heappop, heappush
from typing import List, Awaitable

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
    
    def __init__(self, end_time: float=None, logging_path: str='') -> None:
        
        """
        Initializes a Simulation object
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._event_queue: List[Event] = []
        self._hosts: List[Host] = []
        self._num_hosts: int = 0
        self._sim_time: float = 0.
        self._sim_end_time: float = end_time
        self._resume: asc.Event = asc.Event()
        self._logging_path: str = logging_path
        
        _handle_events = {0: self._handle_event, 1: self._handle_event_end_time}
        self._handle_events: Awaitable = _handle_events[self._sim_end_time is not None]
        
        if logging_path and not os.path.exists(os.path.dirname(self._logging_path)):
            os.makedirs(os.path.dirname(self._logging_path))
        if self._logging_path and os.path.exists(self._logging_path):
            with open(self._logging_path, 'w'):
                pass
        if self._logging_path:
            logging.basicConfig(filename=self._logging_path, level=logging.DEBUG)
    
    def add_host(self, host: Host) -> None:
        
        """
        Adds a host to the simulation
        
        Args:
            host (Host): host to add to simulation
            
        Returns:
            /
        """
        
        self._hosts.append(host)
    
    def set_end_time(self, end_time: float) -> None:
        
        self._sim_end_time = end_time
        
        self._handle_events = self._handle_event_end_time
    
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
        
        logging.info('Stopping Simulation')
        
        self._num_hosts = 0
        for host in self._hosts.values():
            host.stop = True
    
    async def _handle_events(self, _num_hosts: int) -> None:
        
        pass
    
    async def _handle_event(self, _num_hosts: int) -> None:
        
        """
        Handles Events in the event queue
        
        Args:
            _num_hosts (int): number of non terminating hosts
            
        Returns:
            / 
        """
        
        tasks = [asc.create_task(host.run()) for host in self._hosts.values()]
        
        self._num_hosts = len(tasks) - _num_hosts
        
        while self._num_hosts > 0:
            
            await asc.sleep(0)
            
            if not self._event_queue:
                continue
            
            event = heappop(self._event_queue)
            
            logging.info(event)
            
            if not (event._id + 1):
                self._num_hosts -= 1
                continue
            
            self._sim_time = event._end_time
            self._hosts[event._node_id]._resume[event._id].set() 
            
        self.stop_simulation()
    
    async def _handle_event_end_time(self, _num_hosts: int) -> None:
        
        tasks = [asc.create_task(host.run()) for host in self._hosts.values()]
        
        self._num_hosts = len(tasks) - _num_hosts
        
        while self._num_hosts > 0:
            
            await asc.sleep(0)
            
            if not self._event_queue:
                continue
            
            event = heappop(self._event_queue)
            
            if not (event._id + 1):
                self._num_hosts -= 1
                continue
            
            if event._end_time > self._sim_end_time:
                continue
            
            logging.info(event)
            
            self._sim_time = event._end_time
            self._hosts[event._node_id]._resume[event._id].set()
            
        self.stop_simulation()
    
    def run(self, num_hosts: int=0, end_time: float=None) -> None:
        
        """
        Runs the simulation by handling all Events in the event queue
        
        Args:
            num_hosts (int): number of hosts that are not terminating
            
        Returns:
            /
        """
        
        self._hosts = {host._node_id: host for host in self._hosts}
        
        if end_time is not None:
            self.set_end_time(end_time)
        
        asc.run(self._handle_events(num_hosts))
   