import os
import logging
import asyncio as asc
from heapq import heappop, heappush
from typing import List, Awaitable

from qcns.python.components.simulation.event import Event
from qcns.python.components.qubit.qubit import QSystem

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
            end_time (float): end_time of the simulation
            logging_path (str): path to the logging file
            
        Returns:
            /
        """
        
        self._event_queue: List[Event] = []
        self._hosts: List[Host] = []
        self._num_hosts: int = 0
        self._sim_time: float = 0.
        self._sim_end_time: float = end_time
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
    
    @staticmethod
    def create_qsystem(num_qubits: int=1, fidelity: float=1., sparse: bool=0) -> QSystem:
    
        """
        Creates a QSystem with the given properties
        
        Args:
            num_qubits (int): number of qubits in the Qsystem
            fidelity (float): initial fidelity of the QSystem
            sparse (float): whether the QSystem should be sparsely represented
            
        Returns:
            qsystem (QSystem): QSystem with the given properties
        """
        
        return QSystem(num_qubits, fidelity, sparse)
    
    def set_end_time(self, end_time: float) -> None:
        
        """
        Sets the time after that the simulation ends
        
        Args:
            end_time (float): time after which simulation ends
            
        Returns:
            /
        """
        
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
    
    def reset(self) -> None:
        
        """
        Resets the simulation
        
        Args:
            /
            
        Returns:
            /
        """
        
        self._event_queue = []
        self._hosts = [host for host in self._hosts.values()]
        self._num_hosts = 0
        self._sim_time = 0.
        
        if self._logging_path:
            with open(self._logging_path, 'w'):
                pass
        
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
    
    async def _handle_events(self) -> None:
        
        pass
    
    async def _handle_event(self) -> None:
        
        """
        Handles Events in the event queue
        
        Args:
            /
            
        Returns:
            / 
        """
        
        tasks = [asc.create_task(host.run()) for host in self._hosts.values()]
        
        self._num_hosts = len(tasks)
        
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
    
    async def _handle_event_end_time(self) -> None:
        
        """
        Handles events with a given end time
        
        Args:
            /
            
        Returns:
            /
        """
        
        tasks = [asc.create_task(host.run()) for host in self._hosts.values()]
        
        self._num_hosts = len(tasks)
        
        while self._num_hosts > 0:
            
            await asc.sleep(0)
            
            if not self._event_queue:
                continue
            
            event = heappop(self._event_queue)
            
            logging.info(event)
            
            if not (event._id + 1):
                self._num_hosts -= 1
                continue
            
            if event._end_time > self._sim_end_time:
                continue
            
            self._sim_time = event._end_time
            self._hosts[event._node_id]._resume[event._id].set()
            
        self.stop_simulation()
    
    def run(self, end_time: float=None) -> None:
        
        """
        Runs the simulation by handling all Events in the event queue
        
        Args:
            end_time (float): end time of simulation
            
        Returns:
            /
        """
        
        self._hosts = {host._node_id: host for host in self._hosts}
        
        if end_time is not None:
            self.set_end_time(end_time)
        
        asc.run(self._handle_events())
        