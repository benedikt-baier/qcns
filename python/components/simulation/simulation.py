import os
import logging
import asyncio as asc
import hashlib
import numpy as np
from heapq import heappop, heappush
from typing import List, Dict, overload
from numbers import Real
from math import isfinite
from functools import partial
from types import MappingProxyType
from collections.abc import Mapping

from qcns.python.components.simulation.event import Event, CompletionEvent
from qcns.python.components.qubit.qubit import QSystem, DEFAULT_DENSE_BUDGET
from qcns.python.components.qubit.backend import PythonBackend, get_backend

__all__ = ['Simulation']

_USE_CONFIGURED = object()

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
    
    def __init__(self, end_time: float | None = None, logging_path: str='', *, seed: int | None = None,
                 backend: str | PythonBackend = 'python') -> None:
        
        """
        Initializes a Simulation object
        
        Args:
            end_time (float): end_time of the simulation
            logging_path (str): path to the logging file
            
        Returns:
            /
        """
        
        self._backend = get_backend(backend)
        # Keep the historical class-level factory while instance calls inherit
        # this simulation's backend. No global factory or selection is changed.
        self.create_qsystem = partial(type(self).create_qsystem, backend=self._backend)
        self._event_queue: List[Event] = []
        self._hosts: Dict[Host] = {}
        self._num_hosts: int = 0
        self._sim_end_time: float | None = None
        self.set_end_time(end_time)
        self._sim_time = 0.0
        self._stopped = False
        self._logging_path: str = logging_path
        self.seed = np.random.SeedSequence(seed).entropy
        self._rngs: dict[tuple, np.random.Generator] = {}
        
        if logging_path and not os.path.exists(os.path.dirname(self._logging_path)):
            os.makedirs(os.path.dirname(self._logging_path))
        if self._logging_path and os.path.exists(self._logging_path):
            with open(self._logging_path, 'w'):
                pass
            
        if self._logging_path:
            logging.basicConfig(filename=self._logging_path, level=logging.DEBUG)
    
    def rng(self, *component: str | int) -> np.random.Generator:
        """Stable component streams, independent of construction or task order."""
        if component not in self._rngs:
            self._rngs[component] = self._new_rng(component)
        return self._rngs[component]

    def _new_rng(self, component: tuple[str | int, ...]) -> np.random.Generator:
        digest = hashlib.sha256(repr(component).encode('utf8')).digest()
        key = tuple(int.from_bytes(digest[i:i+4], 'little') for i in range(0, 32, 4))
        return np.random.default_rng(np.random.SeedSequence(self.seed, spawn_key=key))

    @property
    def backend(self) -> PythonBackend:
        """Numerical backend inherited by states created in this simulation."""
        return self._backend

    @staticmethod
    def create_qsystem(num_qubits: int=1, fidelity: float=1., *, max_bytes: int = DEFAULT_DENSE_BUDGET,
                       backend: str | PythonBackend = 'python') -> QSystem:
    
        """
        Creates a QSystem with the given properties
        
        Args:
            num_qubits (int): number of qubits in the Qsystem
            fidelity (float): initial fidelity of the QSystem
            backend: numerical kernels; instance calls inherit Simulation.backend
            
        Returns:
            qsystem (QSystem): QSystem with the given properties
        """
        
        return QSystem(num_qubits, fidelity, max_bytes=max_bytes, backend=backend)
    
    def add_host(self, host: Host) -> None:
        
        """
        Adds a host to the simulation
        
        Args:
            host (Host): host to add to simulation
            
        Returns:
            /
        """
        
        self._hosts[host.id] = host
    
    @property
    def hosts(self) -> Mapping[int, Host]:
        """Read-only view of registered hosts for scenario builders and analysis."""
        return MappingProxyType(self._hosts)

    @property
    def time(self) -> float:
        """Time of the last dispatched event, or the reached finite deadline."""
        return self._sim_time

    def set_end_time(self, end_time: float | None) -> None:
        """Set an absolute simulation deadline in seconds; None is unbounded."""
        if end_time is not None and (isinstance(end_time, bool) or not isinstance(end_time, Real)
                                     or not isfinite(end_time) or end_time < 0):
            raise ValueError('end_time must be None or a finite nonnegative number of seconds')
        self._sim_end_time = None if end_time is None else float(end_time)

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
        for component, rng in self._rngs.items():
            rng.bit_generator.state = self._new_rng(component).bit_generator.state
        for host in self._hosts.values():
            host.reset()
        self._num_hosts = 0
        self._sim_time = 0.0
        self._stopped = False
        if self._logging_path:
            with open(self._logging_path, 'w'):
                pass
        
    def stop_simulation(self) -> None:
        
        """
        Stops the simulation
        
        Args:
            /
            
        Returns:
            /
        """
        
        logging.info('Stopping Simulation')
        
        self._num_hosts = 0
        self._stopped = True
    
    @overload
    def run(self) -> None: ...

    @overload
    def run(self, end_time: float | None) -> None: ...

    async def _run_events(self) -> None:
        """Single scheduler for finite and unbounded runs, with cancellable hosts."""
        pending = {host.id for host in self._hosts.values() if host.stop}
        active = set(self._hosts)
        has_completing_hosts = bool(pending)
        failures: list[BaseException] = []
        self._num_hosts = len(pending) if has_completing_hosts else len(active)

        def finish_host(identifier: int) -> None:
            active.discard(identifier)
            pending.discard(identifier)
            self._num_hosts = len(pending) if has_completing_hosts else len(active)

        async def run_host(identifier: int) -> None:
            host = self._hosts[identifier]
            try:
                await host.run()
            except asc.CancelledError:
                raise
            except Exception as error:
                failures.append(error)
                raise
            if not self._stopped:
                # Synchronous operations may have accrued local time ahead of
                # the scheduler. Honor that completion time, not Python return time.
                self.schedule_event(CompletionEvent(max(self._sim_time, host._time), identifier,
                                                        partial(finish_host, identifier)))

        tasks = [asc.create_task(run_host(identifier)) for identifier in self._hosts]
        try:
            while self._num_hosts and not self._stopped:
                await asc.sleep(0)
                if not self._event_queue:
                    # Let runnable child coroutines schedule their next event.
                    await asc.sleep(0)
                if failures:
                    raise failures[0]
                if self._stopped or not self._num_hosts:
                    break
                if not self._event_queue:
                    if self._sim_end_time is None:
                        raise RuntimeError('Simulation deadlock: nodes are waiting without scheduled events')
                    self._sim_time = self._sim_end_time
                    break
                event = heappop(self._event_queue)
                if not getattr(event, 'active', True):
                    continue
                if self._sim_end_time is not None and event._end_time > self._sim_end_time:
                    heappush(self._event_queue, event)
                    self._sim_time = self._sim_end_time
                    break
                if event._id == -1:
                    finish_host(event._node_id)
                    continue
                self._sim_time = max(self._sim_time, event._end_time)
                logging.info(event)
                host = self._hosts[event._node_id]
                host._time = max(host._time, event._end_time)
                if event._id == 5:
                    event.callback()
                if event._id in (1, 4):
                    host._resume.set()
            await asc.sleep(0)
            if failures:
                raise failures[0]
        finally:
            self.stop_simulation()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asc.gather(*tasks, return_exceptions=True)
        if failures:
            raise failures[0]

    def run(self, end_time: float | None | object = _USE_CONFIGURED) -> None:
        """Run until completing hosts finish or the absolute deadline is reached.

        Omit end_time to retain the configured deadline; explicit None clears
        it. Unbounded nonempty networks require at least one host.stop=True.
        Background-only networks require a finite deadline. Zero means time 0.
        """
        if end_time is not _USE_CONFIGURED:
            self.set_end_time(end_time)
        if self._hosts and self._sim_end_time is None and not any(host.stop for host in self._hosts.values()):
            raise ValueError('An unbounded simulation requires at least one completing host (stop=True); set a finite end_time')
        self._stopped = False
        asc.run(self._run_events())

