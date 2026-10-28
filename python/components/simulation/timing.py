"""Configurable processing overhead, in simulation seconds, for node events.

Synchronous events retain immediate local-clock accounting.
Async events yield through Node.wait before executing their body. This is
not a CPU queue or a resource lock; concurrent tasks may overlap their delays.
"""
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import wraps
from inspect import iscoroutinefunction
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Any, TypeVar, cast

from qcns.python.components.simulation.event import GateEvent, WaitEvent

F = TypeVar('F', bound=Callable[..., Any])

def _duration(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
        raise ValueError('Event duration must be a finite nonnegative number of seconds')
    return float(value)


@dataclass(frozen=True)
class EventTimings:
    """Immutable configuration; overrides are copied and checked at creation.

    Keys are decorated method names. Nested events charge independently.
    Zero is an explicit idealization, preserving existing simulator defaults.
    Replace node.event_timings to change subsequent calls, including on routers.
    """
    default_duration: float = 0.0
    durations: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = {}
        for name, duration in self.durations.items():
            if not isinstance(name, str) or not name:
                raise ValueError('Event timing keys must be nonempty method names')
            values[name] = _duration(duration)
        object.__setattr__(self, 'default_duration', _duration(self.default_duration))
        object.__setattr__(self, 'durations', MappingProxyType(values))

    def duration_for(self, name: str) -> float:
        return self.durations.get(name, self.default_duration)


def _advance_local_time(node: Any, duration: float) -> None:
    """Compatibility accounting for immediate synchronous operations only.

    This does not suspend execution; runtime waits must use await node.wait.
    Keep this internal so Node exposes only one waiting primitive.
    """
    node._time += _duration(duration)
    node._sim.schedule_event(WaitEvent(node._time, node.id))


def node_event(function: F) -> F:
    """Charge per-node overhead before a method runs, preserving its signature.

    Failed calls also incur their processing overhead. Async cancellation during
    the delay prevents the body from running. Zero overhead schedules no event.
    Hardware durations remain the responsibility of existing operation methods.
    """
    if iscoroutinefunction(function):
        @wraps(function)
        async def asynchronous(self: Any, *args: Any, **kwargs: Any) -> Any:
            duration = 0.0 if self._event_timing_suspended else self.event_timings.duration_for(function.__name__)
            if duration:
                await self.wait(duration)
            return await function(self, *args, **kwargs)
        return cast(F, asynchronous)

    @wraps(function)
    def synchronous(self: Any, *args: Any, **kwargs: Any) -> Any:
        duration = 0.0 if self._event_timing_suspended else self.event_timings.duration_for(function.__name__)
        if duration:
            _advance_local_time(self, duration)
        return function(self, *args, **kwargs)
    return cast(F, synchronous)

def untimed_events(function: F) -> F:
    """Suspend event overhead during synchronous lifecycle cleanup."""
    @wraps(function)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        previous = self._event_timing_suspended
        self._event_timing_suspended = True
        try:
            return function(self, *args, **kwargs)
        finally:
            self._event_timing_suspended = previous
    return cast(F, wrapped)


def gate_event(function: F) -> F:
    """Charge the gate's operation duration once, for sync and async APIs.

    Precedence: event_timings['gate:X'], event_timings['apply_gate'], then
    the node's existing gate_duration['X']. The processing default does not
    replace known hardware durations. Scheduled completions and apply=False
    queries do not incur a second gate duration.
    """
    def duration_for(node: Any, gate: str) -> float:
        overrides = node.event_timings.durations
        duration = overrides.get(f'gate:{gate}', overrides.get('apply_gate', node._gate_duration.get(gate, 5e-6)))
        return _duration(duration)

    if iscoroutinefunction(function):
        @wraps(function)
        async def asynchronous(self: Any, gate: str, *args: Any, **kwargs: Any) -> Any:
            await self.wait(duration_for(self, gate))
            return await function(self, gate, *args, **kwargs)
        return cast(F, asynchronous)

    @wraps(function)
    def synchronous(self: Any, gate: str, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get('apply', True) and not kwargs.get('_scheduled', False):
            self._time += duration_for(self, gate)
            self._sim.schedule_event(GateEvent(self._time, self.id))
        return function(self, gate, *args, **kwargs)
    return cast(F, synchronous)
