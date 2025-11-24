import numpy as np

__all__ = ['StopEvent', 'SendEvent', 'ReceiveEvent', 'GateEvent']

class Event:
    pass

class Event:
    
    """
    Reprents a generic event
    
    Attr:
        _id (int): ID to distinguish events
        _end_time (float): time when event finishes
        _node_id (int): ID of Host which scheduled event
    """
    
    event_counter: int = 0
    
    def __init__(self, _id: int, _end_time: float, _node_id: int) -> None:
        
        """
        Initializes a generic event
        
        Args:
            _id (int): ID of event
            _end_time (float): time when event finishes
            _node_id (int): ID of host that scheduled event
            
        Returns:
            /
        """
        
        self._id: int = _id
        self._end_time: float = _end_time
        self._node_id: int = _node_id
        self._event_counter = Event.event_counter
        Event.event_counter += 1
        
    def __lt__(self, other: Event) -> bool:
        
        """
        Custom implementation of comparison operator, compares the end time
        
        Args:
            other (Event): event to compare to
        
        Returns:
            less_than (bool): whether this._end_time is less than other._end_time
        """
        
        return self._end_time < other._end_time

class StopEvent(Event):
    
    """
    Represents a StopEvent
    
    Attr:
        _id (int): ID to distinguish events
        _end_time (float): time when event finishes
        _node_id (int): ID of Host which scheduled event
    """
    
    def __init__(self, _node_id: str) -> None:
        
        """
        Initializes a Stop Event
        
        Args:
            /
            
        Returns:
            /
        """
        
        super(StopEvent, self).__init__(-1, np.inf, _node_id)
    
    def __repr__(self) -> str:
        
        """
        Custom print function for Stop Event
        
        Args:
            /
            
        Returns:
            _event (str): printable representation of Event
        """
        
        return f'Stop Event Node: {self._node_id}'

class SendEvent(Event):
    
    """
    Represents a SendEvent
    
    Attr:
        _id (int): ID to distinguish events
        _end_time (float): time when event finishes
        _node_id (int): ID of Host which scheduled event
    """
    
    def __init__(self, _end_time: float, _node_id: int) -> None:
    
        """
        Initializes a Send Event
        
        Args:
            _end_time (float): time at which Event is scheduled
            _node_id (int): ID of Host which created Event
            
        Returns:
            /
        """
    
        super(SendEvent, self).__init__(0, _end_time, _node_id)
        
    def __repr__(self) -> str:
        
        """
        Custom print function for Send Event
        
        Args:
            /
            
        Returns:
            _event (str): printable representation of Event
        """
        
        return f'Send Event Node: {self._node_id} Time: {self._end_time}'

class ReceiveEvent(Event):
    
    """
    Represents a Receive Event
    
    Attr:
        _id (int): ID to distinguish events
        _end_time (float): time when event finishes
        _node_id (int): ID of Host which scheduled event
    """
    
    def __init__(self, _end_time: float, _node_id: int) -> None:
        
        """
        Initializes a Receive Event
        
        Args:
            _end_time (float): time at which Event is scheduled
            _node_id (int): ID of Host which created Event
            
        Returns:
            /
        """
    
        super(ReceiveEvent, self).__init__(1, _end_time, _node_id)
        
    def __repr__(self) -> str:
        
        """
        Custom print function for Receive Event
        
        Args:
            /
            
        Returns:
            _event (str): printable representation of Event
        """
        
        return f'Receive Event Node: {self._node_id} Time: {self._end_time}'
    
class GateEvent(Event):
    
    """
    Represents a Gate Event
    
    Attr:
        _id (int): ID to distinguish events
        _end_time (float): time when event finishes
        _node_id (int): ID of Host which scheduled event
    """
    
    def __init__(self, _end_time: float, _node_id: int) -> None:
        
        """
        Initializes a Gate Event
        
        Args:
            _end_time (float): time at which Event is scheduled
            _node_id (int): Host which schedules Event
            
        Returns:
            / 
        """
        
        super(GateEvent, self).__init__(2, _end_time, _node_id)
        
    def __repr__(self) -> str:
        
        """
        Custom print function for Gate Event
        
        Args:
            /
            
        Returns:
            _event (str): printable representation of Event
        """
        
        return f'Gate Event: Node: {self._node_id} Time: {self._end_time}'

class WaitEvent(Event):
    
    """
    Represents a Wait Event
    
    Attr:
        _id (int): ID to distinguish events
        _end_time (float): time when event finishes
        _node_id (int): ID of Host which scheduled event
    """
    
    def __init__(self, _end_time: float, _node_id: float) -> None:
        
        """
        Initializes a Wait Event
        
        Args:
            _end_time (float): time when event finishes
            _node_id (int): ID of Host  which scheduled event

        Returns:
            /
        """
        
        super(WaitEvent, self).__init__(3, _end_time, _node_id)
        
    def __repr__(self) -> str:
        
        """
        Custom print function for Gate Event
        
        Args:
            /
            
        Returns:
            _event (str): printable representation of Event
        """
        
        return f'Wait Event: Node: {self._node_id} Time: {self._end_time}'