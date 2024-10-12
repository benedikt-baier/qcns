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
        
        self._id: int = 0
        self._end_time: float = np.inf
        self._node_id: int = _node_id

    def __repr__(self) -> str:
        
        """
        Custom print function for Stop Event
        
        Args:
            /
            
        Returns:
            _event (str): printable representation of Event
        """
        
        return f'StopEvent Node: {self._node_id}'

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
    
        self._id: int = 1
        self._end_time: float = _end_time
        self._node_id: int = _node_id
        
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
    
        self._id: int = 2
        self._end_time: float = _end_time
        self._node_id: int = _node_id
        
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
        
        self._id: int = 3
        self._end_time = _end_time
        self._node_id = _node_id
        
    def __repr__(self) -> str:
        
        """
        Custom print function for Gate Event
        
        Args:
            /
            
        Returns:
            _event (str): printable representation of Event
        """
        
        return f'Gate Event: Node: {self._node_id} Time: {self._end_time}'
