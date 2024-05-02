import numpy as np
from typing import List, Any

from python.components.event import ReceiveEvent
from python.components.simulation import Simulation
from python.components.channel import QChannel
from code.qcns.python.components.photon_source import Packet
from python.components.qubit import tensor_operator, remove_qubits

__all__ = ['SinglePhotonSource', 'AtomPhotonSource', 'TwoPhotonSource', 'BSM']

B_0 = np.array([[0.5, 0, 0, 0.5], [0, 0, 0, 0], [0, 0, 0, 0], [0.5, 0, 0, 0.5]], dtype=np.complex128)
M_0 = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)

SINGLE_PHOTON_SOURCE_MODELS = {'standard_model': (5e-6, 0.81, 1.0, 1.0)}
ATOM_PHOTON_SOURCE_MODELS = {'standard_model': (5e-6, 0.81, 1.0, 1.0)}
TWO_PHOTON_SOURCE_MODELS = {'standard_model': (5e-6, 0.81, 1.0, 1.0)}

SEND = 0
RECEIVE = 1

class Host:
    
    pass

class Qubit:
    pass

class SinglePhotonSource:
    
    """
    Represents a Single Photon Source
    """
    
    def __init__(self, _model: str) -> None:
    
        """
        Initializes a Single Photon Source
        
        Args:
            _model (str): single photon source model
            
        Returns:
            /
        """
    
        self._model: Tuple[float, float, float, float] = SINGLE_PHOTON_SOURCE_MODELS[_model]

    def create_qubit(self, _num_requested: int) -> List[Qubit]:
        
        """
        Creates a requested number of qubits
        
        Args:
            _num_requested (int): number of requested qubits
            
        Returns:
            _qubits (list): list of created qubits
        """
        
        if _num_requested == 1:
            qsys = Simulation.create_qsystem(1)
            qsys._state = np.array([[self._model[1], np.sqrt(self._model[1] * (1 - self._model[1]))], [np.sqrt(self._model[1] * (1 - self._model[1])), 1 - self._model[1]]])
            return qsys.qubits
        
        _qubits = []
        for i in range(_num_requested):
            qsys = Simulation.create_qsystem(1)
            qsys._state = np.array([[self._model[1], np.sqrt(self._model[1] * (1 - self._model[1]))], [np.sqrt(self._model[1] * (1 - self._model[1])), 1 - self._model[1]]])
            _qubits.append(qsys.qubits)
        return _qubits

class AtomPhotonSource:
    
    """
    Represents a Atom Photon Entanglement Source
    """
    
    def __init__(self, _sender: Host, _receiver: Host, _length: float, _model: Any, _sim: Simulation) -> None:
        
        """
        Initializes a Atom Photon Entanglement Source
        
        Args:
            _sender (Host): sender of entangled qubits
            _receiver (Host): receiver of entangled qubits
            _length (float): length of connection
            _model (str): model of photon source
            _sim (Simulation): global simulation object
            
        Returns:
            /
        """
        
        self._sender: Host = _sender
        self._receiver: Host = _receiver
        self._sending_time: float = _length * (5e-6)
        self._gate_time: float = self._receiver._gate_duration['CNOT'] + self._receiver._gate_duration['measure']
        self._model: Tuple[float, float, float, float] = ATOM_PHOTON_SOURCE_MODELS[_model[0]]
        self._sim: Simulation = _sim

    def create_bell_pairs(self, _num_requested: int, _num_needed: int) -> None:
        
        """
        Creates a number of needed bell pairs
        
        Args:
            _num_requested (int): number of requested qubits
            _num_needed (int): number of need qubits
            
        Returns:
            /
        """
        
        packet = Packet(self._sender._node_id, self._receiver._node_id, _num_requested, _num_needed)
        
        for i in range(_num_needed):
            
            _curr_time = self._sim._sim_time + self._model[0]
        
            qsys = Simulation.create_qsystem(3)
            qsys._state = np.kron((4 * self._model[1] - 1)/3 * B_0 + (1 - self._model[1])/3 * np.eye(4, dtype=np.complex128), M_0)
            
            q_1, q_2, q_3 = qsys.qubits
            
            self._sender._connections[self._receiver._node_id]['memory'][SEND].l0_store_qubit(q_1, _curr_time, -1)
            self._receiver._connections[self._sender._node_id]['memory'][RECEIVE].l0_store_qubit(q_3, _curr_time, -1)
            
            self._receiver._connections[self._sender._node_id]['eqs'][RECEIVE].put(q_2)
            
            _curr_time += self._sending_time
            
            q_2 = self._receiver._connections[self._sender._node_id]['eqs'][RECEIVE].get()
            
            if q_2 is None:
                continue
            
            q_3 = self._receiver._connections[self._sender._node_id]['memory'][RECEIVE].l0_retrieve_qubit(-1, _curr_time)
            
            q_3.state_transfer(q_2)
            
            _curr_time += self._gate_time
            
            self._receiver._connections[self._sender._node_id]['memory'][RECEIVE].l0_store_qubit(q_3, _curr_time, -1)
            packet.update_l1_success(i)
        
        self._sim.schedule_event(ReceiveEvent(self._sim._sim_time + self._sending_time + self._gate_time, self._receiver._node_id))
        self._receiver._connections[self._sender._node_id]['packet'][RECEIVE].put(packet)

class TwoPhotonSource:
    
    """
    Represents a Two Photon Source
    """
    
    def __init__(self, _sender: Host, _receiver: Host, _length: float, _model: Any, _sim: Simulation) -> None:
        
        """
        Initializes a Two Photon Source
        
        Args:
            _sender (str): _sender of photon source
            _receiver (str): _receiver of photon source
            _length (float): length of whole connection, PS is in the middle
            _model (str): model name for photon source
            _sim (Simulation): simulation object
            
        Returns:
            / 
        """
        
        self._sender: Host = _sender
        self._receiver: Host = _receiver
        self._sending_time: float = _length * 2.5e-6
        self._gate_time_sender: float = self._sender._gate_duration['CNOT'] + self._sender._gate_duration['measure']
        self._gate_time_receiver: float = self._receiver._gate_duration['CNOT'] + self._receiver._gate_duration['measure']
        self._model: Tuple[float, float, float, float] = TWO_PHOTON_SOURCE_MODELS[_model[0]]
        self._sim: Simulation = _sim
    
    def create_bell_pairs(self, _num_requested: int, _num_needed: int) -> None:
        
        """
        Creates a number of needed qubits
        
        Args:
            _num_requested (int): number of requested qubits
            _num_needed (int): number of needed qubits
            
        Returns:
            /
        """
        
        packet_s = Packet(self._receiver._node_id, self._sender._node_id, _num_requested, _num_needed)
        packet_r = Packet(self._sender._node_id, self._receiver._node_id, _num_requested, _num_needed)
        
        for i in range(_num_needed):
            
            _curr_time = self._sim._sim_time + self._model[0]
        
            qsys = Simulation.create_qsystem(4)
            
            qsys._state = tensor_operator(0, [M_0, (4 * self._model[1] - 1)/3 * B_0 + (1 - self._model[1])/3 * np.eye(4, dtype=np.complex128), M_0])
            
            q_1, q_2, q_3, q_4 = qsys.qubits
            
            self._sender._connections[self._receiver._node_id]['memory'][SEND].l0_store_qubit(q_1, _curr_time, -1)
            self._receiver._connections[self._sender._node_id]['memory'][RECEIVE].l0_store_qubit(q_4, _curr_time, -1)
            
            self._sender._connections[self._receiver._node_id]['eqs'][RECEIVE].put(q_2)
            self._receiver._connections[self._sender._node_id]['eqs'][RECEIVE].put(q_3)
            
            _curr_time += self._sending_time
            
            q_2 = self._sender._connections[self._receiver._node_id]['eqs'][RECEIVE].get()
            q_3 = self._receiver._connections[self._sender._node_id]['eqs'][RECEIVE].get()
            
            if q_2 is not None:
                q_1 = self._sender._connections[self._receiver._node_id]['memory'][SEND].l0_retrieve_qubit(-1, _curr_time)
                _new_time = _curr_time + self._gate_time_sender
                q_1.state_transfer(q_2)
                self._sender._connections[self._receiver._node_id]['memory'][SEND].l0_store_qubit(q_1, _new_time, -1)
                packet_s.update_l1_success(i)
                
            if q_3 is not None:
                q_4 = self._receiver._connections[self._sender._node_id]['memory'][RECEIVE].l0_retrieve_qubit(-1, _curr_time)
                _new_time = _curr_time + self._gate_time_receiver
                q_4.state_transfer(q_3)
                self._receiver._connections[self._sender._node_id]['memory'][RECEIVE].l0_store_qubit(q_4, _new_time, -1)
                packet_r.update_l1_success(i)
      
        self._sim.schedule_event(ReceiveEvent(self._sim._sim_time + self._sending_time + self._gate_time_sender, self._sender._node_id))
        self._sim.schedule_event(ReceiveEvent(self._sim._sim_time + self._sending_time + self._gate_time_receiver, self._receiver._node_id))
        self._sender._connections[self._receiver._node_id]['packet'][RECEIVE].put(packet_s)
        self._receiver._connections[self._sender._node_id]['packet'][RECEIVE].put(packet_r)
                
class BSM:
    
    """
    Represents a Bell State measurement connection
    """
    
    def __init__(self, _sender: Host, _receiver: Host, _length: float, _model: Any, _sim: Simulation) -> None:
        
        """
        Initializes a Bell State measurement connection
        
        Args:
            _sender (Host): sender of connection
            _receiver (Host): receiver of connection
            _length (float): length of connection
            _model (Any): model for Bell State measurement connection
            _sim (Simulation): global simulation object
            
        Returns:
            /
        """
        
        self._sender: Host = _sender
        self._receiver: Host = _receiver
        self._sending_time: float = _length * 5e-6
        self._gate_time: float = self._sender._gate_duration['bsm']
        self._model: Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float], float] = (ATOM_PHOTON_SOURCE_MODELS[_model[0]], ATOM_PHOTON_SOURCE_MODELS[_model[1]], _model[2])
        self._sim: Simulation = _sim
        
    def create_bell_pairs(self, _num_requested: int, _num_needed: int) -> None:
        
        """
        Creates a number of needed qubits
        
        Args:
            _num_requested (int): number of requested qubits
            _num_needed (int): number of needed qubits
            
        Returns:
            /
        """
        
        packet_s = Packet(self._receiver._node_id, self._sender._node_id, _num_requested, _num_needed)
        packet_r = Packet(self._sender._node_id, self._receiver._node_id, _num_requested, _num_needed)
        
        for i in range(_num_needed):
            
            _curr_time = self._sim._sim_time + max(self._model[0][0], self._model[1][0])
        
            qsys = Simulation.create_qsystem(4)
            
            qsys._state = tensor_operator(0, [(4 * self._model[0][1] - 1)/3 * B_0 + (1 - self._model[0][1])/3 * np.eye(4, dtype=np.complex128),
                                           (4 * self._model[1][1] - 1)/3 * B_0 + (1 - self._model[1][1])/3 * np.eye(4, dtype=np.complex128)])
            
            q_1, q_2, q_3, q_4 = qsys._qubits
            
            self._sender._connections[self._receiver._node_id]['memory'][SEND].l0_store_qubit(q_1, _curr_time, -1)
            self._receiver._connections[self._sender._node_id]['memory'][RECEIVE].l0_store_qubit(q_4, _curr_time, -1)
            
            self._sender._connections[self._receiver._node_id]['eqs'][RECEIVE].put(q_2)
            self._receiver._connections[self._sender._node_id]['eqs'][RECEIVE].put(q_3)

            q_2 = self._sender._connections[self._receiver._node_id]['eqs'][RECEIVE].get()
            q_3 = self._receiver._connections[self._sender._node_id]['eqs'][RECEIVE].get()
            
            if q_2 is None or q_3 is None:
                continue
            
            res = q_2.prob_bsm(q_3, self._model[2])
            
            remove_qubits([q_2, q_3])
            
            packet_s.update_l1_success(i)
            packet_r.update_l1_success(i)
            
        self._sim.schedule_event(ReceiveEvent(self._sim._sim_time + self._sending_time + self._gate_time, self._sender._node_id))
        self._sim.schedule_event(ReceiveEvent(self._sim._sim_time + self._sending_time + self._gate_time, self._receiver._node_id))
        self._sender._connections[self._receiver._node_id]['packet'][RECEIVE].put(packet_s)
        self._receiver._connections[self._sender._node_id]['packet'][RECEIVE].put(packet_r)
                 