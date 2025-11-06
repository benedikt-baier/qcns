import sys
sys.path.append("../../../")

import numpy as np

import qcns

class Sender(qcns.Node):
    
    def __init__(self, _id, _sim):
        super(Sender, self).__init__(_id, _sim)
        
    async def run(self):
        
        self.attempt_bell_pairs(1, 1)
        
        data_qubit = qcns.QSystem(1).qubits
        
        angle_x, angle_z = np.random.uniform(0, 2 * np.pi, 2)
        
        self.apply_gate('Rx', data_qubit, angle_x)
        self.apply_gate('Rz', data_qubit, angle_z)
        
        com_qubit = self.l3_retrieve_qubit(1, 0)
        
        res = self.apply_gate('bsm', data_qubit, com_qubit)
        
        packet = qcns.Packet(0, 1, 0, 1, l3_needed=1, payload=[angle_x, angle_z])
        packet.l3_update_es(res)
        
        self.send_packet(packet)

class Receiver(qcns.Node):
    
    def __init__(self, _id, _sim):
        super(Receiver, self).__init__(_id, _sim)
        
    async def run(self):
        
        packet = await self.receive_packet()
        
        if packet.is_l1:
            packet = await self.receive_packet()
            
        angle_x, angle_z = packet.payload
        
        com_qubit = self.l3_retrieve_qubit(0, 1)
        
        if packet.l3_es_result[0][0]:
            self.apply_gate('X', com_qubit)
        
        if packet.l3_es_result[1][0]:
            self.apply_gate('Z', com_qubit)
        
        data_qubit = qcns.QSystem(1).qubits
        
        self.apply_gate('Rx', data_qubit, angle_x)
        self.apply_gate('Rz', data_qubit, angle_z)
        
        # print(data_qubit.fidelity(com_qubit))
        
def main():

    sim = qcns.Simulation(logging_path='./debug.log')
    
    sender = Sender(0, sim)
    receiver = Receiver(1, sim)

    sender.set_eqs_connection(receiver)
    
    sim.run()
    
if __name__ == "__main__":
    main()