import sys
sys.path.append("../../../")

import numpy as np

import qcns

class Router(qcns.Host):
    
    def __init__(self, _id, _sim):
        super(Router, self).__init__(_id, _sim, False)
        
        self.routing_table = {1: 1, 2: 2}
        
    async def run(self):
        
        # await self.attempt_bell_pairs(1, 1)
        await self.attempt_bell_pairs(2, 1)
        
        while 1:
        
            packet = await self.receive_packet()
            
            if packet.is_l1:
                continue
            
            packet.l2_dst = self.routing_table[packet.l3_dst]
            
            if not packet.l3_is_cf:
                qubit_src = self.l3_retrieve_qubit(packet.l2_src, 1)
                qubit_dst = self.l3_retrieve_qubit(packet.l2_dst, 0)
                
                res = self.apply_gate('bsm', qubit_src, qubit_dst, combine=True, remove=True)
                
                packet.l3_update_es(res)
                
            packet.l2_src = self.id
            
            await self.send_packet(packet)

class Sender(qcns.Host):
    
    def __init__(self, _id, _sim, qubits):
        super(Sender, self).__init__(_id, _sim)
        
        self.qubits = {q._index: q for q in qubits}
    
    async def send_request(self):
        
        data_qubit = self.qubits[1]
        
        com_qubit = self.l3_retrieve_qubit(0, 0)
        
        self.apply_gate('CNOT', data_qubit, com_qubit, combine=True)
        
        res = self.apply_gate('measure', com_qubit, remove=True)
        
        packet = qcns.Packet(self.id, 0, l3_src=self.id, l3_dst=2, l3_requested=1, payload=[2, 'CNOT'])
        
        packet.l3_update_es(res)
        
        await self.send_packet(packet)
        
        packet = await self.receive_packet()
        
        # while packet.is_l1:
        #     packet = await self.receive_packet()
        
        if packet.l3_es_result[1][0]:
            self.apply_gate('Z', data_qubit)

    async def run(self):
        
        await self.attempt_bell_pairs(0, 1)
        
        self.apply_gate('H', self.qubits[0])
        self.apply_gate('CNOT', self.qubits[0], self.qubits[1])
        
        await self.send_request()

class Receiver(qcns.Host):
    
    def __init__(self, _id, _sim, qubits):
        super(Receiver, self).__init__(_id, _sim)
    
        self.qubits = {q._index: q for q in qubits}
    
    async def receive_request(self):
        
        packet = await self.receive_packet()
        
        # while packet.is_l1:
        #     packet = await self.receive_packet()
        
        data_qubit = self.qubits[packet.payload[0]]
        com_qubit = self.l3_retrieve_qubit(0, 1)
        
        if packet.l3_es_result[0][0]:
            self.apply_gate('X', com_qubit)
        
        self.apply_gate(packet.payload[1], com_qubit, data_qubit)
        
        self.apply_gate('H', com_qubit)
        res = self.apply_gate('measure', com_qubit, remove=True)
        
        packet.l3_update_es(2 * res, 0)
        packet.l3_set_cf()
        packet.l3_switch_src_dst()
        packet.l2_switch_src_dst()
        
        await self.send_packet(packet)
    
    async def run(self):
        
        await self.receive_packet()
        
        await self.receive_request()
        
        self.apply_gate('CNOT', self.qubits[2], self.qubits[3])
        
def main():

    sim = qcns.Simulation(logging_path='./debug.log')
    
    router = Router(0, sim)
    
    qsys = qcns.QSystem(4)
    
    sender = Sender(1, sim, [qsys.qubit(0), qsys.qubit(1)])
    receiver = Receiver(2, sim, [qsys.qubit(2), qsys.qubit(3)])

    router.set_l3_connection(sender, 1)
    router.set_l3_connection(receiver, 1)
    
    sim.run()
    
if __name__ == "__main__":
    main()