import sys
sys.path.append("../../../")

import numpy as np

import qcns

class Router(qcns.Host):
    
    def __init__(self, _id, _sim):
        super(Router, self).__init__(_id, _sim)
        
    async def run(self):
        
        await self.attempt_bell_pairs(2, 1)
        
        packet = await self.receive_packet()
        
        if packet.is_l1:
            packet = await self.receive_packet()
        
        packet.l2_dst = 2
        
        qubit_src = self.l3_retrieve_qubit(packet.l2_src, 1)
        qubit_dst = self.l3_retrieve_qubit(packet.l2_dst, 0)
        
        res = self.apply_gate('bsm', qubit_src, qubit_dst, combine=True, remove=True)
        
        packet.l3_update_es(res)
        packet.l2_src = 0
        
        await self.send_packet(packet)

class Sender(qcns.Host):
    
    def __init__(self, _id, _sim):
        super(Sender, self).__init__(_id, _sim)
        
    async def run(self):
        
        await self.attempt_bell_pairs(0, 1)
        
        packet = qcns.Packet(1, 0, l3_src=1, l3_dst=2, l3_needed=1)
        
        await self.send_packet(packet)

class Receiver(qcns.Host):
    
    def __init__(self, _id, _sim):
        super(Receiver, self).__init__(_id, _sim)
        
    async def run(self):
        
        packet = await self.receive_packet()
        
        if packet.is_l1:
            packet = await self.receive_packet()
        
        qubit = self.l3_retrieve_qubit(packet.l2_src, 1)
        
        if packet.l3_es_result[0][0]:
            self.apply_gate('X', qubit)
            
        if packet.l3_es_result[1][0]:
            self.apply_gate('Z', qubit)
        
def main():

    sim = qcns.Simulation(logging_path='./debug.log')
    
    router = Router(0, sim)
    sender = Sender(1, sim)
    receiver = Receiver(2, sim)

    router.set_eqs_connection(sender)
    router.set_eqs_connection(receiver)
    
    sim.run()
    
if __name__ == "__main__":
    main()