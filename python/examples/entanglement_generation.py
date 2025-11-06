import sys
sys.path.append("../../../")

import numpy as np

import qcns

class Sender(qcns.Node):
    
    def __init__(self, node_id, sim):
        super(Sender, self).__init__(node_id, sim)
        
    async def run(self):
        
        self.attempt_bell_pairs(1, 2) 
        
        packet = await self.receive_packet()
        
        print(packet)

class Receiver(qcns.Node):
    
    def __init__(self, node_id, sim):
        super(Receiver, self).__init__(node_id, sim)
        
    async def run(self):
        
        packet = await self.receive_packet()
        
        print(packet)

def main():
    
    sim = qcns.Simulation(logging_path='./debug.log')
    
    sender = Sender(0, sim)
    receiver = Receiver(1, sim)
    
    # config = qcns.SRC_Model()
    # config = qcns.TPSC_Model()
    # config = qcns.BSMC_Model()
    config = qcns.FSC_Model()
    
    sender.set_eqs_connection(receiver, sender_connection_config=config, receiver_connection_config=config)
    
    sim.run()
    
if __name__ == '__main__':
    main()