
from python.components.packet import Packet

__all__ = ['QProgram', 'L3_CRP', 'L3_FRP']

class Host:
    
    pass

class QProgram:
    
    def __init__(self) -> None:
        
        pass
    
    async def run(self):
        
        pass
    
class L3_CRP(QProgram):
    
    def __init__(self) -> None:
        super(L3_CRP, self).__init__()
        
    async def run(self, host: Host, packet: Packet):
        
        packet.l3_reset_es()
        
        for index in range(packet.l3_num_needed):
        
            qubit_src = host.l3_retrieve_qubit(packet.l2_src, 1)
            qubit_dst = host.l3_retrieve_qubit(packet.l2_dst, 0)
            
            res = await host.apply_gate('prob_bsm', qubit_src, qubit_dst, combine=True, remove=True)
            
            packet.l3_update_es(index, res)  
    
class L3_FRP(QProgram):
    
    def __init__(self) -> None:
        super(L3_FRP, self).__init__()
        
    async def run(self, host: Host, packet: Packet):
        
        for index in range(packet.l3_num_needed):
        
            qubit_src = host.l3_retrieve_qubit(packet.l2_src, 1)
            qubit_dst = host.l3_retrieve_qubit(packet.l2_dst, 0)
            
            res = await host.apply_gate('prob_bsm', qubit_src, qubit_dst, combine=True, remove=True)
            
            packet.l3_update_es(index, res)