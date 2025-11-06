import sys
import numpy as np
import itertools as it
import networkx as nx
sys.path.append("../../../")
import qcns

class Client(qcns.Node):
    
    def __init__(self, node_id, sim, access_router, dst, recv, max_qubits):
        super().__init__(node_id, sim)
        
        self.access_router = access_router
        self.dst = dst
        self.recv = recv
        self.max_qubits = max_qubits
    
    def generate_entanglement(self):
        
        for neigbor, qubits in self.max_qubits.items():
            self.attempt_bell_pairs(neigbor, qubits['send'])
    
    def send_request(self):
        
        packet = qcns.Packet(l2_src=self.id, l2_dst=self.access_router, l3_src=self.id, l3_dst=self.dst, l3_requested=1)
        
        self.send_packet(packet)
    
    async def receive_request(self):
        
        counter = 0
        while counter < self.recv:
            
            packet = await self.receive_packet()
            
            if packet.is_l1:
                continue
            
            # qubit = self.l3_retrieve_qubit(packet.l2_src, 1)
            
            counter += 1
    
    async def run(self):
        
        self.generate_entanglement()
        
        self.send_request()
        
        await self.receive_request()

class Router(qcns.Node):
    
    def __init__(self, node_id, sim, routing_table, max_qubits):
        super().__init__(node_id, sim, False)
        
        self.routing_table = routing_table
        self.max_qubits = max_qubits
        
    def generate_entanglement(self):
        
        for neigbor, qubits in self.max_qubits.items():
            if not qubits['send']:
                continue
            self.attempt_bell_pairs(neigbor, qubits['send'])
    
    def entanglement_swapping(self, packet):
        
        packet.l2_dst = self.routing_table[packet.l3_dst]
        
        qubit_src = self.l3_retrieve_qubit(packet.l2_src, 1)
        qubit_dst = self.l3_retrieve_qubit(packet.l2_dst, 0)
        
        res = self.apply_gate('bsm', qubit_src, qubit_dst)
        
        packet.l3_update_es(res)
        packet.l2_src = self.id
        self.send_packet(packet)
    
    async def handle_packets(self):
        
        while True:
            
            packet = await self.receive_packet()
            
            if packet.is_l1:
                continue
            
            self.entanglement_swapping(packet)
    
    async def run(self):
        
        self.generate_entanglement()
        
        await self.handle_packets()
        
def create_random_graph(num_routers, num_clients, edge_prob):

    # graph_tmp = nx.fast_gnp_random_graph(num_routers, edge_prob, seed=np.random)
    graph_tmp = nx.barabasi_albert_graph(num_routers, edge_prob)
    
    while not nx.is_connected(graph_tmp):
        # graph_tmp = nx.fast_gnp_random_graph(num_routers, edge_prob, seed=np.random)
        graph_tmp = nx.barabasi_albert_graph(num_routers, edge_prob)
    
    graph_tmp.add_nodes_from(range(num_routers, num_clients + num_routers))

    # core 1 .. 5 km
    core_lengths = np.array([0.2, 0.5, 0.8, 1, 2, 3, 4, 5], dtype=np.double)
    edge_length_core = np.random.choice(core_lengths, (len(graph_tmp.edges)), replace=True)
    
    #edge 0.1 ... 0.5 km
    peripherie_lengths = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.double)
    edge_length_peripherie = np.random.choice(peripherie_lengths, num_clients, replace=True)

    graph = nx.Graph()
    graph.add_nodes_from(range(num_routers + num_clients))

    for edge, length in zip(graph_tmp.edges, edge_length_core):
        graph.add_edge(edge[0], edge[1], length=length)

    access_routers = []
    for i in range(num_routers, num_clients + num_routers):
        router = np.random.randint(num_routers)
        graph.add_edge(i, router, length=edge_length_peripherie[i - num_routers])
        access_routers.append(router)
    
    nodes_router = list(graph.nodes)[:num_routers]
    nodes_host = list(graph.nodes)[num_routers:]
    
    return graph, nodes_router, nodes_host, num_routers, access_routers

def generate_traffic_matrix(num_clients, router_id):
    
    traffic_matrix = np.zeros(num_clients, dtype=np.int32)
    recv = np.zeros(num_clients, dtype=np.int32)
    
    for i in range(num_clients):
        
        dst = np.random.randint(num_clients)
        while dst == i:
            dst = np.random.randint(num_clients)
        traffic_matrix[i] = dst + router_id
        recv[dst] += 1

    return traffic_matrix.tolist(), recv.tolist()

def calc_routing_table(graph, router_id, num_nodes):
    
    routing_table = {i: {} for i in range(router_id)}
    needed_qubits = {router: {neighbor: {'send': 0, 'receive': 0} for neighbor in graph.neighbors(router)} for router in range(router_id)}
    
    for src, dst in it.product(range(router_id, num_nodes), range(router_id, num_nodes)):
        if src == dst:
            continue
        path = nx.shortest_path(graph, source=src, target=dst, weight='length')
        for hop_idx in range(1, len(path) - 1):
            routing_table[path[hop_idx]][path[-1]] = path[hop_idx + 1]
            routing_table[path[hop_idx]][path[0]] = path[hop_idx - 1]
            needed_qubits[path[hop_idx]][path[hop_idx + 1]]['send'] += 1
            needed_qubits[path[hop_idx]][path[hop_idx - 1]]['receive'] += 1

    return routing_table, needed_qubits

def create_topology(sim, num_routers, num_clients, edge_prob, coherence) -> None:

    graph, routers, clients, _, access_routers = create_random_graph(num_routers, num_clients, edge_prob)
    traffic_matrix, recv = generate_traffic_matrix(num_clients, num_routers)

    traffic_matrix_n = {k + num_routers: v for k, v in enumerate(traffic_matrix)}
    routing_table, needed_qubits = calc_routing_table(graph, num_routers, num_clients + num_routers)
    
    routers = list(graph.nodes)[:num_routers]
    clients = list(graph.nodes)[num_routers:]
    
    router_l = []
    clients_l = []
    
    for idx, route in routing_table.items():
        router_l.append(Router(idx, sim, route, needed_qubits[idx]))
    for idx, access_router, dst, rec in zip(clients, access_routers, traffic_matrix_n.values(), recv):
        clients_l.append(Client(idx, sim, access_router, dst, rec, {access_router: {'send': 1, 'receive': rec}}))
    
    nodes_l = router_l + clients_l

    for i, j, weight in graph.edges(data=True):
        qchannel = qcns.QChannel_Model(length=weight['length'])
        pchannel = qcns.PChannel_Model(length=weight['length'])
        c_config = qcns.L3C_Model(qchannel=qchannel, pchannel=pchannel, fidelity=np.random.uniform(0.75, 1.))
        m_config = qcns.LQM_Model(depolarization_time=coherence, dephasing_time=coherence)
        nodes_l[i].set_eqs_connection(nodes_l[j], c_config, c_config, m_config, m_config)

    return router_l, clients_l

def main():
    
    num_routers = 20
    num_clients = 80
    edge_prob = 3
    coherence = 1e-1
    
    sim = qcns.Simulation(logging_path='./debug.log')
    
    routers, clients = create_topology(sim, num_routers, num_clients, edge_prob, coherence)
    
    import time
    
    start_time = time.perf_counter()
    
    sim.run()
    
    end_time = time.perf_counter() - start_time
    
    print(end_time)
    
if __name__ == '__main__':
    main()