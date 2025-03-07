import itertools as it
import numpy as np
import networkx as nx

from typing import List, Dict, Tuple, Union, Callable, Any

from qcns.python.components.network.host import Host
from qcns.python.components.simulation.simulation import Simulation

__all__ = []

CORE_GRAPH_FUNCTIONS = {'complete_graph': nx.complete_graph, # first index num nodes
                        'cycle_graph': nx.cycle_graph, # fi
                        'circulant_graph': nx.circulant_graph, # fi
                        'path_graph': nx.path_graph, # fi
                        'star_graph': nx.star_graph, # fi
                        'wheel_graph': nx.wheel_graph, # fi
                        'grid_2d_graph': nx.grid_2d_graph, # fi & si
                        'fast_gnp_random_graph': nx.fast_gnp_random_graph, # fi
                        'newman_watts_strogatz_graph': nx.newman_watts_strogatz_graph, # fi
                        'barabasi_albert_graph': nx.barabasi_albert_graph, # fi
                        'dual_barabasi_albert_graph': nx.dual_barabasi_albert_graph, # fi
                        'extended_barabasi_albert_graph': nx.extended_barabasi_albert_graph} # fi

def create_graph(graph_name: str, num_routers: int, *args: List[Any], num_clients: int=0, **kwargs: Dict[str, Any]) -> nx.Graph:
    
    """
    Creates a graph, either normally or in core periphery topology
    
    Args:
        graph_name (str): name of the core graph to create
        num_routers (int): number of routers in the graph, acts as num_nodes in core graph
        *args (list): list of arguments needed for the graph creation
        num_clients (int): number of clients in the periphery graph
        **kwargs (dict): key word arguments for graph creation
        
    Returns:
        core_graph (nx.Graph): created graph
    """
    
    core_graph = CORE_GRAPH_FUNCTIONS[graph_name](num_routers, *args, **kwargs)
    
    if not num_clients:
        return core_graph
    
    for client in range(num_routers, num_routers + num_clients):
        router = np.random.randint(num_routers)
        core_graph.add_edge(router, client)
        
    return core_graph

def add_attributes_to_node(graph: nx.Graph, node: int, attributes: Dict[str, Any]) -> None:
    
    """
    Adds attributes to the node in the graph
    
    Args:
        graph (nx.Graph): graph of the node
        node (int): ID of node
        attributes (dict): dictonary of attributes to add to node
        
    Returns:
        /
    """
    
    for k, v in attributes.items():
        graph.nodes[node][k] = v

def add_attributes_to_nodes(graph: nx.Graph, attributes: Dict[int, Dict[str, Any]]) -> None:
    
    """
    Adds attributes to nodes in a graph
    
    Args:
        graph (nx.Graph): graph of the nodes
        attributes (dict): dictonary with attributes
        
    Returns:
        /
    """
    
    if attributes is None:
        return
    
    for node, attribute in attributes.items():
        for k, v in attribute.items():
            graph.nodes[node][k] = v

def add_attributes_to_link(graph: nx.Graph, source: int, target: int, attributes: Dict[str, Any]) -> None:
    
    """
    Adds attributes to a link
    
    Args:
        graph (nx.Graph): graph of the edge
        source (int): source of the link
        target (int): target of the link
        attributes (dict): attributes to add to link
        
    Returns:
        /
    """
    
    for k, v in attributes.items():
        graph.edges[source, target][k] = v

def add_attributes_to_links(graph: nx.Graph, attributes: Dict[Tuple[int, int], Dict[str, Any]]) -> None:
    
    """
    Adds attributes to links
    
    Args:
        graph (nx.Graph): graph of the links
        attributes (dict): dictonary of attributes for links
        
    Returns:
        /
    """
    
    if attributes is None:
        return
    
    for (src, dst), attribute in attributes.items():
        for k, v in attribute.items():
            graph.edges[src, dst][k] = v

def create_single_traffic_matrix(src_nodes: List[int], dst_nodes: List[int], max_qubits: int) -> np.array:
    
    """
    Creates a traffic matrix, where each node has only a request to another node
    
    Args:
        src_nodes (list): nodes sending requests
        dst_nodes (list): nodes receiving requests
        max_qubits (int): maximum number of qubits each request can have
        
    Returns:
        traffic_matrix (np.array): matrix indicating which node sends how many qubits to which other node
    """
    
    # TODO see if this works, currently the assumption is that the source and target nodes are from 0 to n
    
    if max_qubits < 2:
        raise ValueError(f'Argument max_qubits must be greater than 1')
    
    index_conversion = {node: index for index, node in enumerate(src_nodes)}
    
    traffic_matrix = np.zeros((len(src_nodes), 2))
    for source in src_nodes:
        target = np.random.choice(dst_nodes)
        while source == target:
            target = np.random.choice(dst_nodes)
        traffic_matrix[index_conversion[source], 0] = target
        traffic_matrix[index_conversion[source], 1] = np.random.randint(1, max_qubits)
        
    return traffic_matrix

def create_multi_traffic_matrix(src_nodes: List[int], dst_nodes: List[int], max_qubits: Union[int, List[int]]) -> np.array:
    
    """
    Creates a traffic matrix where each source node can have a request to multiple targets
    
    Args:
        src_nodes (list): list of source nodes
        dst_nodes (list): list of target nodes
        max_qubits (int/list): maximum number of qubits each source can request
        
    Returns:
        traffic_matrix (np.array): matrix indicating which node sends how many qubits to which other node
    """
    
    if isinstance(max_qubits, int) and max_qubits < 2:
        raise ValueError(f'Argument max_qubits must be greater than 1')
    
    if isinstance(max_qubits, List) and not all([qubit >= 0 for qubit in max_qubits]):
        raise ValueError(f'Argument max_qubits must be greater than 0 for all values')
    
    if isinstance(max_qubits, int):
        max_qubits = np.zeros(len(src_nodes)) + max_qubits
    
    traffic_matrix = np.zeros((len(src_nodes), len(dst_nodes)))
    for source in src_nodes:
        traffic_matrix[source, :] = np.random.randint(0, max_qubits[source], size=len(dst_nodes))
        
    np.fill_diagonal(traffic_matrix, 0)      
            
    return traffic_matrix

def calculate_routing_table(graph: nx.Graph, routers: List[int], clients: List[int], weight: Union[str, Callable]=None) -> Dict[int, Dict[int, int]]:
    
    """
    Calculates the shortest path for every client pair for every router
    
    Args:
        graph (nx.Graph): graph to route on
        routers (list): list of routers
        clients (list): list of clients
        weight (str, func): weight to use in Dijkstra Algorithm
        
    Returns:
        routing_table (dict): routing table of each router, with the next hop for every client
    """
    
    routing_table = {router: {} for router in routers}
    
    for src, dst in it.combinations(clients, 2):
        path = nx.shortest_path(graph, src, dst, weight=weight)
        
        for path_idx in range(1, len(path) - 1):
            routing_table[path[path_idx]][dst] = path[path_idx + 1]
            routing_table[path[path_idx]][src] = path[path_idx - 1]
            
    return routing_table

def calculate_single_packet_count(traffic_matrix: np.array) -> np.array:
    
    """
    Calculates the packet count each client receives from a single traffic matrix
    
    Args:
        traffic_matrix (np.array): matrix indicating which node sends how many qubits to which other node
        
    Returns:
        packet_count (np.array): number of packets each client receives
    """
    
    packet_count = np.zeros(traffic_matrix.shape[0]) # TODO need to find better way as sources might not be targets
    for target, _ in traffic_matrix:
        packet_count[target] += 1
    return packet_count

def calculate_multi_packet_count(traffic_matrix: np.array) -> np.array:
    
    """
    Calculates the packet count each client receives from a multi traffic matrix
    
    Args:
        traffic_matrix (np.array): matrix indicating which node sends how many qubits to which other node
        
    Returns:
        packet_count (np.array): number of packets each client receives
    """
    
    return np.sum(traffic_matrix > 0, axis=0)

def get_access_routers(graph: nx.Graph, clients: List[Union[str, int]]) -> Dict[int, int]:
    
    """
    Calculates the access router for each client
    
    Args:
        graph (nx.Graph): graph of the clients
        clients (list): list of clients
        
    Returns:
        access_router (dict): dictonary indicating which client is connected to which router
    """
    
    return {client: list(graph.neighbors(client))[0] for client in clients}

def set_connection(graph: nx.Graph, nodes: List[Host]) -> None:
    
    """
    Sets a connection between all nodes based on the graph
    
    Args:
        graph (nx.Graph): graph the nodes are in
        nodes (list): list of nodes
        
    Returns:
        /
    """
    
    for (src, dst), data in graph.edges(data=True):
        nodes[src].set_connection(dst, **data)

def set_l3_connection(graph: nx.Graph, nodes: List[Host]) -> None:
    
    """
    Sets a L3 connection between all nodes based on the graph
    
    Args:
        graph (nx.Graph): graph the nodes are in
        nodes (list): list of nodes
        
    Returns:
        /
    """
    
    for (src, dst), data in graph.edges(data=True):
        nodes[src].set_l3_connection(dst, **data)

CONNECTION_DICT = {'l1': set_connection, 'l3': set_l3_connection}

def create_topology(graph_name: str, num_routers: int, router: Host, *args: List[Any], num_clients: int=0, client: Host=None, 
                    node_attributes: Dict[int, Dict[str, Any]]=None, link_attributes: Dict[Tuple[int, int], Dict[str, Any]]=None, 
                    routing_metric: Union[str, Callable]=None, connection_type: str='l3', traffic_type: str='single', max_qubits: Union[int, List[int]]=1, 
                    **kwargs: Dict[str, Any]) -> Tuple[Simulation, List[Host], List[Host]]:
    
    """
    Creates a topology for the simulation based on a graph and node and link attributes
    
    Args:
        graph_name (str): name of the core graph to create
        num_routers (int): number of routers in the graph, acts as num_nodes in core graph
        router (Host): router class to use
        *args (list): list of arguments needed for the graph creation
        num_clients (int): number of clients in the periphery graph
        client (Host): client class to use
        node_attributes (dict): dictonary of node attributes
        link_attributes (dict): dictonary of link attributes
        routing_metric (str/func): routing metric for the routing table
        connection_type (str): type of connection for each link
        traffic_type (str): type of traffic matrix
        max_qubits (int): maximum number of qubits each request can have
        **kwargs (dict): key word arguments for graph creation
        
    Returns:
        sim (Simulation): simulation the routers and clients are in
        routers (list): list of routers
        clients (list): list of clients 
    """
    
    graph = create_graph(graph_name, num_routers, args, num_clients, kwargs)
    add_attributes_to_nodes(graph, node_attributes)
    add_attributes_to_links(graph, link_attributes)
    access_routers = get_access_routers(graph, graph.nodes[num_routers:])
    routing_table = calculate_routing_table(graph, graph.nodes[:num_routers], graph.nodes[num_routers:], routing_metric)
    
    if traffic_type == 'single':
        traffic_matrix = create_single_traffic_matrix(graph.nodes[num_routers:], max_qubits + 1)
        packet_count = calculate_single_packet_count(traffic_matrix)
    else:
        traffic_matrix = create_multi_traffic_matrix(graph.nodes[num_routers:], max_qubits)
        packet_count = calculate_multi_packet_count(traffic_matrix)
    
    sim = Simulation()
    
    routers = []
    
    for router_idx in range(num_routers):
        routers.append(router(node_id=router_idx, sim=sim, 
                              routing_table=routing_table[router_idx],
                              **graph.nodes[router_idx]))
    
    clients = []
    
    if client is None:
        num_clients = 0
    
    for client_idx in range(num_clients):
        clients.append(client(node_id=client_idx + num_routers, sim=sim,
                              access_router=access_routers[client_idx], 
                              traffic_matrix=traffic_matrix[client_idx], packet_count=packet_count[client_idx],
                              **graph.nodes[client_idx + num_routers]))
    
    nodes = routers + clients
    
    CONNECTION_DICT[connection_type](graph, nodes)
            
    return sim, routers, clients