import math
import torch
import numpy as np
import networkx as nx


# an iterative method based on Rayleigh 
# quotient to estimate the largest eigenvalue
# and the associated eigenvector
def power_method(mat, Iter=50, output_format='tensor'):
    """
        mat: input matrix, the default format is tensor in PyTorch.
        Iter: #iterations to estimate the leading eigenvector
        output_format: set this to 'array' if the output should be in numpy array
    """
    if mat.sum() == 0:
        return torch.zeros(len(mat))
    n = len(mat)
    x = torch.rand(n) 
    x = x / torch.norm(x, 2)
    flag = 1e-7
    for i in range(Iter):
        x_new = mat @ x
        x_new = x_new / torch.norm(x_new, 2)
        Diff = torch.norm(x_new - x, 2)
        if Diff <= flag:
            break
        x = x_new
    if output_format == 'tensor':
        return x_new
    elif output_format == 'array':
        return x_new.numpy()
    else:
        raise ValueError("Unknown return type.")


## apply the power method to estimate the spectral norm of a given matrix
def estimate_sym_specNorm(mat, Iter=50, numExp=20):
    """
    mat: input matrix, the default format is tensor in PyTorch.
         Numpy array can also be handled
    Iter: #iterations to estimate the leading eigenvector
    numExp: the #experiments to run in order to estimate the spectral norm
    """
    if mat.sum() == 0:
        return 0
    if type(mat) != torch.Tensor:
        M = torch.tensor(mat, dtype=torch.float32)
    else:
        M = mat.clone()
    ## run the power method multiple times
    ## to make the estimation stable
    spec_norm = 0
    for i in range(numExp):
        v = power_method(M, Iter)
        u = power_method(-M, Iter)
        spec_norm += torch.max( torch.abs(v @ M @ v), torch.abs(u @ (-M) @ u) )
    spec_norm /= numExp
    return spec_norm


## get a sub-matrix indexed by (row_idx, col_idx)
def get_submatrix(mat, row_idx, col_idx):
    return torch.index_select(torch.index_select(mat, 0, row_idx), 1, col_idx)





########################################################################################


## select target subgraph
def select_target_subgraph(graph, graph_type, mapping=None):
    if graph_type == 'Email':
        with open('../data/email-Eu-core-department-labels-cc.txt', 'r') as fid:
            email_data = fid.readlines()
        comm_to_nodes = {}
        for item in email_data:
            nodeID, commID = [int(i) for i in item.rstrip().split()]
            if commID not in comm_to_nodes:
                comm_to_nodes[commID] = [mapping[nodeID]]
            else:
                comm_to_nodes[commID].append(mapping[nodeID])
        comm_size = sorted([(key, len(comm_to_nodes[key])) for key in comm_to_nodes.keys()], key=lambda x: x[1])
        ## the targeted subgraph is selected as the community whose size is 50-quantile among all community sizes
        selected_comm = comm_size[math.floor(len(comm_size) * 0.5)][0]
        comm = comm_to_nodes[selected_comm]
    elif graph_type == 'Airport':
        deg = list(dict(graph.degree()).items())
        deg = sorted(deg, key=lambda x: x[1])
        ## the targeted subgraph is selected as the node (and its neighbors) whose size is 50-quantile among all degrees
        selected_node = deg[math.floor(len(deg) * 0.9)][0] 
        comm = list(graph.neighbors(selected_node)) + [selected_node] 
    elif graph_type == 'Brain':
        ## the last 100 nodes consist of the targeted subgraph
        comm = list(range(len(graph)-100, len(graph)))
    elif graph_type == 'Amazon':
        comm = np.loadtxt('../data/amazon/targeted_set.txt')
        comm = [mapping[i] for i in comm]
    else:
        all_comms = list(greedy_modularity_communities(graph))
        all_comms = sorted(all_comms, key=lambda x: len(x))
        ## the targeted subgraph is selected as the community whose size is 50-quantile among all community sizes
        comm = list(all_comms[math.floor(len(all_comms) * 0.5)])
        assert(len(comm) != 0)
    return comm




## generate synthetic graphs
def gen_graph(graph_type, graph_id=1):
    """
    graph_id: used to index BTER networks
    """
    ## n: #(nodes) for synthetic networks
    n = 375
    if graph_type == 'BA':
        G = nx.barabasi_albert_graph(n, 5)
    elif graph_type == 'Small-World':
        G = nx.watts_strogatz_graph(n, 10, 0.2)
    elif graph_type == 'Email':
        G = nx.read_edgelist('../data/email-Eu-core-cc.txt', nodetype=int)
        G.remove_edges_from(nx.selfloop_edges(G))
    elif graph_type == 'Brain':
        G = nx.from_numpy_array(np.loadtxt('../data/Brain.txt'))
    elif graph_type == 'BTER':
        G = nx.read_edgelist('../data/BTER/BTER_{:02d}.txt'.format(graph_id), nodetype=int)
    elif graph_type == 'Airport':
        G = nx.read_edgelist('../data/US-airport.txt', nodetype=int, create_using=nx.DiGraph, data=(('weight', float),) )
        ## ensure the Airport network is symmetric and normalize the weights
        Adj = nx.adjacency_matrix(G)
        Adj += Adj.T
        Adj /= Adj.max()
        G = nx.from_numpy_matrix(Adj.todense())
        ## pick the largest connected component
        comps = nx.connected_components(G)
        comp_max_idx = max(comps, key=lambda x: len(x))
        G = G.subgraph(comp_max_idx)
    elif graph_type == 'Amazon':
        G = nx.read_edgelist('../data/amazon/amazon_subgraph.edgelist', nodetype=int)
    return G

    
