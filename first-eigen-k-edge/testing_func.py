import os
import sys
import numpy as np
import networkx as nx
from pathlib import Path
from scipy import sparse

module_dur = os.getcwd()
sys.path.append(module_dur)
path = Path(module_dur)
module_path = str(path.parent.absolute())
sys.path.append(module_path)


def generate_graph(n):
    folder = "../../dataset/testing/"
    edge_file = folder + "edgelist.txt"
    G = nx.read_edgelist(edge_file, create_using=nx.DiGraph(), nodetype=int)
    print(G)
    # This is redundant for graph construction
    # for i in range(n):
    #     G.add_node(i)
    nx.draw(G)
    return G


def normalize_laplacian(G):
    adj = nx.adjacency_matrix(G)
    print(adj.shape)
    ind = range(len(graph.nodes()))
    print(graph.nodes())

    # degs = D^(-1/2)
    degs = [1.0 / np.sqrt(graph.out_degree(node) + 1) for node in graph.nodes()]
    print('step 1:', degs)
    degs = sparse.csr_matrix(sparse.coo_matrix((degs, (ind, ind)), shape=adj.shape, dtype=np.float))
    print('step 2:', degs)

    # L = D^(-1/2)*(A+I)*D^(-1/2)??
    L = degs.dot(sparse.eye(adj.shape[0]) + adj)
    L = L.dot(degs)

    return L


graph = generate_graph(6)
L_norm = normalize_laplacian(graph)
print('#####')
print(L_norm)

graph = graph.to_undirected()
L_norm_nx = nx.normalized_laplacian_matrix(graph)
print('#####')
print(L_norm_nx)

L = nx.laplacian_matrix(graph)
print('#####')
print(L)

print('#####')
print(nx.laplacian_spectrum(graph))
print('#####')
print(nx.normalized_laplacian_spectrum(graph))