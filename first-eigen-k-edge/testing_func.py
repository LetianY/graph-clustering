import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
from pathlib import Path


module_dur = os.getcwd()
sys.path.append(module_dur)
path = Path(module_dur)
module_path = str(path.parent.absolute())
sys.path.append(module_path)


def generate_graph():
    folder = "/dataset/testing/"
    edge_file = module_path + folder + "edgelist.txt"
    graph = nx.read_edgelist(edge_file, create_using=nx.DiGraph(), nodetype=int)
    graph = graph.to_undirected()
    return graph


def normalize_laplacian(graph):
    # adjacency Matrix
    adj = nx.adjacency_matrix(graph)
    print(adj.shape)
    ind = range(len(graph.nodes()))

    # degree matrix
    deg = [graph.degree(node) for node in graph.nodes()]
    deg = sparse.csr_matrix(sparse.coo_matrix((deg, (ind, ind)), shape=adj.shape, dtype=float))

    # deg_norm = D^(-1/2)
    deg_norm = [1.0 / np.sqrt(graph.degree(node)) for node in graph.nodes()]
    deg_norm = sparse.csr_matrix(sparse.coo_matrix((deg_norm, (ind, ind)), shape=adj.shape, dtype=float))

    # L = I - D^(-1/2) * A * D^(-1/2)
    laplacian_norm = sparse.eye(adj.shape[0]) - deg_norm * adj * deg_norm
    print(laplacian_norm)

    return laplacian_norm


undirected_graph = generate_graph()
im = nx.draw(undirected_graph, with_labels=True)
plt.show()


print('#####')
largest_cc = max(nx.connected_components(undirected_graph), key=len)
subgraph = undirected_graph.subgraph(largest_cc)
nx.draw(subgraph, with_labels=True)
plt.show()

undirected_graph = subgraph
L = nx.laplacian_matrix(undirected_graph)
L_norm_nx = nx.normalized_laplacian_matrix(undirected_graph)
L_norm = normalize_laplacian(undirected_graph)

print('#####')
print(L)
print('#####')
print(L_norm)
print('#####')
print(L_norm_nx)

print('#####')
print(nx.laplacian_spectrum(undirected_graph))
print('#####')
print(nx.normalized_laplacian_spectrum(undirected_graph))
