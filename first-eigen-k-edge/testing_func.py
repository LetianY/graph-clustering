import os
import sys
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
from pathlib import Path
from itertools import combinations

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


def preprocessing(graph):
    largest_cc = max(nx.connected_components(graph), key=len)

    return graph.subgraph(largest_cc)


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


def cal_laplacian(graph):
    laplacian = nx.laplacian_matrix(graph)
    laplacian_norm_nx = nx.normalized_laplacian_matrix(graph)
    laplacian_norm = normalize_laplacian(graph)

    return laplacian, laplacian_norm_nx, laplacian_norm


def plot_graph(graph):
    im = nx.draw(graph, with_labels=True)
    plt.show()


def save_potential_edges(graph):
    potential_edges = []
    node_list = list(undirected_graph.nodes())
    n = len(node_list)

    for u, v in combinations(node_list, 2):
        if undirected_graph.has_edge(u, v):
            pass
        else:
            potential_edges.append(tuple([u, v]))
    potential_edges = set(potential_edges)

    folder = "/output/unused_edge/testing/"
    file = module_path + folder + "potential_edges.pkl"

    with open(file, 'wb') as f:
        pickle.dump(potential_edges, f, protocol=pickle.HIGHEST_PROTOCOL)


undirected_graph = preprocessing(generate_graph())
save_potential_edges(undirected_graph)
plot_graph(undirected_graph)

temp_graph = undirected_graph.copy()
temp_graph.add_edge(1, 5)
print(temp_graph)
plot_graph(temp_graph)

file_path = module_path + "/output/testing/potential_edges.pkl"
with open(file_path, 'rb') as handle:
    unused_edges = pickle.load(handle)
print(unused_edges)

# initialize dictionary for each iteration
parallel_dict = dict(zip(list(unused_edges), [-1 for i in range(len(unused_edges))]))
# select edge with max increase for update
selected_edge = max(parallel_dict, key=parallel_dict.get)
max_increase = max(parallel_dict.values())



