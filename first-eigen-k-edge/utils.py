import os
import pickle
from itertools import combinations

import networkx as nx
import numpy as np
from scipy import sparse


def generate_graph(args, module_path):
    folder = "/dataset/"
    if not args.data:
        edge_file = module_path + folder + "testing/edgelist.txt"
    else:
        edge_file = module_path + folder + args.data + ".attr/edgelist.txt"

    if not edge_file or not os.path.exists(edge_file):
        raise Exception("edge file not exist!")

    print("loading from " + edge_file)
    graph = nx.read_edgelist(edge_file, create_using=nx.DiGraph(), nodetype=int)

    return graph


def graph_preprocessing(graph):
    graph = graph.to_undirected()

    largest_cc = max(nx.connected_components(graph), key=len)
    subgraph_gcc = graph.subgraph(largest_cc)
    print('Preprocessing finished! Largest Connected Component Info:')

    n = subgraph_gcc.number_of_nodes()
    m = subgraph_gcc.size()
    print('Number of Nodes:', n)
    print('Number of Edges:', m)

    return subgraph_gcc


def normalize_laplacian(graph):
    # adjacency matrix
    adj = nx.adjacency_matrix(graph)
    ind = range(len(graph.nodes()))
    print(adj.shape)

    # normalization: deg_norm = D^(-1/2)
    deg_norm = [1.0 / np.sqrt(graph.degree(node)) for node in graph.nodes()]
    deg_norm = sparse.csr_matrix(sparse.coo_matrix((deg_norm, (ind, ind)), shape=adj.shape, dtype=float))

    # L = I - D^(-1/2) * A * D^(-1/2)
    laplacian_norm = sparse.eye(adj.shape[0]) - deg_norm * adj * deg_norm

    return laplacian_norm


def generate_unused_edges(graph, module_path, args):
    folder = "/output/"
    if not args.data:
        output_file = module_path + folder + "testing/potential_edges.pkl"
    else:
        output_file = module_path + folder + args.data + "/potential_edges.pkl"

    potential_edges = []
    node_list = list(graph.nodes())

    for u, v in combinations(node_list, 2):
        if graph.has_edge(u, v):
            pass
        else:
            potential_edges.append(tuple([u, v]))
    potential_edges = set(potential_edges)

    with open(output_file, 'wb') as f:
        pickle.dump(potential_edges, f, protocol=pickle.HIGHEST_PROTOCOL)


def calculate_spectrum(graph):
    eigen_vals = nx.normalized_laplacian_spectrum(graph)
    first_eigen_val = eigen_vals[1]

    return first_eigen_val


def calculate_new_spectrum(graph, edge: tuple):
    temp_graph = graph.copy()
    temp_graph.add_edge(*edge)
    new_eigen = calculate_spectrum(temp_graph)

    return new_eigen, edge


def random_method_iter(k, graph_gcc, unused_edges, eigen_val_1st, random_list):
    temp_graph = graph_gcc.copy()
    edge_sequence = [unused_edges[i] for i in random_list]
    eigen_val_sequence = [eigen_val_1st]

    for i in range(k):
        edge = edge_sequence[i]
        temp_graph.add_edge(*edge)
        new_eigen = calculate_spectrum(temp_graph)
        eigen_val_sequence.append(new_eigen)

    return eigen_val_sequence


############################################################################
# Unused functions


def load_feature(args, module_path):
    folder = "/dataset/"
    feature_file = module_path + folder + args.data + "/attrs.npz"

    if not feature_file or not os.path.exists(feature_file):
        raise Exception("feature file not exist!")

    print("loading from " + feature_file)
    features = sparse.load_npz(feature_file)

    print(features.shape)
    # n = features.shape[0]

    return features


def load_label(file_name, n):
    if not file_name or not os.path.exists(file_name):
        raise Exception("label file not exist!")
    Y = [set() for i in range(n)]
    is_multiple = False
    with open(file_name, 'r') as f:
        for line in f:
            s = line.strip().split()
            node = int(s[0])
            if node >= n:
                break
            if len(s) > 1:
                for label in s[1:]:
                    label = int(label)
                    Y[node].add(label)
    return Y


def load_edges(file_name):
    if not file_name or not os.path.exists(file_name):
        raise Exception("label file not exist!")

    edges = []
    max_id = 0
    with open(file_name, 'r') as fin:
        for line in fin:
            s, t = line.strip().split()
            s, t = int(s), int(t)
            edges.append((s, t))
            mst = max(s, t)
            if mst > max_id:
                max_id = mst
    return edges, max_id


def calculate_laplacian(graph):
    adj = nx.adjacency_matrix(graph)

    ind = range(len(graph.nodes()))
    deg = [graph.degree(node) for node in graph.nodes()]
    deg = sparse.csr_matrix(sparse.coo_matrix((deg, (ind, ind)), shape=adj.shape, dtype=float))

    laplacian = deg - adj

    return laplacian
