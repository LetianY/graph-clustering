import os
import networkx as nx
from scipy import sparse


def load_data(args):
    folder = "../../dataset/"
    feature_file = folder + args.data + "/attrs.npz"
    edge_file = folder + args.data + ".attr/edgelist.txt"

    if not feature_file or not os.path.exists(feature_file):
        raise Exception("feature file not exist!")

    if not edge_file or not os.path.exists(edge_file):
        raise Exception("edge file not exist!")

    print("loading from " + feature_file)
    features = sparse.load_npz(feature_file)

    print(features.shape)
    # n = features.shape[0]

    print("loading from " + edge_file)
    graph = nx.read_edgelist(edge_file, create_using=nx.DiGraph(), nodetype=int)
    # for i in range(n):
    #     graph.add_node(i)

    return graph, features


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


def normalize_laplacian(graph):
    adj = nx.adjacency_matrix(graph)
    print(adj.shape)
    ind = range(len(graph.nodes()))
    degs = [1.0 / np.sqrt(graph.out_degree(node) + 1) for node in graph.nodes()]
    degs = sparse.csr_matrix(sparse.coo_matrix((degs, (ind, ind)), shape=adj.shape, dtype=np.float))
    L = degs.dot(sparse.eye(adj.shape[0]) + adj)
    L = L.dot(degs)

    return L
