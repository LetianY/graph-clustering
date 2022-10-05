from utils import *
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process...')
    parser.add_argument('--data', type=str, help='graph dataset name')
    args = parser.parse_args()

    print("loading data...")
    graph, features = load_data(args)
    n = features.shape[0]

    A = nx.adjacency_matrix(graph)
    L = nx.laplacian_matrix(graph)
    L_norm = nx.laplacian_matrix(graph)
    eigen_vals = nx.laplacian_spectrum(graph)
    eigen_vals_norm = nx.normalized_laplacian_spectrum(graph)


    # adj = normalize_transition(graph)
