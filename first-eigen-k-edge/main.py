import argparse
import sys
import time
import warnings

from os.path import exists
from pathlib import Path
from algorithms import *
from utils import *

module_dur = os.getcwd()
sys.path.append(module_dur)
path = Path(module_dur)
module_path = str(path.parent.absolute())
sys.path.append(module_path)

"""
FutureWarning: normalized_laplacian_matrix will return 
a scipy.sparse array instead of a matrix in Networkx 3.0.
"""
warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process...')
    parser.add_argument('--data', type=str, help='graph dataset name')
    parser.add_argument('--method', type=str, help='algorithm used')
    args = parser.parse_args()

    start_time = time.time()

    print("loading data...")
    graph = generate_graph(args, module_path)
    graph_gcc = graph_preprocessing(graph)
    eigen_val_1st = calculate_spectrum(graph_gcc)
    print("smallest eigenvalue of gcc graphs' normalized laplacian:", eigen_val_1st)

    if not args.data:
        output_folder = module_path + '/output/testing'
    else:
        output_folder = module_path + '/output/' + args.data
    potential_edge_file = output_folder + '/potential_edges.pkl'

    if exists(potential_edge_file):
        pass
    else:
        print("generating potential edge list...")
        generate_unused_edges(graph_gcc, module_path, args)

    print("loading potential edge list...")
    with open(potential_edge_file, 'rb') as f:
        unused_edges = pickle.load(f)
    # print("unused_edges:", unused_edges)

    print("calculating eigen increase...")
    if not args.method:
        raise Exception("please input an algorithm for computation")
    elif args.method == 'greedy':
        greedy_method(unused_edges=unused_edges,
                      eigen_val_1st=eigen_val_1st,
                      graph_gcc=graph_gcc,
                      output_folder=output_folder,
                      method='greedy')
    elif args.method == 'random':
        random_method(unused_edges=unused_edges,
                      eigen_val_1st=eigen_val_1st,
                      graph_gcc=graph_gcc,
                      output_folder=output_folder,
                      method='random')
    elif args.method == 'edge_degree_min':
        edge_degree_greedy(unused_edges=unused_edges,
                           eigen_val_1st=eigen_val_1st,
                           graph_gcc=graph_gcc,
                           output_folder=output_folder,
                           method='edge_degree_min')
    elif args.method == 'edge_degree_max':
        edge_degree_greedy(unused_edges=unused_edges,
                           eigen_val_1st=eigen_val_1st,
                           graph_gcc=graph_gcc,
                           output_folder=output_folder,
                           method='edge_degree_max')
    else:
        raise Exception("input method not exist!")

    end_time = time.time()
    print('processing time:', end_time - start_time)
