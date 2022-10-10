import sys
import time
import argparse
import warnings
from utils import *
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial


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

    print("generating potential edge list...")
    generate_unused_edges(graph_gcc, module_path, args)

    print("loading potential edge list...")
    if not args.data:
        output_folder = module_path + '/output/testing'
    else:
        output_folder = module_path + '/output/' + args.data
    potential_edge_file = output_folder + '/potential_edges.pkl'
    with open(potential_edge_file, 'rb') as f:
        unused_edges = pickle.load(f)
    # print("unused_edges:", unused_edges)

    print("calculating eigen increase...")
    k = len(unused_edges)
    edge_sequence = []
    eigen_increase_sequence = []
    eigen_val_sequence = [eigen_val_1st]
    temp_graph = graph_gcc.copy()

    with Pool(processes=cpu_count()) as pool:
        for i in range(k):
            original_eigen = calculate_spectrum(temp_graph)

            # parallel computing, iterate over current unused edges
            new_spectrum = partial(calculate_new_spectrum, temp_graph)
            parallel_result = pool.map(new_spectrum, unused_edges)

            # select edge with max increase for update
            selected_edge = max(parallel_result)[1]
            new_eigen = max(parallel_result)[0]
            edge_sequence.append(selected_edge)
            eigen_val_sequence.append(new_eigen)

            max_increase = new_eigen - original_eigen
            eigen_increase_sequence.append(max_increase)
            if i % int(k/100 + 1) == 0:
                print(f"iteration {i}: max increase = {max_increase}, selected edge = {selected_edge}")

            # Delete from unused edge, update graph
            temp_graph.add_edge(*selected_edge)
            unused_edges.remove(selected_edge)

    print("saving results...")
    edge_sequence_path = output_folder + '/greedy_edge_sequence.pkl'
    eigen_increase_path = output_folder + '/greedy_eigen_increase_sequence.pkl'
    eigen_val_sequence_path = output_folder + '/greedy_eigen_val_sequence.pkl'

    with open(edge_sequence_path, 'wb') as f:
        pickle.dump(edge_sequence, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(eigen_val_sequence_path, 'wb') as f:
        pickle.dump(eigen_val_sequence, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(eigen_increase_path, 'wb') as f:
        pickle.dump(eigen_increase_sequence, f, protocol=pickle.HIGHEST_PROTOCOL)

    end_time = time.time()
    print('processing time:', end_time - start_time)

    # print(edge_sequence)
    # print(eigen_val_sequence)
    # print(eigen_increase_sequence)
