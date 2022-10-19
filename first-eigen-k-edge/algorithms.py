from functools import partial
from multiprocessing import Pool, cpu_count

from utils import *


def greedy_method(unused_edges, eigen_val_1st, graph_gcc, output_folder, method):
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
            if i % int(k / 100 + 1) == 0:
                print(f"iteration {i}: max increase = {max_increase}, selected edge = {selected_edge}")

            # Delete from unused edge, update graph
            temp_graph.add_edge(*selected_edge)
            unused_edges.remove(selected_edge)

    print("saving results...")
    edge_sequence_path = output_folder + f'/{method}_edge_sequence.pkl'
    eigen_increase_path = output_folder + f'/{method}_eigen_increase_sequence.pkl'
    eigen_val_sequence_path = output_folder + f'/{method}_eigen_val_sequence.pkl'

    with open(edge_sequence_path, 'wb') as f:
        pickle.dump(edge_sequence, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(eigen_val_sequence_path, 'wb') as f:
        pickle.dump(eigen_val_sequence, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(eigen_increase_path, 'wb') as f:
        pickle.dump(eigen_increase_sequence, f, protocol=pickle.HIGHEST_PROTOCOL)


def random_method(unused_edges, eigen_val_1st, graph_gcc, output_folder, method):
    k = len(unused_edges)
    unused_edges = list(unused_edges)

    iter_num = 500
    random_seq = [np.random.permutation(k) for i in range(iter_num)]

    with Pool(processes=cpu_count()) as pool:
        eigen_val_sequence = partial(random_method_iter, k, graph_gcc, unused_edges, eigen_val_1st)
        eigen_val_sequence_iter = pool.map(eigen_val_sequence, random_seq)

    print("saving results...")
    result_sequence = np.array(eigen_val_sequence_iter)
    quantile_min = np.quantile(result_sequence, 0, axis=0)
    quantile_1st = np.quantile(result_sequence, 0.25, axis=0)
    quantile_median = np.quantile(result_sequence, 0.5, axis=0)
    quantile_3st = np.quantile(result_sequence, 0.75, axis=0)
    quantile_max = np.quantile(result_sequence, 1, axis=0)
    result_mean = np.mean(result_sequence, axis=0)

    var_list = [result_sequence, result_mean, quantile_min, quantile_1st,
                quantile_median, quantile_3st, quantile_max]
    name_list = ['result_sequence', 'result_mean', 'quantile_min', 'quantile_1st',
                 'quantile_median', 'quantile_3st', 'quantile_max']

    for i in range(len(var_list)):
        export_path = output_folder + f'/{method}_eigen_val_{name_list[i]}_iter{iter_num}.pkl'
        with open(export_path, 'wb') as f:
            pickle.dump(var_list[i], f, protocol=pickle.HIGHEST_PROTOCOL)


def greedy_by_degree(unused_edges, eigen_val_1st, graph_gcc, output_folder, method):
    print(unused_edges, eigen_val_1st, graph_gcc, output_folder, method)
    return
