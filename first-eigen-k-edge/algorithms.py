from utils import *
from functools import partial
from multiprocessing import Pool, cpu_count


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
    temp_graph = graph_gcc.copy()

    iter_num = 1
    random_list = np.random.permutation(k)
    edge_sequence = [unused_edges[i] for i in random_list]
    eigen_val_sequence = [eigen_val_1st]

    for i in range(k):
        edge = edge_sequence[i]
        temp_graph.add_edge(*edge)
        new_eigen = calculate_spectrum(temp_graph)
        eigen_val_sequence.append(new_eigen)

    print("saving results...")
    eigen_val_sequence_path = output_folder + f'/{method}_eigen_val_sequence_iter{iter_num}.pkl'

    with open(eigen_val_sequence_path, 'wb') as f:
        pickle.dump(eigen_val_sequence, f, protocol=pickle.HIGHEST_PROTOCOL)
