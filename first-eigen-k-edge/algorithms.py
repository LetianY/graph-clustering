from functools import partial
from multiprocessing import Pool, cpu_count

from utils import *


def greedy_method(unused_edges, eigen_val_1st, graph_gcc, output_folder, method, edge_pct):
    k = len(unused_edges)
    m = graph_gcc.number_of_edges()
    num_add_edges = min(k, int(edge_pct * m))
    edge_sequence = []
    eigen_increase_sequence = []
    eigen_val_sequence = [eigen_val_1st]
    temp_graph = graph_gcc.copy()
    cpu_num = min(cpu_count(), 72)

    with Pool(processes=cpu_num) as pool:
        for i in range(num_add_edges):
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
            if i % int(num_add_edges / 100 + 1) == 0:
                print(f"iteration {i}: max increase = {max_increase}, selected edge = {selected_edge}")

            # Delete from unused edge, update graph
            temp_graph.add_edge(*selected_edge)
            unused_edges.remove(selected_edge)

    print("saving results...")
    edge_sequence_path = output_folder + f'/{method}/edge_sequence_epct{int(edge_pct*100)}.pkl'
    eigen_val_sequence_path = output_folder + f'/{method}/eigen_val_sequence_epct{int(edge_pct*100)}.pkl'
    eigen_increase_path = output_folder + f'/{method}/eigen_increase_sequence_epct{int(edge_pct*100)}.pkl'

    with open(edge_sequence_path, 'wb') as f:
        pickle.dump(edge_sequence, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(eigen_val_sequence_path, 'wb') as f:
        pickle.dump(eigen_val_sequence, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(eigen_increase_path, 'wb') as f:
        pickle.dump(eigen_increase_sequence, f, protocol=pickle.HIGHEST_PROTOCOL)


def random_method(unused_edges, eigen_val_1st, graph_gcc, output_folder, method, edge_pct):
    k = len(unused_edges)
    unused_edges = list(unused_edges)

    m = graph_gcc.number_of_edges()
    num_add_edges = min(k, int(edge_pct * m))

    iter_num = 30
    cpu_num = min(cpu_count(), 72)
    random_seq = [np.random.choice(k, num_add_edges, replace=False) for i in range(iter_num)]

    with Pool(processes=cpu_num) as pool:
        eigen_val_sequence = partial(random_method_iter, num_add_edges, graph_gcc, unused_edges, eigen_val_1st)
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
        export_path = output_folder + f'/{method}/eigen_val_epct{int(edge_pct*100)}_iter{iter_num}_{name_list[i]}.pkl'
        with open(export_path, 'wb') as f:
            pickle.dump(var_list[i], f, protocol=pickle.HIGHEST_PROTOCOL)


def edge_degree_greedy(unused_edges, eigen_val_1st, graph_gcc, output_folder, method, rank, edge_pct):
    # Initialization
    edge_degree = {}
    edge_map = dict.fromkeys(list(graph_gcc.nodes()), [])

    for edge in unused_edges:
        if rank == 'sum':
            edge_degree[edge] = graph_gcc.degree(edge[0]) + graph_gcc.degree(edge[1])
        elif rank == 'mul':
            edge_degree[edge] = graph_gcc.degree(edge[0]) * graph_gcc.degree(edge[1])
        else:
            raise Exception("rank type not exist!")
        edge_map[edge[0]].append(edge)
        edge_map[edge[1]].append(edge)

    # greedy by degree sum
    k = len(unused_edges)
    m = graph_gcc.number_of_edges()
    num_add_edges = min(k, int(edge_pct * m))

    temp_graph = graph_gcc.copy()
    eigen_val_sequence = [eigen_val_1st]
    edge_sequence = []

    for i in range(num_add_edges):
        if method == 'edge_degree_min':
            selected_edge = min(edge_degree, key=edge_degree.get)
        elif method == 'edge_degree_max':
            selected_edge = max(edge_degree, key=edge_degree.get)
        else:
            raise Exception("method not exist!")
        edge_sequence.append(selected_edge)
        # update dictionary
        edge_degree.pop(selected_edge, None)
        edge_map[selected_edge[0]].remove(selected_edge)
        edge_map[selected_edge[1]].remove(selected_edge)
        for edge in edge_map[selected_edge[0]]:
            edge_degree[edge] += 1
        for edge in edge_map[selected_edge[1]]:
            edge_degree[edge] += 1
        # calculate eigenvalues
        temp_graph.add_edge(*selected_edge)
        new_eigen = calculate_spectrum(temp_graph)
        eigen_val_sequence.append(new_eigen)
        unused_edges.remove(selected_edge)

    print("saving results...")
    edge_sequence_path = output_folder + f'/{method}/edge_sequence_epct{int(edge_pct*100)}.pkl'
    eigen_val_sequence_path = output_folder + f'/{method}/eigen_val_sequence_epct{int(edge_pct*100)}.pkl'

    with open(edge_sequence_path, 'wb') as f:
        pickle.dump(edge_sequence, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(eigen_val_sequence_path, 'wb') as f:
        pickle.dump(eigen_val_sequence, f, protocol=pickle.HIGHEST_PROTOCOL)
