import os
import pickle
import time

import numpy as np
import torch
from torch_geometric.utils import get_laplacian
from torch_geometric.utils.convert import from_networkx

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def calculate_spectrum_gpu(edge_index, n):
    lap_sym = get_laplacian(edge_index=edge_index, normalization='sym')
    laplacian = torch.sparse_coo_tensor(lap_sym[0], lap_sym[1], (n, n)).to_dense()
    eigen_vals = torch.linalg.eigvalsh(laplacian)
    first_eigen_val = eigen_vals[1]
    return first_eigen_val.detach().cpu().numpy()


def calculate_new_spectrum_gpu(edge_index, edge, n):
    device = torch.device('cuda')
    selected_edge = list(edge)
    add_edge = torch.tensor([[selected_edge[0], selected_edge[1]],
                             [selected_edge[1], selected_edge[0]]]).to(device)
    edge_index_new = torch.cat([edge_index, add_edge], 1)
    new_eigen = calculate_spectrum_gpu(edge_index_new, n)
    return new_eigen


def greedy_method_gpu(unused_edges, eigen_val_1st, graph_gcc, output_folder, method, edge_pct):
    k = len(unused_edges)
    m = graph_gcc.number_of_edges()
    n = graph_gcc.number_of_nodes()
    num_add_edges = min(k, int(edge_pct * m))
    unused_edges = list(unused_edges)
    edge_sequence = []
    eigen_increase_sequence = []
    eigen_val_sequence = [eigen_val_1st]

    # Transfer data object to GPU.
    device = torch.device('cuda')
    temp_graph = from_networkx(graph_gcc)
    edge_index = temp_graph.edge_index.to(device)

    for i in range(num_add_edges):
        temp_start = time.time()
        original_eigen = calculate_spectrum_gpu(edge_index, n)

        k = len(unused_edges)
        max_increase = -10
        new_eigen = 0
        selected_edge = None

        for j in range(k):
            edge = unused_edges[j]
            new_spectrum = calculate_new_spectrum_gpu(edge_index, edge, n)
            if (new_spectrum - original_eigen) > max_increase:
                max_increase = new_spectrum - original_eigen
                new_eigen = new_spectrum
                selected_edge = edge
            if j % 1000 == 0:
                print(f'{j + 1} edges processed in this round!, still {k - j - 1} edges to go, time = {time.time() - temp_start}')

        # select edge with max increase for update
        edge_sequence.append(selected_edge)
        eigen_val_sequence.append(new_eigen)
        eigen_increase_sequence.append(max_increase)

        # if i % int(num_add_edges / 100 + 1) == 0:
        print(f"iteration {i}: max increase = {max_increase}, selected edge = {selected_edge}")

        # Delete from unused edge, update graph
        unused_edges.remove(selected_edge)
        temp_edge = torch.tensor([[selected_edge[0], selected_edge[1]],
                                  [selected_edge[1], selected_edge[0]]]).to(device)
        edge_index = torch.cat([edge_index, temp_edge], 1)
        print(f"iteration {i}: time = {time.time() - temp_start}")

    print("saving results...")
    edge_sequence_path = output_folder + f'/{method}/edge_sequence_epct{int(edge_pct * 100)}.pkl'
    eigen_val_sequence_path = output_folder + f'/{method}/eigen_val_sequence_epct{int(edge_pct * 100)}.pkl'
    eigen_increase_path = output_folder + f'/{method}/eigen_increase_sequence_epct{int(edge_pct * 100)}.pkl'

    with open(edge_sequence_path, 'wb') as f:
        pickle.dump(edge_sequence, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(eigen_val_sequence_path, 'wb') as f:
        pickle.dump(eigen_val_sequence, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(eigen_increase_path, 'wb') as f:
        pickle.dump(eigen_increase_sequence, f, protocol=pickle.HIGHEST_PROTOCOL)


def random_method_iter_gpu(k, n, edge_index, unused_edges, eigen_val_1st, random_list):
    edge_sequence = [unused_edges[i] for i in random_list]
    eigen_val_sequence = [eigen_val_1st]
    device = torch.device('cuda')

    for i in range(k):
        edge = edge_sequence[i]
        new_eigen = calculate_new_spectrum_gpu(edge_index, edge, n)
        eigen_val_sequence.append(new_eigen)

        # Delete from unused edge, update graph
        selected_edge = edge
        temp_edge = torch.tensor([[selected_edge[0], selected_edge[1]],
                                  [selected_edge[1], selected_edge[0]]]).to(device)
        edge_index = torch.cat([edge_index, temp_edge], 1)

    return eigen_val_sequence


def random_method_gpu(unused_edges, eigen_val_1st, graph_gcc, output_folder, method, edge_pct):
    k = len(unused_edges)
    m = graph_gcc.number_of_edges()
    n = graph_gcc.number_of_nodes()
    num_add_edges = min(k, int(edge_pct * m))
    unused_edges = list(unused_edges)

    # Transfer data object to GPU.
    device = torch.device('cuda')
    temp_graph = from_networkx(graph_gcc)
    edge_index = temp_graph.edge_index.to(device)

    iter_num = 100
    eigen_val_sequence_iter = []

    temp_start = time.time()

    for i in range(iter_num):
        random_list = np.random.choice(k, num_add_edges, replace=False)
        eigen_val_sequence = random_method_iter_gpu(num_add_edges, n, edge_index,
                                                    unused_edges, eigen_val_1st, random_list)
        eigen_val_sequence_iter.append(eigen_val_sequence)
        print(f'iter{i}: time = {time.time()-temp_start}')

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
        export_path = output_folder + f'/{method}/eigen_val_epct{int(edge_pct * 100)}_iter{iter_num}_{name_list[i]}.pkl'
        with open(export_path, 'wb') as f:
            pickle.dump(var_list[i], f, protocol=pickle.HIGHEST_PROTOCOL)


def edge_degree_greedy_gpu(unused_edges, eigen_val_1st, graph_gcc, output_folder, method, rank, edge_pct):
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
    n = graph_gcc.number_of_nodes()
    m = graph_gcc.number_of_edges()
    num_add_edges = min(k, int(edge_pct * m))

    # Transfer data object to GPU.
    device = torch.device('cuda')
    temp_graph = from_networkx(graph_gcc)
    edge_index = temp_graph.edge_index.to(device)

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
        new_eigen = calculate_new_spectrum_gpu(edge_index, selected_edge, n)
        eigen_val_sequence.append(new_eigen)
        unused_edges.remove(selected_edge)

        # update graph
        temp_edge = torch.tensor([[selected_edge[0], selected_edge[1]],
                                  [selected_edge[1], selected_edge[0]]]).to(device)
        edge_index = torch.cat([edge_index, temp_edge], 1)

    print("saving results...")
    edge_sequence_path = output_folder + f'/{method}/edge_sequence_epct{int(edge_pct*100)}.pkl'
    eigen_val_sequence_path = output_folder + f'/{method}/eigen_val_sequence_epct{int(edge_pct*100)}.pkl'

    with open(edge_sequence_path, 'wb') as f:
        pickle.dump(edge_sequence, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(eigen_val_sequence_path, 'wb') as f:
        pickle.dump(eigen_val_sequence, f, protocol=pickle.HIGHEST_PROTOCOL)
