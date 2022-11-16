import os
import torch
import pickle
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import get_laplacian

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def calculate_spectrum_gpu(edge_index):
    lap_sym = get_laplacian(edge_index=edge_index, normalization='sym')
    laplacian = torch.sparse_coo_tensor(lap_sym[0], lap_sym[1], (6, 6)).to_dense()
    eigen_vals = torch.linalg.eigvalsh(laplacian)
    first_eigen_val = eigen_vals[1]
    return first_eigen_val.detach().cpu().numpy()


def calculate_new_spectrum_gpu(edge_index, edge):
    device = torch.device('cuda')
    selected_edge = list(edge)
    add_edge = torch.tensor([[selected_edge[0], selected_edge[1]],
                             [selected_edge[1], selected_edge[0]]]).to(device)
    edge_index_new = torch.cat([edge_index, add_edge], 1)
    new_eigen = calculate_spectrum_gpu(edge_index_new)
    return new_eigen


def greedy_method_gpu(unused_edges, eigen_val_1st, graph_gcc, output_folder, method, edge_pct):
    k = len(unused_edges)
    m = graph_gcc.number_of_edges()
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
        original_eigen = calculate_spectrum_gpu(edge_index)

        k = len(unused_edges)
        max_increase = -10
        new_eigen = 0
        selected_edge = None

        for j in range(k):
            edge = unused_edges[j]
            new_spectrum = calculate_new_spectrum_gpu(edge_index, edge)
            if (new_spectrum - original_eigen) > max_increase:
                max_increase = new_spectrum - original_eigen
                new_eigen = new_spectrum
                selected_edge = edge

        # select edge with max increase for update
        edge_sequence.append(selected_edge)
        eigen_val_sequence.append(new_eigen)
        eigen_increase_sequence.append(max_increase)

        if i % int(num_add_edges / 100 + 1) == 0:
            print(f"iteration {i}: max increase = {max_increase}, selected edge = {selected_edge}")

        # Delete from unused edge, update graph
        unused_edges.remove(selected_edge)
        temp_edge = torch.tensor([[selected_edge[0], selected_edge[1]],
                                  [selected_edge[1], selected_edge[0]]]).to(device)
        edge_index = torch.cat([edge_index, temp_edge], 1)

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