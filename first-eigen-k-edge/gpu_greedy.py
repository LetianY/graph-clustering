import torch
from torch_geometric.utils import get_laplacian
from torch_geometric.utils.convert import from_networkx

def calculate_spectrum_gpu(edge_index):
    device = torch.device('cuda')
    lap_sym = get_laplacian(edge_index=edge_index, normalization='sym')
    lap_sym = lap_sym.to(device)



def greedy_method_gpu(unused_edges, eigen_val_1st, graph_gcc, output_folder, method, edge_pct):
    k = len(unused_edges)
    m = graph_gcc.number_of_edges()
    num_add_edges = min(k, int(edge_pct * m))
    edge_sequence = []
    eigen_increase_sequence = []
    eigen_val_sequence = [eigen_val_1st]

    # Transfer data object to GPU.
    device = torch.device('cuda')
    temp_graph = from_networkx(graph_gcc)
    temp_graph = temp_graph.to(device)
    edge_index = temp_graph.edge_index().to(device)

    for i in range(num_add_edges):
        original_eigen = calculate_spectrum(edge_index)

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