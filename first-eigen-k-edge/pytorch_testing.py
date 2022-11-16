import os
import sys
import pickle
from pathlib import Path

import networkx as nx
import torch
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import get_laplacian

from utils import graph_preprocessing

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

module_dur = os.getcwd()
sys.path.append(module_dur)
path = Path(module_dur)
module_path = str(path.parent.absolute())
sys.path.append(module_path)

folder = "/dataset/"
edge_file = module_path + folder + "testing/edgelist.txt"
graph = nx.read_edgelist(edge_file, create_using=nx.DiGraph(), nodetype=int)
graph_gcc = graph_preprocessing(graph)
pyg_graph = from_networkx(graph_gcc)
print(pyg_graph)
print(pyg_graph.is_undirected())
print(pyg_graph.edge_index)

device = torch.device('cuda')
edge_idx = pyg_graph.edge_index.to(device)
lap_sym = get_laplacian(edge_index=edge_idx, normalization='sym')
print(lap_sym)
s = torch.sparse_coo_tensor(lap_sym[0], lap_sym[1], (6, 6)).to_dense()
print(s)
print(torch.linalg.eigvalsh(s))

output_folder = module_path + '/output/testing'
potential_edge_file = output_folder + '/potential_edges.pkl'
with open(potential_edge_file, 'rb') as f:
    unused_edges = pickle.load(f)
unused_edges = list(unused_edges)
selected_edge = list(unused_edges[0])
add_edge = torch.tensor([[selected_edge[0], selected_edge[1]], [selected_edge[1], selected_edge[0]]]).to(device)

edge_idx = torch.cat([edge_idx, add_edge], 1)
print(edge_idx)
lap_sym = get_laplacian(edge_index=edge_idx, normalization='sym')
print(lap_sym)
s = torch.sparse_coo_tensor(lap_sym[0], lap_sym[1], (6, 6)).to_dense()
print(s)

result = torch.linalg.eigvalsh(s)[1]
print(result.detach().cpu().numpy())
