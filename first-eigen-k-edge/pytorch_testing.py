import os
import sys
from pathlib import Path

import networkx as nx
from torch_geometric.utils.convert import from_networkx

from utils import graph_preprocessing

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
