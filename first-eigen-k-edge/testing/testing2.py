import argparse
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt

module_dur = os.getcwd()
sys.path.append(module_dur)
path = Path(module_dur)
module_path = str(path.parent.absolute())
sys.path.append(module_path)

edge_pct = 1
iter_num = 100
output_folder = module_path + '/output/random'

potential_edge_file = output_folder + '/potential_edges.pkl'
spectrum_file_greedy = output_folder + f'/greedy/eigen_val_sequence_epct{int(edge_pct*100)}.pkl'
spectrum_file_min_edge_degree = output_folder + f'/edge_degree_min/eigen_val_sequence_epct{int(edge_pct*100)}.pkl'
spectrum_file_max_edge_degree = output_folder + f'/edge_degree_max/eigen_val_sequence_epct{int(edge_pct*100)}.pkl'


with open(spectrum_file_greedy, 'rb') as f:
    spectrum_greedy = pickle.load(f)
with open(spectrum_file_max_edge_degree, 'rb') as f:
    spectrum_edge_degree_max = pickle.load(f)
with open(spectrum_file_min_edge_degree, 'rb') as f:
    spectrum_edge_degree_min = pickle.load(f)

print(spectrum_edge_degree_min)

name_list = ['result_mean', 'quantile_min', 'quantile_1st',
             'quantile_median', 'quantile_3st', 'quantile_max']

for i in range(len(name_list)):
    import_path = output_folder + f'/random/eigen_val_epct{int(edge_pct * 100)}_iter{iter_num}_{name_list[i]}.pkl'
    with open(import_path, 'rb') as f:
        spectrum_random = pickle.load(f)
    print(name_list[i], len(spectrum_random))