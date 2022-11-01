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


def result_analysis(args, itr_name=''):
    if not args.data:
        output_folder = module_path + '/output/testing'
    else:
        output_folder = module_path + '/output/' + args.data

    if not args.edge_pct:
        raise Exception("please input edge percentage!!")
    else:
        edge_pct = args.edge_pct

    potential_edge_file = output_folder + '/potential_edges.pkl'
    spectrum_file_greedy = output_folder + f'/greedy/eigen_val_sequence_epct{int(edge_pct*100)}.pkl'
    spectrum_file_min_edge_degree = output_folder + f'/edge_degree_min/eigen_val_sequence_epct{int(edge_pct*100)}.pkl'
    spectrum_file_max_edge_degree = output_folder + f'/edge_degree_max/eigen_val_sequence_epct{int(edge_pct*100)}.pkl'

    with open(potential_edge_file, 'rb') as f:
        unused_edges = pickle.load(f)
    with open(spectrum_file_greedy, 'rb') as f:
        spectrum_greedy = pickle.load(f)
    with open(spectrum_file_max_edge_degree, 'rb') as f:
        spectrum_edge_degree_max = pickle.load(f)
    with open(spectrum_file_min_edge_degree, 'rb') as f:
        spectrum_edge_degree_min = pickle.load(f)

    plt.figure(figsize=(10.5, 6.5))

    if (args.method == 'random') or (not args.method):
        name_list = ['result_mean', 'quantile_min', 'quantile_1st',
                     'quantile_median', 'quantile_3st', 'quantile_max']
        color_list = ['slateblue', 'lightsteelblue', 'darkgrey',
                      'cornflowerblue', 'slategrey', 'black']

        for i in range(len(name_list)):
            import_path = output_folder + f'/random/eigen_val_epct{int(edge_pct*100)}_iter{iter_num}_{name_list[i]}.pkl'

            with open(import_path, 'rb') as f:
                spectrum_random = pickle.load(f)

            plt.plot(range(len(spectrum_random)+1), spectrum_random,
                     label=f'random_{name_list[i]}_{itr_name}',
                     color=color_list[i])

    if (args.method == 'greedy') or (not args.method):
        plt.plot(range(len(spectrum_greedy)+1), spectrum_greedy, label=f'greedy', color='lightcoral')

    if (args.method == 'edge_degree_min') or (not args.method):
        plt.plot(range(len(spectrum_edge_degree_min)+1), spectrum_edge_degree_min, label=f'greedy_edge_degree_min', color='lightgreen')

    if (args.method == 'edge_degree_max') or (not args.method):
        plt.plot(range(len(spectrum_edge_degree_max)+1), spectrum_edge_degree_max, label=f'greedy_edge_degree_max', color='orange')

    plt.xlabel("# of added edges k")
    plt.ylabel("value of the smallest eigenvalue")
    plt.legend()

    if not args.method:
        plt.title(f"eigenvalue plot: {args.data} dataset {int(edge_pct*100)}% edge")
        plt.savefig(output_folder + f'/result_analysis_epct{int(edge_pct*100)}_{itr_name}.png')
    else:
        plt.title(f"eigenvalue plot: {args.data} dataset {int(edge_pct*100)}% edge, {args.method} method")
        plt.savefig(output_folder + f'/{args.method}/result_analysis_epct{int(edge_pct*100)}_{itr_name}.png')

###############################################################################################################

parser = argparse.ArgumentParser(description='Process...')
parser.add_argument('--data', type=str, help='graph dataset name')
parser.add_argument('--method', type=str, help='algorithm used')
parser.add_argument('--edge_pct', type=str, help='percent of original edges to be added')
arguments = parser.parse_args()

iter_num = 500
if not arguments.method:
    result_analysis(args=arguments, itr_name=f'iter{iter_num}')
elif arguments.method == 'random':
    result_analysis(args=arguments, itr_name=f'iter{iter_num}')
else:
    result_analysis(args=arguments)
