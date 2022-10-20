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

    potential_edge_file = output_folder + '/potential_edges.pkl'
    spectrum_file_greedy = output_folder + f'/greedy_eigen_val_sequence.pkl'
    spectrum_file_min_edge_degree = output_folder + f'/edge_degree_min_eigen_val_sequence.pkl'
    spectrum_file_max_edge_degree = output_folder + f'/edge_degree_max_eigen_val_sequence.pkl'

    with open(potential_edge_file, 'rb') as f:
        unused_edges = pickle.load(f)
    with open(spectrum_file_greedy, 'rb') as f:
        spectrum_greedy = pickle.load(f)
    with open(spectrum_file_max_edge_degree, 'rb') as f:
        spectrum_edge_degree_max = pickle.load(f)
    with open(spectrum_file_min_edge_degree, 'rb') as f:
        spectrum_edge_degree_min = pickle.load(f)

    k = len(unused_edges)
    k_value = range(k + 1)

    plt.figure(figsize=(10.5, 6.5))

    if (args.method == 'random') or (not args.method):
        name_list = ['result_mean', 'quantile_min', 'quantile_1st',
                     'quantile_median', 'quantile_3st', 'quantile_max']
        color_list = ['slateblue', 'lightsteelblue', 'darkgrey',
                      'cornflowerblue', 'slategrey', 'black']

        for i in range(len(name_list)):
            import_path = output_folder + f'/random_eigen_val_{name_list[i]}_iter{iter_num}.pkl'

            with open(import_path, 'rb') as f:
                spectrum_random = pickle.load(f)

            plt.plot(k_value, spectrum_random,
                     label=f'random_{name_list[i]}_{itr_name}',
                     color=color_list[i])

    if (args.method == 'greedy') or (not args.method):
        plt.plot(k_value, spectrum_greedy, label=f'greedy', color='lightcoral')

    if (args.method == 'edge_degree_min') or (not args.method):
        plt.plot(k_value, spectrum_edge_degree_min, label=f'greedy_edge_degree_min', color='lightgreen')

    if (args.method == 'edge_degree_max') or (not args.method):
        plt.plot(k_value, spectrum_edge_degree_max, label=f'greedy_edge_degree_max', color='orange')

    plt.xlabel("# of added edges k")
    plt.ylabel("value of the smallest eigenvalue")
    plt.legend()

    if not args.method:
        plt.title(f"eigenvalue plot: {args.data} dataset")
        plt.savefig(output_folder + f'/result_analysis_{itr_name}.png')
    else:
        plt.title(f"eigenvalue plot: {args.data} dataset, {args.method} method")
        plt.savefig(output_folder + f'/{args.method}_result_analysis_{itr_name}.png')


parser = argparse.ArgumentParser(description='Process...')
parser.add_argument('--data', type=str, help='graph dataset name')
parser.add_argument('--method', type=str, help='algorithm used')
arguments = parser.parse_args()

iter_num = 500
if not arguments.method:
    result_analysis(args=arguments, itr_name=f'iter{iter_num}')
elif arguments.method == 'random':
    result_analysis(args=arguments, itr_name=f'iter{iter_num}')
else:
    result_analysis(args=arguments)
