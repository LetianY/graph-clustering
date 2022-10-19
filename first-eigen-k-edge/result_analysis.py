import sys
import os
import pickle
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


module_dur = os.getcwd()
sys.path.append(module_dur)
path = Path(module_dur)
module_path = str(path.parent.absolute())
sys.path.append(module_path)


def result_analysis(args, itr_name=None):
    if not args.data:
        output_folder = module_path + '/output/testing'
    else:
        output_folder = module_path + '/output/' + args.data

    if not args.method:
        potential_edge_file = output_folder + '/potential_edges.pkl'
        spectrum_file_greedy = output_folder + f'/greedy_eigen_val_sequence.pkl'
        spectrum_file_random = output_folder + f'/random_eigen_val_sequence_{itr_name}.pkl'

        with open(potential_edge_file, 'rb') as f:
            unused_edges = pickle.load(f)

        with open(spectrum_file_greedy, 'rb') as f:
            spectrum_greedy = pickle.load(f)

        with open(spectrum_file_random, 'rb') as f:
            spectrum_random = pickle.load(f)

        k = len(unused_edges)
        k_value = range(k + 1)

        plt.figure(figsize=(10.5, 6.5))
        plt.plot(k_value, spectrum_greedy, label=f'greedy', color='cornflowerblue')
        plt.plot(k_value, spectrum_random, label=f'random_{itr_name}', color='lightsteelblue')

        plt.xlabel("# of added edges k")
        plt.ylabel("value of the smallest eigenvalue")
        plt.title(f"eigenvalue plot: {args.data} dataset")
        plt.legend()

        plt.savefig(output_folder + f'/result_analysis.png')
        plt.show()

    else:
        potential_edge_file = output_folder + '/potential_edges.pkl'
        spectrum_file = output_folder + f'/{args.method}_eigen_val_sequence.pkl'

        if args.method == 'random':
            spectrum_file = output_folder + f'/{args.method}_eigen_val_sequence_{itr_name}.pkl'

        with open(potential_edge_file, 'rb') as f:
            unused_edges = pickle.load(f)

        with open(spectrum_file, 'rb') as f:
            result_spectrum = pickle.load(f)

        k = len(unused_edges)
        k_value = range(k + 1)

        plt.figure(figsize=(10.5, 6.5))
        plt.plot(k_value, result_spectrum, label=args.method, color='cornflowerblue')

        plt.xlabel("# of added edges k")
        plt.ylabel("value of the smallest eigenvalue")
        plt.title(f"eigenvalue plot: {args.data} dataset, {args.method} method")
        plt.legend()

        plt.savefig(output_folder + f'/{args.method}_result_analysis.png')
        plt.show()


parser = argparse.ArgumentParser(description='Process...')
parser.add_argument('--data', type=str, help='graph dataset name')
parser.add_argument('--method', type=str, help='algorithm used')
arguments = parser.parse_args()

iter_num = 1
if not arguments.method:
    result_analysis(args=arguments, itr_name=f'iter{iter_num}')
elif arguments.method == 'random':
    result_analysis(args=arguments, itr_name=f'iter{iter_num}')
else:
    result_analysis(args=arguments)
