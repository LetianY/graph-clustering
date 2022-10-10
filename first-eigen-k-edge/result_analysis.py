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

parser = argparse.ArgumentParser(description='Process...')
parser.add_argument('--data', type=str, help='graph dataset name')
parser.add_argument('--method', type=str, help='algorithm used')
args = parser.parse_args()


if not args.data:
    output_folder = module_path + '/output/testing'
else:
    output_folder = module_path + '/output/' + args.data

potential_edge_file = output_folder + '/potential_edges.pkl'
spectrum_file = output_folder + '/greedy_eigen_val_sequence.pkl'

with open(potential_edge_file, 'rb') as f:
    unused_edges = pickle.load(f)

with open(spectrum_file, 'rb') as f:
    greedy_spectrum = pickle.load(f)

k = len(unused_edges)
k_value = range(k+1)


plt.figure(figsize=(10.5, 6.5))
plt.plot(k_value, greedy_spectrum, label="greedy", color='cornflowerblue')

plt.xlabel("# of added edges k")
plt.ylabel("value of the smallest eigenvalue")
plt.title(f"eigenvalue plot: {args.data} dataset, {args.method} method")
plt.legend()
plt.show()

plt.savefig(output_folder+'/result_analysis.png')
