import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt

from algorithms import *
from utils import *

warnings.simplefilter(action='ignore', category=FutureWarning)

module_dur = os.getcwd()
sys.path.append(module_dur)
path = Path(module_dur)
module_path = str(path.parent.absolute())
sys.path.append(module_path)


def plot_graph(graph):
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_title('Cora with Normalized Algebraic Connectivity = 0.0047')
    nx.draw(graph, node_color='lightgreen', ax=ax)
    _ = ax.axis('off')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process...')
    parser.add_argument('--data', type=str, help='graph dataset name')
    args = parser.parse_args()

    if not args.data:
        output_folder = module_path + '/output/testing'
    else:
        output_folder = module_path + '/output/' + args.data
    potential_edge_file = output_folder + '/potential_edges.pkl'

    graph = generate_graph(args, module_path)
    graph_gcc = graph_preprocessing(graph)
    eigen_val_1st = calculate_spectrum(graph_gcc)

    print("loading potential edge list...")
    with open(potential_edge_file, 'rb') as f:
        unused_edges = pickle.load(f)

    '''
    print("Calculating Spectrum")
    with Pool(processes=cpu_count()) as pool:
        temp_graph = graph_gcc.copy()
        new_spectrum = partial(calculate_new_spectrum, temp_graph)
        parallel_result = pool.map(new_spectrum, unused_edges)

    selected_edge = max(parallel_result)[1]
    new_eigen = max(parallel_result)[0]
    temp_graph = max(parallel_result)[2]
    '''
    print(eigen_val_1st)
    plot_graph(graph_gcc)

    '''
    print(new_eigen, selected_edge)
    plot_graph(temp_graph)
    '''