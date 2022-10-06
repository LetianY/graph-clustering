import sys
import argparse

import networkx as nx

from utils import *
from pathlib import Path
import matplotlib.pyplot as plt


module_dur = os.getcwd()
sys.path.append(module_dur)
path = Path(module_dur)
module_path = str(path.parent.absolute())
sys.path.append(module_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process...')
    parser.add_argument('--data', type=str, help='graph dataset name')
    args = parser.parse_args()

    print("loading data...")
    graph = generate_graph(args, module_path)
    graph_gcc = graph_preprocessing(graph)

    eigen_val_1st = calculate_spectrum_gap(graph_gcc)
    print(eigen_val_1st)
