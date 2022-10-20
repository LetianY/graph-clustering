# graph-clustering
This repo gives my final year project topic on graph learning. The codes are still under development.

## Usage
```commandline
python main.py --data dataset --method method
```
Currently, testing would be the default dataset / output folder if no argument is given.
```python
{'method': ['greedy', 'random', 'max_edge_degree'], 'dataset': ['cora', 'facebook', 'random']}
```
Process visualization using cprofile and snakeviz:
```commandline
python -m cprofile -o output main.py --data dataset --method method
snakeviz output
```

## Processed Data Source
@article{yang2020scaling,
  title={Scaling Attributed Network Embedding to Massive Graphs},
  author={Yang, Renchi and Shi, Jieming and Xiao, Xiaokui and Yang, Yin and Liu, Juncheng and Bhowmick, Sourav S},
  journal={Proceedings of the VLDB Endowment},
  volume={14},
  number={1},
  pages={37--49},
  year={2021},
  publisher={VLDB Endowment}
}

The datasets are also available in [Pytorch-Geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html#torch_geometric.datasets.AttributedGraphDataset). 

Node attributes can be loaded as a sparse matrix using the following code

```python
from scipy import sparse
features = sparse.load_npz("attrs.npz")
```