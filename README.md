# graph-clustering
This repo gives my final year project topic on graph learning. The codes are still under development.

## Usage
```commandline
python main.py --data dataset --method method --rank rank --edge_pct edge_pct
```
Currently, testing would be the default dataset / output folder if no argument is given.
The argument rank-type is only required for edge_degree_min/max method.
The default edge_pct will be 0.1, and user can specify an arbitrary portion for calculation.

```python
{'method': ['greedy', 'random', 'edge_degree_min', 'edge_degree_max'], 
 'dataset': ['cora', 'facebook', 'random'], 
 'rank': ['sum', 'mul']}
```
Process visualization using cprofile and snakeviz:
```commandline
python -m cprofile -o output main.py --data dataset --method method
snakeviz output
```

## Environment: 
```commandline
conda create --prefix env python=3.8
conda install networkx=2.8.4
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install pyg -c pyg
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