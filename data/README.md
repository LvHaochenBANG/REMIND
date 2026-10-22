# REMIND dataset package

## Download

The processed datasets are available on
[Zenodo (DOI: 10.5281/zenodo.23559180)](https://doi.org/10.5281/zenodo.23559180).
Download `ENERGY.zip`, `CAMELS.zip`, `PEMS.zip`, or `PEMS-7years.zip`
as needed. From the code repository root, extract the downloaded archives into
`data/`, for example:

```bash
mkdir -p data
unzip CAMELS.zip -d data/
```

The resulting directory should be `data/CAMELS/`, with `adj.npy` and the
split files directly inside it. Each archive includes its dataset directory;
do not add another directory with the same name. Then follow the preparation
instructions below to build retrieval caches for the selected dataset.

## Layout

| Directory | Time-series files | Nodes | Retrieval Top-K |
|---|---|---:|---:|
| ENERGY | `0.npy`–`3.npy`, each `(8640, 103)` | 103 | 2 |
| CAMELS | `hydrology.npy`, `(12699, 7)` | 7 | 22 |
| PEMS | `2015.npy`–`2017.npy`, each `(15264, 581)` | 581 | 12 |
| PEMS-7years | `2011.npy`–`2017.npy`, each `(15264, 581)` | 581 | 12 |

Each dataset requires an `adj.npy` adjacency matrix whose rows and columns
match the node order in the time-series arrays.

The package includes the source NPY arrays, adjacency matrices, precomputed
`train.npz`, `val.npz`, `test.npz`, and `normalization_params.npz`.
Retrieval caches, model weights and prediction outputs are not included.

## Array formats

- Time-series NPY: `[time, node]`, before model standardization.
- Split NPZ: `history` and `targets`, each `[sample, 12, node]`, standardized.
- Normalization NPZ: scalar `mean` and `std`, fitted on training histories.
  Recover original values with `values * std + mean`.
- `PEMS-7years/node_indices.json`: zero-based source-column indices.

Data acquisition units, verified timestamps and sensor identities are not
provided where unavailable; per-dataset READMEs describe known limitations.

## Preparation

Place the four dataset directories under `data/` in the REMIND code repository,
then run these commands from the code repository root:

```bash
bash scripts/train_energy.sh --prepare
bash scripts/train_camels.sh --prepare
bash scripts/train_pems.sh --prepare
bash scripts/train_pems-7years.sh --prepare
```

These commands reuse existing preprocessing files and rebuild retrieval caches.
Use a fresh directory when changing source data or window settings.


## Data sources and references

The packaged arrays are processed subsets used for REMIND experiments, rather
than complete, unmodified releases of the source datasets. Please cite the
corresponding source publications when using these data.

- **ENERGY**: the Energy-Stream benchmark described in Expand and Compress [1].
- **PEMS / PEMS-7years**: the traffic-data lineage of TrafficStream [2] and the
  PEMS-Stream benchmark described in Expand and Compress [1].
- **CAMELS**: the CAMELS dataset described by Addor et al. [3].

1. Wei Chen and Yuxuan Liang. “Expand and Compress: Exploring Tuning Principles for Continual Spatio-Temporal Graph Forecasting.” ICLR, 2025. [Paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/cb2266111eadcfa2c02187ace64e2183-Paper-Conference.pdf).
2. Xu Chen, Junshan Wang, and Kunqing Xie. “TrafficStream: A Streaming Traffic Flow Forecasting Framework Based on Graph Neural Networks and Continual Learning.” IJCAI, pp. 3620–3626, 2021. [DOI](https://doi.org/10.24963/ijcai.2021/498).
3. Nans Addor, Andrew J. Newman, Naoki Mizukami, and Martyn P. Clark. “The CAMELS data set: catchment attributes and meteorology for large-sample studies.” Hydrology and Earth System Sciences, 21(10):5293–5313, 2017. [DOI](https://doi.org/10.5194/hess-21-5293-2017).
