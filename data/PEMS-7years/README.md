# PEMS-7years

## Files

- `2011.npy`–`2017.npy`: seven annual arrays, each with shape `(15264, 581)`
  and dtype `float64`. Rows represent time steps and columns represent nodes. The reported sampling
  interval is 5 minutes: each file contains 53 days of observations, not a full year.
- `adj.npy`: a `(581, 581)` adjacency matrix using the same node order.
- `node_indices.json`: zero-based source-column indices; entry `i` identifies
  the source column corresponding to output node `i`.

## Node selection

All seven years use a fixed set of 581 columns in the same order. Selection is
based on adjacency-submatrix consistency across years, using
`numpy.allclose(atol=1e-6)` with its default relative tolerance. T



## Source and citation

This processed seven-period traffic subset follows the TrafficStream / PEMS-Stream data lineage described in the following publications.

Xu Chen, Junshan Wang, and Kunqing Xie. “TrafficStream: A Streaming Traffic Flow Forecasting Framework Based on Graph Neural Networks and Continual Learning.” IJCAI, pp. 3620–3626, 2021. [DOI](https://doi.org/10.24963/ijcai.2021/498).

Wei Chen and Yuxuan Liang. “Expand and Compress: Exploring Tuning Principles for Continual Spatio-Temporal Graph Forecasting.” ICLR, 2025. [Paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/cb2266111eadcfa2c02187ace64e2183-Paper-Conference.pdf).
