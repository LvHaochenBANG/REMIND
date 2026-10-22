# ENERGY

Four arrays (`0.npy` through `3.npy`) each contain 8,640 time steps and
103 nodes. The reported sampling interval is 10 minutes, corresponding to
240 days of observations across the four files; calendar timestamps are not included.
`adj.npy` is a symmetric 103 × 103 adjacency matrix with self-loops.
The retrieval similarity temperature used in the experiment is 1.7.

## Source and citation

This processed subset uses the Energy-Stream benchmark described in the following publication.

Wei Chen and Yuxuan Liang. “Expand and Compress: Exploring Tuning Principles for Continual Spatio-Temporal Graph Forecasting.” ICLR, 2025. [Paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/cb2266111eadcfa2c02187ace64e2183-Paper-Conference.pdf).
