# PEMS (three years)

`2015.npy`, `2016.npy`, and `2017.npy` each contain 15,264 time steps
and 581 nodes. The reported variable is traffic flow at 5-minute intervals.
Each annual file represents 53 days of observations, not a full calendar year.

`adj.npy` is a 581 × 581 adjacency matrix with self-loops; node order matches
the data arrays. The retrieval similarity temperature used in the experiment is 1.0.


## Source and citation

This processed traffic subset follows the TrafficStream / PEMS-Stream data lineage described in the following publications.

Xu Chen, Junshan Wang, and Kunqing Xie. “TrafficStream: A Streaming Traffic Flow Forecasting Framework Based on Graph Neural Networks and Continual Learning.” IJCAI, pp. 3620–3626, 2021. [DOI](https://doi.org/10.24963/ijcai.2021/498).

Wei Chen and Yuxuan Liang. “Expand and Compress: Exploring Tuning Principles for Continual Spatio-Temporal Graph Forecasting.” ICLR, 2025. [Paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/cb2266111eadcfa2c02187ace64e2183-Paper-Conference.pdf).
