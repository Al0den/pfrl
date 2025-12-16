# Market Geometry & Potential Fields

A geometric and spectral representation of financial markets based on correlation graphs, manifold embeddings, and induced potential fields.



## Overview

This project models a universe of assets as a **dynamic geometric system**:

1. Statistical dependencies between assets are estimated from rolling windows of returns.
2. These dependencies define a **weighted similarity graph**.
3. The graph is embedded into a low-dimensional continuous space using **spectral methods**.
4. A scalar signal defined on assets induces a **smooth potential field** over the embedding.
5. The resulting geometry can be visualized, sampled, and reused as structured features.

The approach is deterministic, interpretable, and grounded in spectral graph theory and kernel methods.



## Mathematical Summary

This section provides a concise mathematical overview of the core construction.



### 1. Returns and Correlation

Log-returns are computed as:

$$
r_{t,i} = \log P_{t,i} - \log P_{t-1,i}
$$

On a rolling window of length \( L \), a correlation matrix is estimated:

$$
C \in \mathbb{R}^{N \times N}
$$

where \( N \) is the number of assets.



### 2. Similarity Graph

A weighted undirected graph is constructed from the correlation matrix:

- Nodes correspond to assets
- Edge weights correspond to positive correlations (k-nearest neighbors)

This yields a symmetric adjacency matrix:

$$
W \ge 0
$$

and a diagonal degree matrix:

$$
D_{ii} = \sum_{j=1}^{N} W_{ij}
$$

<p align="center">
  <img src="https://github.com/user-attachments/assets/d1948367-36b4-4167-a7b4-2330aee33b9e" alt="example_graph" />
  <br/>
  <em>Figure 1: Example graph</em>
</p>


### 3. Normalized Graph Laplacian

The geometry of the graph is encoded via the **symmetric normalized Laplacian**:

$$
L = I - D^{-1/2} W D^{-1/2}
$$

This operator is invariant to degree scaling and approximates the Laplace–Beltrami operator of an underlying continuous manifold.



### 4. Spectral Embedding (Laplacian Eigenmaps)

The embedding is obtained from the eigen-decomposition:

$$
L v_k = \lambda_k v_k,
\quad
0 = \lambda_1 \le \lambda_2 \le \dots
$$

The first non-trivial eigenvectors define a low-dimensional embedding:

$$
X \in \mathbb{R}^{N \times d}
$$

Each asset is represented as a point in a latent geometric space.

To ensure temporal consistency, embeddings are aligned across time using **orthogonal Procrustes alignment**.



### 5. Scalar Signal on Assets

A scalar signal is defined per asset:

$$
\phi \in \mathbb{R}^N
$$

Typical examples include momentum or risk-adjusted momentum.  
The signal is standardized to zero mean and unit variance.



### 6. Potential Field

Given embedded asset positions \( x_i \in \mathbb{R}^d \), the signal induces a smooth potential field:

$$
U(g) = - \sum_{i=1}^{N} \phi_i
\exp\left(-\frac{\|g - x_i\|^2}{2\sigma^2}\right),
\quad g \in \mathbb{R}^d
$$

The associated vector field is given by the gradient:

$$
F(g) = - \nabla U(g)
$$

This field reveals collective structure, attractors, and repulsive regions in the embedded space.

![potential_landscape](https://github.com/user-attachments/assets/7a14bcb3-eb63-4b57-8327-6c01b72daf7f)


<p align="center">
  <img src="https://github.com/user-attachments/assets/7a14bcb3-eb63-4b57-8327-6c01b72daf7f" alt="potential_landscape" />
  <br/>
  <em>Figure 1: Potential Landscape</em>
</p>




## Visualizations

The project includes scripts to:

- plot correlation graphs,
- visualize 2D and 3D potential landscapes,
- generate animated time-evolving embeddings and fields.

These scripts are located in `scripts/visuals/`.



## Design Principles

- **Geometry-first**: structure emerges from correlations, not heuristics.
- **Spectral methods**: stable, well-founded, interpretable.
- **Deterministic core**: no stochastic components in the geometry.
- **Separation of concerns**: data, geometry, fields, and downstream usage are cleanly decoupled.
