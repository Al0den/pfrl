# Market Geometry & Potential Fields

A geometric and spectral representation of financial markets based on correlation graphs, manifold embeddings, and induced potential fields.



## Overview

This project models a universe of assets as a **dynamic geometric system**:

1. Asset dependencies are estimated from rolling windows of returns.
2. These dependencies define a **weighted similarity graph**.
3. The graph is embedded into a low-dimensional continuous space using **spectral methods**.
4. A scalar signal defined on assets induces a **smooth potential field** over the embedding.
5. The resulting geometry can be visualized, sampled, and reused as structured features.

The approach is deterministic, interpretable, and grounded in spectral graph theory and kernel methods.



## Mathematical Summary (Quick)

Let \( N \) assets be observed over time.

### 1. Returns and Correlation

Log-returns are computed as:
\[
r_{t,i} = \log P_{t,i} - \log P_{t-1,i}.
\]

On a rolling window, a correlation matrix \( C \in \mathbb{R}^{N \times N} \) is estimated.



### 2. Similarity Graph

A weighted undirected graph is constructed from correlations:
- Nodes: assets
- Edge weights: positive correlations (k-nearest neighbors)

This yields a symmetric adjacency matrix \( W \ge 0 \), and a degree matrix
\[
D_{ii} = \sum_j W_{ij}.
\]



### 3. Normalized Graph Laplacian

The geometry of the graph is encoded via the **symmetric normalized Laplacian**:
\[
L = I - D^{-1/2} W D^{-1/2}.
\]

This operator is invariant to degree scaling and approximates the Laplace–Beltrami operator of an underlying manifold.



### 4. Spectral Embedding (Laplacian Eigenmaps)

The embedding is obtained from the eigen-decomposition:
\[
L v_k = \lambda_k v_k,
\quad
0 = \lambda_1 \le \lambda_2 \le \dots
\]

The first non-trivial eigenvectors define an embedding:
\[
X \in \mathbb{R}^{N \times d}.
\]

Each asset is represented as a point in a low-dimensional latent space.

To ensure temporal consistency, embeddings are aligned across time using **orthogonal Procrustes alignment**.



### 5. Scalar Signal on Assets

A scalar signal \( \phi \in \mathbb{R}^N \) is computed per asset, e.g.:
- momentum,
- risk-adjusted momentum.

It is normalized to zero mean and unit variance.



### 6. Potential Field

Given asset positions \( x_i \in \mathbb{R}^d \), the signal induces a smooth potential field:
\[
U(g) = - \sum_{i=1}^N \phi_i
\exp\!\left(-\frac{\|g - x_i\|^2}{2\sigma^2}\right),
\quad g \in \mathbb{R}^d.
\]

The associated vector field is:
\[
F(g) = - \nabla U(g).
\]

This field reveals collective structure, attractors, and repulsive regions in the embedded space.



## Visualizations

The project includes scripts to:
- plot correlation graphs,
- visualize 2D and 3D potential landscapes,
- generate animated time-evolving embeddings and fields.

These are located in `scripts/visuals/`.



## Design Principles

- **Geometry-first**: structure emerges from correlations, not heuristics.
- **Spectral methods**: stable, well-founded, interpretable.
- **Deterministic core**: no stochastic components in the geometry.
- **Separation of concerns**: data, geometry, fields, and downstream usage are cleanly decoupled.



## References

- Belkin, M., & Niyogi, P.  
  *Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering*  
  NIPS, 2001.

- Chung, F.  
  *Spectral Graph Theory*  
  AMS, 1997.

- Coifman, R. R., & Lafon, S.  
  *Diffusion Maps*  
  Appl. Comput. Harmonic Analysis, 2006.



## Notes

This repository focuses on the **mathematical and geometric layer**.
Downstream usage (e.g. optimization or learning) is optional and treated as a separate concern.
