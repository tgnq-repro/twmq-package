# twmq

**Two-way Grouped Network Quantile (TGNQ) Autoregression**

An R package implementing the TGNQ autoregression model proposed in

> *Two-way Homogeneity Pursuit for Quantile Network Vector Autoregression.*

The package fits a vector autoregressive quantile model on time series observed
over a directed network, in which each node is endowed with **two latent group
memberships**:

- a **row (G) group** capturing the node's *susceptibility* (how it is affected by others), and
- a **column (H) group** capturing the node's *influence* (how it affects others).

This two-way grouping captures asymmetric, directional interactions that the
usual one-way grouped network autoregression cannot represent. The model
performs node clustering and parameter estimation simultaneously, balancing
flexibility and interpretability, and supports valid statistical inference at
multiple quantile levels.

---

## Requirements

The package was developed and tested under the following environment:

- **R version:** 4.5.1 (2025-06-13)
- **Platform:** aarch64-apple-darwin20 (macOS Sequoia 15.4.1)

The required R packages and the specific versions used are:

| Package          | Version     | Role                                                     |
|------------------|-------------|----------------------------------------------------------|
| `Rcpp`           | 1.1.0       | C++ interface to R                                       |
| `RcppArmadillo`  | 15.2.3-1    | C++ linear algebra backend (Armadillo)                   |
| `quantreg`       | 6.1         | Quantile regression solvers                              |
| `conquer`        | 1.3.3       | Convolution-type smoothed quantile regression            |
| `CEoptim`        | 1.3         | Cross-entropy discrete optimization                      |
| `foreach`        | 1.5.2       | Parallel `for`-loop backend (parallel routines only)     |
| `doSNOW`         | 1.0.20      | SNOW parallel backend for `foreach` (parallel routines)  |
| `RhpcBLASctl`    | 0.23-42     | Per-worker control of BLAS threads (parallel routines)   |

The packages `foreach`, `doSNOW`, and `RhpcBLASctl` are only required for the
parallel routines (`twmq.estimate.auto.parallel`, `update_NARG_twmq_parallel`,
`Refine_G_parallel`, `Refine_H_parallel`); the core functionality runs without
them.

Other versions of R (≥ 4.1.0) and the listed packages are likely to work, but
have not been formally tested.

---

## Installation

Install the `remotes` package first if needed. Then install the development version of `twmq` from GitHub:

`install.packages("remotes")`

`remotes::install_github("wenyangliu0110/twmq-package")`

The package depends on `Rcpp`, `RcppArmadillo`, `quantreg`, `conquer`, and `CEoptim`. For the parallel routines, also install `doSNOW`, `foreach`, and `RhpcBLASctl`.

---

## Model

For each node `i` and quantile level `τ`, TGNQ specifies the conditional
quantile of `Y_{it}` given the past as a sum of three components:

- a **network term** `Σ_j θ_{g_i h_j}(τ) · w_{ij} · Y_{j,t-1}`, where `g_i` and
  `h_j` are unobserved group memberships shared across quantiles, and `w_{ij}`
  are entries of a row-normalized weight matrix derived from the network
  adjacency matrix;
- an **autoregressive term** `ν_{g_i}(τ) · Y_{i,t-1}`;
- a **covariate term** `x_{it}^⊤ γ_{g_i}(τ)`.

Three forms of the network effect `θ_{gh}(τ)` are supported:

| `method`           | Form                                  |
|--------------------|---------------------------------------|
| `"general"`        | `θ_{gh}(τ)` is free for each `(g,h)` |
| `"additive"`       | `θ_{gh}(τ) = α_g(τ) + β_h(τ)`         |
| `"multiplicative"` | `θ_{gh}(τ) = α_g(τ) × β_h(τ)`         |

The general specification is the most flexible. The additive and multiplicative
specifications further decompose the directional effect into a "susceptibility"
component of the receiver and an "influence" component of the sender.

---

## Inputs

A typical fit requires:

- `Ymat`: an `N × T` matrix of responses (rows = nodes, columns = time points);
- `X_tensor`: an `N × p × (T−1)` array of time-varying covariates;
- `W`: an `N × N` row-normalized weight matrix (adjacency matrix divided by
  out-degree, with zero diagonal);
- `taus`: a vector of pre-specified quantile levels;
- `G`, `H`: numbers of row- and column-groups (or a candidate grid for selection).

---

## Estimation procedure

The package implements both algorithms described in the paper:

- **Vanilla Algorithm (Algorithm 1).** Coordinate descent that alternates
  between (i) updating model parameters given memberships, (ii) updating row
  memberships, and (iii) updating column memberships. To mitigate sensitivity
  to initialization, a multi-start strategy is provided: candidate initial
  memberships are generated from per-node LASSO quantile regression followed
  by k-means clustering, and the best-loss solution among all starts is
  returned.

- **Enhanced Algorithm (Algorithm 2).** Designed to escape local minima
  encountered when updating the column membership `H`. It iteratively
  constructs proposal H-vectors over an active set of nodes, accepts those
  that strictly decrease the loss, and reduces the active set as the
  algorithm progresses. Cross-entropy discrete optimization (`CEoptim`) is
  used as a fallback when full enumeration of follower configurations is
  infeasible.

Two loss functions are supported throughout: the standard quantile check loss
and the convolution-type smoothed quantile loss (the *conquer* loss of
He et al., 2023), which improves computational scalability on large networks.

---

## Selecting the number of groups

The number of row and column groups can be chosen by minimizing a quantile
information criterion (QIC) of the form

```
QIC(G, H) = log( L̂(ψ̂, Ĝ, Ĥ) ) + λ_{NT} · G · (H + p + 1),
```

where `L̂` is the multi-quantile loss at the fitted memberships and parameters,
and `λ_{NT}` is a tuning parameter that vanishes at an appropriate rate.
Following the paper, a practical default is

```
λ_{NT} = N^{1/10} · T^{-1} · log(T) / (10 · min(n̄, 10)),
```

where `n̄` is the average out-degree of the network. The package fits the
model over a grid of `(G, H)` values; the pair minimizing QIC is selected as
`(Ĝ, Ĥ)`.

---

## Inference

When the group numbers are correctly specified, the post-estimation parameter
estimators are asymptotically normal at all pre-specified quantile levels.
The package provides utilities to construct 95% confidence intervals for the
group-level parameters (network effects, autoregressive effects, and covariate
effects), based on the asymptotic variance derived in the paper.

In addition, per-node refinement diagnostics are provided. For each node, the
loss attained under each candidate G- (or H-) membership is computed while
optimizing over neighboring memberships. These diagnostics are used in the
theoretical refinement procedure of the paper, and can also serve as
sensitivity checks for individual node assignments.

---

## Parallel computing

For larger networks, parallel versions of both the multi-start vanilla
algorithm and the enhanced algorithm are provided, built on `doSNOW` /
`foreach`. The per-node H-optimization step in the enhanced algorithm and the
multi-start initialization in the vanilla algorithm are the most expensive
steps and benefit most from parallelization.

---

## Main exported functions

| Function | Purpose |
|----------|---------|
| `twmq.estimate.auto`           | Vanilla Algorithm 1 with multi-start LASSO + k-means initialization |
| `twmq.estimate.auto.parallel`  | Parallel version of the multi-start vanilla algorithm |
| `update_NARG_twmq`             | Enhanced Algorithm 2 (escape local minima for `H`) |
| `update_NARG_twmq_parallel`    | Parallel version of the enhanced algorithm |
| `twmq_ci`                      | 95% confidence intervals for the parameters |
| `twmq.label.switch`            | Align estimated group labels with reference labels |
| `Refine_G`, `Refine_H`         | Per-node membership refinement diagnostics |
| `Refine_G_parallel`, `Refine_H_parallel` | Parallel versions of the refinement diagnostics |
| `err_rate_mapping`             | Mis-clustering rate after best label permutation |

The C++ workhorses (compiled via `Rcpp` / `RcppArmadillo`) live in
`src/twq.cpp` and are called automatically by the R interface.

---


## Citation

If you use this package, please cite:

> *Two-way Homogeneity Pursuit for Quantile Network Vector Autoregression.*

A BibTeX entry will be added once the paper is published.

---

## License

This package is released for academic and research use. See the `DESCRIPTION`
file for details.