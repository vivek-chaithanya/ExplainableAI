# Results: Stability of GNNExplainer under graph perturbations

*Dataset*: BA-House synthetic graph (700 nodes, 4 classes), GCN accuracy train/val/test = 0.983/0.993/0.964. 30 target nodes, budget k = 6 undirected edges, 10 explainer seeds, 200 epochs.

## 1. Seed stability (graph fixed, different explainer initialisations)

| Method | mean pairwise Jaccard | 95% bootstrap CI |
|---|---|---|
| GNNExplainer | 0.863 | [0.819, 0.906] |
| Random-edge baseline | 0.314 | [0.232, 0.404] |

Paired Wilcoxon (GNNExplainer vs Random, per node): p = 1.86e-09.

Fraction of target nodes whose explanation is *not* identical across all seeds (mean J < 1): 0.67.

**H0 check** (overlap consistently high, threshold J >= 0.8): one-sided Wilcoxon for median J < 0.8, p = 0.996.

### Effect of budget k (seed stability)

| k | GNNExplainer J | Random J |
|---|---|---|
| 3 | 0.653 | 0.142 |
| 6 | 0.863 | 0.316 |
| 9 | 0.797 | 0.475 |
| 12 | 0.876 | 0.622 |

## 2. Perturbation stability (prediction-preserving perturbations, explainer seed fixed)

| Perturbation | magnitude | GNNExplainer J | Random J | p (Wilcoxon, per node) | acceptance rate |
|---|---|---|---|---|---|
| add | 5% of subgraph edges | 0.795 | 0.209 | 1.73e-06 | 1.00 |
| add | 10% of subgraph edges | 0.743 | 0.195 | 1.86e-09 | 1.00 |
| drop | 5% of subgraph edges | 0.868 | 0.308 | 1.70e-06 | 1.00 |
| drop | 10% of subgraph edges | 0.846 | 0.306 | 1.69e-06 | 1.00 |

## 3. Fidelity, sufficiency, sparsity and ground-truth agreement (clean graph)

| Method | Fidelity (prob drop on removal) | Sufficiency (pred. kept) | P(class) when retaining only explanation | Sparsity (#edges) | Precision vs. house motif |
|---|---|---|---|---|---|
| GNNExplainer | 0.537 | 0.600 | 0.453 | 6.0 | 0.912 |
| Random | 0.399 | 0.367 | 0.349 | 6.0 | 0.416 |

- Wilcoxon fidelity: GNNExplainer vs Random p = 4.28e-02

- Wilcoxon sufficiency: GNNExplainer vs Random p = 3.24e-04

- Wilcoxon gt_precision: GNNExplainer vs Random p = 1.73e-06

### Fidelity / sufficiency of GNNExplainer on perturbed graphs

|                |   fidelity |   sufficiency |
|:---------------|-----------:|--------------:|
| ('add', 0.05)  |      0.538 |         0.453 |
| ('add', 0.1)   |      0.531 |         0.393 |
| ('drop', 0.05) |      0.488 |         0.607 |
| ('drop', 0.1)  |      0.506 |         0.58  |

Spearman correlation between per-node fidelity and seed stability: rho = -0.188 (p = 0.32).

## 4. Least stable target nodes (failure cases)

|   node |   n_cand |   jacc_gnnexp |   jacc_random |
|-------:|---------:|--------------:|--------------:|
|    568 |      169 |         0.623 |         0.022 |
|    402 |       13 |         0.626 |         0.301 |
|    463 |       17 |         0.702 |         0.219 |
|    472 |       16 |         0.721 |         0.259 |
|    587 |       14 |         0.729 |         0.287 |

