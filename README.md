# Exact Invariance from Rank-Coordinate Scoring: Differentiable Ranking without Permutation Relaxation

Exact invariance to monotone feature reparameterization for neural rankers, by
scoring in rank coordinates through an exact sort.

A ranker built this way produces an ordering that does not change when any input
feature is passed through a strictly monotone transformation (a gamma warp, a
log, a unit change, a gain or calibration shift). The invariance is exact and
structural, not learned, and it holds to numerical precision. Standard rankers,
including the NeuralSort and Sinkhorn differentiable sorters, do not have it,
because they rank by a scorer that reads feature magnitudes.

This repository contains the method, the baselines, and scripts that reproduce
the results in the paper.

## Idea in one paragraph

Sorting is not differentiable, so order structure is usually put into neural
networks by relaxing the permutation into a soft, magnitude-dependent matrix
(NeuralSort, Sinkhorn). We do not relax the permutation. We keep the sort exact
and route gradients through the values being sorted. Each item is scored from
the within-group ranks of its features rather than from their magnitudes, with a
small smooth curve per feature. Because a strictly monotone map preserves all
within-group ranks, the score, and therefore the ranking, is unchanged. The
invariance comes from scoring in rank coordinates, not from how the sort is made
differentiable.

## Results

Same test groups throughout; only the feature calibration is changed by a
monotone transform. Kendall tau reported.

Invariance across datasets (maximum deviation of the rank-coordinate ranker vs.
worst-case drop of an MLP):

| Dataset            | Curve handle (max tau deviation) | MLP (worst tau drop) |
|--------------------|----------------------------------|----------------------|
| California Housing | 0.0 (exact)                      | 0.578                |
| Wine Quality       | 0.0 (exact)                      | 0.225                |
| Diabetes           | 0.0 (exact)                      | 0.152                |

Four-way comparison on California Housing (tau range across all transforms):

| Method            | tau range |
|-------------------|-----------|
| Curve handle      | **0.000** |
| MLP               | 0.578     |
| NeuralSort        | 0.423     |
| Sinkhorn          | 0.588     |

All four methods are competitive at the identity transform (Curve 0.444, MLP
0.473, NeuralSort 0.474, Sinkhorn 0.456). Only the curve handle stays flat.

## Install

```bash
git clone https://github.com/aphorion/curve-rank-invariance
cd curve-rank-invariance
pip install torch numpy scipy scikit-learn
```

Python 3.9+ and any recent PyTorch. A GPU is optional; the experiments run on
CPU in a few minutes.

## Run

```bash
python exp_invariance.py     # curve handle vs MLP, across datasets
python exp_relaxation.py     # four-way: curve vs MLP vs NeuralSort vs Sinkhorn
```

`exp_invariance.py` prints, per dataset, the Kendall tau of both methods under
every transform, the curve handle's maximum deviation (expected 0.0), and the
MLP's worst drop. `exp_relaxation.py` prints the four-way table and the tau
range per method.

Wine Quality is downloaded from UCI at runtime, with a mirror fallback; if both
sources are unreachable the dataset is skipped and the others still run.

## Code

The logic is in one library module; the experiment scripts are thin entry
points that import it.

- `rankcore.py` — data loaders, the float64 within-group rank transform, the
  models (`CurveScorer`, `MLPScorer`), the relaxation baselines (`neuralsort_P`,
  `sinkhorn_P`), the listwise loss, training and evaluation, and the monotone
  transforms. The public API is documented at the top of the file.
- `exp_invariance.py` — multi-dataset invariance experiment.
- `exp_relaxation.py` — four-way exact-vs-relaxation comparison.

Minimal use:

```python
import rankcore as rc
from sklearn.preprocessing import StandardScaler

X, y, groups, name = rc.load_housing()
train, test = rc.split_by_group(groups, 0.7, seed=42)

X_rank = rc.within_group_ranks(X, groups)
model = rc.train_listwise(rc.CurveScorer(X.shape[1]),
                          X_rank[train], y[train], groups[train])

# evaluate under a monotone transform of the test features
Xt = rc.monotone_transform(X[test], "per-feat random", seed=42)
tau = rc.evaluate_kendall(model, rc.within_group_ranks(Xt, groups[test]),
                          y[test], groups[test])
print(tau)  # identical to the untransformed test tau
```

## Notes on exactness

Within-group ranks are computed in float64 with a stable sort. Single-precision
ranks deviate only under transforms that compress distinct values below machine
epsilon (for example a strong gamma); double precision removes this and makes
the invariance exact rather than precision-limited.

The method requires commensurable groups, where members are comparable on a
common scale (documents in a query, houses in a neighborhood). It is a ranker
for comparison groups, not a general tabular learner. It trades a small amount
of clean-data accuracy for the invariance, so it is suited to ranking under
calibration shift, batch effects, sensor drift, and cross-site transfer, and not
to stable, well-calibrated problems where magnitude carries signal.

## Paper

The accompanying paper, *Exact Invariance from Rank-Coordinate Scoring:
Differentiable Ranking Without Permutation Relaxation*, is included as
`paper_invariance.pdf` with LaTeX source in `paper_invariance.tex`.

## Citation

```bibtex
@misc{nguthiru2025curverank,
  title  = {Exact Invariance from Rank-Coordinate Scoring:
            Differentiable Ranking Without Permutation Relaxation},
  author = {Nguthiru, Edwin},
  year   = {2025},
  note   = {Aphorion},
  url    = {https://github.com/aphorion/curve-rank-invariance}
}
```

## Author

Edwin Nguthiru, Aphorion. edwin@aphorion.co
