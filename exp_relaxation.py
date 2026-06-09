"""
exp_relaxation.py
=================
Four-way comparison under monotone feature transforms:
curve handle (rank-coordinate, exact sort) vs MLP, NeuralSort, Sinkhorn.

Reproduces: all three magnitude-based methods degrade under monotone transforms,
including the two differentiable sorters; only the rank-coordinate curve handle
is invariant (tau-range 0.0). Shows the invariance is from rank-coordinate
scoring, not from making the sort differentiable.

Run:  python exp_relaxation.py
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

import rankcore as rc

SEED = 42
DATASETS = [rc.load_housing, rc.load_diabetes]


def run_one(loader):
    X, y, groups, name = loader()
    trm, tem = rc.split_by_group(groups, 0.7, SEED)
    nf = X.shape[1]

    X_rank = rc.within_group_ranks(X, groups)
    scaler = StandardScaler().fit(X[trm])
    X_std = scaler.transform(X).astype(np.float32)

    curve = rc.train_listwise(rc.CurveScorer(nf), X_rank[trm], y[trm], groups[trm], seed=SEED)
    mlp = rc.train_listwise(rc.MLPScorer(nf), X_std[trm], y[trm], groups[trm], seed=SEED)
    nsort = rc.train_relax(rc.MLPScorer(nf), X_std[trm], y[trm], groups[trm], rc.neuralsort_P, seed=SEED)
    sink = rc.train_relax(rc.MLPScorer(nf), X_std[trm], y[trm], groups[trm], rc.sinkhorn_P, seed=SEED)

    X_te, y_te, g_te = X[tem], y[tem], groups[tem]
    methods = ["curve", "MLP", "NeuralSort", "Sinkhorn"]
    res = {k: [] for k in methods}

    print(f"\n{'='*68}\n{name}  ({len(X)} items, {len(set(groups))} groups, {nf} feat)\n{'='*68}")
    print(f"{'transform':<18}" + "".join(f"{k:>12}" for k in methods))
    print("-" * (18 + 12 * 4))
    for t in rc.TRANSFORMS:
        Xt = rc.monotone_transform(X_te, t, seed=SEED)
        Xr = rc.within_group_ranks(Xt, g_te)
        Xs = scaler.transform(Xt).astype(np.float32)
        row = {
            "curve": rc.evaluate_kendall(curve, Xr, y_te, g_te),
            "MLP": rc.evaluate_kendall(mlp, Xs, y_te, g_te),
            "NeuralSort": rc.evaluate_kendall(nsort, Xs, y_te, g_te),
            "Sinkhorn": rc.evaluate_kendall(sink, Xs, y_te, g_te),
        }
        for k in methods:
            res[k].append(row[k])
        print(f"{t:<18}" + "".join(f"{row[k]:>12.4f}" for k in methods))

    print("-" * (18 + 12 * 4))
    print(f"{'tau-range':<18}" + "".join(f"{(max(res[k])-min(res[k])):>12.4f}" for k in methods))
    return name, {k: (max(res[k]) - min(res[k])) for k in methods}


def main():
    rows = [run_one(L) for L in DATASETS]
    print("\n" + "=" * 68)
    print("VERDICT  (tau-range across transforms; 0.0 = invariant)")
    print("=" * 68)
    for name, ranges in rows:
        print(f"\n  {name}:")
        for k, r in ranges.items():
            tag = "INVARIANT" if r < 1e-6 else f"degrades ({r:.3f})"
            print(f"    {k:<12} {tag}")


if __name__ == "__main__":
    main()
