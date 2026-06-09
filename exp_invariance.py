"""
exp_invariance.py
=================
Curve-handle vs MLP under monotone feature transforms, across datasets.

Reproduces: curve-handle Kendall tau is bit-identical (deviation 0.0) under all
monotone transforms; the MLP degrades. Same test groups throughout, so there is
no difficulty confound.

Run:  python exp_invariance.py
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

import rankcore as rc

SEED = 42
DATASETS = [rc.load_housing, rc.load_diabetes, rc.load_wine]  # wine last (network)


def run_one(loader):
    try:
        X, y, groups, name = loader()
    except Exception as e:
        print(f"[skip {loader.__name__}: {type(e).__name__}]")
        return None

    trm, tem = rc.split_by_group(groups, 0.7, SEED)

    X_rank = rc.within_group_ranks(X, groups)
    scaler = StandardScaler().fit(X[trm])
    X_std = scaler.transform(X).astype(np.float32)

    curve = rc.train_listwise(rc.CurveScorer(X.shape[1]), X_rank[trm], y[trm], groups[trm], seed=SEED)
    mlp = rc.train_listwise(rc.MLPScorer(X.shape[1]), X_std[trm], y[trm], groups[trm], seed=SEED)

    X_te, y_te, g_te = X[tem], y[tem], groups[tem]
    print(f"\n{'='*60}\n{name}  ({len(X)} items, {len(set(groups))} groups, {X.shape[1]} feat)\n{'='*60}")
    print(f"{'transform':<18}{'curve tau':>11}{'MLP tau':>11}")
    print("-" * 40)
    c_taus, m_taus = [], []
    for t in rc.TRANSFORMS:
        Xt = rc.monotone_transform(X_te, t, seed=SEED)
        c = rc.evaluate_kendall(curve, rc.within_group_ranks(Xt, g_te), y_te, g_te)
        m = rc.evaluate_kendall(mlp, scaler.transform(Xt).astype(np.float32), y_te, g_te)
        c_taus.append(c); m_taus.append(m)
        print(f"{t:<18}{c:>11.5f}{m:>11.5f}")
    dev = float(np.abs(np.array(c_taus) - c_taus[0]).max())
    drop = float(m_taus[0] - min(m_taus))
    print(f"curve max deviation: {dev:.2e}   MLP worst drop: {drop:+.4f}")
    return name, dev, drop


def main():
    rows = [r for r in (run_one(L) for L in DATASETS) if r is not None]
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'dataset':<22}{'curve invariant?':>18}{'MLP worst drop':>16}")
    print("-" * 56)
    for name, dev, drop in rows:
        inv = "exact (0.0)" if dev < 1e-6 else f"~{dev:.1e}"
        print(f"{name:<22}{inv:>18}{drop:>+16.4f}")


if __name__ == "__main__":
    main()
