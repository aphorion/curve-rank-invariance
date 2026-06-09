"""
rankcore.py
===========
Reusable components for rank-coordinate ranking experiments.

Public API
----------
Data:
    load_housing(), load_wine(), load_diabetes()  -> (X, y, groups, name)
    split_by_group(groups, frac_train, seed)       -> (train_mask, test_mask)

Features:
    within_group_ranks(X, groups)                  -> rank features in [0,1] (float64 ranks)

Models:
    CurveScorer(n_features, K)                      rank-coordinate scorer (the method)
    MLPScorer(n_features, hidden)                   magnitude baseline

Relaxation baselines:
    neuralsort_P(scores, tau)                       NeuralSort soft permutation
    sinkhorn_P(scores, tau, iters)                  Sinkhorn soft permutation

Losses / training / eval:
    listwise_loss(scores, targets, groups)
    train_listwise(model, X, y, groups, ...)
    train_relax(model, X, y, groups, method, ...)
    evaluate_kendall(model, X, y, groups)

Transforms:
    monotone_transform(X, name, seed)               apply a monotone per-feature map
    TRANSFORMS                                       list of transform names

All randomness is seeded. Ranks use float64 + stable sort so that the
invariance is exact rather than precision-limited.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def _filter_groups(X, y, g, min_size):
    uniq, cnt = np.unique(g, return_counts=True)
    valid = set(uniq[cnt >= min_size])
    keep = np.array([gi in valid for gi in g])
    return X[keep], y[keep], g[keep]


def load_housing():
    from sklearn.datasets import fetch_california_housing
    d = fetch_california_housing()
    X = d.data.astype(np.float32)
    y = d.target.astype(np.float32)
    lat, lon = X[:, 6], X[:, 7]
    g = np.floor(lat / 0.25).astype(int) * 10000 + np.floor(lon / 0.25).astype(int)
    X, y, g = _filter_groups(X[:, :6], y, g, min_size=6)
    return X, y, g, "California Housing"


def load_wine():
    import urllib.request
    sources = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/",
        "https://raw.githubusercontent.com/zygmuntz/wine-quality/master/winequality/",
    ]
    files = ["winequality-red.csv", "winequality-white.csv"]
    frames = None
    for base in sources:
        try:
            fr = []
            for fn in files:
                raw = urllib.request.urlopen(base + fn, timeout=15).read().decode()
                rows = [r.split(";") for r in raw.strip().split("\n")]
                fr.append(np.array(rows[1:], dtype=np.float32))
            frames = fr
            break
        except Exception:
            continue
    if frames is None:
        raise RuntimeError("wine quality unavailable from all sources")
    data = np.vstack(frames)
    feat, qual = data[:, :11], data[:, 11]
    g = np.floor(feat[:, 10] / 0.5).astype(int)        # alcohol bands
    feat = feat[:, :10]                                # drop alcohol (grouping var)
    X, y, g = _filter_groups(feat, qual, g, min_size=8)
    return X, y, g, "Wine Quality"


def load_diabetes():
    from sklearn.datasets import load_diabetes as _ld
    d = _ld()
    X = d.data.astype(np.float32)
    y = d.target.astype(np.float32)
    bmi = X[:, 2]
    g = np.floor((bmi - bmi.min()) / ((bmi.max() - bmi.min()) / 12 + 1e-9)).astype(int)
    X, y, g = _filter_groups(np.delete(X, 2, axis=1), y, g, min_size=6)
    return X, y, g, "Diabetes"


def split_by_group(groups, frac_train=0.7, seed=42):
    rng = np.random.RandomState(seed)
    gs = np.array(list(set(groups)))
    rng.shuffle(gs)
    cut = int(frac_train * len(gs))
    tr = set(gs[:cut])
    train_mask = np.array([g in tr for g in groups])
    return train_mask, ~train_mask


# ----------------------------------------------------------------------------
# Rank-coordinate features (float64 ranks for exact invariance)
# ----------------------------------------------------------------------------
def within_group_ranks(X, groups):
    X = X.astype(np.float64)
    Xr = np.empty_like(X, dtype=np.float32)
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        sub = X[idx]
        order = np.argsort(np.argsort(sub, axis=0, kind="stable"), axis=0, kind="stable")
        Xr[idx] = (order.astype(np.float64) / max(len(idx) - 1, 1)).astype(np.float32)
    return Xr


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
class CurveScorer(nn.Module):
    """Rank-coordinate scorer. One smooth K-knot curve per feature, evaluated
    at the feature's within-group rank, summed across features."""
    def __init__(self, n_features, K=16):
        super().__init__()
        self.curves = nn.Parameter(torch.randn(n_features, K) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_rank):
        B, Fn = x_rank.shape
        K = self.curves.shape[-1]
        q = x_rank.clamp(1e-6, 1 - 1e-6) * (K - 1)
        i = q.long().clamp(0, K - 2)
        fr = q - i.float()
        feat = torch.arange(Fn, device=x_rank.device).expand(B, Fn)
        lo = self.curves[feat, i]
        hi = self.curves[feat, i + 1]
        return (lo + fr * (hi - lo)).sum(-1) + self.bias


class MLPScorer(nn.Module):
    """Magnitude baseline. Scores each item from its raw (standardized) features."""
    def __init__(self, n_features, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ----------------------------------------------------------------------------
# Relaxation baselines
# ----------------------------------------------------------------------------
def neuralsort_P(scores, tau=1.0):
    """NeuralSort soft permutation (Grover et al. 2019).
    Row i is a soft one-hot for the i-th largest element."""
    B, n = scores.shape
    s = scores.unsqueeze(-1)
    A = (s - s.transpose(1, 2)).abs()
    A_sum = A.sum(dim=2, keepdim=True)
    scaling = (n + 1 - 2 * (torch.arange(n, device=scores.device) + 1)).float()
    C = s * scaling.view(1, 1, n)
    P = C.transpose(1, 2) - A_sum.transpose(1, 2)
    return F.softmax(P / tau, dim=-1)


def sinkhorn_P(scores, tau=0.05, iters=30):
    """Sinkhorn soft permutation. Element i assigned to rank position j.
    Cost from normalized scores to target positions; log-domain normalization."""
    B, n = scores.shape
    s_min = scores.min(dim=1, keepdim=True).values
    s_max = scores.max(dim=1, keepdim=True).values
    s_norm = (scores - s_min) / (s_max - s_min + 1e-9)
    r = torch.linspace(0, 1, n, device=scores.device).view(1, 1, n)
    C = (s_norm.unsqueeze(2) - r) ** 2
    log_P = -C / tau
    for _ in range(iters):
        log_P = log_P - torch.logsumexp(log_P, dim=2, keepdim=True)
        log_P = log_P - torch.logsumexp(log_P, dim=1, keepdim=True)
    return log_P.exp()


# ----------------------------------------------------------------------------
# Losses, training, evaluation
# ----------------------------------------------------------------------------
def listwise_loss(scores, targets, groups):
    """Negative mean per-group correlation between scores and targets."""
    losses = []
    for g in groups.unique():
        m = groups == g
        s, t = scores[m], targets[m]
        if s.numel() < 2:
            continue
        s = (s - s.mean()) / (s.std() + 1e-6)
        t = (t - t.mean()) / (t.std() + 1e-6)
        losses.append(-(s * t).mean())
    if not losses:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)
    return torch.stack(losses).mean()


def _relax_group_loss(scorer, X_g, t_g, method, tau):
    """Soft-permutation cross-entropy for one group, with per-method orientation."""
    s = scorer(X_g).unsqueeze(0)
    n = t_g.numel()
    P = method(s, tau=tau)
    tgt = torch.zeros(1, n, n, device=X_g.device)
    if method is neuralsort_P:
        desc = (n - 1) - torch.argsort(torch.argsort(t_g))   # 0 = largest, row index
        tgt[0, desc, torch.arange(n, device=X_g.device)] = 1.0
    else:
        asc = torch.argsort(torch.argsort(t_g))              # element -> ascending position
        tgt[0, torch.arange(n, device=X_g.device), asc] = 1.0
    return F.binary_cross_entropy(P.clamp(1e-6, 1 - 1e-6), tgt)


def _to_tensors(X, y, groups):
    return (torch.from_numpy(X).to(DEVICE),
            torch.from_numpy(y).to(DEVICE),
            torch.from_numpy(groups).long().to(DEVICE))


def train_listwise(model, X, y, groups, epochs=120, lr=5e-3, group_batch=32, seed=42):
    torch.manual_seed(seed)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt, gt = _to_tensors(X, y, groups)
    gl = list(np.unique(groups))
    rng = np.random.RandomState(seed)
    for _ in range(epochs):
        rng.shuffle(gl)
        for i in range(0, len(gl), group_batch):
            chosen = torch.tensor(gl[i:i + group_batch], device=DEVICE, dtype=gt.dtype)
            m = (gt.unsqueeze(1) == chosen.unsqueeze(0)).any(1)
            if not m.any():
                continue
            loss = listwise_loss(model(Xt[m]), yt[m], gt[m])
            opt.zero_grad(); loss.backward(); opt.step()
    return model


def train_relax(model, X, y, groups, method, epochs=80, lr=5e-3, tau=None,
                group_batch=16, seed=42):
    if tau is None:
        tau = 1.0 if method is neuralsort_P else 0.05
    torch.manual_seed(seed)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xt, yt, gt = _to_tensors(X, y, groups)
    gl = list(np.unique(groups))
    rng = np.random.RandomState(seed)
    for _ in range(epochs):
        rng.shuffle(gl)
        for i in range(0, len(gl), group_batch):
            chunk = gl[i:i + group_batch]
            losses = []
            for g in chunk:
                m = gt == g
                if m.sum() < 2:
                    continue
                losses.append(_relax_group_loss(model, Xt[m], yt[m], method, tau))
            if not losses:
                continue
            opt.zero_grad(); torch.stack(losses).mean().backward(); opt.step()
    return model


def evaluate_kendall(model, X, y, groups):
    """Mean per-group Kendall tau between model scores and targets."""
    model.eval()
    with torch.no_grad():
        s = model(torch.from_numpy(X).to(DEVICE)).cpu().numpy()
    taus = []
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 2 or np.std(s[m]) < 1e-9 or np.std(y[m]) < 1e-9:
            continue
        taus.append(kendalltau(s[m], y[m]).correlation)
    return float(np.nanmean(taus))


# ----------------------------------------------------------------------------
# Monotone transforms
# ----------------------------------------------------------------------------
TRANSFORMS = ["identity", "gamma 1.5", "gamma 2.5", "gamma 4", "log1p", "exp",
              "per-feat random"]


def _gamma(v, g):
    v = v.astype(np.float64)
    a, b = v.min(), v.max()
    if b - a < 1e-12:
        return v
    return (b - a) * ((v - a) / (b - a)) ** g + a


def monotone_transform(X, name, seed=0):
    """Apply a strictly monotone per-feature transform. Returns float64."""
    Xo = X.copy().astype(np.float64)
    if name == "identity":
        return Xo
    if name == "per-feat random":
        rng = np.random.RandomState(seed)
        for j in range(Xo.shape[1]):
            Xo[:, j] = _gamma(Xo[:, j], rng.uniform(0.3, 4.0))
        return Xo
    if name.startswith("gamma"):
        g = float(name.split()[1])
        for j in range(Xo.shape[1]):
            Xo[:, j] = _gamma(Xo[:, j], g)
        return Xo
    if name == "log1p":
        for j in range(Xo.shape[1]):
            Xo[:, j] = np.log1p(Xo[:, j] - Xo[:, j].min() + 1e-6)
        return Xo
    if name == "exp":
        for j in range(Xo.shape[1]):
            v = Xo[:, j]
            nrm = (v - v.min()) / (v.max() - v.min() + 1e-9)
            Xo[:, j] = np.expm1(nrm * 2.0)
        return Xo
    raise ValueError(f"unknown transform: {name}")
