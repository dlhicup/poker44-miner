#!/usr/bin/env python3
"""EXPERIMENT: does a WIDER feature set for the (scale-invariant) rank branch
improve cross-date transfer?

Rationale: the transfer gate (2-SD mean-shift / variance-ratio / band) exists
to protect the RAW branch, whose trees split on ABSOLUTE feature values and
saturate when the live level shifts. But the RANK branch ranks each feature
WITHIN the request, so a pure location/scale shift is invisible to it — a
mean-shifted feature still orders chunks correctly. Today both branches use
the same 98 strictly-gated features. This tests whether giving the rank
branch a WIDE set (drop only truly degenerate features) raises LORO reward
and/or lowers cross-date std.

Compares the rank-branch GBDT alone under identical LORO folds:
  STRICT: the 98 transfer-gated features (current)
  WIDE:   all candidate features minus degenerate-live/degenerate-bench ones
  MID:    wide minus features with an EXTREME (>6 SD) mean shift (belt-and-braces)

Run:
    cd /home/client_7075_3/Projects/Poker44-subnet
    PYTHONPATH=. .venv/bin/python local_test/exp_wide_rank.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

REPO = Path("/home/client_7075_3/Projects/Poker44-subnet")
sys.path.insert(0, str(REPO))

from neurons.detector import CANDIDATE_FEATURES, percentile_ranks, _calibrate  # noqa: E402
from poker44.score.scoring import reward  # noqa: E402
import local_test.train_detector as T  # noqa: E402

TARGET_FPR = T.TARGET_HUMAN_FPR
FULL_MIN = T.FULL_SIZE_MIN_GROUPS
GBM = T.GBM_PARAMS


def rank_matrix(X):
    cols = [percentile_ranks([float(v) for v in X[:, j]]) for j in range(X.shape[1])]
    return np.asarray(cols, dtype=float).T


def _one_fold(date, releases_rank, releases_y_dates, yb):
    X_tr = np.vstack([releases_rank[d] for d, _ in releases_y_dates if d != date])
    y_tr = np.concatenate([yb[d] for d, _ in releases_y_dates if d != date])
    X_te, y_te = releases_rank[date], yb[date]
    oof = np.zeros(len(y_tr))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X_tr, y_tr):
        m = GradientBoostingClassifier(**GBM).fit(X_tr[tr], y_tr[tr])
        oof[te] = m.predict_proba(X_tr[te])[:, 1]
    thr = min(max(float(np.quantile(oof[y_tr == 0], 1.0 - TARGET_FPR)), 0.05), 0.95)
    model = GradientBoostingClassifier(**GBM).fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    pc = np.array([_calibrate(v, thr) for v in p])
    r, _ = reward(pc, y_te.astype(bool))
    return r


def loro_rank(releases_rank, releases_y_dates):
    """LORO reward for a rank-branch GBDT over the given per-release rank views.
    Folds run in parallel (6 workers) like the main trainer."""
    full_dates = [d for d, y in releases_y_dates if len(y) >= FULL_MIN]
    yb = dict(releases_y_dates)
    rewards = Parallel(n_jobs=6)(
        delayed(_one_fold)(date, releases_rank, releases_y_dates, yb)
        for date in full_dates)
    return np.array(rewards)


def build_colsets(releases, X_live, X_bench):
    """Return {name: [col indices]} for STRICT / WIDE / MID feature sets."""
    strict, _ = T.feature_transfer_gate(X_live, X_bench)
    strict_cols = [CANDIDATE_FEATURES.index(n) for n in strict]

    wide_cols, mid_cols = [], []
    for j, name in enumerate(CANDIDATE_FEATURES):
        live, bench = X_live[:, j], X_bench[:, j]
        ls, bs = float(np.std(live)), float(np.std(bench))
        if ls < 1e-6 or bs < 1e-6:
            continue  # degenerate on either side — useless even ranked
        wide_cols.append(j)
        shift = abs(float(np.mean(live)) - float(np.mean(bench))) / max(bs, 1e-9)
        if shift <= 6.0:
            mid_cols.append(j)
    return {"STRICT (98, current)": strict_cols,
            "WIDE (drop degenerate only)": wide_cols,
            "MID (wide, drop >6SD shift)": mid_cols}


def main():
    print("[load] releases (cached)...")
    releases = T.load_all_releases()
    for cd in [REPO / "local_test" / "captures",
               REPO / "local_test" / "captures_archive_20260717"]:
        if sorted(cd.glob("query_*.json")):
            T.CAPTURES_DIR = cd
            break
    _, _, X_live_full = T.load_live_capture_features()
    X_all = np.vstack([Xr for _, Xr, _ in releases])
    rng = np.random.RandomState(0)
    X_bench = X_all[rng.choice(len(X_all), size=min(T.GATE_BENCH_SAMPLE, len(X_all)), replace=False)]

    colsets = build_colsets(releases, X_live_full, X_bench)
    y_dates = [(d, y) for d, _, y in releases]

    print("\n== Rank-branch LORO by feature-set width ==", flush=True)
    base = None
    for name, cols in colsets.items():
        print(f"  [running] {name} ({len(cols)} feat)...", flush=True)
        rank_views = {d: rank_matrix(Xr[:, cols]) for d, Xr, _ in releases}
        r = loro_rank(rank_views, y_dates)
        score = r.mean() - 0.5 * r.std()
        tag = ""
        if base is None:
            base = r
        else:
            tag = f"  d(mean)={r.mean()-base.mean():+.4f} d(std)={r.std()-base.std():+.4f}"
        print(f"  {name:32s} nfeat={len(cols):3d}  mean={r.mean():.4f} "
              f"std={r.std():.4f} min={r.min():.4f} mean-0.5std={score:.4f}{tag}")

    print("\n  (base = STRICT; want WIDE/MID mean up and/or std down to justify a wider rank branch)")


if __name__ == "__main__":
    main()
