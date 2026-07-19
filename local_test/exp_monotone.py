#!/usr/bin/env python3
"""EXPERIMENT: does sign-stability monotone constraining reduce cross-date
transfer variance? (Weapon-5 prototype — decide BEFORE the HistGBM export
rewrite.)

Thesis (from the top-miner teardown): our offline model already equals the
leaders' (~0.92 AP); the gap is live-transfer VARIANCE — how stable the
within-request chunk order stays across the benchmark->live shift. The top
codebases (uid239, uid172) lock feature-effect SIGN directions that are
stable across dates so a live marginal shift can't flip a split's meaning.

This measures whether that actually lowers cross-date reward STD (the
transfer proxy) on OUR data, comparing three raw learners under identical
leave-one-release-out folds:
  A. GBM   = current GradientBoostingClassifier (no monotone support)
  B. HGB   = HistGradientBoostingClassifier, UNCONSTRAINED
  C. HGB+M = HistGradientBoostingClassifier, sign-stability monotone_cst

Decision rule printed at the end: build the production HistGBM path only if
C reduces LORO std (and/or raises mean-0.5*std) meaningfully vs A.

Run:
    cd /home/client_7075_3/Projects/Poker44-subnet
    PYTHONPATH=. .venv/bin/python local_test/exp_monotone.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

REPO = Path("/home/client_7075_3/Projects/Poker44-subnet")
sys.path.insert(0, str(REPO))

from neurons.detector import CANDIDATE_FEATURES, _calibrate  # noqa: E402
from poker44.score.scoring import reward  # noqa: E402
import local_test.train_detector as T  # noqa: E402

TARGET_FPR = T.TARGET_HUMAN_FPR
FULL_MIN = T.FULL_SIZE_MIN_GROUPS

# Sign-stability parameters (uid239-style)
SIGN_MIN_DATES_FRAC = 0.70   # feature's rho sign must agree on >= this frac of dates
SIGN_RHO_FLOOR = 0.03        # ignore near-zero correlations when voting the sign
GBM_PARAMS = T.GBM_PARAMS
HGB_PARAMS = dict(max_iter=400, learning_rate=0.05, max_depth=3,
                  l2_regularization=1.0, random_state=0)


def _thr(oof, y):
    t = float(np.quantile(oof[y == 0], 1.0 - TARGET_FPR))
    return min(max(t, 0.05), 0.95)


def _reward_at(p_te, y_te, thr):
    p = np.array([_calibrate(v, thr) for v in p_te])
    r, _ = reward(p, y_te.astype(bool))
    return r


def sign_stability_constraints(releases):
    """monotonic_cst vector over the gated feature order: +1/-1 for features
    whose per-date Spearman(feature,label) sign is stable across dates, else
    0. Uses only full-size releases (enough labels for a stable rho)."""
    full = [(d, X, y) for d, X, y in releases if len(y) >= FULL_MIN]
    ncol = full[0][1].shape[1]
    signs = np.zeros((len(full), ncol))
    for i, (_, X, y) in enumerate(full):
        for j in range(ncol):
            col = X[:, j]
            if np.std(col) < 1e-12:
                continue
            rho = spearmanr(col, y).statistic
            if np.isfinite(rho) and abs(rho) >= SIGN_RHO_FLOOR:
                signs[i, j] = np.sign(rho)
    cst = np.zeros(ncol, dtype=int)
    n_dates = len(full)
    for j in range(ncol):
        votes = signs[:, j]
        pos = np.sum(votes > 0)
        neg = np.sum(votes < 0)
        if pos >= SIGN_MIN_DATES_FRAC * n_dates and pos > neg:
            cst[j] = 1
        elif neg >= SIGN_MIN_DATES_FRAC * n_dates and neg > pos:
            cst[j] = -1
    return cst


def loro(releases, make_model, monotone=None):
    """Leave-one-release-out cross-date reward, threshold from train-side OOF."""
    full = [(d, X, y) for d, X, y in releases if len(y) >= FULL_MIN]
    rewards = []
    for date, X_te, y_te in full:
        X_tr = np.vstack([X for d, X, _ in releases if d != date])
        y_tr = np.concatenate([y for d, _, y in releases if d != date])
        # train-side OOF threshold
        oof = np.zeros(len(y_tr))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X_tr, y_tr):
            m = make_model(monotone)
            m.fit(X_tr[tr], y_tr[tr])
            oof[te] = m.predict_proba(X_tr[te])[:, 1]
        thr = _thr(oof, y_tr)
        model = make_model(monotone)
        model.fit(X_tr, y_tr)
        p_te = model.predict_proba(X_te)[:, 1]
        rewards.append(_reward_at(p_te, y_te, thr))
    return np.array(rewards)


def main():
    print("[load] featurizing releases (parity-transformed, cached)...")
    releases = T.load_all_releases()
    # gate exactly as production (needs a live capture; use newest available
    # or fall back to the archived captures if the fresh dir is still empty)
    cap_dirs = [REPO / "local_test" / "captures",
                REPO / "local_test" / "captures_archive_20260717"]
    caps = []
    for cd in cap_dirs:
        caps = sorted(cd.glob("query_*.json"))
        if caps:
            T.CAPTURES_DIR = cd
            break
    _, live_chunks, X_live = T.load_live_capture_features()
    X_all = np.vstack([Xr for _, Xr, _ in releases])
    rng = np.random.RandomState(0)
    idx = rng.choice(len(X_all), size=min(T.GATE_BENCH_SAMPLE, len(X_all)), replace=False)
    features, _ = T.feature_transfer_gate(X_live, X_all[idx])
    cols = [CANDIDATE_FEATURES.index(n) for n in features]
    releases = [(d, Xr[:, cols], yr) for d, Xr, yr in releases]
    print(f"[gate] {len(features)} features survive")

    cst = sign_stability_constraints(releases)
    n_pos = int(np.sum(cst > 0)); n_neg = int(np.sum(cst < 0)); n_free = int(np.sum(cst == 0))
    print(f"[signs] monotone constraints: +1={n_pos}  -1={n_neg}  free={n_free} "
          f"(of {len(cst)}; stable if sign agrees on >={SIGN_MIN_DATES_FRAC:.0%} of dates)")

    def mk_gbm(_):  return GradientBoostingClassifier(**GBM_PARAMS)
    def mk_hgb(_):  return HistGradientBoostingClassifier(**HGB_PARAMS)
    def mk_hgb_m(m): return HistGradientBoostingClassifier(monotonic_cst=m, **HGB_PARAMS)

    print("\n[loro] running three learners over identical folds...")
    rA = loro(releases, mk_gbm)
    rB = loro(releases, mk_hgb)
    rC = loro(releases, mk_hgb_m, monotone=cst)

    def summ(tag, r):
        score = r.mean() - 0.5 * r.std()
        print(f"  {tag:24s} mean={r.mean():.4f}  std={r.std():.4f}  "
              f"min={r.min():.4f}  mean-0.5std={score:.4f}")
        return score

    print("\n== Cross-date (LORO) transfer stability ==")
    sA = summ("A GBM (current)", rA)
    sB = summ("B HistGBM unconstrained", rB)
    sC = summ("C HistGBM + monotone", rC)

    print("\n== Decision ==")
    dstd = rC.std() - rA.std()
    dscore = sC - sA
    print(f"  monotone vs current:  d(std)={dstd:+.4f} (want negative)  "
          f"d(mean-0.5std)={dscore:+.4f} (want positive)")
    verdict = ("BUILD — monotone reduces cross-date variance"
               if (dstd < -0.002 or dscore > 0.003)
               else "SKIP — no meaningful transfer-variance reduction on our data")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
