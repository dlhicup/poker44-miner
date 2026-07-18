#!/usr/bin/env python3
"""Train the Poker44 bot detector and emit neurons/detector_params.py.

MODEL (v3): gradient-boosted trees on the expanded candidate feature set
(neurons.detector.CANDIDATE_FEATURES, 177 names), filtered by a live-vs-
benchmark FEATURE TRANSFER GATE so only features whose live distribution
matches the (parity-transformed) benchmark distribution are trained on.
Chunk-size subsampling to 35 hands and the <5-action hand filter live
INSIDE extract_features, so train and serve share them automatically.

DATA: every release banked in local_test/data/release_*.json (real_eval.py
banks the newest release daily; backfill_releases.py recovers history).
Before featurizing, every benchmark hand is re-canonicalized through
poker44.validator.payload_view.build_miner_payload_hand (see
load_all_releases for why). AUGMENTATION: a fraction of each release's real
human groups is "roboticized" (bot-like regularity injected) and added to
the TRAINING side as hard positives — see the ROBO_* block; disable with
--no-robo. Held-out validation folds never contain augmented data.

VALIDATION: leave-one-release-out over the FULL-SIZE releases (>=100
groups). Small pilot releases still contribute training data to the final
fit, but are too noisy to serve as validation folds. All reported numbers
and the calibration threshold come from out-of-sample predictions only —
tree models overfit train probabilities badly, so calibrating on train
scores collapses in production (verified: reward 0.85 -> 0.75).

Run:
    cd /home/client_7075_3/Projects/Poker44-subnet
    PYTHONPATH=. .venv/bin/python local_test/train_detector.py
"""
from __future__ import annotations

import copy
import json
import sys
import zlib
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path("/home/client_7075_3/Projects/Poker44-subnet")
sys.path.insert(0, str(REPO))

from neurons.detector import (  # noqa: E402
    CANDIDATE_FEATURES,
    extract_features,
    percentile_ranks,
    _calibrate,
)
from poker44.score.scoring import reward  # noqa: E402
from poker44.validator.payload_view import build_miner_payload_hand  # noqa: E402

DATA_DIR = REPO / "local_test" / "data"
CAPTURES_DIR = REPO / "local_test" / "captures"
# Bumped whenever the featurisation input changes; stale caches are ignored.
# v4: expanded CANDIDATE_FEATURES matrix + parity re-canonicalization of the
# benchmark hands through build_miner_payload_hand (see load_all_releases).
# v5: action n-gram token features added to CANDIDATE_FEATURES.
CACHE_VERSION = 5
TARGET_HUMAN_FPR = 0.07     # <=7% humans over 0.5 (validator gate allows 10%)
FULL_SIZE_MIN_GROUPS = 100  # releases smaller than this are pilot-era noise
GATE_BENCH_SAMPLE = 200     # benchmark chunks sampled for the transfer gate
GBM_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=0,
)
# Heterogeneous ensemble members (the structure every top miner runs):
# decorrelated tree families generalize better than one GBDT on a small,
# shifting training set. Depth/leaf caps keep the pure-python export a few MB.
FOREST_PARAMS = dict(
    n_estimators=200,
    max_depth=7,
    min_samples_leaf=20,
    random_state=0,
    n_jobs=1,
)
# Blend-weight grid: all (w_gb, w_et, w_rf) on the 0.1-step simplex.
BLEND_GRID = [
    (a / 10, b / 10, (10 - a - b) / 10)
    for a in range(11)
    for b in range(11 - a)
]
# Raw-vs-rank branch blend grid. The rank-branch experiment showed a smooth
# plateau (every w in 0.4-0.8 beat both single branches), so a coarse grid is
# enough and less overfit-prone.
W_RAW_GRID = (0.3, 0.4, 0.5, 0.6, 0.7)
PARAMS_PATH = REPO / "neurons" / "detector_params.py"

# Gate acceptance criteria (reported honestly; no workarounds if they fail).
GATE1_LORO_MIN = 0.84
GATE2_STD_MIN = 0.08
GATE2_MID_FRAC_MIN = 0.5
GATE2_MID_LO, GATE2_MID_HI = 0.15, 0.70

# ---- Roboticized hard positives (data augmentation) ------------------------
# For a fraction of each release's REAL HUMAN groups, generate one
# "roboticized" variant: the same session with bot-like regularity injected
# (repeated action-line templates, street-modal action smoothing, bet-size
# quantization to a small grid). The variant is labeled BOT and added to the
# TRAINING side only — LORO held-out folds stay pure benchmark data, so the
# reported numbers remain comparable across model generations. These are hard
# positives: 55-75% human residue with robotic regularity layered on top,
# which sharpens the decision boundary exactly where well-disguised real bots
# operate (the benchmark's own synthetic bots are mostly easy positives).
ROBO_ENABLED = "--no-robo" not in sys.argv
ROBO_VERSION = 1            # bump to invalidate cached robo features
ROBO_FRACTION = 0.5         # fraction of each release's human groups augmented
ROBO_SEVERITIES = (0.25, 0.45)  # cycled deterministically across picked groups


def _robo_rng(date: str, tag: str) -> np.random.RandomState:
    """Deterministic per-release RNG (no global seed coupling)."""
    return np.random.RandomState(zlib.crc32(f"robo|{tag}|{date}".encode()) & 0x7FFFFFFF)


def _remap_template_actions(tmpl_actions: list, hand: dict) -> list:
    """Copy a template action line onto `hand`, remapping the template's
    actor seats onto the hand's own seats (order of first appearance,
    cycling). Keeps the repeated PATTERN while staying structurally
    plausible for this table."""
    acts = copy.deepcopy(tmpl_actions)
    own_seats = sorted({
        int(p.get("seat")) for p in hand.get("players", [])
        if isinstance(p, dict) and p.get("seat")
    })
    if not own_seats:
        return acts
    tmpl_seats: list = []
    for act in acts:
        seat = act.get("actor_seat")
        if seat is not None and seat not in tmpl_seats:
            tmpl_seats.append(seat)
    mapping = {s: own_seats[i % len(own_seats)] for i, s in enumerate(tmpl_seats)}
    for act in acts:
        if act.get("actor_seat") in mapping:
            act["actor_seat"] = mapping[act["actor_seat"]]
    return acts


def _roboticize_group(hands: list, severity: float, rng: np.random.RandomState) -> list:
    """Return a bot-labeled variant of a real human group (hard positive).

    Three graded transforms, each applied with probability ~ `severity`:
      (a) template repetition — a small pool of action lines gets reused
          across hands (raises n-gram concentration, lowers cross-hand
          variance: the classic scripted-bot signature);
      (b) street-modal action smoothing — actions drift toward the group's
          modal action for that street (lowers action entropy);
      (c) bet-size quantization — nonzero amounts snap to a 3-level grid
          from the group's own quantiles (kills bet-size uniqueness).
    Action COUNTS are never changed, so the <5-action hand filter and the
    5-8 action live windowing see the same structure as the original."""
    hands = copy.deepcopy(hands)
    usable = [
        h for h in hands
        if isinstance(h, dict) and isinstance(h.get("actions"), list) and h["actions"]
    ]
    if len(usable) < 4:
        return hands

    # (a) template repetition
    n_templates = max(2, int(round(6 * (1.0 - severity))))
    t_idx = rng.choice(len(usable), size=min(n_templates, len(usable)), replace=False)
    templates = [copy.deepcopy(usable[i]["actions"]) for i in t_idx]
    for hand in usable:
        if rng.random_sample() < severity:
            tmpl = templates[rng.randint(len(templates))]
            hand["actions"] = _remap_template_actions(tmpl, hand)

    # street-modal actions + canonical amount grid, from the (post-template)
    # group itself so every robo group is self-consistent
    by_street: dict = {}
    amounts: list = []
    normed: list = []
    for hand in usable:
        for act in hand["actions"]:
            a_type = str(act.get("action_type") or "")
            if a_type:
                by_street.setdefault(str(act.get("street") or ""), []).append(a_type)
            if float(act.get("amount") or 0.0) > 0:
                amounts.append(float(act["amount"]))
            if float(act.get("normalized_amount_bb") or 0.0) > 0:
                normed.append(float(act["normalized_amount_bb"]))
    modal = {}
    for street, types in by_street.items():
        vals, counts = np.unique(types, return_counts=True)
        modal[street] = str(vals[np.argmax(counts)])
    canon_amt = np.quantile(amounts, [0.25, 0.5, 0.75]).tolist() if amounts else []
    canon_nrm = np.quantile(normed, [0.25, 0.5, 0.75]).tolist() if normed else []

    for hand in usable:
        for act in hand["actions"]:
            # (b) street-modal action smoothing
            street = str(act.get("street") or "")
            if street in modal and rng.random_sample() < severity * 0.6:
                new_type = modal[street]
                act["action_type"] = new_type
                if new_type in ("fold", "check"):
                    act["amount"] = 0
                    act["normalized_amount_bb"] = 0
                    act["raise_to"] = None
                    act["call_to"] = None
            # (c) bet-size quantization
            if canon_amt and float(act.get("amount") or 0.0) > 0 \
                    and rng.random_sample() < severity:
                val = float(act["amount"])
                act["amount"] = min(canon_amt, key=lambda c: abs(c - val))
                if canon_nrm and float(act.get("normalized_amount_bb") or 0.0) > 0:
                    nv = float(act["normalized_amount_bb"])
                    act["normalized_amount_bb"] = min(canon_nrm, key=lambda c: abs(c - nv))
    return hands


def load_robo_features() -> dict:
    """Roboticized hard positives per release: {date: X_robo} over the full
    CANDIDATE_FEATURES matrix. Deterministic (per-release seeded RNG) and
    cached per (CACHE_VERSION, ROBO_VERSION). Groups are parity-transformed
    BEFORE roboticization so the transforms operate on the exact miner-
    visible view the model trains and serves on."""
    if not ROBO_ENABLED:
        return {}
    robo: dict = {}
    for path in sorted(DATA_DIR.glob("release_*.json")):
        date = path.stem.replace("release_", "")
        cache_path = DATA_DIR / f"features_robo_{date}.json"
        X = None
        if cache_path.exists():
            try:
                with open(cache_path) as fh:
                    cached = json.load(fh)
                if (cached.get("features") == CANDIDATE_FEATURES
                        and cached.get("cache_version") == CACHE_VERSION
                        and cached.get("robo_version") == ROBO_VERSION):
                    X = np.asarray(cached["X"], dtype=float)
            except Exception:
                X = None
        if X is None:
            with open(path) as fh:
                data = json.load(fh)
            labels = np.asarray(data["labels"], dtype=int)
            human_idx = np.flatnonzero(labels == 0)
            if not len(human_idx):
                robo[date] = np.zeros((0, len(CANDIDATE_FEATURES)))
                continue
            sel_rng = _robo_rng(date, "select")
            n_pick = max(1, int(round(ROBO_FRACTION * len(human_idx))))
            picked = sel_rng.choice(human_idx, size=n_pick, replace=False)
            rows = []
            for k, gi in enumerate(sorted(picked.tolist())):
                grp = [build_miner_payload_hand(h) for h in data["groups"][gi]]
                severity = ROBO_SEVERITIES[k % len(ROBO_SEVERITIES)]
                grp = _roboticize_group(grp, severity, _robo_rng(date, f"g{gi}"))
                feats = extract_features(grp)
                rows.append([feats[k2] for k2 in CANDIDATE_FEATURES])
            X = np.asarray(rows, dtype=float)
            with open(cache_path, "w") as fh:
                json.dump({"features": CANDIDATE_FEATURES,
                           "cache_version": CACHE_VERSION,
                           "robo_version": ROBO_VERSION,
                           "X": X.tolist()}, fh)
        robo[date] = X
    return robo


def load_all_releases():
    """Load every banked release: list of (date, X, y) with X over the full
    CANDIDATE_FEATURES matrix. Feature caching: each immutable release is
    featurized once per CACHE_VERSION.

    PARITY TRANSFORM: before featurizing, every benchmark hand is passed
    through poker44.validator.payload_view.build_miner_payload_hand. Live
    payloads are windowed to 5-8 visible actions by the CURRENT
    canonicalizer, while the benchmark releases were produced by an older
    canonicalizer (hands carry 1-22 actions); re-canonicalizing the
    benchmark hands restores train/serve parity (verified: 5-8 action
    bucket coverage 44.7% -> 78.2%). This re-censoring is intentional NOW
    because no bet-size/amount features are used anymore — the re-bucketing
    and re-noising that made a second censor pass harmful before (v2 note)
    only corrupted amount-derived features, which the transfer gate
    excludes by construction.
    """
    paths = sorted(DATA_DIR.glob("release_*.json"))
    if len(paths) < 2:
        sys.exit(f"Need >=2 banked releases in {DATA_DIR}; found {len(paths)}.")
    releases = []
    for path in paths:
        date = path.stem.replace("release_", "")
        cache_path = DATA_DIR / f"features_{date}.json"
        X = y = None
        if cache_path.exists():
            try:
                with open(cache_path) as fh:
                    cached = json.load(fh)
                if (cached.get("features") == CANDIDATE_FEATURES
                        and cached.get("cache_version") == CACHE_VERSION):
                    X = np.asarray(cached["X"], dtype=float)
                    y = np.asarray(cached["y"], dtype=int)
            except Exception:
                X = y = None
        if X is None:
            with open(path) as fh:
                data = json.load(fh)
            y = np.asarray(data["labels"], dtype=int)
            rows = []
            for grp in data["groups"]:
                # Parity transform (see docstring): re-canonicalize each
                # benchmark hand exactly as the live validator censors hands.
                grp = [build_miner_payload_hand(hand) for hand in grp]
                feats = extract_features(grp)
                rows.append([feats[k] for k in CANDIDATE_FEATURES])
            X = np.asarray(rows, dtype=float)
            with open(cache_path, "w") as fh:
                json.dump({"features": CANDIDATE_FEATURES,
                           "cache_version": CACHE_VERSION,
                           "X": X.tolist(), "y": y.tolist()}, fh)
        releases.append((date, X, y))
    return releases


def load_live_capture_features():
    """Featurize the newest captured live validator payload (100 chunks).

    Captured chunks are ALREADY miner-visible live payloads (the current
    canonicalizer produced them), so they are featurized as-delivered.
    Returns (capture_path, chunks, X_live over CANDIDATE_FEATURES).
    """
    captures = sorted(CAPTURES_DIR.glob("query_*.json"))
    if not captures:
        sys.exit(f"No live capture query_*.json found in {CAPTURES_DIR}.")
    capture_path = captures[-1]
    with open(capture_path) as fh:
        capture = json.load(fh)
    chunks = capture["chunks"]
    rows = []
    for chunk in chunks:
        feats = extract_features(chunk)
        rows.append([feats[k] for k in CANDIDATE_FEATURES])
    return capture_path, chunks, np.asarray(rows, dtype=float)


def feature_transfer_gate(X_live: np.ndarray, X_bench: np.ndarray):
    """Drop candidate features whose LIVE distribution does not transfer.

    A feature survives only if it varies on both populations AND its live
    mean sits inside (a slightly widened) benchmark support with a
    standardized mean shift <= 4 benchmark SDs. Everything else is exactly
    the failure mode that sank the v1 bet-size features: strong on the
    synthetic benchmark, out-of-distribution live, saturating every score.
    Returns (survivors, dropped) where dropped is a list of dicts.
    """
    survivors, dropped = [], []
    for j, name in enumerate(CANDIDATE_FEATURES):
        live = X_live[:, j]
        bench = X_bench[:, j]
        live_std = float(np.std(live))
        bench_std = float(np.std(bench))
        live_mean = float(np.mean(live))
        bench_mean = float(np.mean(bench))
        bench_q05 = float(np.quantile(bench, 0.05))
        bench_q95 = float(np.quantile(bench, 0.95))
        band = bench_q95 - bench_q05
        reasons = []
        if live_std < 1e-6:
            reasons.append("degenerate live (std<1e-6)")
        if bench_std < 1e-6:
            reasons.append("degenerate bench (std<1e-6)")
        shift = abs(live_mean - bench_mean) / max(bench_std, 1e-9)
        if shift > 2.0:
            reasons.append(f"mean shift {shift:.1f} SD > 2")
        # Central-band containment: the live mean must sit inside the
        # benchmark's q05-q95 band (+15% slack) — full min/max support over a
        # 200-chunk sample is far too permissive and let a residual
        # multivariate shift through (v3 live median 0.81).
        if band > 1e-9 and not (
            bench_q05 - 0.15 * band <= live_mean <= bench_q95 + 0.15 * band
        ):
            reasons.append("live mean outside bench q05-q95 band+15%")
        # Variance-ratio canary: a live spread collapse (or explosion) means
        # the feature behaves differently live even if its mean matches.
        if live_std >= 1e-6 and bench_std >= 1e-6:
            ratio = live_std / bench_std
            if ratio < 0.33 or ratio > 3.0:
                reasons.append(f"variance ratio {ratio:.2f} outside [0.33, 3]")
        if reasons:
            dropped.append({
                "name": name,
                "reasons": "; ".join(reasons),
                "live_mean": live_mean,
                "live_std": live_std,
                "bench_mean": bench_mean,
                "bench_std": bench_std,
            })
        else:
            survivors.append(name)

    print(f"\n== Feature transfer gate: {len(survivors)} kept, "
          f"{len(dropped)} dropped of {len(CANDIDATE_FEATURES)} ==")
    if dropped:
        w = max(len(d["name"]) for d in dropped)
        print(f"  {'feature'.ljust(w)}  live_mean  bench_mean  bench_std  reason")
        for d in dropped:
            print(f"  {d['name'].ljust(w)}  {d['live_mean']:9.4f}  "
                  f"{d['bench_mean']:10.4f}  {d['bench_std']:9.4f}  {d['reasons']}")
    return survivors, dropped


def oof_threshold(X: np.ndarray, y: np.ndarray) -> float:
    """Calibration threshold from OUT-OF-FOLD human scores (never train
    scores — GBM train probabilities saturate near 0/1 and would collapse
    the calibration in production)."""
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        model = GradientBoostingClassifier(**GBM_PARAMS).fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
    threshold = float(np.quantile(oof[y == 0], 1.0 - TARGET_HUMAN_FPR))
    return min(max(threshold, 0.05), 0.95)


def _fit_members(X: np.ndarray, y: np.ndarray) -> dict:
    """Fit the three ensemble members."""
    return {
        "gb": GradientBoostingClassifier(**GBM_PARAMS).fit(X, y),
        "et": ExtraTreesClassifier(**FOREST_PARAMS).fit(X, y),
        "rf": RandomForestClassifier(**FOREST_PARAMS).fit(X, y),
    }


def _oof_member_probs(X: np.ndarray, y: np.ndarray) -> dict:
    """5-fold out-of-fold class-1 probabilities for each member family."""
    oof = {k: np.zeros(len(y)) for k in ("gb", "et", "rf")}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        fold_models = _fit_members(X[tr], y[tr])
        for key, model in fold_models.items():
            oof[key][te] = model.predict_proba(X[te])[:, 1]
    return oof


def _tune_blend(oof: dict, y: np.ndarray) -> tuple:
    """Pick blend weights + calibration threshold on OOF predictions,
    maximizing the authoritative validator reward. Returns
    (weights_dict, threshold, oof_reward)."""
    best = (None, 0.5, -1.0)
    for w_gb, w_et, w_rf in BLEND_GRID:
        blend = w_gb * oof["gb"] + w_et * oof["et"] + w_rf * oof["rf"]
        thr = float(np.quantile(blend[y == 0], 1.0 - TARGET_HUMAN_FPR))
        thr = min(max(thr, 0.05), 0.95)
        p = np.array([_calibrate(v, thr) for v in blend])
        rew, _ = reward(p, y.astype(bool))
        if rew > best[2]:
            best = ({"gb": w_gb, "et": w_et, "rf": w_rf}, thr, rew)
    return best


def _blend_probs(models: dict, weights: dict, X: np.ndarray) -> np.ndarray:
    return sum(
        weights[k] * models[k].predict_proba(X)[:, 1] for k in ("gb", "et", "rf")
    )


def _rank_matrix(X: np.ndarray) -> np.ndarray:
    """Within-group percentile-rank transform, one column at a time, using the
    SAME pure-python ranker the miner serves with (imported from
    neurons.detector) so train and serve match by construction. Applied per
    RELEASE at training time; the serving analog ranks within the incoming
    100-chunk request."""
    cols = [percentile_ranks([float(v) for v in X[:, j]]) for j in range(X.shape[1])]
    return np.asarray(cols, dtype=float).T


def _gb_oof(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """5-fold OOF probabilities for a single GBDT (rank-branch helper)."""
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        model = GradientBoostingClassifier(**GBM_PARAMS).fit(X[tr], y[tr])
        oof[te] = model.predict_proba(X[te])[:, 1]
    return oof


def _tune_branch_blend(oof_raw: np.ndarray, oof_rank: np.ndarray, y: np.ndarray):
    """Pick w_raw + threshold on OOF predictions of the two branches,
    maximizing the authoritative reward. Returns (w_raw, threshold, reward)."""
    best = (0.5, 0.5, -1.0)
    for w in W_RAW_GRID:
        blend = w * oof_raw + (1.0 - w) * oof_rank
        thr = float(np.quantile(blend[y == 0], 1.0 - TARGET_HUMAN_FPR))
        thr = min(max(thr, 0.05), 0.95)
        p = np.array([_calibrate(v, thr) for v in blend])
        rew, _ = reward(p, y.astype(bool))
        if rew > best[2]:
            best = (w, thr, rew)
    return best


# ---- Human Manifold branch (one-class, real humans only) -------------------

def _fit_manifold(X_humans: np.ndarray) -> dict:
    """Fit the human-typicality model on REAL human chunks only.

    Ledoit-Wolf shrinkage covariance -> Mahalanobis distance from the human
    center; d0/s map distance to a probability via sigmoid((d - d0) / s).
    d0/s come from the human distance distribution itself (q90 anchor,
    spread-scaled) — synthetic bots never shape any parameter here.
    """
    lw = LedoitWolf().fit(X_humans)
    mu = lw.location_
    prec = lw.precision_
    diff = X_humans - mu
    d = np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", diff, prec, diff)))
    d0 = float(np.quantile(d, 0.90))
    s = float(max(1e-6, (np.quantile(d, 0.99) - np.quantile(d, 0.50)) / 3.0))
    return {"mu": mu, "prec": prec, "d0": d0, "s": s}


def _manifold_dists(man: dict, X: np.ndarray) -> np.ndarray:
    diff = X - man["mu"]
    return np.sqrt(np.maximum(0.0, np.einsum("ij,jk,ik->i", diff, man["prec"], diff)))


def _oof_manifold_dists(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """5-fold OOF manifold DISTANCES (fit on train-fold HUMANS only)."""
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
        man = _fit_manifold(X[tr][y[tr] == 0])
        oof[te] = _manifold_dists(man, X[te])
    return oof


def _ranks_within_blocks(values: np.ndarray, block_sizes: list) -> np.ndarray:
    """Percentile-rank `values` within each contiguous block (release).

    Mirrors serving, where the manifold's distances are ranked within the
    incoming request: absolute Mahalanobis distances shift between benchmark
    and live (joint covariance drift — a fixed sigmoid saturated to 1.0 on
    every live chunk), but the ordering within one batch survives."""
    out = np.zeros(len(values))
    start = 0
    for size in block_sizes:
        block = values[start:start + size]
        out[start:start + size] = np.asarray(percentile_ranks([float(v) for v in block]))
        start += size
    return out


def _tune_tri_blend(
    oof_raw: np.ndarray,
    oof_rank: np.ndarray,
    oof_man: np.ndarray,
    y: np.ndarray,
):
    """Tune (raw, rank, manifold) weights + threshold on OOF predictions.

    0.1-step simplex with the manifold capped at 0.4 — a one-class branch
    complements the supervised branches, it must not dominate them.
    Returns (weights_dict, threshold, oof_reward)."""
    best = ({"raw": 0.6, "rank": 0.4, "manifold": 0.0}, 0.5, -1.0)
    for wr10 in range(0, 11):
        for wk10 in range(0, 11 - wr10):
            wm10 = 10 - wr10 - wk10
            if wm10 > 4:
                continue
            blend = (
                (wr10 / 10) * oof_raw
                + (wk10 / 10) * oof_rank
                + (wm10 / 10) * oof_man
            )
            thr = float(np.quantile(blend[y == 0], 1.0 - TARGET_HUMAN_FPR))
            thr = min(max(thr, 0.05), 0.95)
            p = np.array([_calibrate(v, thr) for v in blend])
            rew, _ = reward(p, y.astype(bool))
            if rew > best[2]:
                best = (
                    {"raw": wr10 / 10, "rank": wk10 / 10, "manifold": wm10 / 10},
                    thr,
                    rew,
                )
    return best


def report(tag: str, p: np.ndarray, y: np.ndarray) -> None:
    rew, met = reward(p, y)
    print(f"  {tag}: AUC={roc_auc_score(y, p):.3f} reward={rew:.4f} "
          f"ap={met['ap_score']:.3f} recall@fpr5={met['bot_recall']:.3f} "
          f"bots>=0.5 {np.mean(p[y == 1] >= 0.5):.1%} "
          f"humans<0.5 {np.mean(p[y == 0] < 0.5):.1%}")


def main() -> None:
    releases = load_all_releases()
    n_total = sum(len(y) for _, _, y in releases)
    print(f"[data] {len(releases)} releases banked, {n_total} labeled groups "
          f"(candidate matrix: {len(CANDIDATE_FEATURES)} features)")

    # ---- feature transfer gate (live capture vs parity-transformed bench) --
    capture_path, live_chunks, X_live = load_live_capture_features()
    print(f"[gate] live capture: {capture_path.name} ({len(live_chunks)} chunks)")
    X_all_cand = np.vstack([Xr for _, Xr, _ in releases])
    sample_rng = np.random.RandomState(0)
    bench_idx = sample_rng.choice(
        len(X_all_cand), size=min(GATE_BENCH_SAMPLE, len(X_all_cand)),
        replace=False)
    features, dropped = feature_transfer_gate(X_live, X_all_cand[bench_idx])
    if not features:
        sys.exit("Transfer gate dropped every candidate feature — aborting.")
    cols = [CANDIDATE_FEATURES.index(name) for name in features]
    releases = [(d, Xr[:, cols], yr) for d, Xr, yr in releases]
    X_live = X_live[:, cols]
    # Rank-branch training views: percentile ranks WITHIN each release (the
    # serving analog ranks within the incoming request).
    releases_rank = {d: _rank_matrix(Xr) for d, Xr, _ in releases}

    # ---- roboticized hard positives (TRAIN-side augmentation only) ----
    # aug[d] = (X_aug, y_aug, rank_aug): the release's real groups plus its
    # robo variants (labeled bot), with the rank view computed over the
    # COMBINED block — mimicking a live request that contains subtle bots.
    # Held-out LORO views never touch aug (test purity).
    robo_feats = {d: Xb[:, cols] for d, Xb in load_robo_features().items()}
    aug = {}
    n_robo = 0
    for d, Xr, yr in releases:
        Xb = robo_feats.get(d)
        if Xb is not None and len(Xb):
            X_aug = np.vstack([Xr, Xb])
            y_aug = np.concatenate([yr, np.ones(len(Xb), dtype=int)])
            n_robo += len(Xb)
        else:
            X_aug, y_aug = Xr, yr
        aug[d] = (X_aug, y_aug, _rank_matrix(X_aug))
    if ROBO_ENABLED:
        print(f"[robo] {n_robo} roboticized hard positives across "
              f"{sum(1 for v in robo_feats.values() if len(v))} releases "
              f"(fraction={ROBO_FRACTION}, severities={ROBO_SEVERITIES}; "
              f"train-side only)")

    full = [(d, X, y) for d, X, y in releases if len(y) >= FULL_SIZE_MIN_GROUPS]
    print(f"[data] {len(full)} full-size validation folds | "
          f"{len(features)} gated features")

    # ---- honest leave-one-release-out over the full-size releases ----
    print("\n== Leave-one-release-out (full-size releases; "
          "threshold from train-side OOF only) ==")
    def _loro_fold(date, X_te):
        """One fold of the full three-branch pipeline, train-side-only tuning:
        raw ensemble (OOF-tuned internal weights) + rank-branch GBDT
        (within-release rank views) + human-manifold one-class (fit on
        train-side HUMANS only) + tri-blend weights and threshold from
        train-side OOF. Scores the held-out date exactly as serving would."""
        # Train side uses the AUGMENTED per-release blocks (real + robo);
        # the held-out release stays pure benchmark data, ranked among its
        # own real groups only.
        train_dates = [d for d, _, _ in releases if d != date]
        X_tr = np.vstack([aug[d][0] for d in train_dates])
        y_tr = np.concatenate([aug[d][1] for d in train_dates])
        X_tr_rank = np.vstack([aug[d][2] for d in train_dates])
        X_te_rank = releases_rank[date]

        models = _fit_members(X_tr, y_tr)
        oof_members = _oof_member_probs(X_tr, y_tr)
        weights, _, _ = _tune_blend(oof_members, y_tr)
        oof_raw = sum(weights[k] * oof_members[k] for k in ("gb", "et", "rf"))

        rank_model = GradientBoostingClassifier(**GBM_PARAMS).fit(X_tr_rank, y_tr)
        oof_rank = _gb_oof(X_tr_rank, y_tr)

        man = _fit_manifold(X_tr[y_tr == 0])
        tr_sizes = [len(aug[d][1]) for d in train_dates]
        oof_man = _ranks_within_blocks(_oof_manifold_dists(X_tr, y_tr), tr_sizes)

        tri_w, thr, _ = _tune_tri_blend(oof_raw, oof_rank, oof_man, y_tr)
        p_raw = _blend_probs(models, weights, X_te)
        p_rank = rank_model.predict_proba(X_te_rank)[:, 1]
        p_man = _ranks_within_blocks(_manifold_dists(man, X_te), [len(X_te)])
        blend = (
            tri_w["raw"] * p_raw
            + tri_w["rank"] * p_rank
            + tri_w["manifold"] * p_man
        )
        return thr, np.array([_calibrate(v, thr) for v in blend])

    fold_out = Parallel(n_jobs=6)(
        delayed(_loro_fold)(date, X_te) for date, X_te, _ in full)
    fold_rewards = []
    for (date, X_te, y_te), (thr, p) in zip(full, fold_out):
        rew, _ = reward(p, y_te)
        fold_rewards.append(rew)
        report(f"held-out {date} (thr={thr:.3f})", p, y_te)
    loro_mean = float(np.mean(fold_rewards))
    print(f"\n  LORO mean reward = {loro_mean:.4f}  "
          f"min={min(fold_rewards):.4f}  max={max(fold_rewards):.4f}")

    # ---- final model: fit on ALL releases (augmented blocks), OOF-tuned ----
    all_dates = [d for d, _, _ in releases]
    X = np.vstack([aug[d][0] for d in all_dates])
    y = np.concatenate([aug[d][1] for d in all_dates])
    X_rank = np.vstack([aug[d][2] for d in all_dates])

    final_models = _fit_members(X, y)
    final_oof = _oof_member_probs(X, y)
    weights, _, _ = _tune_blend(final_oof, y)
    oof_raw = sum(weights[k] * final_oof[k] for k in ("gb", "et", "rf"))

    final_rank = GradientBoostingClassifier(**GBM_PARAMS).fit(X_rank, y)
    oof_rank = _gb_oof(X_rank, y)

    final_man = _fit_manifold(X[y == 0])
    all_sizes = [len(aug[d][1]) for d in all_dates]
    oof_man = _ranks_within_blocks(_oof_manifold_dists(X, y), all_sizes)

    tri_w, threshold, oof_rew = _tune_tri_blend(oof_raw, oof_rank, oof_man, y)
    w_raw = tri_w["raw"]  # legacy key, kept in the export for compatibility
    n_humans = int((y == 0).sum())
    print(f"\n[final] fit on {len(y)} groups ({n_humans} real-human for the "
          f"manifold) | raw member weights={weights} | tri weights={tri_w} "
          f"| threshold={threshold:.4f} | OOF blend reward={oof_rew:.4f}")
    if tri_w["manifold"] == 0.0:
        print("[manifold] OOF tuner selected weight 0.0 — manifold ships "
              "disabled; promotion would need the live A/B path "
              "(INNOVATION_PLAN Weapon 3).")

    # ---- export every member as flat tree arrays (pure-Python inference) ----
    def _export_regression_trees(gbc):
        """GBDT stages: leaf value = raw additive contribution."""
        out = []
        for stage in gbc.estimators_:        # (n_stages, 1) regressor trees
            t = stage[0].tree_
            out.append({
                # full-precision thresholds: sklearn compares float32(x)
                # against float64 thresholds; rounding can flip a split
                "f": [int(v) for v in t.feature],
                "t": [float(v) for v in t.threshold],
                "l": [int(v) for v in t.children_left],
                "r": [int(v) for v in t.children_right],
                "v": [float(v[0][0]) for v in t.value],
            })
        return out

    def _export_forest_trees(forest):
        """Averaging classifiers: leaf value = class-1 probability at leaf."""
        out = []
        for est in forest.estimators_:
            t = est.tree_
            values = []
            for node in t.value:             # shape (1, 2): class counts/fracs
                c0, c1 = float(node[0][0]), float(node[0][1])
                values.append(c1 / max(1e-12, c0 + c1))
            out.append({
                "f": [int(v) for v in t.feature],
                "t": [float(v) for v in t.threshold],
                "l": [int(v) for v in t.children_left],
                "r": [int(v) for v in t.children_right],
                "v": values,
            })
        return out

    prior = float(np.clip(np.mean(y), 1e-9, 1 - 1e-9))
    init = float(np.log(prior / (1.0 - prior)))
    raw_members = [
        {
            "kind": "gbdt",
            "weight": weights["gb"],
            "init": init,
            "learning_rate": GBM_PARAMS["learning_rate"],
            "trees": _export_regression_trees(final_models["gb"]),
        },
        {
            "kind": "forest",
            "weight": weights["et"],
            "trees": _export_forest_trees(final_models["et"]),
        },
        {
            "kind": "forest",
            "weight": weights["rf"],
            "trees": _export_forest_trees(final_models["rf"]),
        },
    ]
    # Drop zero-weight members entirely — smaller file, faster inference.
    raw_members = [m for m in raw_members if m["weight"] > 0]
    rank_members = [
        {
            "kind": "gbdt",
            "weight": 1.0,
            "init": init,
            "learning_rate": GBM_PARAMS["learning_rate"],
            "trees": _export_regression_trees(final_rank),
        },
    ]

    dates = ", ".join(d for d, _, _ in releases)
    payload = {
        "model": "rank_blend",
        "feature_names": features,
        "threshold": threshold,
        "w_raw": w_raw,
        "weights": tri_w,
        "raw_members": raw_members,
        "rank_members": rank_members,
    }
    if tri_w["manifold"] > 0.0:
        payload["manifold"] = {
            "output": "rank",  # serve as within-request distance ranks
            "mu": [float(v) for v in final_man["mu"]],
            "prec": [[float(v) for v in row] for row in final_man["prec"]],
            "d0": final_man["d0"],  # sigmoid fallback for single-chunk path
            "s": final_man["s"],
        }
    body = json.dumps(payload, separators=(",", ":"))
    PARAMS_PATH.write_text(
        '"""Auto-generated by local_test/train_detector.py — do not edit by hand.\n'
        "\n"
        f"Two-branch rank_blend: raw tree ensemble (member weights {weights}) "
        f"blended w_raw={w_raw}\n"
        "with a request-relative percentile-rank GBDT branch "
        "(scale-drift-immune by construction),\n"
        f"on {len(features)} transfer-gated features "
        "(parity-transformed benchmark hands).\n"
        f"Trained on {len(releases)} banked releases ({len(y)} labeled groups): "
        f"{dates}\n"
        '"""\n'
        "import json as _json\n\n"
        f"PARAMS = _json.loads({body!r})\n"
    )
    size_kb = PARAMS_PATH.stat().st_size / 1024
    n_trees = sum(len(m["trees"]) for m in raw_members + rank_members)
    print(f"[write] {PARAMS_PATH} ({size_kb:.0f} KB, {n_trees} trees, "
          f"{len(raw_members)} raw + {len(rank_members)} rank members)")

    # ---- parity: exported pure-Python branches vs sklearn ----
    from importlib import reload
    import neurons.detector_params
    import neurons.detector as det
    reload(neurons.detector_params)
    reload(det)
    sk_raw = _blend_probs(final_models, weights, X[:200])
    for row, expect in zip(X[:200], sk_raw):
        got = det._members_probability(
            det.PARAMS["raw_members"], [float(v) for v in row]
        )
        if abs(got - expect) > 1e-4:
            sys.exit(f"RAW PARITY FAILURE: pure-python {got} vs sklearn {expect}")
    sk_rank = final_rank.predict_proba(X_rank[:200])[:, 1]
    for row, expect in zip(X_rank[:200], sk_rank):
        got = det._members_probability(
            det.PARAMS["rank_members"], [float(v) for v in row]
        )
        if abs(got - expect) > 1e-4:
            sys.exit(f"RANK PARITY FAILURE: pure-python {got} vs sklearn {expect}")
    if det.PARAMS.get("manifold"):
        np_d = _manifold_dists(final_man, X[:200])
        for row, expect in zip(X[:200], np_d):
            got = det._manifold_distance(
                det.PARAMS["manifold"], [float(v) for v in row]
            )
            if abs(got - expect) > 1e-6 * max(1.0, abs(expect)):
                sys.exit(
                    f"MANIFOLD PARITY FAILURE: pure-python {got} vs numpy {expect}"
                )
    gate3 = True
    print("[parity] pure-Python raw + rank"
          + (" + manifold" if det.PARAMS.get("manifold") else "")
          + " inference matches reference on 200 samples each ✓")

    # ---- manifold kill-criteria probe (INNOVATION_PLAN Phase A) ----
    # Decorrelation is measured on DISTANCES (the served signal is their
    # within-request rank, a monotone transform, so Spearman is identical).
    from scipy.stats import spearmanr
    live_raw_p = np.array([
        det._members_probability(det.PARAMS["raw_members"], [float(v) for v in row])
        for row in X_live
    ])
    live_man_d = _manifold_dists(final_man, X_live)
    live_man_rank = _ranks_within_blocks(live_man_d, [len(X_live)])
    rho = float(spearmanr(live_raw_p, live_man_d).statistic)
    print(f"\n== Manifold probe (live capture, {len(X_live)} chunks) ==")
    print(f"  live Spearman manifold-vs-raw = {rho:.4f} "
          f"(kill if > 0.9: {'KILL — nothing new' if rho > 0.9 else 'OK — new signal'})")
    print(f"  manifold served ranks: std={float(np.std(live_man_rank)):.4f} "
          f"(uniform by construction) | raw distance spread: "
          f"std={float(np.std(live_man_d)):.4f} "
          f"deciles={np.round(np.quantile(live_man_d, [.1,.5,.9]), 2).tolist()}")

    # ---- GATE 2: live replay through the TRUE serving path (batch scorer
    # with the within-request rank branch) on the captured 100-chunk query ----
    live_scores = np.array(det.score_chunks_batch(live_chunks))
    live_std = float(np.std(live_scores))
    mid_frac = float(np.mean((live_scores >= GATE2_MID_LO)
                             & (live_scores <= GATE2_MID_HI)))
    print(f"\n== Live replay ({len(live_scores)} captured chunks, new params) ==")
    print(f"  std={live_std:.4f} (need >= {GATE2_STD_MIN}) | "
          f"frac in [{GATE2_MID_LO},{GATE2_MID_HI}]={mid_frac:.2f} "
          f"(need >= {GATE2_MID_FRAC_MIN})")
    print(f"  min={live_scores.min():.3f} q25={np.quantile(live_scores, .25):.3f} "
          f"median={np.median(live_scores):.3f} "
          f"q75={np.quantile(live_scores, .75):.3f} max={live_scores.max():.3f}")
    edges = np.arange(0.0, 1.01, 0.1)
    hist, _ = np.histogram(live_scores, bins=edges)
    for lo, hi, c in zip(edges[:-1], edges[1:], hist):
        print(f"  [{lo:.1f},{hi:.1f}) {'#' * int(c)} {c}")

    # ---- gate verdicts (report honestly; no workarounds) ----
    gate1 = loro_mean >= GATE1_LORO_MIN
    gate2 = live_std >= GATE2_STD_MIN and mid_frac >= GATE2_MID_FRAC_MIN
    print(f"\n== Gates ==")
    print(f"  GATE 1 (LORO mean >= {GATE1_LORO_MIN}): "
          f"{'PASS' if gate1 else 'FAIL'} ({loro_mean:.4f})")
    print(f"  GATE 2 (live std >= {GATE2_STD_MIN} & mid-frac >= "
          f"{GATE2_MID_FRAC_MIN}): {'PASS' if gate2 else 'FAIL'} "
          f"(std={live_std:.4f}, mid_frac={mid_frac:.2f})")
    print(f"  GATE 3 (sklearn parity): {'PASS' if gate3 else 'FAIL'}")
    print(f"  ALL GATES: {'PASS' if (gate1 and gate2 and gate3) else 'FAIL'}")
    print("[done] Rerun local_test/real_eval.py to grade the new model, then "
          "commit+push neurons/detector_params.py.")


if __name__ == "__main__":
    main()
