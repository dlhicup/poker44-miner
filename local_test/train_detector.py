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
load_all_releases for why).

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

import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
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

    full = [(d, X, y) for d, X, y in releases if len(y) >= FULL_SIZE_MIN_GROUPS]
    print(f"[data] {len(full)} full-size validation folds | "
          f"{len(features)} gated features")

    # ---- honest leave-one-release-out over the full-size releases ----
    print("\n== Leave-one-release-out (full-size releases; "
          "threshold from train-side OOF only) ==")
    def _loro_fold(date, X_te):
        """One fold of the full two-branch pipeline, train-side-only tuning:
        raw ensemble (3 members, OOF-tuned internal weights) + rank-branch
        GBDT (within-release rank views) + raw/rank blend weight and
        threshold from train-side OOF. Scores the held-out date exactly as
        serving would (its rank view computed within itself)."""
        X_tr = np.vstack([X for d, X, _ in releases if d != date])
        y_tr = np.concatenate([y for d, _, y in releases if d != date])
        X_tr_rank = np.vstack([releases_rank[d] for d, _, _ in releases if d != date])
        X_te_rank = releases_rank[date]

        models = _fit_members(X_tr, y_tr)
        oof_members = _oof_member_probs(X_tr, y_tr)
        weights, _, _ = _tune_blend(oof_members, y_tr)
        oof_raw = sum(weights[k] * oof_members[k] for k in ("gb", "et", "rf"))

        rank_model = GradientBoostingClassifier(**GBM_PARAMS).fit(X_tr_rank, y_tr)
        oof_rank = _gb_oof(X_tr_rank, y_tr)

        w_raw, thr, _ = _tune_branch_blend(oof_raw, oof_rank, y_tr)
        p_raw = _blend_probs(models, weights, X_te)
        p_rank = rank_model.predict_proba(X_te_rank)[:, 1]
        blend = w_raw * p_raw + (1.0 - w_raw) * p_rank
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

    # ---- final two-branch model: fit on ALL releases, OOF-tuned ----
    X = np.vstack([Xr for _, Xr, _ in releases])
    y = np.concatenate([yr for _, _, yr in releases])
    X_rank = np.vstack([releases_rank[d] for d, _, _ in releases])

    final_models = _fit_members(X, y)
    final_oof = _oof_member_probs(X, y)
    weights, _, _ = _tune_blend(final_oof, y)
    oof_raw = sum(weights[k] * final_oof[k] for k in ("gb", "et", "rf"))

    final_rank = GradientBoostingClassifier(**GBM_PARAMS).fit(X_rank, y)
    oof_rank = _gb_oof(X_rank, y)

    w_raw, threshold, oof_rew = _tune_branch_blend(oof_raw, oof_rank, y)
    print(f"\n[final] fit on {len(y)} groups | raw member weights={weights} "
          f"| w_raw={w_raw} | threshold={threshold:.4f} "
          f"| OOF blend reward={oof_rew:.4f}")

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
    body = json.dumps(
        {
            "model": "rank_blend",
            "feature_names": features,
            "threshold": threshold,
            "w_raw": w_raw,
            "raw_members": raw_members,
            "rank_members": rank_members,
        },
        separators=(",", ":"),
    )
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
    gate3 = True
    print("[parity] pure-Python raw + rank branch inference matches sklearn "
          "on 200 samples each ✓")

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
