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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path("/home/client_7075_3/Projects/Poker44-subnet")
sys.path.insert(0, str(REPO))

from neurons.detector import CANDIDATE_FEATURES, extract_features, _calibrate  # noqa: E402
from poker44.score.scoring import reward  # noqa: E402
from poker44.validator.payload_view import build_miner_payload_hand  # noqa: E402

DATA_DIR = REPO / "local_test" / "data"
CAPTURES_DIR = REPO / "local_test" / "captures"
# Bumped whenever the featurisation input changes; stale caches are ignored.
# v4: expanded CANDIDATE_FEATURES matrix + parity re-canonicalization of the
# benchmark hands through build_miner_payload_hand (see load_all_releases).
CACHE_VERSION = 4
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

    full = [(d, X, y) for d, X, y in releases if len(y) >= FULL_SIZE_MIN_GROUPS]
    print(f"[data] {len(full)} full-size validation folds | "
          f"{len(features)} gated features")

    # ---- honest leave-one-release-out over the full-size releases ----
    print("\n== Leave-one-release-out (full-size releases; "
          "threshold from train-side OOF only) ==")
    def _loro_fold(date, X_te):
        """One fold: fit + OOF threshold on the train side, score held-out.
        Identical math to the serial loop (fixed random_state everywhere);
        folds are independent so they run in parallel for wall-time only."""
        X_tr = np.vstack([X for d, X, _ in releases if d != date])
        y_tr = np.concatenate([y for d, _, y in releases if d != date])
        model = GradientBoostingClassifier(**GBM_PARAMS).fit(X_tr, y_tr)
        thr = oof_threshold(X_tr, y_tr)
        return thr, np.array([_calibrate(v, thr)
                              for v in model.predict_proba(X_te)[:, 1]])

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

    # ---- final model: fit on ALL releases, OOF-calibrated ----
    X = np.vstack([Xr for _, Xr, _ in releases])
    y = np.concatenate([yr for _, _, yr in releases])
    final = GradientBoostingClassifier(**GBM_PARAMS).fit(X, y)
    threshold = oof_threshold(X, y)
    print(f"\n[final] fit on {len(y)} groups | calibration threshold={threshold:.4f}")

    # ---- export every tree as flat arrays (pure-Python inference) ----
    trees = []
    for stage in final.estimators_:          # (n_stages, 1) regressor trees
        t = stage[0].tree_
        trees.append({
            # full-precision thresholds: sklearn compares float32(x) against
            # float64 thresholds, and any rounding here can flip a split
            "f": [int(v) for v in t.feature],
            "t": [float(v) for v in t.threshold],
            "l": [int(v) for v in t.children_left],
            "r": [int(v) for v in t.children_right],
            "v": [float(v[0][0]) for v in t.value],
        })
    # sklearn GBC initial raw prediction = log-odds of the base rate
    prior = float(np.clip(np.mean(y), 1e-9, 1 - 1e-9))
    init = float(np.log(prior / (1.0 - prior)))

    dates = ", ".join(d for d, _, _ in releases)
    body = json.dumps(
        {
            "model": "gbdt",
            "feature_names": features,
            "init": init,
            "learning_rate": GBM_PARAMS["learning_rate"],
            "threshold": threshold,
            "trees": trees,
        },
        separators=(",", ":"),
    )
    PARAMS_PATH.write_text(
        '"""Auto-generated by local_test/train_detector.py — do not edit by hand.\n'
        "\n"
        f"Gradient-boosted trees ({GBM_PARAMS['n_estimators']} x depth "
        f"{GBM_PARAMS['max_depth']}) on {len(features)} transfer-gated features\n"
        "(parity-transformed benchmark hands via build_miner_payload_hand).\n"
        f"Trained on {len(releases)} banked releases ({len(y)} labeled groups): "
        f"{dates}\n"
        '"""\n'
        "import json as _json\n\n"
        f"PARAMS = _json.loads({body!r})\n"
    )
    size_kb = PARAMS_PATH.stat().st_size / 1024
    print(f"[write] {PARAMS_PATH} ({size_kb:.0f} KB, {len(trees)} trees)")

    # ---- parity check: exported pure-Python path vs sklearn ----
    from importlib import reload
    import neurons.detector_params
    import neurons.detector as det
    reload(neurons.detector_params)
    reload(det)
    sk_prob = final.predict_proba(X[:200])[:, 1]
    for row, expect in zip(X[:200], sk_prob):
        feats = dict(zip(features, row))
        logit = det.PARAMS["init"] + det.PARAMS["learning_rate"] * sum(
            det._eval_tree(tree, [feats[n] for n in features])
            for tree in det.PARAMS["trees"])
        got = 1.0 / (1.0 + np.exp(-logit))
        if abs(got - expect) > 1e-4:
            sys.exit(f"PARITY FAILURE: pure-python {got} vs sklearn {expect}")
    gate3 = True
    print("[parity] pure-Python tree inference matches sklearn on 200 samples ✓")

    # ---- GATE 2: live replay with the NEW exported params ----
    live_scores = np.array([det.score_chunk(chunk) for chunk in live_chunks])
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
