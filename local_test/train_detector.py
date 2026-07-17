#!/usr/bin/env python3
"""Train the Poker44 bot detector and emit neurons/detector_params.py.

MODEL (v2): gradient-boosted trees on the RAW 10 chunk features.
Forensics (verified adversarially) showed the bot population has three
families; the largest ("near-human sticky caller", 41% of bots) deviates
from humans by only 0.5-0.8 SD in a CONSISTENT DIRECTION, which the old
two-sided |z| -> logistic design structurally cannot see (the abs() erases
the sign). Trees on raw features keep directional signal AND learn
two-sided splits natively; honest LORO reward ~0.85 vs ~0.79 for the
legacy shape.

DATA: every release banked in local_test/data/release_*.json (real_eval.py
banks the newest release daily; backfill_releases.py recovers history).

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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path("/home/client_7075_3/Projects/Poker44-subnet")
sys.path.insert(0, str(REPO))

from neurons.detector import extract_features, _calibrate  # noqa: E402
from poker44.score.scoring import reward  # noqa: E402

DATA_DIR = REPO / "local_test" / "data"
# Bumped whenever the featurisation input changes; stale caches are ignored.
# v2: benchmark groups are fed AS-DELIVERED (they are already the validator's
# miner-visible payload). v1 wrongly ran prepare_hand_for_miner over them a
# second time, training the model on a distribution production never sends.
CACHE_VERSION = 3
# Only features whose PRODUCTION distribution matches the benchmark training
# distribution (verified against a captured live validator payload). The four
# dropped features were the model's strongest on the synthetic benchmark but
# are catastrophically out-of-distribution live: real hands bet ~1-2bb in tight
# standard sizes, while the synthetic benchmark bots bet 8-190bb wildly, so
# betsize_std_bb (0.14 live vs ~48 train), betsize_entropy_norm,
# betsize_frac_unique and xhand_fold_ratio_std all collapsed and saturated
# every live score to ~0.98 (all-True). No rescaling (CV, bet/pot) transfers
# either. What's left are scale-free per-street rates and reach fractions.
FEATURES = [
    "flop_fold_rate",
    "turn_fold_rate",
    "river_fold_rate",
    "flop_raise_rate",
    "frac_hands_reach_flop",
    "frac_hands_reach_river",
]
TARGET_HUMAN_FPR = 0.07     # <=7% humans over 0.5 (validator gate allows 10%)
FULL_SIZE_MIN_GROUPS = 100  # releases smaller than this are pilot-era noise
GBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=0,
)
PARAMS_PATH = REPO / "neurons" / "detector_params.py"


def load_all_releases():
    """Load every banked release: list of (date, X, y). Feature caching:
    each immutable release is featurized once per CACHE_VERSION.

    Groups are featurized AS-DELIVERED. The benchmark API already returns the
    validator's miner-visible payload (seat_N aliases, hole_cards=None, empty
    board, zeroed outcome, sb/bb=0.01/0.02), i.e. prepare_hand_for_miner has
    already been applied upstream. Running it again re-buckets and re-noises
    bet sizes and re-samples the action window, producing features that
    production never sends.
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
                if (cached.get("features") == FEATURES
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
                feats = extract_features(grp)
                rows.append([feats[k] for k in FEATURES])
            X = np.asarray(rows, dtype=float)
            with open(cache_path, "w") as fh:
                json.dump({"features": FEATURES, "cache_version": CACHE_VERSION,
                           "X": X.tolist(), "y": y.tolist()}, fh)
        releases.append((date, X, y))
    return releases


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
    full = [(d, X, y) for d, X, y in releases if len(y) >= FULL_SIZE_MIN_GROUPS]
    print(f"[data] {len(releases)} releases banked, {n_total} labeled groups "
          f"| {len(full)} full-size validation folds")

    # ---- honest leave-one-release-out over the full-size releases ----
    print("\n== Leave-one-release-out (full-size releases; "
          "threshold from train-side OOF only) ==")
    fold_rewards = []
    for date, X_te, y_te in full:
        X_tr = np.vstack([X for d, X, _ in releases if d != date])
        y_tr = np.concatenate([y for d, _, y in releases if d != date])
        model = GradientBoostingClassifier(**GBM_PARAMS).fit(X_tr, y_tr)
        thr = oof_threshold(X_tr, y_tr)
        p = np.array([_calibrate(v, thr)
                      for v in model.predict_proba(X_te)[:, 1]])
        rew, _ = reward(p, y_te)
        fold_rewards.append(rew)
        report(f"held-out {date} (thr={thr:.3f})", p, y_te)
    print(f"\n  LORO mean reward = {float(np.mean(fold_rewards)):.4f}  "
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
            "feature_names": FEATURES,
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
        f"{GBM_PARAMS['max_depth']}) on raw censored features.\n"
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
        feats = dict(zip(FEATURES, row))
        logit = det.PARAMS["init"] + det.PARAMS["learning_rate"] * sum(
            det._eval_tree(tree, [feats[n] for n in FEATURES])
            for tree in det.PARAMS["trees"])
        got = 1.0 / (1.0 + np.exp(-logit))
        if abs(got - expect) > 1e-4:
            sys.exit(f"PARITY FAILURE: pure-python {got} vs sklearn {expect}")
    print("[parity] pure-Python tree inference matches sklearn on 200 samples ✓")
    print("[done] Rerun local_test/real_eval.py to grade the new model, then "
          "commit+push neurons/detector_params.py.")


if __name__ == "__main__":
    main()
