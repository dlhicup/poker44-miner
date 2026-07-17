#!/usr/bin/env python3
"""Train the Poker44 dispersion detector and emit neurons/detector_params.py.

DATA: every release banked in local_test/data/release_*.json is used.
(Each daily run of local_test/real_eval.py banks that day's labeled release
automatically, so the training corpus grows over time; just keep testing.)

Pipeline
--------
1. Load ALL banked releases.
2. Censor every hand with the EXACT production censor
   (poker44.validator.payload_view.prepare_hand_for_miner).
3. Extract features with neurons.detector.extract_features — the very
   function the deployed miner runs, so train == serve by construction.
4. Model: two-sided robust z (|x - median| / scale, centers fit on TRAINING
   data only) -> logistic regression.
5. Honest validation: LEAVE-ONE-RELEASE-OUT — fit on all releases except one,
   test on the held-out release, rotate. All reported numbers and the
   calibration threshold come from these out-of-sample predictions only.
6. Final artifact: refit on all releases pooled, fold the scaler into
   per-feature weights, and write neurons/detector_params.py.

Run:
    cd /home/client_7075_3/Projects/Poker44-subnet
    PYTHONPATH=. .venv/bin/python local_test/train_detector.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO = Path("/home/client_7075_3/Projects/Poker44-subnet")
sys.path.insert(0, str(REPO))

from neurons.detector import extract_features, _calibrate  # noqa: E402
from poker44.score.scoring import reward  # noqa: E402
from poker44.validator.payload_view import prepare_hand_for_miner  # noqa: E402

DATA_DIR = REPO / "local_test" / "data"
FEATURES = [
    "flop_fold_rate",
    "turn_fold_rate",
    "river_fold_rate",
    "flop_raise_rate",
    "xhand_fold_ratio_std",
    "betsize_std_bb",
    "betsize_entropy_norm",
    "betsize_frac_unique",
    "frac_hands_reach_flop",
    "frac_hands_reach_river",
]
TARGET_HUMAN_FPR = 0.07  # calibration target: <=7% humans over 0.5 (gate allows 10%)
PARAMS_PATH = REPO / "neurons" / "detector_params.py"


def load_all_releases():
    """Load every banked release: returns list of (date, X, y).

    Featurization (censor + extract) is cached per release in
    local_test/data/features_<date>.json — releases are immutable, so each
    one is censored exactly once, ever. The cache stores the FEATURES list
    it was built with and is invalidated automatically if features change.
    """
    paths = sorted(DATA_DIR.glob("release_*.json"))
    if len(paths) < 2:
        sys.exit(
            f"Need at least 2 banked releases in {DATA_DIR} for honest "
            f"validation; found {len(paths)}. Run local_test/real_eval.py "
            "on more days first (or local_test/backfill_releases.py for history)."
        )
    releases = []
    for path in paths:
        date = path.stem.replace("release_", "")
        cache_path = DATA_DIR / f"features_{date}.json"
        X = y = None
        if cache_path.exists():
            try:
                with open(cache_path) as fh:
                    cached = json.load(fh)
                if cached.get("features") == FEATURES:
                    X = np.asarray(cached["X"], dtype=float)
                    y = np.asarray(cached["y"], dtype=int)
            except Exception:
                X = y = None  # unreadable cache -> refeaturize
        if X is None:
            with open(path) as fh:
                data = json.load(fh)
            y = np.asarray(data["labels"], dtype=int)
            rows = []
            for grp in data["groups"]:
                censored = [prepare_hand_for_miner(h) for h in grp]
                feats = extract_features(censored)
                rows.append([feats[k] for k in FEATURES])
            X = np.asarray(rows, dtype=float)
            with open(cache_path, "w") as fh:
                json.dump({"features": FEATURES, "X": X.tolist(),
                           "y": y.tolist()}, fh)
        print(f"[{date}] {len(y)} groups | humans={int((y == 0).sum())} "
              f"bots={int(y.sum())}")
        releases.append((date, X, y))
    return releases


def fit_model(X: np.ndarray, y: np.ndarray, C: float):
    center = np.median(X, axis=0)
    scale = np.maximum(1e-9, X.std(axis=0))
    Z = np.abs((X - center) / scale)
    scaler = StandardScaler().fit(Z)
    lr = LogisticRegression(max_iter=5000, C=C).fit(scaler.transform(Z), y)
    return center, scale, scaler, lr


def predict(model, X: np.ndarray) -> np.ndarray:
    center, scale, scaler, lr = model
    Z = np.abs((X - center) / scale)
    return lr.predict_proba(scaler.transform(Z))[:, 1]


def report(tag: str, p: np.ndarray, y: np.ndarray) -> None:
    rew, met = reward(p, y)
    print(f"  {tag}: AUC={roc_auc_score(y, p):.3f} reward={rew:.4f} "
          f"ap={met['ap_score']:.3f} recall@fpr5={met['bot_recall']:.3f} "
          f"bots>=0.5 {np.mean(p[y == 1] >= 0.5):.1%} "
          f"humans<0.5 {np.mean(p[y == 0] < 0.5):.1%}")


def leave_one_release_out(releases, C: float):
    """Fit on all releases but one, predict the held-out one; rotate."""
    oos_pred, oos_y, rewards = [], [], []
    for i, (date, X_te, y_te) in enumerate(releases):
        X_tr = np.vstack([X for j, (_, X, _) in enumerate(releases) if j != i])
        y_tr = np.concatenate([y for j, (_, _, y) in enumerate(releases) if j != i])
        p = predict(fit_model(X_tr, y_tr, C), X_te)
        rew, _ = reward(p, y_te)
        rewards.append((date, rew, p, y_te))
        oos_pred.append(p)
        oos_y.append(y_te)
    return np.concatenate(oos_pred), np.concatenate(oos_y), rewards


def main() -> None:
    releases = load_all_releases()
    n_total = sum(len(y) for _, _, y in releases)
    print(f"[data] {len(releases)} releases, {n_total} labeled groups total")

    # ---- pick C by mean leave-one-release-out reward ----
    print("\n== Leave-one-release-out validation ==")
    best_c, best_rew, best_run = None, -1.0, None
    for C in (0.1, 0.3, 1.0, 3.0):
        oos_pred, oos_y, per_release = leave_one_release_out(releases, C)
        rews = [r for _, r, _, _ in per_release]
        mean_rew = float(np.mean(rews))
        print(f"  C={C:<4}: mean={mean_rew:.4f}  "
              f"min={min(rews):.4f}  max={max(rews):.4f}  (n={len(rews)} folds)")
        if mean_rew > best_rew:
            best_c, best_rew = C, mean_rew
            best_run = (oos_pred, oos_y, per_release)
    print(f"  -> selected C={best_c} (mean out-of-sample reward {best_rew:.4f})")

    oos_pred, oos_y, per_release = best_run
    print("\n== Out-of-sample per-release detail at selected C (pre-calibration) ==")
    for date, _, p, y in per_release:
        report(f"held-out {date}", p, y)

    # ---- calibration threshold from OUT-OF-SAMPLE human predictions ----
    human_scores = oos_pred[oos_y == 0]
    threshold = float(np.quantile(human_scores, 1.0 - TARGET_HUMAN_FPR))
    threshold = min(max(threshold, 0.05), 0.95)
    print(f"\n[calibration] threshold={threshold:.4f} "
          f"(out-of-sample human {100 * (1 - TARGET_HUMAN_FPR):.0f}th percentile)")
    cal = np.vectorize(lambda p: _calibrate(p, threshold))
    print("== Out-of-sample detail AFTER calibration ==")
    for date, _, p, y in per_release:
        report(f"held-out {date} calibrated", cal(p), y)

    # ---- final model: refit on ALL releases pooled, fold scaler in ----
    X = np.vstack([Xr for _, Xr, _ in releases])
    y = np.concatenate([yr for _, _, yr in releases])
    center, scale, scaler, lr = fit_model(X, y, best_c)
    coef = lr.coef_[0] / scaler.scale_
    bias = float(lr.intercept_[0] - np.sum(lr.coef_[0] * scaler.mean_ / scaler.scale_))

    dates = ", ".join(d for d, _, _ in releases)
    lines = [
        '"""Auto-generated by local_test/train_detector.py — do not edit by hand.',
        "",
        f"Trained on banked Poker44 benchmark releases: {dates}",
        f"({len(y)} labeled chunk groups). Regenerate by rerunning the script.",
        '"""',
        "",
        "PARAMS = {",
        '    "features": [',
    ]
    for i, name in enumerate(FEATURES):
        lines.append(
            f'        ("{name}", {float(center[i])!r}, {float(scale[i])!r}, '
            f"{float(coef[i])!r}),"
        )
    lines += [
        "    ],",
        f'    "bias": {bias!r},',
        f'    "threshold": {threshold!r},',
        "}",
        "",
    ]
    PARAMS_PATH.write_text("\n".join(lines))
    print(f"\n[write] {PARAMS_PATH}")
    print("[done] The deployed miner now uses the refreshed parameters. "
          "Rerun local_test/real_eval.py to grade the updated model.")


if __name__ == "__main__":
    main()
