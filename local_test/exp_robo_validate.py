#!/usr/bin/env python3
"""VALIDATION: are our v2 roboticized bots really similar to the validator's
real bots?

Hard limit: live bots are UNLABELED, so we cannot compare to them directly.
But three honest, independent signals bound the answer:

  (1) SIGNATURE DIRECTION vs REAL bots. The benchmark's labeled bots ARE the
      validator's own bots. Compute the mean displacement human->bot and
      human->robo on the transfer-stable features; cosine similarity says
      whether our synthetic bots move along the SAME signature axes as real
      bots (direction), and the norm ratio says whether they are milder
      (as live bots are). Target: high cosine (same kind of bot), norm < 1
      (subtle like live, not cartoonish like the benchmark).

  (2) LIVE IN-DISTRIBUTION. A realistic synthetic bot must be a plausible
      LIVE chunk, i.e. lie inside the live feature ranges (not OOD). For
      each population, the fraction of transfer-stable features landing
      inside live's [p5,p95] band. Robo should be >= benchmark bots (we
      calibrated it to live; the benchmark bots are partly OOD live).

  (3) RESEMBLE THE LIVE CHUNKS THE MODEL ALREADY FLAGS. The highest-scoring
      live chunks are the model's best guess at live bots. Nearest-neighbor
      distance (standardized, transfer-stable features) from robo bots to
      those suspicious live chunks vs from benchmark bots to them. Robo
      should be at least as close as the benchmark bots.

Run:
    cd /home/client_7075_3/Projects/Poker44-subnet
    PYTHONPATH=. .venv/bin/python local_test/exp_robo_validate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/client_7075_3/Projects/Poker44-subnet")
sys.path.insert(0, str(REPO))

from neurons.detector import (  # noqa: E402
    CANDIDATE_FEATURES, PARAMS, extract_features, _members_probability,
)
from poker44.validator.payload_view import build_miner_payload_hand  # noqa: E402
import local_test.train_detector as T  # noqa: E402


def feats(groups, idx):
    out = []
    for g in groups:
        try:
            f = extract_features(g)
            out.append([float(f[CANDIDATE_FEATURES[j]]) for j in idx])
        except Exception:
            pass
    return np.asarray(out, dtype=float)


def main():
    fn = PARAMS["feature_names"]                 # transfer-stable features
    idx = [CANDIDATE_FEATURES.index(n) for n in fn]
    rm = PARAMS["raw_members"]

    # populations
    H, B = [], []
    for path in sorted(REPO.glob("local_test/data/release_2026-07-1*.json"))[-5:]:
        d = json.load(open(path))
        for g, y in zip(d["groups"], d["labels"]):
            gt = [build_miner_payload_hand(h) for h in g]
            (B if y == 1 else H).append(gt)
    R = []
    for k, g in enumerate(H):
        a, s = T.ROBO_SCHEDULE[k % len(T.ROBO_SCHEDULE)]
        R.append(T._roboticize_group(g, a, s, T._robo_rng("val", f"g{k}")))
    caps = sorted(REPO.glob("local_test/captures/query_*.json"))
    L = json.load(open(caps[-1]))["chunks"]

    XH, XB, XR = feats(H, idx), feats(B, idx), feats(R, idx)
    XL = feats(L, idx)
    print(f"[n] humans={len(XH)} bench_bots={len(XB)} robo={len(XR)} live={len(XL)}")

    # Robust standardization: scale by the POOLED std across all four
    # populations, and DROP near-degenerate features (pooled std < 1e-3) whose
    # z-scores would otherwise explode (a handful of q10 quantile features are
    # ~0 for almost every human). Clip to keep single outliers from dominating.
    pooled = np.vstack([XH, XB, XR, XL])
    psd = pooled.std(0)
    keep = psd >= 1e-3
    fn = [n for n, k in zip(fn, keep) if k]
    XH, XB, XR, XL = XH[:, keep], XB[:, keep], XR[:, keep], XL[:, keep]
    mu, sd = XH.mean(0), pooled[:, keep].std(0) + 1e-9
    clip = lambda Z: np.clip((Z - mu) / sd, -8, 8)
    zH, zB, zR, zL = clip(XH), clip(XB), clip(XR), clip(XL)
    print(f"[feat] {int(keep.sum())} non-degenerate transfer-stable features used "
          f"(dropped {int((~keep).sum())} near-constant)")

    # (1) signature direction vs REAL (benchmark) bots
    dB = zB.mean(0) - zH.mean(0)      # human->real-bot displacement
    dR = zR.mean(0) - zH.mean(0)      # human->robo displacement
    cos = float(dB @ dR / ((np.linalg.norm(dB)*np.linalg.norm(dR)) + 1e-12))
    print("\n(1) SIGNATURE DIRECTION vs real validator (benchmark) bots")
    print(f"    cosine(human->robo, human->realbot) = {cos:+.3f}  "
          f"(1=identical signature axis, 0=unrelated)")
    print(f"    displacement norm  robo={np.linalg.norm(dR):.2f}  "
          f"realbot={np.linalg.norm(dB):.2f}  ratio={np.linalg.norm(dR)/ (np.linalg.norm(dB)+1e-9):.2f} "
          f"(want <1: subtle like live, not cartoonish)")

    # (2) live in-distribution
    lo, hi = np.quantile(XL, 0.05, 0), np.quantile(XL, 0.95, 0)
    def in_band(X):
        return float(np.mean((X >= lo) & (X <= hi)))
    print("\n(2) LIVE IN-DISTRIBUTION (frac of transfer-stable features in live [p5,p95])")
    print(f"    humans={in_band(XH):.2f}  real bots={in_band(XB):.2f}  "
          f"ROBO={in_band(XR):.2f}  (robo >= real bots ⇒ at least as live-plausible)")

    # (3) resemble the live chunks the model already flags as bots
    rawL = np.array([_members_probability(rm, r.tolist()) for r in XL])
    susp = zL[rawL >= np.quantile(rawL, 0.75)]   # top-25% most bot-like live
    def nn_dist(Z):
        # mean nearest-neighbour euclid distance to the suspicious live set
        d = np.sqrt(((Z[:, None, :] - susp[None, :, :])**2).sum(-1))
        return float(d.min(1).mean())
    print("\n(3) DISTANCE TO THE LIVE CHUNKS THE MODEL FLAGS AS BOTS (top-25%)")
    print(f"    nearest-neighbour dist  real bots={nn_dist(zB):.2f}  "
          f"ROBO={nn_dist(zR):.2f}  humans={nn_dist(zH):.2f}  (robo <= real bots ⇒ as close)")

    # per-feature snapshot on the top real-bot-discriminative axes
    order = np.argsort(-np.abs(dB))[:10]
    print("\n    top real-bot axes           human   realbot    robo     live")
    for j in order:
        print(f"    {fn[j]:26s} {XH[:,j].mean():7.3f} {XB[:,j].mean():8.3f} "
              f"{XR[:,j].mean():7.3f} {XL[:,j].mean():7.3f}")

    # verdict
    ok = (cos >= 0.6 and np.linalg.norm(dR) < np.linalg.norm(dB)
          and in_band(XR) >= in_band(XB) and nn_dist(zR) <= nn_dist(zB) * 1.15)
    print("\n== VERDICT ==")
    print("  " + ("PLAUSIBLE — robo bots share the real bot signature (same axes, "
                  "milder), are as live-in-distribution as real bots, and sit as "
                  "close to the model's flagged live chunks."
                  if ok else
                  "WEAK — robo bots diverge from the real bot signature or live "
                  "distribution on one+ checks; review before trusting v8."))


if __name__ == "__main__":
    main()
