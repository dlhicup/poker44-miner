#!/usr/bin/env python3
"""ANALYSIS: do our roboticized synthetic bots resemble the ambiguous LIVE
chunks (the real bots we miss), or do they just recreate the easy benchmark
bots?

We cannot label live bots, but we CAN position four populations in the same
feature space and compare:
  H  = benchmark HUMANS (label 0)
  B  = benchmark BOTS (label 1) — the "cartoon" easy positives
  R  = our current ROBOTICIZED bots (self-generated hard positives)
  L  = LIVE captured chunks (unlabeled mix of humans + real bots)

Signals:
  (1) manifold distance-from-human (the deployed one-class model): how far
      from the human center does each population sit? Good synthetic bots
      should overlap the *upper* part of the live distribution (the live
      chunks that are plausibly bots), NOT sit far past everything (benchmark
      bots) nor collapse onto humans.
  (2) the deployed model's raw score per population.
  (3) which transfer-stable features separate H from B, and whether R moves
      along those same axes or different ones.

Run:
    cd /home/client_7075_3/Projects/Poker44-subnet
    PYTHONPATH=. .venv/bin/python local_test/exp_robo_realism.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/client_7075_3/Projects/Poker44-subnet")
sys.path.insert(0, str(REPO))

from neurons.detector import (  # noqa: E402
    CANDIDATE_FEATURES, PARAMS, extract_features, _manifold_distance,
    _members_probability, percentile_ranks,
)
from poker44.validator.payload_view import build_miner_payload_hand  # noqa: E402
import local_test.train_detector as T  # noqa: E402


def q(a):
    a = np.asarray(a, dtype=float)
    return (f"n={len(a)} mean={a.mean():.3f} "
            f"p10={np.quantile(a,.1):.3f} p50={np.quantile(a,.5):.3f} "
            f"p90={np.quantile(a,.9):.3f} max={a.max():.3f}")


def feats_of(groups):
    rows = []
    for g in groups:
        try:
            f = extract_features(g)
            rows.append([f[k] for k in CANDIDATE_FEATURES])
        except Exception:
            pass
    return np.asarray(rows, dtype=float)


def main():
    man = PARAMS.get("manifold")
    if not man:
        sys.exit("no manifold in deployed PARAMS")
    raw_members = PARAMS["raw_members"]
    fn = PARAMS["feature_names"]
    fn_idx = [CANDIDATE_FEATURES.index(n) for n in fn]

    # --- load benchmark humans + bots (parity-transformed) from recent releases
    H_groups, B_groups = [], []
    for path in sorted(REPO.glob("local_test/data/release_2026-07-1*.json"))[-4:]:
        d = json.load(open(path))
        for g, y in zip(d["groups"], d["labels"]):
            gt = [build_miner_payload_hand(h) for h in g]
            (B_groups if y == 1 else H_groups).append(gt)
    print(f"[data] benchmark humans={len(H_groups)} bots={len(B_groups)}")

    # --- regenerate current roboticized bots from those humans
    R_groups = []
    for i, g in enumerate(H_groups):
        sev = T.ROBO_SEVERITIES[i % len(T.ROBO_SEVERITIES)]
        R_groups.append(T._roboticize_group(g, sev, T._robo_rng("analysis", f"g{i}")))
    print(f"[data] roboticized bots generated={len(R_groups)}")

    # --- live captures (unlabeled)
    caps = sorted(REPO.glob("local_test/captures/query_*.json")) or \
           sorted(REPO.glob("local_test/captures_archive_20260717/query_*.json"))
    L_chunks = json.load(open(caps[-1]))["chunks"] if caps else []
    print(f"[data] live chunks={len(L_chunks)} from {caps[-1].name if caps else 'NONE'}")

    XH, XB, XR = feats_of(H_groups), feats_of(B_groups), feats_of(R_groups)
    XL = feats_of(L_chunks) if L_chunks else np.zeros((0, len(CANDIDATE_FEATURES)))

    # (1) manifold distance-from-human per population
    def mdist(X):
        return np.array([_manifold_distance(man, [float(v) for v in r]) for r in X])
    print("\n== Manifold distance-from-human (higher = less human) ==")
    print(f"  H humans : {q(mdist(XH))}")
    print(f"  B bots   : {q(mdist(XB))}")
    print(f"  R robo   : {q(mdist(XR))}")
    if len(XL):
        print(f"  L live   : {q(mdist(XL))}")
        # where do R and B sit within the LIVE distance distribution?
        dL = mdist(XL)
        for tag, X in [("B bots", XB), ("R robo", XR)]:
            d = mdist(X)
            pct = float(np.mean(dL[None, :] < d[:, None]))  # frac of live below each
            print(f"    {tag}: mean live-percentile = {pct:.2f} "
                  f"(0.5=looks like median live chunk; ~1.0=more extreme than all live)")

    # (2) deployed raw model score per population
    def rawp(X):
        Xs = X[:, fn_idx]
        return np.array([_members_probability(raw_members, [float(v) for v in r]) for r in Xs])
    print("\n== Deployed RAW model score per population ==")
    print(f"  H humans : {q(rawp(XH))}")
    print(f"  B bots   : {q(rawp(XB))}")
    print(f"  R robo   : {q(rawp(XR))}")
    if len(XL):
        print(f"  L live   : {q(rawp(XL))}")

    # (3) top transfer-stable axes separating H from B, and whether R moves there
    print("\n== Do robo bots move along the same axes as benchmark bots? ==")
    stable = fn_idx  # transfer-gated features = the ones that matter live
    diffs = []
    for j in stable:
        mh, mb = XH[:, j].mean(), XB[:, j].mean()
        s = XH[:, j].std() + 1e-9
        diffs.append((abs(mb - mh) / s, j))
    diffs.sort(reverse=True)
    print(f"  {'feature':32s} {'H->B (SD)':>10s} {'H->R (SD)':>10s}  aligned?")
    for _, j in diffs[:12]:
        name = CANDIDATE_FEATURES[j]
        s = XH[:, j].std() + 1e-9
        hb = (XB[:, j].mean() - XH[:, j].mean()) / s
        hr = (XR[:, j].mean() - XH[:, j].mean()) / s
        aligned = "yes" if (hb * hr > 0 and abs(hr) > 0.15 * abs(hb)) else "NO"
        print(f"  {name:32s} {hb:>10.2f} {hr:>10.2f}  {aligned}")


if __name__ == "__main__":
    main()
