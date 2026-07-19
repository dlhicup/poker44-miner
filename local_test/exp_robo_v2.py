#!/usr/bin/env python3
"""PROTOTYPE: a milder, archetype-diverse roboticizer that lands synthetic
bots in the LIVE-realistic score band (~0.45-0.65), not the cartoon-bot band
(~0.84 — see exp_robo_realism.py).

Design changes vs the current roboticizer:
  1. MILDER severity gradient, and each group gets ONE dominant archetype
     rather than all transforms stacked (stacking is what over-hardens them):
       - "repeat"   : template repetition only (n-gram concentration)
       - "entropy"  : street-modal action smoothing only (low action entropy)
       - "imbalance": nudge the action mix toward a single modal action
       - "mirror"   : make a fraction of hands near-duplicates of one seed
  2. A fine severity curriculum so the near-boundary region is densely
     covered instead of a couple of hard points.
  3. VALIDATION against the live capture: the synthetic bots should overlap
     the upper-middle of the live score/distance distribution (plausible live
     bots), not sit past all of it.

This script only PROTOTYPES + validates; if the band matches, the archetype
roboticizer is wired into train_detector.py.

Run:
    cd /home/client_7075_3/Projects/Poker44-subnet
    PYTHONPATH=. .venv/bin/python local_test/exp_robo_v2.py
"""
from __future__ import annotations

import copy
import json
import sys
import zlib
from pathlib import Path

import numpy as np

REPO = Path("/home/client_7075_3/Projects/Poker44-subnet")
sys.path.insert(0, str(REPO))

from neurons.detector import (  # noqa: E402
    CANDIDATE_FEATURES, PARAMS, extract_features, _manifold_distance,
    _members_probability,
)
from poker44.validator.payload_view import build_miner_payload_hand  # noqa: E402


def _rng(tag):
    return np.random.RandomState(zlib.crc32(tag.encode()) & 0x7FFFFFFF)


def _usable(hands):
    return [h for h in hands
            if isinstance(h, dict) and isinstance(h.get("actions"), list) and h["actions"]]


def roboticize_v2(hands, archetype, severity, rng):
    """One archetype, applied mildly. Action COUNTS never change (keeps the
    5-8 action live-window structure intact)."""
    hands = copy.deepcopy(hands)
    usable = _usable(hands)
    if len(usable) < 4:
        return hands

    if archetype == "repeat":
        # reuse a small pool of action lines across a fraction of hands
        n_t = max(2, int(round(5 * (1 - severity))))
        idx = rng.choice(len(usable), size=min(n_t, len(usable)), replace=False)
        templates = [copy.deepcopy(usable[i]["actions"]) for i in idx]
        own = sorted({int(p["seat"]) for h in usable for p in h.get("players", [])
                      if isinstance(p, dict) and p.get("seat")})
        for h in usable:
            if rng.random_sample() < severity:
                t = copy.deepcopy(templates[rng.randint(len(templates))])
                ts = []
                for a in t:
                    if a.get("actor_seat") is not None and a["actor_seat"] not in ts:
                        ts.append(a["actor_seat"])
                m = {s: own[i % len(own)] for i, s in enumerate(ts)} if own else {}
                for a in t:
                    if a.get("actor_seat") in m:
                        a["actor_seat"] = m[a["actor_seat"]]
                h["actions"] = t

    elif archetype == "entropy":
        # drift actions toward each street's modal action
        by_street = {}
        for h in usable:
            for a in h["actions"]:
                by_street.setdefault(str(a.get("street") or ""), []).append(
                    str(a.get("action_type") or ""))
        modal = {}
        for s, ts in by_street.items():
            v, c = np.unique([t for t in ts if t], return_counts=True)
            if len(v):
                modal[s] = str(v[np.argmax(c)])
        for h in usable:
            for a in h["actions"]:
                s = str(a.get("street") or "")
                if s in modal and rng.random_sample() < severity:
                    a["action_type"] = modal[s]
                    if modal[s] in ("fold", "check"):
                        a["amount"] = 0
                        a["normalized_amount_bb"] = 0

    elif archetype == "imbalance":
        # nudge the whole group toward its single most common action
        allacts = [str(a.get("action_type") or "") for h in usable for a in h["actions"]]
        v, c = np.unique([t for t in allacts if t], return_counts=True)
        if len(v):
            dom = str(v[np.argmax(c)])
            for h in usable:
                for a in h["actions"]:
                    if rng.random_sample() < severity * 0.5:
                        a["action_type"] = dom
                        if dom in ("fold", "check"):
                            a["amount"] = 0
                            a["normalized_amount_bb"] = 0

    elif archetype == "mirror":
        # make a fraction of hands near-clones of one seed hand's action string
        seed = copy.deepcopy(usable[rng.randint(len(usable))]["actions"])
        for h in usable:
            if rng.random_sample() < severity * 0.7:
                n = len(h["actions"])
                clone = copy.deepcopy(seed[:n]) if len(seed) >= n else copy.deepcopy(seed)
                for k, a in enumerate(clone):
                    if k < len(h["actions"]):
                        a["actor_seat"] = h["actions"][k].get("actor_seat")
                h["actions"] = clone + h["actions"][len(clone):]
    return hands


def main():
    man = PARAMS["manifold"]
    raw_members = PARAMS["raw_members"]
    fn_idx = [CANDIDATE_FEATURES.index(n) for n in PARAMS["feature_names"]]

    H = []
    for path in sorted(REPO.glob("local_test/data/release_2026-07-1*.json"))[-4:]:
        d = json.load(open(path))
        for g, y in zip(d["groups"], d["labels"]):
            if y == 0:
                H.append([build_miner_payload_hand(h) for h in g])
    caps = sorted(REPO.glob("local_test/captures/query_*.json"))
    L = json.load(open(caps[-1]))["chunks"]

    def score(groups):
        raw, dist = [], []
        for g in groups:
            try:
                f = extract_features(g)
                row = [f[k] for k in CANDIDATE_FEATURES]
            except Exception:
                continue
            raw.append(_members_probability(raw_members, [float(row[i]) for i in fn_idx]))
            dist.append(_manifold_distance(man, [float(v) for v in row]))
        return np.array(raw), np.array(dist)

    Lr, Ld = score(L)
    print(f"LIVE target band: score p25={np.quantile(Lr,.25):.3f} "
          f"p50={np.quantile(Lr,.5):.3f} p75={np.quantile(Lr,.75):.3f} | "
          f"dist p50={np.quantile(Ld,.5):.1f} p90={np.quantile(Ld,.9):.1f}")
    print("\narchetype  sev   robo_score(p25/p50/p75)   dist_p50   live-pctile   verdict")
    ARCHES = ("repeat", "entropy", "imbalance", "mirror")
    SEVS = (0.08, 0.15, 0.22, 0.30)
    for arch in ARCHES:
        for sev in SEVS:
            R = [roboticize_v2(H[i], arch, sev, _rng(f"{arch}{sev}g{i}"))
                 for i in range(len(H))]
            Rr, Rd = score(R)
            pct = float(np.mean(Ld[None, :] < Rd[:, None]))  # how extreme vs live
            in_band = np.mean((Rr >= 0.45) & (Rr <= 0.70))
            verdict = "GOOD" if 0.45 <= np.quantile(Rr, .5) <= 0.70 else \
                      ("too easy" if np.quantile(Rr, .5) > 0.70 else "too weak")
            print(f"{arch:9s} {sev:.2f}  {np.quantile(Rr,.25):.3f}/"
                  f"{np.quantile(Rr,.5):.3f}/{np.quantile(Rr,.75):.3f}"
                  f"      {np.quantile(Rd,.5):.1f}     {pct:.2f}"
                  f"        in_band={in_band:.0%} {verdict}")


if __name__ == "__main__":
    main()
