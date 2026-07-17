#!/usr/bin/env python3
"""
Poker44 — censored-feature discrimination analysis (real bots vs real humans).

1. Fetches labeled chunk groups from the public benchmark API (same flow as
   local_test/real_eval.py), caching raw data to disk.
2. Censors every hand with the production validator censor
   (poker44.validator.payload_view.prepare_hand_for_miner).
3. Engineers chunk-level features on the CENSORED view only.
4. Reports single-feature ROC AUC vs the real bot/human labels.

Run:
    cd /home/client_7075_3/Projects/Poker44-subnet
    PYTHONPATH=. .venv/bin/python local_test/feature_analysis.py
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import Counter
from typing import Any, Dict, List

from poker44.validator.payload_view import prepare_hand_for_miner

BASE = "https://api.poker44.net/api/v1/benchmark"
MAX_GROUPS = 400  # collect at least 300 if available
PAGE_LIMIT = 24
CACHE_PATH = os.environ.get(
    "P44_CACHE",
    "/tmp/claude-1000/-home-client-7075-3-Projects-Poker44-subnet/"
    "b6b862c4-370e-4b04-9250-4990a40580cc/scratchpad/benchmark_cache.json",
)

VISIBLE_BB = 0.02  # censor renders all money with bb = 0.02
STREET_ORDER = ("preflop", "flop", "turn", "river")


# --------------------------------------------------------------------------
# Data fetching (with disk cache)
# --------------------------------------------------------------------------
def fetch_real_chunk_groups(max_groups: int = MAX_GROUPS):
    import requests

    for attempt in range(3):
        try:
            status = requests.get(BASE, timeout=30).json()["data"]
            break
        except Exception as exc:  # noqa: BLE001
            print(f"[api] status fetch failed (try {attempt + 1}/3): {exc}")
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError("could not reach benchmark API")

    source_date = status["latestSourceDate"]
    print(f"[api] latest release {source_date} | "
          f"{status.get('totalChunks')} chunks, {status.get('totalHands')} hands")

    raw_groups: List[List[Dict[str, Any]]] = []
    labels: List[int] = []
    cursor = None
    while len(raw_groups) < max_groups:
        params: Dict[str, Any] = {"sourceDate": source_date, "limit": PAGE_LIMIT}
        if cursor:
            params["cursor"] = cursor
        page = None
        for attempt in range(3):
            try:
                page = requests.get(f"{BASE}/chunks", params=params, timeout=60).json()["data"]
                break
            except Exception as exc:  # noqa: BLE001
                print(f"[api] page fetch failed (try {attempt + 1}/3): {exc}")
                time.sleep(2 * (attempt + 1))
        if page is None:
            print("[api] giving up on this page; proceeding with what we have")
            break
        for rec in page["chunks"]:
            for grp, y in zip(rec["chunks"], rec["groundTruth"]):
                raw_groups.append(grp)
                labels.append(int(y))
                if len(raw_groups) >= max_groups:
                    break
            if len(raw_groups) >= max_groups:
                break
        cursor = page.get("nextCursor")
        print(f"[api] collected {len(raw_groups)} groups so far")
        if not cursor:
            break
    return raw_groups, labels, source_date


def load_or_fetch():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as fh:
            cache = json.load(fh)
        print(f"[cache] loaded {len(cache['groups'])} groups from {CACHE_PATH}")
        return cache["groups"], cache["labels"]
    groups, labels, source_date = fetch_real_chunk_groups()
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as fh:
        json.dump({"source_date": source_date, "groups": groups, "labels": labels}, fh)
    print(f"[cache] saved {len(groups)} groups to {CACHE_PATH}")
    return groups, labels


# --------------------------------------------------------------------------
# Feature engineering on CENSORED hands
# --------------------------------------------------------------------------
def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            ent -= p * math.log(p)
    return ent


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def chunk_features(censored_hands: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute chunk-level features from a group of censored hands."""
    f: Dict[str, float] = {}
    n_hands = len(censored_hands)

    act_counts = Counter()          # all actors
    hero_counts = Counter()         # hero only
    street_action_counts: Dict[str, Counter] = {s: Counter() for s in STREET_ORDER}
    bet_sizes: List[float] = []     # normalized_amount_bb of bet/raise actions
    stacks_bb: List[float] = []
    players_per_hand: List[float] = []
    actions_per_hand: List[float] = []
    streets_reached = Counter()     # per-hand furthest street presence
    hero_fold_hands = 0
    hero_any_action_hands = 0
    per_hand_aggr: List[float] = []
    per_hand_fold_ratio: List[float] = []
    per_hand_raise_ratio: List[float] = []
    amount_bucket_counter = Counter()  # coarse bucket usage across chunk
    pot_after_vals: List[float] = []
    unique_action_ids_seen = set()

    for hand in censored_hands:
        meta = hand.get("metadata") or {}
        hero_seat = int(meta.get("hero_seat", 0) or 0)
        players = hand.get("players") or []
        actions = hand.get("actions") or []
        streets = hand.get("streets") or []

        players_per_hand.append(float(len(players)))
        actions_per_hand.append(float(len(actions)))
        for p in players:
            st = float(p.get("starting_stack", 0.0) or 0.0) / VISIBLE_BB
            stacks_bb.append(st)

        street_names = {str(s.get("street", "")).lower() for s in streets}
        for s in STREET_ORDER[1:]:
            if s in street_names:
                streets_reached[s] += 1

        h_counts = Counter()
        hero_acted = False
        hero_folded = False
        for a in actions:
            at = str(a.get("action_type", "") or "")
            act_counts[at] += 1
            street = str(a.get("street", "") or "preflop").lower()
            if street in street_action_counts:
                street_action_counts[street][at] += 1
            amt = float(a.get("normalized_amount_bb", 0.0) or 0.0)
            pot_after_vals.append(float(a.get("pot_after", 0.0) or 0.0))
            if at in ("bet", "raise"):
                bet_sizes.append(amt)
                amount_bucket_counter[round(amt, 2)] += 1
            if int(a.get("actor_seat", 0) or 0) == hero_seat and hero_seat > 0:
                hero_counts[at] += 1
                h_counts[at] += 1
                hero_acted = True
                if at == "fold":
                    hero_folded = True
        if hero_acted:
            hero_any_action_hands += 1
            if hero_folded:
                hero_fold_hands += 1

        n_act = sum(1 for a in actions)  # window size for this hand
        if n_act > 0:
            aggr_n = sum(1 for a in actions if a.get("action_type") in ("bet", "raise"))
            pass_n = sum(1 for a in actions if a.get("action_type") in ("call", "check"))
            per_hand_aggr.append(aggr_n / max(1, aggr_n + pass_n))
            per_hand_fold_ratio.append(
                sum(1 for a in actions if a.get("action_type") == "fold") / n_act)
            per_hand_raise_ratio.append(
                sum(1 for a in actions if a.get("action_type") == "raise") / n_act)
        for a in actions:
            unique_action_ids_seen.add(str(a.get("action_id", "")))

    total_actions = sum(act_counts.values()) or 1
    total_hero = sum(hero_counts.values()) or 1

    # --- overall action-type ratios ---
    for at in ("check", "call", "bet", "raise", "fold"):
        f[f"ratio_{at}"] = act_counts[at] / total_actions
        f[f"hero_ratio_{at}"] = hero_counts[at] / total_hero

    # --- aggression factor ---
    aggr = act_counts["bet"] + act_counts["raise"]
    passive = act_counts["call"] + act_counts["check"]
    f["aggression_factor"] = aggr / max(1, passive)
    h_aggr = hero_counts["bet"] + hero_counts["raise"]
    h_pass = hero_counts["call"] + hero_counts["check"]
    f["hero_aggression_factor"] = h_aggr / max(1, h_pass)

    # --- street depth distribution ---
    for s in ("flop", "turn", "river"):
        f[f"frac_hands_reach_{s}"] = streets_reached[s] / max(1, n_hands)

    # --- bet sizing stats (bb) ---
    f["betsize_mean_bb"] = _mean(bet_sizes)
    f["betsize_std_bb"] = _std(bet_sizes)
    f["betsize_bucket_entropy"] = _entropy(amount_bucket_counter)
    if amount_bucket_counter:
        f["betsize_mode_share"] = amount_bucket_counter.most_common(1)[0][1] / sum(
            amount_bucket_counter.values())
        f["betsize_n_unique_buckets"] = float(len(amount_bucket_counter))
    else:
        f["betsize_mode_share"] = 0.0
        f["betsize_n_unique_buckets"] = 0.0

    # --- per-street raise / fold rates ---
    for s in STREET_ORDER:
        sc = street_action_counts[s]
        tot = sum(sc.values())
        f[f"{s}_raise_rate"] = sc["raise"] / max(1, tot)
        f[f"{s}_fold_rate"] = sc["fold"] / max(1, tot)
        f[f"{s}_bet_rate"] = sc["bet"] / max(1, tot)

    # --- starting stack stats ---
    f["stack_mean_bb"] = _mean(stacks_bb)
    f["stack_std_bb"] = _std(stacks_bb)
    f["stack_cv"] = f["stack_std_bb"] / max(1e-9, f["stack_mean_bb"])
    stack_counter = Counter(round(s, 1) for s in stacks_bb)
    f["stack_mode_share"] = (
        stack_counter.most_common(1)[0][1] / max(1, sum(stack_counter.values()))
        if stack_counter else 0.0)
    f["stack_n_unique"] = float(len(stack_counter))
    f["stack_frac_exact_100bb"] = (
        sum(1 for s in stacks_bb if abs(s - 100.0) < 0.5) / max(1, len(stacks_bb)))

    # --- players per hand ---
    f["players_per_hand_mean"] = _mean(players_per_hand)
    f["players_per_hand_std"] = _std(players_per_hand)

    # --- hand-level action-count stats (censored window sizes) ---
    f["actions_per_hand_mean"] = _mean(actions_per_hand)
    f["actions_per_hand_std"] = _std(actions_per_hand)

    # --- cross-hand consistency ---
    f["xhand_aggr_std"] = _std(per_hand_aggr)
    f["xhand_fold_ratio_std"] = _std(per_hand_fold_ratio)
    f["xhand_raise_ratio_std"] = _std(per_hand_raise_ratio)
    f["xhand_aggr_mean"] = _mean(per_hand_aggr)
    # entropy of discretized per-hand aggression (consistency signal)
    aggr_bins = Counter(min(9, int(a * 10)) for a in per_hand_aggr)
    f["xhand_aggr_entropy"] = _entropy(aggr_bins)

    # --- hero fold fraction ---
    f["hero_fold_hand_frac"] = hero_fold_hands / max(1, hero_any_action_hands)
    f["hero_active_hand_frac"] = hero_any_action_hands / max(1, n_hands)

    # --- pot dynamics ---
    f["pot_after_mean"] = _mean(pot_after_vals)
    f["pot_after_std"] = _std(pot_after_vals)

    f["n_hands"] = float(n_hands)
    return f


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def main() -> None:
    groups, labels = load_or_fetch()
    print(f"[data] {len(groups)} chunk groups | humans(0)={labels.count(0)} "
          f"bots(1)={labels.count(1)}")

    print("[censor] applying prepare_hand_for_miner to every hand ...")
    feats: List[Dict[str, float]] = []
    for gi, grp in enumerate(groups):
        censored = [prepare_hand_for_miner(h) for h in grp]
        feats.append(chunk_features(censored))
        if (gi + 1) % 50 == 0:
            print(f"  censored {gi + 1}/{len(groups)} groups")

    import numpy as np
    from sklearn.metrics import roc_auc_score

    names = sorted(feats[0].keys())
    X = np.array([[f.get(k, 0.0) for k in names] for f in feats], dtype=float)
    y = np.array(labels, dtype=int)

    rows = []
    for j, name in enumerate(names):
        col = X[:, j]
        if np.std(col) < 1e-12:
            auc = 0.5
        else:
            auc = roc_auc_score(y, col)
        bot = col[y == 1]
        hum = col[y == 0]
        rows.append({
            "feature": name,
            "auc": float(auc),
            "abs_dev": abs(auc - 0.5),
            "bot_mean": float(bot.mean()), "bot_std": float(bot.std()),
            "hum_mean": float(hum.mean()), "hum_std": float(hum.std()),
        })
    rows.sort(key=lambda r: -r["abs_dev"])

    print("\n" + "=" * 96)
    print(f"{'feature':34s} {'AUC':>7s} {'bot mean±std':>22s} {'human mean±std':>22s}")
    print("-" * 96)
    for r in rows:
        print(f"{r['feature']:34s} {r['auc']:7.4f} "
              f"{r['bot_mean']:11.4f}±{r['bot_std']:8.4f} "
              f"{r['hum_mean']:11.4f}±{r['hum_std']:8.4f}")
    print("=" * 96)

    # ------------------------------------------------------------------
    # Two-sided ("outlier") variants: bots are DISPERSED, not shifted.
    # AUC of |x - median(x)| captures symmetric variance differences that
    # a monotone single-feature AUC cannot.
    # ------------------------------------------------------------------
    two_sided = []
    for j, name in enumerate(names):
        col = X[:, j]
        if np.std(col) < 1e-12:
            continue
        dev = np.abs(col - np.median(col)) / np.std(col)
        two_sided.append({"feature": name, "auc_abs_dev": float(roc_auc_score(y, dev))})
    two_sided.sort(key=lambda r: -abs(r["auc_abs_dev"] - 0.5))
    print("\nTwo-sided |x - median| AUC (top 15):")
    for r in two_sided[:15]:
        print(f"  {r['feature']:34s} {r['auc_abs_dev']:.4f}")

    live = [j for j in range(len(names)) if np.std(X[:, j]) > 1e-12]
    Z = np.abs((X[:, live] - np.median(X[:, live], axis=0)) / (np.std(X[:, live], axis=0)))
    combined_auc = float(roc_auc_score(y, Z.mean(axis=1)))
    print(f"\nCombined mean-|z-from-median| outlier score AUC: {combined_auc:.4f}")

    out = {
        "n_groups": len(groups),
        "n_bots": int(y.sum()),
        "n_humans": int((1 - y).sum()),
        "features": rows,
        "two_sided_features": two_sided,
        "combined_outlier_auc": combined_auc,
    }
    out_path = os.path.join(os.path.dirname(CACHE_PATH), "feature_auc_results.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[out] results saved to {out_path}")


if __name__ == "__main__":
    main()
