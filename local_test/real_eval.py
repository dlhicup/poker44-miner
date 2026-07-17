#!/usr/bin/env python3
"""
Poker44 — REAL poker environment eval (offline grading on REAL recorded hands).

Same pipeline as local_eval.py, but instead of fabricated bots it pulls REAL
labeled poker hands (humans AND bots) from the public benchmark API:

    https://api.poker44.net/api/v1/benchmark      (no auth, no key)

Flow (identical to production grading, minus the blockchain):
    real chunk groups (with real 1=bot/0=human labels), AS-DELIVERED
      -> MODEL: one risk score per chunk group
      -> reward(y_pred, y_true)      (the exact validator metric)

The API's groups are ALREADY the validator's miner-visible payload, so they
are fed to the model untouched — exactly what a validator sends over the wire.

Run:
    cd /home/client_7075_3/Projects/Poker44-subnet
    PYTHONPATH=. python3 local_test/real_eval.py
Needs: `requests` (already in requirements.txt) + internet. numpy/sklearn
optional (falls back to the bit-faithful pure-Python reward()).
"""
from __future__ import annotations

import sys

# Reuse everything already built and verified in local_eval.py.
from local_test.local_eval import (
    MODEL,   # the model under test (your detector via your_model_score_chunk)
    grade,   # exact validator grading (reward once over all chunks)
)

BASE = "https://api.poker44.net/api/v1/benchmark"
MAX_GROUPS = 10_000  # no practical cap: fetch the ENTIRE latest release (~146 groups, ~5,000 hands)


def fetch_real_chunk_groups(max_groups: int = MAX_GROUPS):
    """Return (raw_groups, labels, source_date): real groups + 1=bot/0=human labels."""
    import requests

    status = requests.get(BASE, timeout=30).json()["data"]
    source_date = status["latestSourceDate"]  # source of truth; never hardcode
    print(f"[api] latest release {source_date} | "
          f"{status.get('totalChunks')} chunks, {status.get('totalHands')} hands")

    raw_groups, labels, cursor = [], [], None
    while len(raw_groups) < max_groups:
        params = {"sourceDate": source_date, "limit": 24}
        if cursor:
            params["cursor"] = cursor
        data = requests.get(f"{BASE}/chunks", params=params, timeout=60).json()["data"]
        for rec in data["chunks"]:
            groups = rec["chunks"]          # list of groups; each ~30 hands
            y_ints = rec["groundTruth"]     # 1=bot, 0=human (aligned to groups)
            for grp, y in zip(groups, y_ints):
                raw_groups.append(grp)
                labels.append(int(y))
                if len(raw_groups) >= max_groups:
                    break
            if len(raw_groups) >= max_groups:
                break
        cursor = data.get("nextCursor")
        if not cursor:
            break

    # Bank this release's labeled data permanently (local_test/data/), so every
    # daily test run also grows the training corpus for train_detector.py.
    try:
        import json
        import os
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(data_dir, exist_ok=True)
        bank_path = os.path.join(data_dir, f"release_{source_date}.json")
        if not os.path.exists(bank_path):
            with open(bank_path, "w") as fh:
                json.dump({"source_date": source_date, "groups": raw_groups,
                           "labels": labels}, fh)
            print(f"[bank] saved {len(raw_groups)} labeled groups -> {bank_path}")
        else:
            print(f"[bank] release {source_date} already banked ({bank_path})")
    except Exception as exc:  # banking must never break the eval
        print(f"[bank] could not save release ({type(exc).__name__}: {exc})")

    return raw_groups, labels, source_date


def main() -> None:
    print("=" * 72)
    print("Poker44 REAL poker environment eval  (real recorded hands, offline grade)")
    print("=" * 72)
    try:
        raw_groups, labels, source_date = fetch_real_chunk_groups()
    except Exception as exc:
        print(f"[api] fetch failed ({type(exc).__name__}: {exc}). Need internet + requests.")
        sys.exit(1)

    print(f"[data] {len(raw_groups)} real chunk groups | "
          f"humans(0)={labels.count(0)}  bots(1)={labels.count(1)}")

    # Score each group AS-DELIVERED. The benchmark API already returns the
    # validator's miner-visible payload (prepare_hand_for_miner applied
    # upstream), so re-censoring here would re-bucket/re-noise bet sizes and
    # re-sample the action window -> a distribution production never sends.
    y_pred = [MODEL(grp) for grp in raw_groups]

    rew, metrics, impl = grade(y_pred, labels)

    hs = [p for p, l in zip(y_pred, labels) if l == 0]
    bs = [p for p, l in zip(y_pred, labels) if l == 1]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print("-" * 72)
    print(f"MODEL: {MODEL.__name__}    REWARD IMPL: {impl}")
    print(f"  mean human score: {mean(hs):.4f}   mean bot score: {mean(bs):.4f}")
    print("-" * 72)
    print("GRADE on REAL data (what a validator would compute):")
    print(f"  reward     = {metrics['reward']:.6f}")
    print(f"  ap_score   = {metrics['ap_score']:.6f}")
    print(f"  bot_recall = {metrics['bot_recall']:.6f}   (recall @ FPR<=5%)")
    print(f"  fpr        = {metrics['fpr']:.6f}")
    print("=" * 72)
    print("This is the HONEST baseline: the reference heuristic on real bots.\n"
          "Beat this number with your own model at the plug-in point in local_eval.py.")

    # Append one summary line to local_test/logs/eval_history.jsonl (never fatal).
    try:
        from local_test.eval_logging import log_eval_result

        log_path = log_eval_result(
            script="real_eval",
            model_name=MODEL.__name__,
            metrics=metrics,
            extra={
                "release": source_date,
                "n_groups": len(raw_groups),
                "humans": labels.count(0),
                "bots": labels.count(1),
                "mean_human_score": round(mean(hs), 6),
                "mean_bot_score": round(mean(bs), 6),
                "reward_impl": impl,
            },
        )
        print(f"[log] result appended to {log_path}")
    except Exception as exc:  # logging must never break the eval
        print(f"[log] could not write history ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
