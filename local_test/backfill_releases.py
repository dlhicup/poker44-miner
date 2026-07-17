#!/usr/bin/env python3
"""Backfill ALL available Poker44 benchmark releases into local_test/data/.

The public benchmark API keeps a rolling window of daily releases (currently
~30 days). This script enumerates GET /releases and downloads every release
not already banked locally, saving each as local_test/data/release_<date>.json
in the same format real_eval.py banks ({source_date, groups, labels}).

Already-banked releases are skipped, so reruns are cheap — run this any time
to top up history. Releases are immutable once published.

Run:
    cd /home/client_7075_3/Projects/Poker44-subnet
    PYTHONPATH=. .venv/bin/python local_test/backfill_releases.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

BASE = "https://api.poker44.net/api/v1/benchmark"
DATA_DIR = Path(__file__).resolve().parent / "data"
PAGE_LIMIT = 24
PAUSE_SECONDS = 0.5  # politeness delay between requests


def list_releases() -> list[str]:
    data = requests.get(f"{BASE}/releases", timeout=30).json()["data"]
    releases = data["releases"]
    dates = []
    for r in releases:
        if isinstance(r, dict):
            dates.append(r.get("sourceDate") or r.get("source_date"))
        else:
            dates.append(str(r))
    return sorted(d for d in dates if d)


def fetch_release(source_date: str) -> tuple[list, list]:
    groups, labels, cursor = [], [], None
    while True:
        params = {"sourceDate": source_date, "limit": PAGE_LIMIT}
        if cursor:
            params["cursor"] = cursor
        data = requests.get(f"{BASE}/chunks", params=params, timeout=60).json()["data"]
        for rec in data["chunks"]:
            for grp, y in zip(rec["chunks"], rec["groundTruth"]):
                groups.append(grp)
                labels.append(int(y))
        cursor = data.get("nextCursor")
        if not cursor:
            return groups, labels
        time.sleep(PAUSE_SECONDS)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    dates = list_releases()
    print(f"[api] {len(dates)} releases available: {dates[0]} .. {dates[-1]}")

    have = {p.stem.replace("release_", "") for p in DATA_DIR.glob("release_*.json")}
    todo = [d for d in dates if d not in have]
    print(f"[plan] already banked: {len(have)} | to download: {len(todo)}")

    total_groups = 0
    for i, date in enumerate(todo, 1):
        try:
            groups, labels = fetch_release(date)
        except Exception as exc:
            print(f"[{i}/{len(todo)}] {date}  FAILED ({type(exc).__name__}: {exc}) — "
                  "skipping, rerun later to retry")
            time.sleep(2.0)
            continue
        path = DATA_DIR / f"release_{date}.json"
        with open(path, "w") as fh:
            json.dump({"source_date": date, "groups": groups, "labels": labels}, fh)
        total_groups += len(groups)
        print(f"[{i}/{len(todo)}] {date}  {len(groups)} groups "
              f"(humans={labels.count(0)} bots={labels.count(1)}) -> {path.name}")
        time.sleep(PAUSE_SECONDS)

    banked = sorted(DATA_DIR.glob("release_*.json"))
    print(f"\n[done] downloaded {total_groups} new groups | "
          f"{len(banked)} releases now banked in {DATA_DIR}")


if __name__ == "__main__":
    main()
