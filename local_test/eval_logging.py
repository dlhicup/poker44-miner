"""Tiny result logger for the local eval harnesses.

Every graded run appends ONE json line to local_test/logs/eval_history.jsonl,
so performance can be tracked run-over-run (and release-over-release) with:

    cat local_test/logs/eval_history.jsonl
or a quick table:
    PYTHONPATH=. python3 -c "from local_test.eval_logging import show_history; show_history()"

Logging must never break an eval run: callers wrap log_eval_result in
try/except, and this module itself only uses the stdlib.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

LOG_DIR = Path(__file__).resolve().parent / "logs"
HISTORY_PATH = LOG_DIR / "eval_history.jsonl"


def log_eval_result(
    script: str,
    model_name: str,
    metrics: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Append one summary line for a graded eval run; returns the log path."""
    LOG_DIR.mkdir(exist_ok=True)
    entry: Dict[str, Any] = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "script": script,
        "model": model_name,
    }
    if extra:
        entry.update(extra)
    for key, value in metrics.items():
        try:
            entry[key] = round(float(value), 6)
        except (TypeError, ValueError):
            entry[key] = value
    with open(HISTORY_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return HISTORY_PATH


def show_history(path: Path = HISTORY_PATH) -> None:
    """Print the eval history as an aligned, human-readable list."""
    if not path.exists():
        print(f"No history yet ({path}). Run real_eval.py or local_eval.py first.")
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            print(
                f"{e.get('timestamp', '?'):25s} {e.get('script', '?'):10s} "
                f"release={e.get('release', '-'):10s} groups={e.get('n_groups', '-'):>4} "
                f"reward={e.get('reward', float('nan')):.4f} "
                f"ap={e.get('ap_score', float('nan')):.4f} "
                f"recall@5={e.get('bot_recall', float('nan')):.4f} "
                f"model={e.get('model', '?')}"
            )
