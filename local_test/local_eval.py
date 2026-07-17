#!/usr/bin/env python3
"""
Poker44 — SELF-CONTAINED OFFLINE evaluation harness for miner developers.

WHAT THIS IS
------------
This lets you develop and grade a Poker44 bot-detection model on your laptop
with **NO blockchain, NO validator, NO eval/benchmark API call required**.
It reproduces the production grading pipeline exactly:

    raw hand payloads
      -> prepare_hand_for_miner()      (the same censor the validator applies)
      -> your model: one risk score per CHUNK (a chunk == one player's hands)
      -> reward(y_pred, y_true)         (the same reward the validator uses)
      -> print reward, ap_score, bot_recall, fpr

It imports ONLY the bittensor-free pieces of the repo:
  * poker44.validator.payload_view.prepare_hand_for_miner   (pure stdlib)
  * poker44.score.scoring.reward                            (numpy + sklearn)
The miner's baseline heuristic is COPIED inline (neurons/miner.py imports
bittensor at module top and cannot be imported offline).

DATA
----
  * HUMANS (label 0): the real local corpus
      hands_generator/human_hands/poker_hands_combined.json.gz
    (32,088 hands, every one label == "human"; it contains ZERO bots).
  * BOTS (label 1): SYNTHESIZED here, because there are no local bot hands.
    The synthetic bots are deliberately crude (unnaturally uniform, passive,
    always-reach-showdown) so a baseline model can distinguish them and score
    meaningfully above random. THEY ARE NOT REAL BOTS. For serious training,
    replace them with labeled bot chunks from the public benchmark API — see
    fetch_benchmark_examples() near the bottom (a ready-to-use, commented
    recipe: status -> latestSourceDate -> chunks, no auth required).

RUN
---
    cd /home/client_7075_3/Projects/Poker44-subnet
    PYTHONPATH=/home/client_7075_3/Projects/Poker44-subnet python3 \
        <scratchpad>/local_eval.py

DEPENDENCIES
------------
Only the official reward() needs numpy + scikit-learn. If they are not
installed this harness AUTOMATICALLY falls back to a bit-faithful pure-Python
reimplementation of reward()/average_precision/recall@fpr, so it still runs
end-to-end with zero heavy deps. Install the real deps for the authoritative
path:  pip install -r requirements.txt   (or: pip install numpy scikit-learn).
"""

from __future__ import annotations

import gzip
import json
import os
import random
import sys

# --------------------------------------------------------------------------- #
# 0. Repo import path                                                          #
# --------------------------------------------------------------------------- #
REPO_ROOT = "/home/client_7075_3/Projects/Poker44-subnet"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

HUMAN_HANDS_GZ = os.path.join(
    REPO_ROOT, "hands_generator", "human_hands", "poker_hands_combined.json.gz"
)

# The censor is pure stdlib and must always import; fail loudly & helpfully.
try:
    from poker44.validator.payload_view import prepare_hand_for_miner
except Exception as exc:  # pragma: no cover - only hit on a broken PYTHONPATH
    sys.stderr.write(
        "FATAL: could not import poker44.validator.payload_view.\n"
        f"       ({type(exc).__name__}: {exc})\n"
        "       Run from the repo root with "
        f"PYTHONPATH={REPO_ROOT}\n"
    )
    raise

# The official reward() needs numpy + scikit-learn. Use it when present,
# otherwise fall back to the pure-Python reimplementation defined below.
try:
    import numpy as _np
    from poker44.score.scoring import reward as _official_reward

    HAVE_OFFICIAL_REWARD = True
    _REWARD_IMPORT_ERR = None
except Exception as exc:  # numpy / sklearn (or scoring.py) unavailable
    HAVE_OFFICIAL_REWARD = False
    _REWARD_IMPORT_ERR = f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- #
# 1. Load real HUMAN hands (verified loader)                                   #
# --------------------------------------------------------------------------- #
def load_human_hands(path: str = HUMAN_HANDS_GZ, n: int | None = None) -> list[dict]:
    """Return a list of full hand-payload dicts.

    The .gz is ONE pretty-printed JSON array (NOT JSONL), so json.load parses
    the whole ~32k-hand document at once (~8.5 MB compressed, modest RAM).
    Each element already matches HandHistory.to_payload() /
    prepare_hand_for_miner input shape.
    """
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        hands = json.load(fh)  # -> list[dict], 32,088 items
    return hands if n is None else hands[:n]


# --------------------------------------------------------------------------- #
# 2. SYNTHESIZE crude "bot-like" hands (label 1)                               #
#                                                                             #
#    There are NO local bot hands, so we fabricate obviously-uniform,          #
#    passive hands that always reach showdown. This is ONLY so the harness     #
#    runs offline with a labeled 2-class set. Real bots are far subtler --     #
#    swap these for benchmark-API bot chunks (see fetch_benchmark_examples).   #
#                                                                             #
#    Shape mirrors the real corpus exactly (metadata/players/streets/actions/  #
#    outcome/label) so it flows through prepare_hand_for_miner identically.    #
# --------------------------------------------------------------------------- #
def _synth_bot_hand(rng: random.Random) -> dict:
    """One fabricated bot hand: 6 seats, reaches the river, and every non-blind
    action is a passive call/check with metronomic bet sizing. Post-censor this
    yields high street-depth + high call/check ratios + ~zero fold/raise, which
    the baseline heuristic (below) reads as 'bot-like'."""
    sb, bb = 0.01, 0.02
    seats = [1, 2, 3, 4, 5, 6]
    button = rng.choice(seats)
    hero = rng.choice(seats)

    players = [
        {
            "player_uid": f"bot_seat_{s}",
            "seat": s,
            "starting_stack": 2.0,       # unnaturally identical stacks
            "hole_cards": None,
            "showed_hand": False,
        }
        for s in seats
    ]

    streets = [
        {"street": "flop", "board_cards": ["4s", "6h", "Js"]},
        {"street": "turn", "board_cards": ["4s", "6h", "Js", "Ad"]},
        {"street": "river", "board_cards": ["4s", "6h", "Js", "Ad", "9d"]},
    ]

    actions: list[dict] = []
    aid = 0
    pot = 0.0

    def add(street: str, seat: int, atype: str, amount: float) -> None:
        nonlocal aid, pot
        aid += 1
        before = pot
        pot = round(pot + amount, 4)
        actions.append(
            {
                "action_id": str(aid),
                "street": street,
                "actor_seat": seat,
                "action_type": atype,
                "amount": amount,
                "raise_to": None,
                "call_to": None,
                "normalized_amount_bb": round(amount / bb, 4),
                "pot_before": before,
                "pot_after": pot,
            }
        )

    # Blinds (prepare_hand_for_miner drops these; included for realism).
    add("preflop", 2, "small_blind", sb)
    add("preflop", 3, "big_blind", bb)
    # Two bots limp/call preflop then passively call/check every street.
    contenders = [4, 5]
    for s in contenders:
        add("preflop", s, "call", bb)             # metronomic flat call
    for street in ("flop", "turn", "river"):
        for s in contenders:
            # alternate check / small identical call so actions survive the
            # de-dup collapse and fill the 5-8 action censor window.
            if (s + len(actions)) % 2 == 0:
                add(street, s, "check", 0.0)
            else:
                add(street, s, "call", 0.04)      # identical 2bb call, always

    outcome = {
        "winners": [f"bot_seat_{contenders[0]}"],
        "payouts": {f"bot_seat_{contenders[0]}": round(pot, 4)},
        "total_pot": round(pot, 4),
        "rake": 0.0,
        "result_reason": "showdown",
        "showdown": True,
    }

    return {
        "metadata": {
            "game_type": "Hold'em",
            "limit_type": "No Limit",
            "max_seats": 6,
            "hero_seat": hero,
            "hand_ended_on_street": "river",
            "button_seat": button,
            "sb": sb,
            "bb": bb,
            "ante": 0.0,
            "rng_seed_commitment": None,
        },
        "players": players,
        "streets": streets,
        "actions": actions,
        "outcome": outcome,
        "label": "bot",   # stripped by prepare_hand_for_miner; here for clarity
    }


def synth_bot_hands(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    return [_synth_bot_hand(rng) for _ in range(n)]


# --------------------------------------------------------------------------- #
# 3a. BASELINE MODEL — copied verbatim from neurons/miner.py                    #
#     (that module imports bittensor at the top, so we copy the heuristic       #
#      rather than import it). This is the reference miner. REPLACE IT with     #
#      your own model at the plug-in point in section 3b.                       #
# --------------------------------------------------------------------------- #
from collections import Counter  # noqa: E402  (kept next to the code that uses it)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _score_hand(hand: dict) -> float:
    actions = hand.get("actions") or []
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    outcome = hand.get("outcome") or {}

    action_counts = Counter(action.get("action_type") for action in actions)
    meaningful_actions = max(
        1,
        sum(action_counts.get(k, 0) for k in ("call", "check", "bet", "raise", "fold")),
    )
    call_ratio = action_counts.get("call", 0) / meaningful_actions
    check_ratio = action_counts.get("check", 0) / meaningful_actions
    fold_ratio = action_counts.get("fold", 0) / meaningful_actions
    raise_ratio = action_counts.get("raise", 0) / meaningful_actions
    street_depth = len(streets) / 3.0
    showdown_flag = 1.0 if outcome.get("showdown") else 0.0

    player_count_signal = 0.0
    if players:
        player_count_signal = (6 - min(len(players), 6)) / 4.0

    score = 0.0
    score += 0.32 * street_depth
    score += 0.22 * showdown_flag
    score += 0.18 * _clamp01(call_ratio / 0.35)
    score += 0.12 * _clamp01(check_ratio / 0.30)
    score += 0.08 * _clamp01(player_count_signal)
    score -= 0.18 * _clamp01(fold_ratio / 0.55)
    score -= 0.10 * _clamp01(raise_ratio / 0.20)
    return _clamp01(score)


def score_chunk(chunk: list[dict]) -> float:
    """One bot-risk score in [0,1] for a whole chunk (list of CENSORED hands).

    NOTE: after prepare_hand_for_miner() every hand's outcome.showdown is False,
    so the 0.22*showdown term contributes 0 in practice — street depth and the
    call/check vs fold/raise ratios do the real separating work.
    """
    if not chunk:
        return 0.5
    hand_scores = [_score_hand(h) for h in chunk]
    return round(_clamp01(sum(hand_scores) / len(hand_scores)), 6)


# --------------------------------------------------------------------------- #
# 3b. >>> PLUG IN YOUR OWN MODEL HERE <<<                                       #
#     Implement a function that takes ONE chunk (a list of CENSORED hand        #
#     dicts, exactly what prepare_hand_for_miner produces) and returns ONE      #
#     float risk score in [0,1] (higher == more bot-like). Then point MODEL     #
#     at it. That is the only change needed to grade YOUR detector offline.     #
# --------------------------------------------------------------------------- #
def your_model_score_chunk(chunk: list[dict]) -> float:
    # The trained dispersion detector (pure stdlib, bittensor-free).
    from neurons.detector import score_chunk as detector_score_chunk

    return detector_score_chunk(chunk)


# The dispersion detector is the model under test; `score_chunk` is the old
# reference heuristic, kept for comparison.
MODEL = your_model_score_chunk


# --------------------------------------------------------------------------- #
# 4a. Pure-Python fallback reward() — bit-faithful to poker44.score.scoring     #
#     Used only when numpy/scikit-learn are unavailable.                        #
# --------------------------------------------------------------------------- #
def _average_precision_py(y_true: list[int], y_score: list[float]) -> float:
    """Match sklearn.metrics.average_precision_score for binary labels:
    AP = sum_k (R_k - R_{k-1}) * P_k, evaluated at each distinct score
    threshold from highest to lowest, ties grouped together, R_0 = 0."""
    positives = sum(1 for t in y_true if t == 1)
    if positives == 0 or not y_score:
        return 0.0
    pairs = sorted(zip(y_score, y_true), key=lambda p: p[0], reverse=True)
    tp = fp = 0
    prev_recall = 0.0
    ap = 0.0
    i, n = 0, len(pairs)
    while i < n:
        score_val = pairs[i][0]
        j = i
        while j < n and pairs[j][0] == score_val:  # group tied scores
            if pairs[j][1] == 1:
                tp += 1
            else:
                fp += 1
            j += 1
        recall = tp / positives
        precision = tp / (tp + fp)
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j
    return ap


def _recall_at_fpr_py(y_score, y_true, max_fpr: float = 0.05):
    """Pure-Python twin of poker44.score.scoring._recall_at_fpr: best bot recall
    reachable while human FPR stays <= max_fpr (first argmax on ties)."""
    labels = [int(v) for v in y_true]
    scores = [float(v) for v in y_score]
    positives = sum(1 for l in labels if l == 1)
    negatives = sum(1 for l in labels if l == 0)
    if positives <= 0 or negatives <= 0 or not scores:
        return 0.0, 0.0
    # argsort(-scores) with a stable sort; ties keep original order.
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = fp = 0
    best_recall = 0.0
    best_fpr = 0.0
    found = False
    for idx in order:
        if labels[idx] == 1:
            tp += 1
        else:
            fp += 1
        fpr = fp / negatives
        recall = tp / positives
        if fpr <= max_fpr and (not found or recall > best_recall):
            best_recall, best_fpr, found = recall, fpr, True
    if not found:
        return 0.0, 0.0
    return best_recall, best_fpr


def _reward_py(y_pred: list[float], y_true) -> tuple[float, dict]:
    preds = [float(v) for v in y_pred]
    truth = [1 if bool(v) else 0 for v in y_true]
    if preds and any(t == 1 for t in truth):
        ap_score = _average_precision_py(truth, preds)
    else:
        ap_score = 0.0
    bot_recall, fpr = _recall_at_fpr_py(preds, truth, max_fpr=0.05)
    human_safety_penalty = 1.0
    base_score = 0.75 * ap_score + 0.25 * bot_recall
    rew = base_score * human_safety_penalty
    return rew, {
        "fpr": fpr,
        "bot_recall": bot_recall,
        "ap_score": ap_score,
        "human_safety_penalty": human_safety_penalty,
        "base_score": base_score,
        "reward": rew,
    }


# --------------------------------------------------------------------------- #
# 4b. grade(): the exact validator grading call (reward once over ALL chunks)   #
# --------------------------------------------------------------------------- #
def grade(y_pred: list[float], y_true: list[int]) -> tuple[float, dict, str]:
    """Grade one miner. reward() is computed ONCE over the whole per-chunk
    arrays (never per chunk) — mirroring forward.py:622-624. Returns
    (reward, metrics, which_impl)."""
    if HAVE_OFFICIAL_REWARD:
        rew, metrics = _official_reward(
            _np.asarray(y_pred, dtype=float), _np.asarray(y_true, dtype=bool)
        )
        return float(rew), {k: float(v) for k, v in metrics.items()}, "official (numpy+sklearn)"
    rew, metrics = _reward_py(y_pred, y_true)
    return float(rew), {k: float(v) for k, v in metrics.items()}, "pure-python fallback"


# --------------------------------------------------------------------------- #
# 5. Build the labeled offline eval set and run the pipeline                    #
# --------------------------------------------------------------------------- #
HANDS_PER_CHUNK = 30      # ~30 hands per chunk == production chunk-group size
NUM_HUMAN_CHUNKS = 20     # label 0
NUM_BOT_CHUNKS = 20       # label 1


def build_chunks() -> tuple[list[list[dict]], list[int]]:
    """Return (raw_chunks, labels). Each chunk is one single-label list of raw
    hand payloads. 1 == bot, 0 == human (validator convention: forward.py:127
    with runtime_provider.py:453 is_human = not is_bot)."""
    need_humans = NUM_HUMAN_CHUNKS * HANDS_PER_CHUNK
    humans = load_human_hands(n=need_humans)
    if len(humans) < need_humans:
        raise RuntimeError(
            f"needed {need_humans} human hands but only {len(humans)} available"
        )
    bots = synth_bot_hands(NUM_BOT_CHUNKS * HANDS_PER_CHUNK, seed=0)

    raw_chunks: list[list[dict]] = []
    labels: list[int] = []

    for c in range(NUM_HUMAN_CHUNKS):
        raw_chunks.append(humans[c * HANDS_PER_CHUNK : (c + 1) * HANDS_PER_CHUNK])
        labels.append(0)
    for c in range(NUM_BOT_CHUNKS):
        raw_chunks.append(bots[c * HANDS_PER_CHUNK : (c + 1) * HANDS_PER_CHUNK])
        labels.append(1)

    return raw_chunks, labels


def censor_chunk(raw_chunk: list[dict]) -> list[dict]:
    """Apply the EXACT production censor to every hand in the chunk.
    prepare_hand_for_miner expects a payload dict (HandHistory.to_payload()
    shape); our raw hands already are dicts."""
    return [prepare_hand_for_miner(hand) for hand in raw_chunk]


def main() -> None:
    print("=" * 72)
    print("Poker44 OFFLINE eval harness  (no blockchain, no eval API)")
    print("=" * 72)

    if not HAVE_OFFICIAL_REWARD:
        print(
            "[deps] numpy/scikit-learn not importable -> using the pure-Python\n"
            f"       reward() fallback (bit-faithful). Reason: {_REWARD_IMPORT_ERR}\n"
            "       Install the authoritative path with: "
            "pip install -r requirements.txt\n"
        )
    else:
        print("[deps] using the OFFICIAL poker44.score.scoring.reward "
              "(numpy + scikit-learn)\n")

    # 1) Build a labeled 2-class set of raw chunks.
    raw_chunks, labels = build_chunks()
    print(
        f"[data] {len(raw_chunks)} chunks x {HANDS_PER_CHUNK} hands "
        f"= {len(raw_chunks) * HANDS_PER_CHUNK} hands  "
        f"| humans(0)={labels.count(0)}  bots(1)={labels.count(1)}"
    )

    # 2) Censor every hand exactly as the validator does before the miner sees it.
    censored_chunks = [censor_chunk(rc) for rc in raw_chunks]
    print("[censor] prepare_hand_for_miner applied to every hand "
          "(label/hole_cards/outcome stripped, seats aliased, bb-bucketed)")

    # 3) Score: one risk score per chunk with the selected MODEL.
    y_pred = [MODEL(chunk) for chunk in censored_chunks]
    y_true = labels

    # 4) Grade: reward() ONCE over all per-chunk scores/labels.
    rew, metrics, impl = grade(y_pred, y_true)

    # Separation diagnostics (not part of grading, just to see the model works).
    human_scores = [p for p, l in zip(y_pred, y_true) if l == 0]
    bot_scores = [p for p, l in zip(y_pred, y_true) if l == 1]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")

    print("\n" + "-" * 72)
    print(f"MODEL: {MODEL.__name__}    REWARD IMPL: {impl}")
    print("-" * 72)
    print(f"  mean human-chunk score : {mean(human_scores):.4f}  "
          f"(range {min(human_scores):.4f}..{max(human_scores):.4f})")
    print(f"  mean bot-chunk   score : {mean(bot_scores):.4f}  "
          f"(range {min(bot_scores):.4f}..{max(bot_scores):.4f})")
    print("-" * 72)
    print("GRADE (what the validator would compute for this miner):")
    print(f"  reward     = {metrics['reward']:.6f}")
    print(f"  ap_score   = {metrics['ap_score']:.6f}")
    print(f"  bot_recall = {metrics['bot_recall']:.6f}   (recall @ FPR<=5%)")
    print(f"  fpr        = {metrics['fpr']:.6f}")
    print(f"  base_score = {metrics['base_score']:.6f}   "
          f"(= 0.75*ap_score + 0.25*bot_recall)")
    print("=" * 72)
    print(
        "Random baseline reward ~= prevalence-driven AP (~0.5 here) blended with\n"
        "0.25*recall; a reward well above that means the model separates the two\n"
        "classes. The crude synthetic bots make this easy on purpose — swap in\n"
        "real benchmark-API bot chunks (fetch_benchmark_examples) for a true test."
    )

    # Append one summary line to local_test/logs/eval_history.jsonl (never fatal).
    try:
        from local_test.eval_logging import log_eval_result

        log_path = log_eval_result(
            script="local_eval",
            model_name=MODEL.__name__,
            metrics=metrics,
            extra={
                "release": "local-corpus+synth-bots",
                "n_groups": len(raw_chunks),
                "humans": y_true.count(0),
                "bots": y_true.count(1),
                "mean_human_score": round(mean(human_scores), 6),
                "mean_bot_score": round(mean(bot_scores), 6),
                "reward_impl": impl,
            },
        )
        print(f"[log] result appended to {log_path}")
    except Exception as exc:  # logging must never break the eval
        print(f"[log] could not write history ({type(exc).__name__}: {exc})")


# --------------------------------------------------------------------------- #
# 6. REAL bots: public benchmark API recipe (COMMENTED — needs internet only,   #
#    no auth). Use this to REPLACE synth_bot_hands() with genuine labeled bot   #
#    (and human) chunks. Every response is wrapped in {success, data:{...}} so  #
#    you must index ['data']. Labels live at the RECORD level (groundTruth ints #
#    1=bot/0=human), positionally aligned to record['chunks'] — one label and   #
#    one prediction per chunk GROUP (~30 hands), never per hand.                #
# --------------------------------------------------------------------------- #
def fetch_benchmark_examples(limit_per_page: int = 24, split: str | None = None):
    """Return [(group_hands, y_int, y_str, chunk_hash), ...] from the public
    Poker44 benchmark API. NEEDS INTERNET (no credentials). Each group is the
    miner-visible model input; feed group_hands straight into MODEL() after
    censoring, exactly like the synthetic path. Requires the `requests` package.

    Wire the results into build_chunks() to replace the synthetic bots:
        examples = fetch_benchmark_examples()
        raw_chunks = [g for (g, y, ys, h) in examples]
        labels     = [y for (g, y, ys, h) in examples]
    """
    import requests  # not imported at module top so offline runs never need it

    base = "https://api.poker44.net/api/v1/benchmark"  # public, no auth

    # (A) status -> latest release date (source of truth; do NOT hardcode).
    status = requests.get(base, timeout=30).json()["data"]
    source_date = status["latestSourceDate"]

    # (B) page through ALL chunk records for that release via nextCursor.
    records, cursor = [], None
    while True:
        params = {"sourceDate": source_date, "limit": limit_per_page}
        if split:
            params["split"] = split          # optional: "train" | "validation"
        if cursor:
            params["cursor"] = cursor
        data = requests.get(f"{base}/chunks", params=params, timeout=60).json()["data"]
        records.extend(data["chunks"])
        cursor = data.get("nextCursor")
        if not cursor:
            break

    # (C) one labeled example per chunk GROUP (aligned to groundTruth).
    examples = []
    for rec in records:
        groups = rec["chunks"]              # list of groups; each ~30 hand dicts
        y_ints = rec["groundTruth"]         # 1=bot, 0=human
        y_strs = rec["groundTruthLabels"]   # "bot"/"human"
        assert len(groups) == len(y_ints) == len(y_strs)
        for grp, y, ys in zip(groups, y_ints, y_strs):
            examples.append((grp, y, ys, rec["chunkHash"]))
    return examples


if __name__ == "__main__":
    main()
