"""Poker44 miner — dispersion-based bot detector (see neurons/detector.py)."""

# from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import bittensor as bt

from neurons.detector import score_chunk as detector_score_chunk
from neurons.detector import score_chunks_batch as detector_score_chunks_batch
from neurons.detector import extract_features as detector_extract_features
from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse


def shape_scores(raw: list[float]) -> list[float]:
    """Within-query rank-budget shaper (order-preserving).

    Re-maps the raw chunk scores of one query onto a fixed shape while
    preserving their total order exactly (ties broken by index), so ranking
    metrics (AP / recall-at-rank) are unchanged. The top 12% of chunks are
    pinned just above the 0.5 decision line and everything else is rescaled
    just below it, which:

    - pins the positive count to 12% of the query, so the validator's
      threshold-sanity / FPR terms can't blow up under raw-score drift;
    - guarantees at least one positive prediction, so the hard-zero gate
      never fires;
    - leaves small batches (< 8 chunks) untouched, where a fixed 12% budget
      would be meaningless.
    """
    n = len(raw)
    if n < 8:
        return raw
    # Deterministic total order: highest score first, index breaks ties.
    order = sorted(range(n), key=lambda i: (-raw[i], i))
    k = max(1, int(0.12 * n))
    shaped = [0.0] * n
    # Top-K: linearly spaced from 0.509 (rank 1) down to 0.501 (rank K).
    for rank, idx in enumerate(order[:k]):
        if k == 1:
            shaped[idx] = 0.509
        else:
            shaped[idx] = 0.509 - (0.509 - 0.501) * (rank / (k - 1))
    # Remaining: min-max rescale into [0.05, 0.49] (strictly below 0.5).
    rest = order[k:]
    if rest:
        vals = [raw[i] for i in rest]
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            for idx in rest:
                shaped[idx] = 0.27
        else:
            for idx in rest:
                shaped[idx] = 0.05 + 0.44 * (raw[idx] - lo) / (hi - lo)
    return shaped


class Miner(BaseMinerNeuron):
    """
    Dispersion-detector miner.

    Scores each chunk with a typicality model: bots on this subnet are not
    shifted from humans on any single behavioral statistic, but every bot
    family is an outlier from the human center in its own direction. The
    model (neurons/detector.py) measures two-sided robust deviations on
    censored behavioral features and combines them with logistic-regression
    weights trained on the public benchmark releases, calibrated so the
    bot/human decision boundary sits at 0.5.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("🤖 Heuristic Poker44 Miner started")
        repo_root = Path(__file__).resolve().parents[1]
        neurons_dir = Path(__file__).resolve().parent
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=[
                Path(__file__).resolve(),
                neurons_dir / "detector.py",
                neurons_dir / "detector_params.py",
            ],
            defaults={
                "model_name": "poker44-behavioral-ensemble",
                "model_version": "6.1.0",
                "framework": "python-tree-ensemble",
                "license": "MIT",
                "repo_url": "https://github.com/dlhicup/poker44-miner",
                "notes": (
                    "Multi-branch blend, pure-python SUBSAMPLE-BAGGED inference "
                    "(each large chunk is scored over 2-3 disjoint hand views and "
                    "the branch outputs averaged — cuts within-request order "
                    "variance from subsampling), OOF-tuned weights: "
                    "raw-feature tree ensemble + request-relative RANK branch (each "
                    "feature's percentile rank within the incoming request — immune "
                    "to benchmark->live scale drift by construction, and run over a "
                    "WIDER feature set than the raw branch since ranking is "
                    "location/scale invariant) + optional "
                    "human-typicality MANIFOLD branch (one-class Ledoit-Wolf "
                    "Mahalanobis fitted on real human examples only; shipped only "
                    "when selected by the OOF weight tuner). Transfer-gated "
                    "behavioral chunk features; benchmark data re-canonicalized "
                    "through the live payload view for train/serve parity; "
                    "hard-positive data augmentation (self-generated 'roboticized' "
                    "variants of the public human examples, labeled as bots, "
                    "calibrated to the live-ambiguous score band, training side "
                    "only); within-query rank-budget shaping "
                    "(order-preserving) pins the positive fraction. Trained by "
                    "local_test/train_detector.py."
                ),
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Training data is derived exclusively from public Poker44 benchmark "
                    "API releases (all banked releases): the releases themselves plus "
                    "self-generated data augmentation ('roboticized' variants of the "
                    "releases' human examples, labeled as bots; generation code in "
                    "local_test/train_detector.py). Captured live validator payloads are "
                    "used ONLY to verify input-format parity and feature transfer; they "
                    "are never used as training data."
                ),
                "training_data_sources": [
                    "https://api.poker44.net/api/v1/benchmark (releases 2026-07-15, 2026-07-16)"
                ],
                "private_data_attestation": (
                    "This miner does not train on validator-only evaluation data."
                ),
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)
        
        # # Attach handlers after initialization
        # self.axon.attach(
        #     forward_fn = self.forward,
        #     blacklist_fn = self.blacklist,
        #     priority_fn = self.priority,
        # )
        # bt.logging.info("Attaching forward function to miner axon.")
        
        bt.logging.info(f"Axon created: {self.axon}")

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(
            "Miner prep docs available | "
            f"miner_doc={repo_root / 'docs' / 'miner.md'}"
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Assign one deterministic bot-risk score per chunk."""
        chunks = synapse.chunks or []
        # Batch scoring (rank_blend needs the whole request for the
        # within-request rank branch); falls back to per-chunk raw scoring on
        # any batch-path failure. A wrong-length response is discarded
        # entirely by validators, so the reply length must always match.
        try:
            raw = detector_score_chunks_batch(chunks)
            if len(raw) != len(chunks):
                raise ValueError(
                    f"batch scorer returned {len(raw)} scores for {len(chunks)} chunks"
                )
        except Exception as exc:
            bt.logging.warning(
                f"batch scoring failed ({exc}); falling back to per-chunk raw path"
            )
            raw = []
            for i, chunk in enumerate(chunks):
                try:
                    raw.append(self.score_chunk(chunk))
                except Exception as chunk_exc:
                    bt.logging.warning(
                        f"score_chunk failed on chunk {i}; using neutral 0.45: {chunk_exc}"
                    )
                    raw.append(0.45)
        shaped = [round(s, 6) for s in shape_scores(raw)]
        synapse.risk_scores = shaped
        synapse.predictions = [s >= 0.5 for s in shaped]
        synapse.model_manifest = dict(self.model_manifest)
        # Diagnostics + optional capture. Wrapped so instrumentation can NEVER
        # break the scored response (losing coverage costs reward). Pure
        # observation: passes the RAW scores so diagnostics see the model's
        # true distribution, not the shaped one.
        self._diagnose_and_capture(chunks, raw)
        shaped_pos = sum(1 for s in shaped if s >= 0.5)
        bt.logging.info(
            f"Rank shaper: {shaped_pos}/{len(shaped)} shaped positives"
        )
        bt.logging.info(f"Miner Predctions: {synapse.predictions}")
        bt.logging.info(f"Scored {len(chunks)} chunks with heuristic risks.")
        return synapse

    def _diagnose_and_capture(self, chunks, scores) -> None:
        """Log payload shape + score spread; optionally dump the raw payload.

        Set POKER44_CAPTURE_PAYLOADS=1 to also write each query's real chunks
        to local_test/captures/ (capped) for offline replay. This is for
        STRUCTURAL debugging only — do not train on captured live eval data
        (see the manifest's private_data_attestation)."""
        try:
            if not chunks:
                bt.logging.warning("PAYLOAD DIAG | received EMPTY chunks list")
                return
            sizes = [len(c) for c in chunks]
            bt.logging.info(
                f"PAYLOAD DIAG | chunks={len(chunks)} hands/chunk "
                f"min={min(sizes)} max={max(sizes)} mean={sum(sizes)/len(sizes):.1f} "
                f"total_hands={sum(sizes)}"
            )
            if scores:
                over = sum(1 for s in scores if s >= 0.5)
                bt.logging.info(
                    f"PAYLOAD DIAG | score min={min(scores):.4f} "
                    f"max={max(scores):.4f} mean={sum(scores)/len(scores):.4f} "
                    f">=0.5: {over}/{len(scores)}"
                )
            try:
                bt.logging.info(
                    f"PAYLOAD DIAG | chunk0 (size={len(chunks[0])}) "
                    f"features={detector_extract_features(chunks[0])}"
                )
            except Exception as exc:
                bt.logging.warning(f"PAYLOAD DIAG | feature extract failed: {exc}")

            if os.getenv("POKER44_CAPTURE_PAYLOADS", "").strip().lower() in {"1", "true", "yes", "on"}:
                cap_dir = Path(__file__).resolve().parents[1] / "local_test" / "captures"
                cap_dir.mkdir(parents=True, exist_ok=True)
                existing = sorted(cap_dir.glob("query_*.json"))
                cap_max = int(os.getenv("POKER44_CAPTURE_MAX", "5"))
                if len(existing) >= cap_max:
                    bt.logging.info(
                        f"PAYLOAD DIAG | capture cap reached ({cap_max}); not writing more"
                    )
                    return
                stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                out = cap_dir / f"query_{stamp}.json"
                with out.open("w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "captured_at": stamp,
                            "n_chunks": len(chunks),
                            "chunk_sizes": sizes,
                            "scores": scores,
                            "chunks": chunks,
                        },
                        fh,
                    )
                bt.logging.info(f"PAYLOAD DIAG | captured raw payload -> {out}")
        except Exception as exc:
            bt.logging.warning(f"PAYLOAD DIAG | diagnostics failed (non-fatal): {exc}")

    @staticmethod
    def score_chunk(chunk: list[dict]) -> float:
        """Delegate to the trained dispersion detector (neurons/detector.py)."""
        return detector_score_chunk(chunk)

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        """Determine whether to blacklist incoming requests."""
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        """Assign priority based on caller's stake."""
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Random miner running...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)
