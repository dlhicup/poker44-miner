"""Poker44 miner — dispersion-based bot detector (see neurons/detector.py)."""

# from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import bittensor as bt

from neurons.detector import score_chunk as detector_score_chunk
from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse


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
                "model_name": "poker44-dispersion-detector",
                "model_version": "1.0.0",
                "framework": "python-logistic-dispersion",
                "license": "MIT",
                "repo_url": "https://github.com/Poker44/Poker44-subnet",
                "notes": (
                    "Two-sided dispersion (typicality) detector: robust |z| of censored "
                    "behavioral features vs the human center, logistic weights, piecewise "
                    "calibration at 0.5. Trained offline via local_test/train_detector.py."
                ),
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained exclusively on the public Poker44 benchmark API releases "
                    "2026-07-15 and 2026-07-16 (labeled chunk groups, validator-censored "
                    "payload view). No validator-only evaluation data used."
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
        scores = [self.score_chunk(chunk) for chunk in chunks]
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(f"Miner Predctions: {synapse.predictions}")
        bt.logging.info(f"Scored {len(chunks)} chunks with heuristic risks.")
        return synapse

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
