"""Central eval-api dataset adapter for validator-side evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import bittensor as bt

from poker44.core.models import LabeledHandBatch


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, str(default))).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _normalize_base_url(value: str) -> str:
    return value.strip().rstrip("/")


def _compute_batches_hash(batches: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(list(batches), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProviderRuntimeConfig:
    api_base_url: str
    internal_secret: str
    validator_id: str
    min_eval_hands: int = 70
    max_eval_hands: int = 120
    require_mixed: bool = True
    attempt_publish_current: bool = True
    mark_evaluated: bool = True
    request_timeout_seconds: int = 15

    @classmethod
    def from_env(cls, *, default_validator_id: str) -> "ProviderRuntimeConfig":
        api_base_url = _normalize_base_url(
            os.getenv(
                "POKER44_EVAL_API_BASE_URL",
                os.getenv("POKER44_PROVIDER_API_BASE_URL", "http://185.196.20.208:4001"),
            )
        )
        internal_secret = str(os.getenv("POKER44_PROVIDER_INTERNAL_SECRET", "")).strip()
        if not internal_secret:
            raise RuntimeError(
                "POKER44_PROVIDER_INTERNAL_SECRET is required when POKER44_RUNTIME_MODE=provider_runtime."
            )

        validator_id = (
            str(os.getenv("POKER44_PROVIDER_VALIDATOR_ID", default_validator_id)).strip()
            or default_validator_id
        )
        return cls(
            api_base_url=api_base_url,
            internal_secret=internal_secret,
            validator_id=validator_id,
            min_eval_hands=max(0, int(os.getenv("POKER44_PROVIDER_MIN_EVAL_HANDS", "70"))),
            max_eval_hands=max(0, int(os.getenv("POKER44_PROVIDER_MAX_EVAL_HANDS", "120"))),
            require_mixed=_env_bool("POKER44_PROVIDER_REQUIRE_MIXED", True),
            attempt_publish_current=_env_bool("POKER44_PROVIDER_ATTEMPT_PUBLISH_CURRENT", True),
            mark_evaluated=_env_bool("POKER44_PROVIDER_MARK_EVALUATED", True),
            request_timeout_seconds=int(os.getenv("POKER44_PROVIDER_REQUEST_TIMEOUT_SECONDS", "15")),
        )

    def public_summary(self) -> Dict[str, Any]:
        return {
            "mode": "provider_runtime",
            "api_base_url": self.api_base_url,
            "validator_id": self.validator_id,
            "min_eval_hands": self.min_eval_hands,
            "max_eval_hands": self.max_eval_hands,
            "require_mixed": self.require_mixed,
            "attempt_publish_current": self.attempt_publish_current,
            "mark_evaluated": self.mark_evaluated,
        }


class _EvalApiClient:
    def __init__(self, cfg: ProviderRuntimeConfig):
        self.cfg = cfg

    def _request(
        self,
        method: str,
        path: str,
        *,
        query: Optional[Mapping[str, Any]] = None,
        payload: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        url = f"{self.cfg.api_base_url}{path}"
        if query:
            url = f"{url}?{urlencode({k: v for k, v in query.items() if v is not None})}"

        headers = {
            "accept": "application/json",
            "x-eval-secret": self.cfg.internal_secret,
        }
        body_bytes = None
        if payload is not None:
            body_bytes = json.dumps(payload).encode("utf-8")
            headers["content-type"] = "application/json"

        request = Request(url, data=body_bytes, headers=headers, method=method.upper())
        with urlopen(request, timeout=self.cfg.request_timeout_seconds) as response:
            raw = response.read().decode("utf-8")

        decoded = json.loads(raw) if raw else None
        if isinstance(decoded, dict) and decoded.get("success") is True and "data" in decoded:
            return decoded["data"]
        return decoded

    def get(self, path: str, *, query: Optional[Mapping[str, Any]] = None) -> Any:
        return self._request("GET", path, query=query)

    def post(self, path: str, *, payload: Optional[Mapping[str, Any]] = None) -> Any:
        return self._request("POST", path, payload=payload)


class ProviderRuntimeManager:
    """Health-checks and metadata for the central eval API."""

    def __init__(self, cfg: ProviderRuntimeConfig):
        self.cfg = cfg
        self.client = _EvalApiClient(cfg)
        self.status: Dict[str, Any] = {
            "runtime_ready": False,
            "last_error": "",
        }

    def ensure_runtime_ready(self) -> bool:
        try:
            eval_health = self.client.get(
                "/internal/eval/health",
                query={"minHands": self.cfg.min_eval_hands},
            )
            ok = isinstance(eval_health, dict) and bool(eval_health.get("ok"))
            self.status["runtime_ready"] = ok
            self.status["last_error"] = ""
            if isinstance(eval_health, dict):
                self.status["available_hands"] = int(eval_health.get("availableHands") or 0)
                self.status["ready_for_evaluation"] = bool(eval_health.get("readyForEvaluation"))
                self.status["window_start"] = str(eval_health.get("windowStart") or "")
                self.status["window_end"] = str(eval_health.get("windowEnd") or "")
            return ok
        except Exception as exc:
            self.status["runtime_ready"] = False
            self.status["last_error"] = str(exc)
            return False


class ProviderRuntimeDatasetProvider:
    """Consumes central labeled eval batches from platform-backend."""

    def __init__(self, cfg: ProviderRuntimeConfig):
        self.cfg = cfg
        self.manager = ProviderRuntimeManager(cfg)
        self._dataset_hash: str = ""
        self._stats: Dict[str, Any] = cfg.public_summary()
        self._pending_hand_ids: List[str] = []
        self._active_chunk_id: str = ""

    @property
    def dataset_hash(self) -> str:
        return self._dataset_hash

    @property
    def stats(self) -> Dict[str, Any]:
        merged = dict(self._stats)
        merged.update(self.manager.status)
        return merged

    def refresh_if_due(self) -> None:
        self.manager.ensure_runtime_ready()

    def fetch_hand_batch(
        self,
        *,
        limit: int = 80,
        include_integrity: bool = True,
    ) -> List[LabeledHandBatch]:
        _ = include_integrity
        self._pending_hand_ids = []
        self._active_chunk_id = ""

        if not self.manager.ensure_runtime_ready():
            self._stats.update(
                {
                    "batch_count": 0,
                    "last_fetch_status": "eval_api_not_ready",
                    "last_fetch_at": int(time.time()),
                }
            )
            return []

        available_hands = int(self.manager.status.get("available_hands") or 0)
        ready_for_evaluation = bool(self.manager.status.get("ready_for_evaluation"))
        self._stats.update(
            {
                "available_hands": available_hands,
                "min_eval_hands": self.cfg.min_eval_hands,
            }
        )
        if self.cfg.min_eval_hands > 0 and not ready_for_evaluation:
            self._stats.update(
                {
                    "batch_count": 0,
                    "last_fetch_status": "waiting_for_min_hands",
                    "last_fetch_at": int(time.time()),
                }
            )
            bt.logging.info(
                f"Central eval API waiting for enough hands | have={available_hands} need={self.cfg.min_eval_hands}"
            )
            return []

        try:
            if self.cfg.attempt_publish_current:
                try:
                    publish_result = self.manager.client.post(
                        "/internal/eval/publish-current",
                        payload={
                            "validatorId": self.cfg.validator_id,
                            "handCount": max(
                                self.cfg.min_eval_hands,
                                min(self.cfg.max_eval_hands, limit or self.cfg.max_eval_hands),
                            ),
                            "requireMixed": self.cfg.require_mixed,
                        },
                    )
                    if isinstance(publish_result, dict):
                        self._stats.update(
                            {
                                "publish_reason": str(publish_result.get("reason") or ""),
                                "publish_chunk_id": str(publish_result.get("chunkId") or ""),
                                "publish_chunk_hash": str(publish_result.get("chunkHash") or ""),
                            }
                        )
                except Exception as exc:
                    bt.logging.warning(f"Central eval publish-current failed: {exc}")

            payload = self.manager.client.get("/internal/eval/current")
            batches_raw = payload.get("batches", []) if isinstance(payload, dict) else []
            if not isinstance(batches_raw, list):
                batches_raw = []
            if limit > 0:
                batches_raw = batches_raw[:limit]

            canonical_chunk_hash = (
                str(payload.get("chunkHash") or "").strip() if isinstance(payload, dict) else ""
            )
            self._dataset_hash = (
                canonical_chunk_hash
                if canonical_chunk_hash
                else (_compute_batches_hash(batches_raw) if batches_raw else "")
            )
            self._stats.update(
                {
                    "runtime_mode": "provider_runtime",
                    "batch_count": len(batches_raw),
                    "requested_limit": limit,
                    "last_fetch_status": "ok" if batches_raw else "waiting_for_active_chunk",
                    "last_fetch_at": int(time.time()),
                    "active_chunk_id": str(payload.get("chunkId") or "") if isinstance(payload, dict) else "",
                    "active_chunk_hash": str(payload.get("chunkHash") or "") if isinstance(payload, dict) else "",
                    "active_chunk_producer": str(payload.get("producerValidatorId") or "") if isinstance(payload, dict) else "",
                    "active_window_start": str(payload.get("windowStart") or "") if isinstance(payload, dict) else "",
                    "active_window_end": str(payload.get("windowEnd") or "") if isinstance(payload, dict) else "",
                }
            )
            self._active_chunk_id = (
                str(payload.get("chunkId") or "").strip() if isinstance(payload, dict) else ""
            )

            if not batches_raw:
                return []

            batches: List[LabeledHandBatch] = []
            hand_ids: List[str] = []
            for entry in batches_raw:
                if not isinstance(entry, dict):
                    continue
                hands_raw = entry.get("hands")
                if not isinstance(hands_raw, list):
                    continue
                normalized_hands = [hand for hand in hands_raw if isinstance(hand, dict)]
                if not normalized_hands:
                    continue
                is_human = bool(entry.get("is_human", False))
                batches.append(LabeledHandBatch(hands=normalized_hands, is_human=is_human))  # type: ignore[arg-type]
                for hand in normalized_hands:
                    hand_id = str(hand.get("hand_id") or "").strip()
                    if hand_id:
                        hand_ids.append(hand_id)

            self._pending_hand_ids = sorted(set(hand_ids))
            return batches
        except Exception as exc:
            self._stats.update(
                {
                    "batch_count": 0,
                    "last_fetch_status": f"error:{exc}",
                    "last_fetch_at": int(time.time()),
                }
            )
            bt.logging.warning(f"Central eval fetch failed: {exc}")
            return []

    def mark_last_batch_evaluated(self) -> None:
        if not self.cfg.mark_evaluated:
            self._pending_hand_ids = []
            return
        if not self._pending_hand_ids:
            return
        try:
            result = self.manager.client.post(
                "/internal/eval/mark-evaluated",
                payload={
                    "hand_ids": list(self._pending_hand_ids),
                    "chunkId": self._active_chunk_id,
                    "validatorId": self.cfg.validator_id,
                },
            )
            updated = 0
            if isinstance(result, dict):
                updated = int(result.get("updated") or 0)
            bt.logging.info(
                f"Central eval API marked evaluated hands | requested={len(self._pending_hand_ids)} updated={updated}"
            )
        except Exception as exc:
            bt.logging.warning(f"Central eval mark-evaluated failed: {exc}")
        finally:
            self._pending_hand_ids = []
