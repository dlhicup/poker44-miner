# Public Benchmark and W&B

This document describes the public benchmark flow intended for miner training and offline reference.

## Purpose

The public benchmark exists to give miners:

- a reproducible labeled dataset for local experimentation;
- a schema aligned with the sanitized validator payloads miners see today;
- an artifact that can be published to Weights & Biases without exposing live evaluation data.

## Important Boundary

The public benchmark is **not** the same thing as production validator evaluation.

Production validators now evaluate miners using:

- live hands generated on Poker44 benchmark tables;
- centralized SQL persistence;
- sanitizer-built batches exposed by the central eval API;
- unseen batches delivered through the validator runtime.

The public benchmark remains a development artifact, not a copy of the live validator stream.

## Current Public Benchmark Inputs

The public benchmark is built only from:

- the public human dataset committed in the repo:
  `hands_generator/human_hands/poker_hands_combined.json.gz`
- offline-generated bot chunks derived from the public dataset

It does **not** use:

- live provider-table SQL
- `/internal/eval/current`
- validator live batches
- central platform eval windows

## Output

The benchmark builder produces a labeled dataset with:

- `train` / `validation` split
- ground-truth labels
- sanitized hands matching the miner-visible schema
- aggregate dataset statistics
- dataset hash for versioning

Default output path:

`data/public_miner_benchmark.json.gz`

## Build Locally

```bash
python scripts/publish/publish_public_benchmark.py --skip-wandb
```

Example with explicit output path:

```bash
python scripts/publish/publish_public_benchmark.py \
  --skip-wandb \
  --output-path /tmp/p44-public-benchmark.json.gz
```

## Publish to W&B

Offline test:

```bash
WANDB_MODE=offline python scripts/publish/publish_public_benchmark.py --offline
```

Online publish:

```bash
export WANDB_API_KEY=...

python scripts/publish/publish_public_benchmark.py \
  --wandb-project poker44-miner-benchmarks \
  --wandb-entity <your-team>
```

## What W&B Publishes

The publish script logs:

- the benchmark artifact file
- dataset hash
- split counts
- aggregate benchmark metadata

It does not publish live provider-runtime evaluation batches.

## Relationship to the Live Miner Contract

The public benchmark should be treated as:

- schema familiarization;
- local training material;
- offline evaluation support.

The live miner contract is documented in [Miner Guide](./miner.md).

Current production contract:

- validators send `DetectionSynapse(chunks=...)`;
- miners return one score per chunk;
- each chunk may contain one or many sanitized hands.

That production contract is sourced from live platform tables, not from this public artifact.
