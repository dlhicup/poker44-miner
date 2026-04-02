# Poker44 Validator Guide

Validator guide for Poker44 subnet `126`.

## Current Model

The validator now has one intended operating model:

- a single real poker table runs on Poker44 platform infrastructure
- that table persists hands in the central platform SQL
- `poker44-platform-backend` builds sanitized evaluation chunks from those hands
- validators do **not** run their own tables
- validators do **not** bootstrap provider frontend/backend locally
- validators consume chunks from the central eval API
- validators send those chunks to miners, compute rewards, and set weights on-chain

The old `mixed_dataset` mode still exists in code for compatibility, but it is no longer the target operating path.

## Pull + Restart Contract

When a validator operator does only:

1. `git pull`
2. restart the validator process

the validator should be able to resume evaluation against the central Poker44 eval API.

Concretely, `pull + restart` means:

- the validator starts in `provider_runtime`
- it does **not** create a local table
- it does **not** clone `poker44-platform-*`
- it does **not** configure `nginx`, `certbot`, `ufw`, frontend, or provider backend
- it calls the central eval API on Poker44 platform
- it checks whether enough real hands exist to build a chunk
- it can ask the central backend to publish the current chunk if needed
- it fetches the active canonical chunk
- it sends that same chunk to miners
- it computes rewards and sets weights
- it marks the consumed hand tokens as evaluated back in the central backend

## Separation of Responsibilities

`poker44-platform-*` owns:

- the live poker table
- the game runtime
- SQL persistence of hands/events
- chunk generation
- canonical chunk publication
- `eval_chunks`, `eval_chunk_epochs`, `eval_used_chunks`

`poker44-subnet` owns:

- validator polling
- miner queries
- reward calculation
- weight updates

This keeps the validator focused on evaluation, not infrastructure.

## Requirements

- Linux server
- Python 3.10+
- PM2
- registered validator hotkey on netuid `126`
- network access from the validator to the central Poker44 eval API

No local provider stack is required for the validator in this model.

## Install

```bash
git clone https://github.com/Poker44/Poker44-subnet
cd Poker44-subnet
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install bittensor-cli
```

Or use:

```bash
./scripts/validator/main/setup.sh
```

## Registration

```bash
btcli subnet register \
  --wallet.name p44_cold \
  --wallet.hotkey p44_validator \
  --netuid 126 \
  --subtensor.network finney

btcli wallet overview --wallet.name p44_cold --subtensor.network finney
```

## Required Environment

Mandatory:

- `POKER44_RUNTIME_MODE=provider_runtime`
- `WALLET_NAME`
- `HOTKEY`
- `POKER44_PROVIDER_INTERNAL_SECRET`

Important defaults:

- `POKER44_EVAL_API_BASE_URL=http://185.196.20.208:4001`
- `POKER44_PROVIDER_MIN_EVAL_HANDS=70`
- `POKER44_PROVIDER_MAX_EVAL_HANDS=120`
- `POKER44_PROVIDER_ATTEMPT_PUBLISH_CURRENT=true`

Notes:

- `POKER44_EVAL_API_BASE_URL` is the central `poker44-platform-backend` that exposes `/internal/eval/*`
- the validator talks directly to that central API
- the central backend itself remains responsible for SQL and chunk generation

## Shared Coordinator Rule

All validators should read from the same Poker44 platform backend / coordinator path so they evaluate miners with the same active chunk.

Today, the intended central source is:

- eval API: `http://185.196.20.208:4001`

## What Gets Automated

In this model, the validator automates only validator-side evaluation:

- polling the central eval API
- asking for chunk publication when appropriate
- fetching the active chunk
- querying miners
- computing rewards
- setting weights
- marking hand tokens evaluated

It does **not** automate:

- provider DNS
- provider TLS
- provider frontend/backend deployment
- local room creation
- local SQL/Redis

Those belong to Poker44 platform infrastructure, not to the validator.

## Run Validator

Preferred command:

```bash
WALLET_NAME=p44_cold \
HOTKEY=p44_validator \
POKER44_RUNTIME_MODE=provider_runtime \
POKER44_PROVIDER_INTERNAL_SECRET=force-start-secret \
POKER44_EVAL_API_BASE_URL=http://185.196.20.208:4001 \
./scripts/validator/run/run_vali.sh
```

Script path:

- `scripts/validator/run/run_vali.sh`

## PM2

```bash
pm2 logs poker44_validator
pm2 restart poker44_validator
pm2 stop poker44_validator
pm2 delete poker44_validator
```

## Canonical Chunk Behavior

The intended chunk lifecycle is:

- real hands are generated on the Poker44 platform table
- raw hands stay in platform SQL
- the platform backend builds sanitized labeled batches
- the active chunk is stored centrally
- validators fetch that chunk through `/internal/eval/current`
- validators score miners against the same chunk
- the platform backend tracks chunk publication and usage centrally

## Related Docs

- [VALIDATOR_PROVIDER_SETUP.md](/Users/mac/poker44-launch/documentacion/operaciones/VALIDATOR_PROVIDER_SETUP.md)
- [ENV_MATRIX.md](/Users/mac/poker44-launch/documentacion/operaciones/ENV_MATRIX.md)
- [RUNBOOK.md](/Users/mac/poker44-launch/documentacion/operaciones/RUNBOOK.md)
