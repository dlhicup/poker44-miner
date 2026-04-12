# Poker44 Validator Guide

Validator guide for Poker44 subnet `126`.

## Current Architecture

Poker44 validators are now intended to run in a **consumer-only** model.

That means:

- validators do **not** run their own poker tables;
- validators do **not** bootstrap provider frontend/backend locally;
- validators do **not** build live evaluation data from local JSON on the production path;
- validators consume canonical evaluation batches from the central Poker44 eval API;
- validators query miners, compute rewards, and set weights on-chain.

The old `mixed_dataset` mode still exists in code for compatibility and local experimentation,
but it is no longer the target production operating path.

## Separation of Responsibilities

### Poker44 platform infrastructure owns

- the live provider table;
- bots seated at that table;
- real-time gameplay;
- SQL persistence of hands and events;
- sanitization of evaluation payloads;
- chunk publication through `/internal/eval/*`.

### `poker44-subnet` validator owns

- polling the eval API;
- fetching the active canonical batch set;
- querying miners;
- scoring miner responses;
- updating weights on-chain;
- marking evaluated hand IDs back to the API.

This is the key design boundary: live table/runtime logic lives in `poker44-platform-*`, not in
the validator.

## What the Validator Actually Sends to Miners

Current validator behavior is important to understand precisely.

The validator fetches `batches` from the central eval API. Each returned batch currently looks like:

- one hidden label (`is_human`) on the validator side only;
- one list of `hands`;
- today, each batch typically contains a single sanitized hand/example.

Then the validator converts those batches into:

- `DetectionSynapse(chunks=...)`

Where:

- `chunks` is a list of chunks;
- each chunk is a list of sanitized hands;
- today, each chunk is usually a one-hand chunk;
- miners return one score per chunk.

So the current production path is **not** “one label for the entire epoch payload”.
Instead:

- the active eval payload contains many labeled batches;
- the validator scores miners batch-by-batch;
- each batch currently corresponds to one sanitized hand/example.

Relevant code:

- [validator entrypoint](/Users/mac/poker44-launch/poker44-subnet/neurons/validator.py)
- [runtime provider](/Users/mac/poker44-launch/poker44-subnet/poker44/validator/runtime_provider.py)
- [forward cycle](/Users/mac/poker44-launch/poker44-subnet/poker44/validator/forward.py)
- [synapse](/Users/mac/poker44-launch/poker44-subnet/poker44/validator/synapse.py)

## Where the Eval Data Comes From

The current production source is:

1. a single live provider table runs on Poker44 platform infrastructure;
2. that table contains both human and bot seats;
3. all hands are persisted to platform SQL;
4. `poker44-platform-backend` scans unconsumed hands and builds sanitized evaluation batches;
5. if `requireMixed=true`, only source hands that include both human and bot participation are eligible;
6. the backend publishes an active canonical chunk for the epoch/window;
7. validators read that active chunk through `/internal/eval/current`.

## Observability And Competition Signals

In `dev`, the validator also publishes two signed observability payloads:

- `validator_runtime.json`
- `network_snapshot.json`

These are best-effort and are not part of the scoring path. They exist so the
platform can expose:

- validator runtime alignment;
- live network/miner state from validator-signed metagraph snapshots;
- a weekly competition surface built on top of the canonical eval feed.

The intended competition model is:

- weekly epoch (Monday 20:00 UTC to Monday 20:00 UTC);
- continuous evaluation on canonical live hands during the epoch;
- public provisional leaderboard during the week;
- winner-take-all settlement after the epoch closes.

Settlement behavior in `dev` now follows a platform-decided pattern:

- during the week, validators continue using local score-based weights;
- after the backend settles the latest closed epoch, validators fetch the
  canonical settlement vector from `/internal/competition/current/weights`;
- if no epoch has been settled yet, validators fall back safely rather than
  blocking `set_weights()`.

Important nuance:

- source hands come from mixed live tables;
- the published payload can contain both human-labeled and bot-labeled batches;
- but each delivered batch/chunk is still currently a one-example unit from the validator’s point of view.

## Pull + Restart Contract

When a validator operator does only:

1. `git pull`
2. restart the validator

the validator should resume normal evaluation against the central eval API.

Concretely:

- it starts in `provider_runtime`;
- it connects to the central Poker44 eval API;
- it checks whether enough real hands exist;
- it may request publication of the current canonical chunk;
- it fetches the active chunk;
- it sends that chunk set to miners;
- it computes rewards;
- it sets weights;
- it marks evaluated hand IDs back to the API.

## Requirements

- Linux server
- Python 3.10+
- PM2
- registered validator hotkey on netuid `126`
- network access to the central Poker44 eval API

No local provider stack is required in the target production model.

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

## Runtime Modes

### Production target

- `POKER44_RUNTIME_MODE=provider_runtime`

### Legacy compatibility mode

- `POKER44_RUNTIME_MODE=mixed_dataset`

`mixed_dataset` still exists for compatibility, but production validators should be treated as
`provider_runtime` consumers of central eval data.

## Required Environment

Mandatory for production:

- `POKER44_RUNTIME_MODE=provider_runtime`
- `WALLET_NAME`
- `HOTKEY`
- `POKER44_EVAL_API_BASE_URL`
- `POKER44_PROVIDER_INTERNAL_SECRET`

Optional observability/reporting:

- `POKER44_VALIDATOR_RUNTIME_REPORT_URL`
- `POKER44_VALIDATOR_NETWORK_SNAPSHOT_REPORT_URL`

Important defaults in the current script:

- `POKER44_CHUNK_COUNT=80`
- `POKER44_REWARD_WINDOW=40`
- `POKER44_POLL_INTERVAL_SECONDS=300`
- `POKER44_MINERS_PER_CYCLE=16`
- `POKER44_PROVIDER_MIN_EVAL_HANDS=40`
- `POKER44_PROVIDER_MAX_EVAL_HANDS=70`
- `POKER44_PROVIDER_ATTEMPT_PUBLISH_CURRENT=true`

Notes:

- `POKER44_EVAL_API_BASE_URL` points at the central `poker44-platform-backend`;
- `POKER44_PROVIDER_INTERNAL_SECRET` is required for `/internal/eval/*`;
- `POKER44_CHUNK_COUNT` controls how many batches/chunks the validator will forward to miners in
  one cycle;
- today those batches are usually one sanitized hand/example each.

## Run Validator

Script path:

- `scripts/validator/run/run_vali.sh`

Example:

```bash
WALLET_NAME=p44_cold \
HOTKEY=p44_validator \
POKER44_RUNTIME_MODE=provider_runtime \
POKER44_PROVIDER_INTERNAL_SECRET=force-start-secret \
POKER44_EVAL_API_BASE_URL=http://185.196.20.208:4001 \
./scripts/validator/run/run_vali.sh
```

## Canonical Chunk Lifecycle

The current lifecycle is:

1. platform table generates real hands;
2. hands are persisted in SQL;
3. backend selects eligible unconsumed hands;
4. backend builds sanitized labeled batches from those hands;
5. backend publishes an active canonical chunk for the current window;
6. validator fetches it through `/internal/eval/current`;
7. validator sends the resulting chunk list to miners;
8. validator scores miner responses against the hidden labels;
9. validator marks the evaluated hand IDs back to the eval API.

## Current Scoring Granularity

Current scoring granularity is:

- one returned score per chunk;
- one validator label per chunk;
- today, one chunk is usually one sanitized hand/example.

This matters for miner/operator expectations:

- the live source is mixed-table gameplay;
- the current validator scoring contract is still chunk-level;
- the chunk-level contract is currently implemented as many one-example chunks.

## What the Validator Does Not Do

The production validator does **not**:

- run a local poker table;
- deploy provider frontend/backend;
- manage DNS or TLS;
- manage local SQL/Redis for provider runtime;
- generate production eval data from local JSON.

Those are platform responsibilities.

## PM2

```bash
pm2 logs poker44_validator
pm2 restart poker44_validator
pm2 stop poker44_validator
pm2 delete poker44_validator
```

## Related Docs

- [Miner guide](./miner.md)
- [Public benchmark](./public-benchmark.md)
- [Anti-leakage policy](./anti-leakage.md)
