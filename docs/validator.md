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

- the live benchmark tables;
- bots seated at those tables;
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
- one chunk-sized evaluation unit that may contain one or many sanitized hands.

Then the validator converts those batches into:

- `DetectionSynapse(chunks=...)`

Where:

- `chunks` is a list of chunks;
- each chunk is a list of sanitized hands;
- each chunk may contain one or many sanitized hands;
- miners return one score per chunk.

So the current production path is **not** “one label for the entire epoch payload”.
Instead:

- the active eval payload contains many labeled batches;
- the validator scores miners batch-by-batch;
- each individual batch/chunk is homogeneous, so the hands inside it are all human or all bot;
- each batch/chunk remains one scoring unit from the validator’s point of view.

Relevant code:

- [validator entrypoint](/Users/mac/poker44-launch/poker44-subnet/neurons/validator.py)
- [runtime provider](/Users/mac/poker44-launch/poker44-subnet/poker44/validator/runtime_provider.py)
- [forward cycle](/Users/mac/poker44-launch/poker44-subnet/poker44/validator/forward.py)
- [synapse](/Users/mac/poker44-launch/poker44-subnet/poker44/validator/synapse.py)

## Where the Eval Data Comes From

The current production source is:

1. live benchmark tables run on Poker44 platform infrastructure;
2. those tables contain both human and bot seats;
3. all hands are persisted to platform SQL;
4. `poker44-platform-backend` builds sanitized evaluation batches from those benchmark-table hands;
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
- canonical eval windows of 2 hours inside that weekly epoch;
- continuous evaluation on canonical live hands during the epoch;
- public provisional leaderboard during the week;
- winner-take-all settlement after the epoch closes.

Settlement behavior in `dev` now follows a platform-decided pattern:

- validators fetch the canonical competition vector from
  `/internal/competition/current/weights`;
- once the backend has settled at least one weekly winner, the latest settled
  winner becomes the canonical competition vector for the current/vigente
  period, but validators apply a Swarm-style burn on top of it:
  `97%` to `uid 0`, `3%` to the backend-provided winner vector;
- before the first weekly settlement exists, the backend returns its explicit
  fallback vector (typically `uid 0`, which remains `100%` burned);
- validators only fall back to local score-based weights if the backend is
  unavailable or returns no usable positive vector.

Important nuance:

- source hands come from live benchmark tables;
- the backend rotates the active canonical chunk on 2-hour windows;
- validators poll that runtime continuously (`POKER44_POLL_INTERVAL_SECONDS`,
  300s by default) and can score the active window multiple times before the
  next window opens;
- the published payload can contain both human-labeled and bot-labeled batches;
- each delivered batch/chunk is still one scoring unit from the validator’s point of view.

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
- each batch/chunk may contain one or many sanitized hands.

## Run Validator

Script path:

- `scripts/validator/run/run_vali.sh`

Example:

```bash
WALLET_NAME=p44_cold \
HOTKEY=p44_validator \
POKER44_RUNTIME_MODE=provider_runtime \
POKER44_PROVIDER_INTERNAL_SECRET=replace-with-real-shared-secret \
POKER44_EVAL_API_BASE_URL=https://api-dev.poker44.net \
./scripts/validator/run/run_vali.sh
```

## Canonical Chunk Lifecycle

The current lifecycle is:

1. live benchmark tables generate real hands;
2. hands are persisted in SQL;
3. backend selects eligible benchmark-table hands;
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
- one chunk may contain one or many sanitized hands.

This matters for miner/operator expectations:

- the live source is benchmark-table gameplay;
- the current validator scoring contract is still chunk-level;
- the chunk-level contract is implemented as `list[list[hand]]`, with one score expected per chunk.

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
