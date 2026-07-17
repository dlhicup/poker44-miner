#!/bin/bash

# Poker44 Miner Startup Script

NETUID="${NETUID:-126}"
WALLET_NAME="${WALLET_NAME:-poker44-miner-ck}"
HOTKEY="${HOTKEY:-poker44-miner-hk}"
NETWORK="${NETWORK:-finney}"
MINER_SCRIPT="${MINER_SCRIPT:-./neurons/miner.py}"
PM2_NAME="${PM2_NAME:-poker44_miner}"  ##  name of Miner, as you wish
AXON_PORT="${AXON_PORT:-8091}"
ALLOWED_VALIDATOR_HOTKEYS="${ALLOWED_VALIDATOR_HOTKEYS:-}"
# Use the project venv's python (has bittensor); override with PYTHON_BIN.
PYTHON_BIN="${PYTHON_BIN:-$(pwd)/.venv/bin/python}"

if [ ! -f "$MINER_SCRIPT" ]; then
    echo "Error: Miner script not found at $MINER_SCRIPT"
    exit 1
fi

if ! command -v pm2 &> /dev/null; then
    echo "Error: PM2 is not installed"
    exit 1
fi

pm2 delete $PM2_NAME 2>/dev/null || true

export PYTHONPATH="$(pwd)"
# bittensor >= 10 disables CLI arg parsing unless this is set; without it the
# neuron gets a default config (no netuid/wallet/neuron.*) and crashes.
export BT_NO_PARSE_CLI_ARGS=false
# Stamp the manifest with the commit actually being served (compliance:
# manifest repo_commit must match the public repo).
# Unconditional on purpose: git HEAD is the ONLY source of truth here. A
# ":-" fallback would let a stale exported value survive across restarts and
# make the manifest claim a commit that no longer matches the served model.
export POKER44_MODEL_REPO_COMMIT="$(git rev-parse HEAD 2>/dev/null)"

MINER_ARGS=(
  --netuid "$NETUID"
  --wallet.name "$WALLET_NAME"
  --wallet.hotkey "$HOTKEY"
  --subtensor.network "$NETWORK"
  --axon.port "$AXON_PORT"
  --logging.debug
)

if [ -n "$ALLOWED_VALIDATOR_HOTKEYS" ]; then
  read -r -a VALIDATOR_HOTKEY_ARRAY <<< "$ALLOWED_VALIDATOR_HOTKEYS"
  MINER_ARGS+=(--blacklist.allowed_validator_hotkeys "${VALIDATOR_HOTKEY_ARRAY[@]}")
else
  MINER_ARGS+=(--blacklist.force_validator_permit)
fi

pm2 start $MINER_SCRIPT \
  --name $PM2_NAME \
  --interpreter "$PYTHON_BIN" -- \
  "${MINER_ARGS[@]}"

pm2 save

echo "Miner started: $PM2_NAME"
echo "View logs: pm2 logs $PM2_NAME"
echo "Config: netuid=$NETUID network=$NETWORK wallet=$WALLET_NAME hotkey=$HOTKEY axon_port=$AXON_PORT"
if [ -n "$ALLOWED_VALIDATOR_HOTKEYS" ]; then
    echo "Access mode: validator allowlist"
else
    echo "Access mode: validator_permit fallback"
fi
