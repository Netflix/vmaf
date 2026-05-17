#!/usr/bin/env bash
# SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
# Copyright 2026 Lusoris and Claude (Anthropic)
#
# dev/scripts/dev-mcp-entrypoint.sh — container entrypoint
#
# Starts the embedded MCP UDS server at ${VMAF_MCP_UDS_PATH}
# (default: /tmp/vmaf-mcp.sock) and tails the log.
#
# The server is spawned as a background daemon; this script then
# exec-tails the log so Docker's log collector sees MCP output and
# the container stays foreground.

set -euo pipefail

SOCKET_PATH="${VMAF_MCP_UDS_PATH:-/tmp/vmaf-mcp.sock}"
LOG_FILE="${VMAF_MCP_LOG:-/tmp/vmaf-mcp.log}"
MODEL_PATH="${VMAF_MODEL_PATH:-/workspace/model}"

# Remove stale socket from a previous run
if [ -S "${SOCKET_PATH}" ]; then
  rm -f "${SOCKET_PATH}"
fi

echo "[dev-mcp-entrypoint] Starting vmaf-mcp UDS server at ${SOCKET_PATH}" | tee -a "${LOG_FILE}"
echo "[dev-mcp-entrypoint] Model path: ${MODEL_PATH}" | tee -a "${LOG_FILE}"
echo "[dev-mcp-entrypoint] Build info: $(vmaf --version 2>&1 || echo 'vmaf CLI not in PATH')" | tee -a "${LOG_FILE}"

# Start the MCP server in the background.
# vmaf-mcp reads the UDS transport config from environment:
#   VMAF_MCP_UDS_PATH — socket path
#   VMAF_MODEL_PATH   — directory containing .json and .onnx model files
vmaf-mcp \
  --transport uds \
  --socket "${SOCKET_PATH}" \
  --model-dir "${MODEL_PATH}" \
  >>"${LOG_FILE}" 2>&1 &

MCP_PID=$!
echo "[dev-mcp-entrypoint] MCP server PID: ${MCP_PID}" | tee -a "${LOG_FILE}"

# Wait for socket to appear (max 30 s)
for i in $(seq 1 30); do
  if [ -S "${SOCKET_PATH}" ]; then
    echo "[dev-mcp-entrypoint] Socket ready after ${i}s" | tee -a "${LOG_FILE}"
    break
  fi
  sleep 1
done

if [ ! -S "${SOCKET_PATH}" ]; then
  echo "[dev-mcp-entrypoint] ERROR: socket never appeared at ${SOCKET_PATH}" | tee -a "${LOG_FILE}"
  exit 1
fi

# Tail the log (keeps container foreground; Docker log collector reads stdout)
exec tail -F "${LOG_FILE}"
