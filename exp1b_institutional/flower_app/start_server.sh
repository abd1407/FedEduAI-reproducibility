#!/usr/bin/env bash
set -euo pipefail
: "${FEDEDUAI_CERT_DIR:?Set directory containing pilot certificates}"
: "${FEDEDUAI_SERVER_OUTPUT:?Set a private output directory}"
export FEDEDUAI_EXECUTION_MODE=distributed_pilot
mkdir -p "$FEDEDUAI_SERVER_OUTPUT"
exec flower-superlink \
 --ssl-ca-certfile "$FEDEDUAI_CERT_DIR/ca.crt" \
 --ssl-certfile "$FEDEDUAI_CERT_DIR/server.crt" \
 --ssl-keyfile "$FEDEDUAI_CERT_DIR/server.key" \
 --auth-list-public-keys "$FEDEDUAI_CERT_DIR/allowed_nodes.csv" \
 --fleet-api-address 0.0.0.0:9092 \
 --control-api-address 127.0.0.1:9093 \
 --serverappio-api-address 127.0.0.1:9091
