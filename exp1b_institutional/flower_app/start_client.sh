#!/usr/bin/env bash
set -euo pipefail
: "${FEDEDUAI_CERT_DIR:?Set this node's certificate directory}"
: "${FEDEDUAI_DATA_DIR:?Set this node's private shard directory}"
: "${FEDEDUAI_SERVER_HOST:?Set server DNS name or IP matching certificate}"
: "${FEDEDUAI_PARTITION_ID:?Set this node's numeric partition ID}"
exec flower-supernode \
 --superlink "$FEDEDUAI_SERVER_HOST:9092" \
 --root-certificates "$FEDEDUAI_CERT_DIR/ca.crt" \
 --auth-supernode-private-key "$FEDEDUAI_CERT_DIR/node$FEDEDUAI_PARTITION_ID.key" \
 --auth-supernode-public-key "$FEDEDUAI_CERT_DIR/node$FEDEDUAI_PARTITION_ID.pub" \
 --node-config "partition-id=$FEDEDUAI_PARTITION_ID" \
 --clientappio-api-address 127.0.0.1:9094
