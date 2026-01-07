#!/bin/bash
set -e

# 1. Download weights if the volume is empty
if [ ! -e "/app/weights/DONE" ]; then
    echo "Checkpoints not found in volume. Downloading..."
    sh ./scripts/download_all_weights.sh
fi

# 2. Start the FastAPI server
echo "Starting FastAPI Server..."
exec "$@"
