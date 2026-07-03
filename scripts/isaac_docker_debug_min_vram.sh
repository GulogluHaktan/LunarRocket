#!/bin/bash
set -e

# Resolve paths
WORKSPACE_DIR="/home/haktan/Projects/LunarRocket"
OUTPUT_DIR="$WORKSPACE_DIR/outputs"
mkdir -p "$OUTPUT_DIR"

echo "[LunarRocket] Running low-VRAM debug baseline in Isaac Sim..."
docker run --rm \
  --entrypoint /bin/bash \
  --gpus all \
  --network=host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v "$WORKSPACE_DIR:/workspace/LunarRocket" \
  -w /workspace/LunarRocket \
  --name lunar-rocket-isaac-debug-vram \
  nvcr.io/nvidia/isaac-sim:6.0.1 \
  -lc "cd /isaac-sim && ./python.sh /workspace/LunarRocket/app/debug_min_vram.py"

echo "[LunarRocket] Run complete! Stats:"
if [ -f "$OUTPUT_DIR/debug_min_vram_stats.json" ]; then
  cat "$OUTPUT_DIR/debug_min_vram_stats.json"
else
  echo "Error: Stats file was not created!"
  exit 1
fi
