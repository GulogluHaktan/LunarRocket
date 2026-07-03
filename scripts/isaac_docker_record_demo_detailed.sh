#!/bin/bash
set -e

# Resolve paths
WORKSPACE_DIR="/home/haktan/Projects/LunarRocket"
OUTPUT_DIR="$WORKSPACE_DIR/outputs/record_demo_detailed"
mkdir -p "$OUTPUT_DIR/frames"

echo "[LunarRocket] Running low-VRAM detailed demo video capture in Isaac Sim..."
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
  --name lunar-rocket-isaac-record-demo-detailed \
  nvcr.io/nvidia/isaac-sim:6.0.1 \
  -lc "cd /isaac-sim && ./python.sh /workspace/LunarRocket/app/record_demo_detailed.py"

echo "[LunarRocket] Encoding frames with ffmpeg on host..."
FFMPEG_START=$(date +%s.%N)
ffmpeg -y -framerate 24 -i "$OUTPUT_DIR/frames/frame_%04d.png" \
  -c:v libx264 -pix_fmt yuv420p "$OUTPUT_DIR/demo_detailed.mp4"
FFMPEG_END=$(date +%s.%N)
FFMPEG_ENCODE_TIME=$(python3 -c "print($FFMPEG_END - $FFMPEG_START)")
echo "[LunarRocket] ffmpeg encode time: $FFMPEG_ENCODE_TIME s"

# Update ffmpeg encode time in stats.json using python on host
python3 -c "
import json
path = '$OUTPUT_DIR/stats.json'
with open(path, 'r') as f:
    stats = json.load(f)
stats['ffmpeg encode time (s)'] = float('$FFMPEG_ENCODE_TIME')
with open(path, 'w') as f:
    json.dump(stats, f, indent=4)
"

echo "[LunarRocket] Run complete! Stats:"
if [ -f "$OUTPUT_DIR/stats.json" ]; then
  cat "$OUTPUT_DIR/stats.json"
else
  echo "Error: Stats file was not created!"
  exit 1
fi
