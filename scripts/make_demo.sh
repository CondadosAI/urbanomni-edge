#!/usr/bin/env bash
# Demo video for one clip: blur plates -> upstream geometry (--refine --labels)
# -> empty-road plate -> dashboard with top-down minimap.
#
# Usage: scripts/make_demo.sh <name> <segment.mp4> <background.mp4> [model-dir] [imgsz]
#   segment     the clip to show (1920 wide)
#   background  footage of the same fixed camera for the empty-road median
set -eu
cd "$(dirname "$0")/.."
NAME=${1:?name}
SEG=${2:?segment clip}
BGCLIP=${3:?background clip}
MODEL=${4:-models/v2m_640_int8head_openvino_model}
IMGSZ=${5:-640}
ROOT=$(pwd)

mkdir -p runs/labels runs/plate runs/render
BLUR=data/clips/${NAME}_blur.mp4
[ -f "$BLUR" ] || uv run python scripts/blur_plates.py --src "$SEG" --out "$BLUR" --width 1920

(cd upstream && ../.venv/bin/python bev_realtime_v2.py --input "$ROOT/$BLUR" \
  --kp-model "$ROOT/$MODEL" --kp-imgsz $IMGSZ --kp-conf 0.1 --kp-vis 0.5 \
  --class-conf 0:0.4,1:0.5,3:0.5 --suppress-nested 0.85 --smooth 11 --scale-lock \
  --fixed-camera auto --bev-fit --refine --layout dashboard --device intel:gpu.0 \
  --labels "$ROOT/runs/labels/${NAME}.json.gz" --output "$ROOT/runs/render/${NAME}_upstream.mp4") \
  2>&1 | grep -E "fixed-camera|tracks|refine\] demoted|processed|Error|Traceback"

uv run python scripts/clean_plate.py --clip "$BGCLIP" --out runs/plate/${NAME}.png --every 1 --max-frames 360 --mask-model "$MODEL"

uv run python scripts/compose.py --clip "$BLUR" --labels runs/labels/${NAME}.json.gz \
  --plate runs/plate/${NAME}.png --out runs/render/demo_${NAME}.mp4 \
  --line1 "Intel Iris Xe iGPU (laptop)  |  OpenVINO INT8" \
  --line2 "UrbanOmniDetect-2 M @640  |  detector 33 ms per frame (30 fps)" \
  ${CREDIT:+--credit "$CREDIT"}
