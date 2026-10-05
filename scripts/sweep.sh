#!/usr/bin/env bash
# One benchmark session over every size x resolution x backend.
# Usage: scripts/sweep.sh <session-id> [sizes] [imgszs]
set -u
cd "$(dirname "$0")/.."
SESSION=${1:?session id}
SIZES=${2:-"n s m l x"}
IMGSZS=${3:-"640 960"}
CLIP=data/clips/karol_seg_1080.mp4
B="uv run python scripts/bench.py --clip $CLIP --frames 200 --warmup 20 --session $SESSION"

for imgsz in $IMGSZS; do
  for s in $SIZES; do
    pre="v2${s}_${imgsz}"
    $B --model checkpoints/urbanomnidetect_v2_${s}_640.pt --device 0 --imgsz $imgsz --tag pt-cuda-fp32-$s$imgsz
    [ -f models/${pre}_fp16.engine ] && \
      $B --model models/${pre}_fp16.engine --device 0 --imgsz $imgsz --tag trt-fp16-$s$imgsz
    $B --model models/${pre}_fp16_openvino_model --device intel:gpu.0 --imgsz $imgsz --tag ov-igpu-fp16-$s$imgsz
    $B --model models/${pre}_fp16_openvino_model --device intel:cpu --imgsz $imgsz --tag ov-cpu-fp32-$s$imgsz
  done
done 2>&1 | grep '^{' | python3 -c "
import json,sys
for l in sys.stdin:
    r=json.loads(l); print(f\"{r['tag']:24s} inf {r['inf_ms']:7.2f} ms  total {r['total_ms']:7.2f} ms  fps {r['wall_fps']:6.1f}\", flush=True)"
echo SWEEP_DONE $SESSION
