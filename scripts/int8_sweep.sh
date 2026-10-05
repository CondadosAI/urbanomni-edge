#!/usr/bin/env bash
# Quantize (full INT8 and INT8 with float head), then time + parity on iGPU and CPU.
# Usage: scripts/int8_sweep.sh <session-id> [sizes] [imgszs]
set -u
cd "$(dirname "$0")/.."
SESSION=${1:?session id}
SIZES=${2:-"n s m"}
IMGSZS=${3:-"640 960"}
CALIB=data/clips/karol_calib_1080.mp4
CLIP=data/clips/karol_seg_1080.mp4
B="uv run python scripts/bench.py --clip $CLIP --frames 200 --warmup 20 --session $SESSION"

for imgsz in $IMGSZS; do
  for s in $SIZES; do
    pre="v2${s}_${imgsz}"
    for v in int8 int8head; do
      ir=models/${pre}_${v}_openvino_model
      if [ ! -d $ir ]; then
        flag=""; [ $v = int8head ] && flag="--ignore-head"
        uv run python scripts/quant_ov.py --ir models/${pre}_fp16_openvino_model --clip $CALIB \
          --n 300 --imgsz $imgsz $flag 2>&1 | grep -E "SAVED|Error|Traceback"
      fi
      $B --model $ir --device intel:gpu.0 --imgsz $imgsz --tag ov-igpu-$v-$s$imgsz | grep '^{'
      $B --model $ir --device intel:cpu --imgsz $imgsz --tag ov-cpu-$v-$s$imgsz | grep '^{'
      uv run python scripts/parity.py --ref runs/bench/pt-cuda-fp32-$s${imgsz}_s1_preds.npy \
        runs/bench/ov-igpu-$v-$s${imgsz}_${SESSION}_preds.npy runs/bench/ov-cpu-$v-$s${imgsz}_${SESSION}_preds.npy \
        --out runs/parity_int8.jsonl
    done
  done
done 2>&1
echo INT8_DONE $SESSION
