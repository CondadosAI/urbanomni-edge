#!/usr/bin/env bash
# Featured rows only, each started from a cooled machine (CPU package < COOL C,
# or after 180 s), so iGPU throttling from the previous run does not leak in.
# Usage: scripts/featured.sh <session-id>
set -u
cd "$(dirname "$0")/.."
S=${1:?session}
COOL=${COOL:-55}
CLIP=data/clips/karol_seg_1080.mp4
B="uv run python scripts/bench.py --clip $CLIP --frames 200 --warmup 20 --session $S"

cool() {
  for _ in $(seq 36); do
    t=$(for h in /sys/class/hwmon/hwmon*; do [ "$(cat $h/name)" = coretemp ] && cat $h/temp1_input; done)
    [ "$t" -lt $((COOL * 1000)) ] && return
    sleep 5
  done
}

run() { cool; $B "$@" | grep '^{'; }

run --model checkpoints/urbanomnidetect_v2_m_640.pt --device 0 --imgsz 640 --tag pt-cuda-fp32-m640
run --model models/v2m_640_fp16.engine --device 0 --imgsz 640 --tag trt-fp16-m640
run --model models/v2m_640_fp16_openvino_model --device intel:gpu.0 --imgsz 640 --tag ov-igpu-fp16-m640
run --model models/v2m_640_int8head_openvino_model --device intel:gpu.0 --imgsz 640 --tag ov-igpu-int8head-m640
run --model models/v2m_640_fp16_openvino_model --device intel:cpu --imgsz 640 --tag ov-cpu-fp32-m640
run --model models/v2m_640_int8_openvino_model --device intel:cpu --imgsz 640 --tag ov-cpu-int8-m640
run --model models/v2m_640_int8head_openvino_model --device intel:cpu --imgsz 640 --tag ov-cpu-int8head-m640
run --model models/v2s_640_int8head_openvino_model --device intel:gpu.0 --imgsz 640 --tag ov-igpu-int8head-s640
run --model models/v2x_960_fp16_openvino_model --device intel:gpu.0 --imgsz 960 --tag ov-igpu-fp16-x960
run --model models/v2x_960_fp16.engine --device 0 --imgsz 960 --tag trt-fp16-x960
echo FEATURED_DONE $S
