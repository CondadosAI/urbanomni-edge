#!/usr/bin/env bash
# Build TRT FP16 engines and OpenVINO FP16 IRs for every size at 640 and 960.
set -u
cd "$(dirname "$0")/.."
for imgsz in 640 960; do
  for s in n s m l x; do
    [ -d "models/v2${s}_${imgsz}_fp16_openvino_model" ] || \
      uv run python scripts/export.py --size $s --imgsz $imgsz --fmt openvino --half 2>&1 | grep EXPORTED
    [ -f "models/v2${s}_${imgsz}_fp16.engine" ] || \
      uv run python scripts/export.py --size $s --imgsz $imgsz --fmt engine --half 2>&1 | grep -E "EXPORTED|Error|error" | tail -2
  done
done
echo BUILD_DONE
