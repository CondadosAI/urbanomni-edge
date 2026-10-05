# UrbanOmniDetect-2 on a laptop iGPU

This repository measures how fast UrbanOmniDetect-2, a detector that draws a 3D box around every road user from a single uncalibrated camera, runs on an ordinary laptop, and how much its output changes when it is converted for that laptop's hardware. The model is exported with stock Ultralytics to OpenVINO (for the Intel CPU and the Intel Iris Xe integrated GPU, with and without INT8 quantization) and to TensorRT (for the NVIDIA laptop GPU). Every converted model is timed on the same traffic clip and compared, detection by detection and corner by corner, against the original PyTorch model. The scripts that render the demo videos (plate blurring, an empty-road background, and a dashboard with a top-down map) are here too.

Companion code for the CondadosAI article on UrbanOmniDetect-2 at the edge. Source code is Apache-2.0; the model, Ultralytics and the footage are not, see [Licences](#licences-what-apache-20-here-does-not-cover).

## Results

The rows the article features. "Model ms" is the time the model itself takes per 1920x1080 frame (the Ultralytics inference stage: letterboxing before it and decoding after it are not included). "Corner error" is how far, in pixels of the 1920x1080 frame, the eight projected 3D-box corners move against PyTorch FP32 on the RTX 3060. "Dets matched" is the share of PyTorch FP32's detections that the converted model also finds. Numbers come from [`output/results_all.md`](output/results_all.md), which holds every size and backend.

| Model | Input | Hardware | Runtime | Model ms (median) | min-max over sessions | Sessions | Corner err. median / p90 (px) | Dets matched |
|---|--:|---|---|--:|--:|--:|--:|--:|
| M | 640 | RTX 3060 Laptop | PyTorch FP32 | 15.4 | 15.3-15.7 | 3 | reference | reference |
| M | 640 | RTX 3060 Laptop | TensorRT FP16 | 4.0 | 3.9-4.0 | 3 | 0.18 / 0.37 | 0.997 |
| M | 640 | Iris Xe iGPU | OpenVINO FP16 | 57.8 | 55.8-68.5 | 3 | 0.22 / 0.56 | 0.995 |
| M | 640 | Iris Xe iGPU | OpenVINO INT8 | 42.1 | 31.3-42.1 | 2 | 0.96 / 3.04 | 0.982 |
| M | 640 | Iris Xe iGPU | OpenVINO INT8, float head | 33.2 | 26.1-47.1 | 3 | 0.39 / 0.98 | 0.989 |
| M | 640 | i7-12700H CPU | OpenVINO FP32 | 160.9 | 156.1-165.8 | 3 | 0.01 / 0.02 | 0.999 |
| M | 640 | i7-12700H CPU | OpenVINO INT8 | 48.2 | 46.2-53.6 | 3 | 0.94 / 3.05 | 0.983 |
| M | 640 | i7-12700H CPU | OpenVINO INT8, float head | 62.7 | 59.6-69.8 | 3 | 0.33 / 0.90 | 0.988 |
| S | 640 | Iris Xe iGPU | OpenVINO INT8, float head | 16.7 | 14.4-25.7 | 3 | 0.50 / 2.14 | 0.985 |
| X | 960 | RTX 3060 Laptop | TensorRT FP16 | 20.9 | 20.9-21.2 | 3 | 0.19 / 0.33 | 0.998 |
| X | 960 | Iris Xe iGPU | OpenVINO FP16 | 303.9 | 260.6-402.8 | 3 | 0.24 / 0.87 | 0.996 |

How to read the columns:

- **Model ms (median)**: each session reports the median of 200 frames; the table shows the median of those per-session medians. With three sessions that is the middle one. With two sessions `scripts/summarize.py` takes the upper of the two, so a two-session row shows the slower session. The full INT8 iGPU row above is the only featured row with two sessions.
- **INT8, float head**: NNCF post-training INT8 with the Pose26 head (module 23, the layers that output the boxes and the cuboid corners) left out of quantization, so it runs in the IR's float precision. Plain "INT8" quantizes the head too.
- **OpenVINO FP32 on the CPU** loads the same FP16 IR as the iGPU rows. OpenVINO's CPU plugin computes in FP32 on this processor, which is why the row is labelled FP32 and why its corner error against PyTorch FP32 is 0.01 px.
- **Dets matched** is `recall_vs_ref` in the parity files: detections at score 0.25 or more, matched to the reference per frame by class and box IoU 0.5 or more (Hungarian assignment). Corner error counts only corners the reference itself is confident in (corner confidence 0.5 or more). The reference is the session-1 PyTorch FP32 run.
- Parity is agreement with PyTorch, not accuracy. There is no ground truth for these clips.

### A second camera: Shanghai crossing

The INT8 models were calibrated on footage from the footbridge clip only. To check that the INT8 parity holds on a camera the calibration never saw, the M model at 640 was run once on a crossing in Shanghai. One session, started hot (the PyTorch row records a CPU package at 87 C at its first timed frame) and with no cool-down, so treat the latency as indicative. Parity from [`output/parity_shanghai.jsonl`](output/parity_shanghai.jsonl), latency from [`output/latency_shanghai.jsonl`](output/latency_shanghai.jsonl).

| Model | Input | Hardware | Runtime | Model ms (one session) | Corner err. median / p90 (px) | Dets matched |
|---|--:|---|---|--:|--:|--:|
| M | 640 | RTX 3060 Laptop | PyTorch FP32 | 15.2 | reference | reference |
| M | 640 | Iris Xe iGPU | OpenVINO FP16 | 57.2 | 0.31 / 0.53 | 0.996 |
| M | 640 | Iris Xe iGPU | OpenVINO INT8 | 25.7 | 0.95 / 2.64 | 0.975 |
| M | 640 | Iris Xe iGPU | OpenVINO INT8, float head | 27.4 | 0.50 / 1.21 | 0.985 |

## What is in this repository

| Path | What it does |
|---|---|
| `stubs/v2/hybrid_v2.py` | An inference-only `HybridPoseModel26`, an empty subclass of Ultralytics' `PoseModel`. The released checkpoints are pickled as that class; upstream it only changes the training loss, and the forward pass is the stock YOLO26 pose model. With this stub the AGPL checkpoints load without importing any upstream code. Every script that loads a `.pt` puts `stubs/` on `sys.path` and imports it first. |
| `scripts/export.py` | Exports `checkpoints/urbanomnidetect_v2_<size>_640.pt` to OpenVINO IR, a TensorRT engine or ONNX with stock Ultralytics (static shape, batch 1), and names it `models/v2<size>_<imgsz>_<precision>...`. |
| `scripts/build_engines.sh` | Runs `export.py` for every size (N, S, M, L, X) at 640 and 960: OpenVINO FP16 IR and TensorRT FP16 engine. |
| `scripts/quant_ov.py` | NNCF post-training INT8 quantization of an OpenVINO IR. Calibration frames get the same letterbox, RGB order and scaling as the Ultralytics predictor applies. `--ignore-head` keeps the Pose26 head (module 23) in float. |
| `scripts/bench.py` | Times one model on a clip through the Ultralytics predictor with a square letterbox (`rect=False`) for every backend, appends a row to `runs/bench/results.jsonl` and saves the predictions for parity. Also records the iGPU clock and CPU package temperature from sysfs. |
| `scripts/parity.py` | Compares saved predictions against the PyTorch FP32 reference (definitions above). |
| `scripts/sweep.sh` | One session over every size, resolution and float backend (PyTorch CUDA FP32, TensorRT FP16, OpenVINO iGPU FP16, OpenVINO CPU). |
| `scripts/int8_sweep.sh` | Quantizes N, S and M at 640 and 960 (full INT8 and INT8 with float head), times them on the iGPU and CPU, and runs parity. |
| `scripts/session.sh` | One full session (`sweep.sh` then `int8_sweep.sh`), aborted if an NVMe drive passes 75 C. |
| `scripts/featured.sh` | The featured rows only, each started after the CPU package cools below 55 C or after 180 s, whichever comes first. |
| `scripts/summarize.py` | Builds the results table from `runs/bench/results.jsonl` and `runs/parity_*.jsonl`. |
| `scripts/export_evidence.py` | Copies those logs into `output/`, dropping the machine's hostname and nothing else. |
| `scripts/draw_frame.py` | Draws the cuboids on one frame (sanity check). |
| `scripts/render.py` | Renders cuboids on a clip with the measured inference time on screen. |
| `scripts/blur_plates.py` | Blurs licence plates: open-image-models' YOLOv9 plate detector on the 4K source in overlapping tiles, boxes padded and held two frames either side, Gaussian blur. |
| `scripts/clean_plate.py` | Empty-road image of a fixed camera: the per-pixel median over many frames, optionally with detected vehicles masked out first (for queues that never clear). |
| `scripts/compose.py` | The demo dashboard: camera view with cuboids, an info strip, and a top-down map made by warping the empty-road image through the ground homography that the upstream renderer writes with `--labels`. |
| `scripts/conflicts.py` | Time-to-collision between tracked vehicles on the ground plane, from the same labels. Optional; `compose.py --conflicts` draws pairs under a threshold in red. |
| `scripts/make_demo.sh` | Blur, upstream geometry, empty-road image and dashboard for one clip. Needs the upstream repository (see step 8). Its info-strip text states this machine's hardware and the 33 ms figure; edit `--line1`/`--line2` for yours. |
| `output/results_all.md` | Every size, resolution and backend. |
| `output/latency.jsonl` | Every timing row on the footbridge clip (sessions s1, s2, s3). |
| `output/parity_footbridge.jsonl` | Parity rows on the footbridge clip: float backends (session 1), then INT8 for sessions 1 and 2. |
| `output/latency_shanghai.jsonl`, `output/parity_shanghai.jsonl` | The Shanghai check. |

## Reproduce

You need Linux, [uv](https://docs.astral.sh/uv/), ffmpeg with libx264, an Intel GPU driver that OpenVINO can see (for `intel:gpu.0`) and an NVIDIA driver (for PyTorch CUDA and TensorRT). Run everything from the repository root.

### 1. Environment

```bash
uv sync
```

This installs the pinned versions in `uv.lock`, including Ultralytics (AGPL-3.0) and TensorRT (NVIDIA proprietary licence).

### 2. Weights

```bash
uv run hf download mehmetkeremturkcan/UrbanOmniDetect-2 \
  checkpoints/urbanomnidetect_v2_n_640.pt checkpoints/urbanomnidetect_v2_s_640.pt \
  checkpoints/urbanomnidetect_v2_m_640.pt checkpoints/urbanomnidetect_v2_l_640.pt \
  checkpoints/urbanomnidetect_v2_x_640.pt \
  --revision be4f69a300c43f6d2c7dd35f99b6c2a828912817 --local-dir .
```

The weights are AGPL-3.0. They land in `checkpoints/`, which is git-ignored.

### 3. Footage

The clips are not in this repository. Download them from YouTube and cut them as below. The 4K streams were fetched with ffmpeg from yt-dlp's format list; the `yt-dlp` lines here are an equivalent way to get the same 2160p video.

```bash
mkdir -p data/clips && cd data/clips

# Karol Majek, "4K Road traffic video for object detection and tracking", CC BY 3.0
uvx yt-dlp -f "bv*[height<=2160][ext=mp4]" -o karol-road-4k.mp4 https://youtu.be/MNn9qKG2UFI
# benchmark segment: 26 s to 61 s, then 1920 wide
ffmpeg -ss 26 -t 35 -i karol-road-4k.mp4 -an -c:v libx264 -crf 18 -pix_fmt yuv420p -movflags +faststart karol-road-4k_seg.mp4
ffmpeg -i karol-road-4k_seg.mp4 -vf scale=1920:-2:flags=lanczos -c:v libx264 -crf 16 -an karol_seg_1080.mp4
# INT8 calibration and empty-road footage: 70 s to the end at 3 fps (no overlap with the segment)
ffmpeg -ss 70 -i karol-road-4k.mp4 -vf "fps=3,scale=1920:-2:flags=lanczos" -c:v libx264 -crf 16 -an karol_calib_1080.mp4

# Thompson Fang, Yan'an Road East / Tibet Road Middle crossing, Shanghai, CC BY 3.0
uvx yt-dlp -f "bv*[height<=2160][ext=mp4]" -o shanghai.mp4 https://youtu.be/O0LyetdH4Lg
ffmpeg -ss 238 -i shanghai.mp4 -t 22 -an -vf "scale=1920:-2:flags=lanczos" -c:v libx264 -crf 16 -pix_fmt yuv420p shanghai_seg_1080.mp4
ffmpeg -ss 238 -i shanghai.mp4 -t 22 -c:v libx264 -crf 16 -an shanghai_seg_4k.mp4    # demo 1
ffmpeg -ss 42  -i shanghai.mp4 -t 22 -c:v libx264 -crf 16 -an shanghai2_seg_4k.mp4   # demo 2
# empty-road footage: 100 s to 220 s at 3 fps
ffmpeg -ss 100 -i shanghai.mp4 -t 120 -an -vf "fps=3,scale=1920:-2:flags=lanczos" -c:v libx264 -crf 16 -pix_fmt yuv420p shanghai_bg_1080.mp4
cd ../..
```

`bench.py` reads the first 220 frames of `karol_seg_1080.mp4` (20 warmup, 200 timed), so the timing and parity frames are the first 7.3 s of the segment.

### 4. Export

```bash
bash scripts/build_engines.sh     # OpenVINO FP16 IR + TensorRT FP16 engine, every size at 640 and 960
```

The checkpoints were trained at 640; the 960 exports run the same weights on a larger input. TensorRT engines are tied to the GPU and TensorRT version that built them.

### 5. Benchmark sessions

```bash
bash scripts/session.sh s1        # sweep.sh + int8_sweep.sh (quantizes on first run)
bash scripts/session.sh s2        # a separate sitting
bash scripts/featured.sh s3       # featured rows only, with cool-down
```

`int8_sweep.sh` calls `quant_ov.py` with 300 frames from `karol_calib_1080.mp4` and preset `mixed`, once per model, then reuses the IR in later sessions. It also runs INT8 parity into `runs/parity_int8.jsonl`.

### 6. Parity of the float backends

```bash
for s in n s m l x; do for i in 640 960; do
  uv run python scripts/parity.py --ref runs/bench/pt-cuda-fp32-$s${i}_s1_preds.npy \
    runs/bench/trt-fp16-$s${i}_s1_preds.npy runs/bench/ov-igpu-fp16-$s${i}_s1_preds.npy \
    runs/bench/ov-cpu-fp32-$s${i}_s1_preds.npy --out runs/parity_s1.jsonl
done; done
```

### 7. Table

```bash
uv run python scripts/summarize.py > output/results_all.md
uv run python scripts/export_evidence.py
```

To rebuild the table from the shipped logs alone, copy `output/latency.jsonl` to `runs/bench/results.jsonl` and `output/parity_footbridge.jsonl` to `runs/parity_s1.jsonl`, then run `summarize.py`; the output matches `output/results_all.md` byte for byte. The INT8 parity rows appear once per session. They are identical between sessions for 23 of the 24 INT8 tags; for M at 960 on the iGPU (full INT8) session 2 found 2 more detections than session 1 (dets matched 0.9711 against 0.9705), and the table uses the last row, session 2.

The Shanghai check:

```bash
B="uv run python scripts/bench.py --clip data/clips/shanghai_seg_1080.mp4 --frames 200 --warmup 20 --session sh1 --out runs/bench_shanghai"
$B --model checkpoints/urbanomnidetect_v2_m_640.pt --device 0 --imgsz 640 --tag pt-cuda-fp32-m640
$B --model models/v2m_640_int8head_openvino_model --device intel:gpu.0 --imgsz 640 --tag ov-igpu-int8head-m640
$B --model models/v2m_640_int8_openvino_model --device intel:gpu.0 --imgsz 640 --tag ov-igpu-int8-m640
$B --model models/v2m_640_fp16_openvino_model --device intel:gpu.0 --imgsz 640 --tag ov-igpu-fp16-m640
uv run python scripts/parity.py --ref runs/bench_shanghai/pt-cuda-fp32-m640_sh1_preds.npy \
  runs/bench_shanghai/ov-igpu-fp16-m640_sh1_preds.npy runs/bench_shanghai/ov-igpu-int8-m640_sh1_preds.npy \
  runs/bench_shanghai/ov-igpu-int8head-m640_sh1_preds.npy --out runs/parity_shanghai.jsonl
```

### 8. Demo videos (needs the upstream code, AGPL-3.0)

The cuboid tracking, smoothing, ground homography and offline refinement in the videos come from the author's `bev_realtime_v2.py`, which this repository does not include. Clone it yourself into `upstream/` (git-ignored); by doing so you take its code under its own licence, AGPL-3.0.

```bash
git clone https://github.com/mkturkcan/urbanomnidetect.git upstream
git -C upstream checkout caa8537803f61b5d04602a25432575f36fce085d
```

The renders used that commit unmodified, run with this repository's virtual environment (`../.venv/bin/python`, which covers its requirements).

Footbridge demo (0 to 17 s of the segment):

```bash
uv run python scripts/blur_plates.py --src data/clips/karol-road-4k_seg.mp4 --out data/clips/karol_seg_1080_blur.mp4 --width 1920
ffmpeg -i data/clips/karol_seg_1080_blur.mp4 -t 17 -c:v libx264 -crf 16 -an data/clips/karol_cut_1080_blur.mp4
mkdir -p runs/labels
(cd upstream && ../.venv/bin/python bev_realtime_v2.py --input ../data/clips/karol_cut_1080_blur.mp4 \
  --kp-model ../models/v2m_640_int8head_openvino_model --kp-imgsz 640 --kp-conf 0.1 --kp-vis 0.5 \
  --class-conf 0:0.4,1:0.5,3:0.5 --suppress-nested 0.85 --smooth 11 --scale-lock --fixed-camera auto \
  --bev-fit --refine --layout dashboard --device intel:gpu.0 \
  --labels ../runs/labels/m640_cut.json.gz --output ../runs/render/bev_m640_cut_refine.mp4)
uv run python scripts/clean_plate.py --clip data/clips/karol_calib_1080.mp4 --out runs/plate/road.png --every 3 --max-frames 236
uv run python scripts/compose.py --clip data/clips/karol_cut_1080_blur.mp4 --labels runs/labels/m640_cut.json.gz \
  --plate runs/plate/road.png --out runs/render/demo_m640.mp4 \
  --line1 "Intel Iris Xe iGPU (laptop)  |  OpenVINO INT8" \
  --line2 "UrbanOmniDetect-2 M @640  |  detector 33 ms per frame (30 fps)"
```

Shanghai demos (the second reuses the first one's empty-road image, same camera):

```bash
CREDIT="Footage: Thompson Fang, CC BY 3.0  |  Model: UrbanOmniDetect-2 (M. K. Turkcan), AGPL-3.0"
CREDIT="$CREDIT" bash scripts/make_demo.sh shanghai  data/clips/shanghai_seg_4k.mp4  data/clips/shanghai_bg_1080.mp4
CREDIT="$CREDIT" bash scripts/make_demo.sh shanghai2 data/clips/shanghai2_seg_4k.mp4 data/clips/shanghai_bg_1080.mp4
for n in shanghai shanghai2; do
  uv run python scripts/compose.py --clip data/clips/${n}_blur.mp4 --labels runs/labels/$n.json.gz \
    --plate runs/plate/shanghai.png --out runs/render/demo_$n.mp4 \
    --line1 "Intel Iris Xe iGPU (laptop)  |  OpenVINO INT8" \
    --line2 "UrbanOmniDetect-2 M @640  |  detector 33 ms per frame (30 fps)" \
    --credit "$CREDIT" --edge 4 --map-min-box-h 50
done
```

Optional time-to-collision overlay:

```bash
uv run python scripts/conflicts.py --labels runs/labels/shanghai.json.gz --out runs/conflicts/shanghai.json
# then add --conflicts runs/conflicts/shanghai.json --ttc-max 1.0 to compose.py
```

## How these numbers were made

| | |
|---|---|
| CPU | Intel Core i7-12700H, 20 threads |
| iGPU | Intel Iris Xe (on the i7-12700H) |
| dGPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6 GB, driver 580.173.02 |
| RAM, OS | 31 GB; Ubuntu 22.04.5 LTS, kernel 6.8.0-138-generic |
| Versions | Python 3.12.9, torch 2.14.1+cu130, openvino 2026.4.1, nncf 3.4.0, tensorrt-cu13 10.16.1.11, ultralytics 8.4.61 (all pinned in `uv.lock`) |
| Model | UrbanOmniDetect-2 N/S/M/L/X, checkpoints `urbanomnidetect_v2_<size>_640.pt` from [mehmetkeremturkcan/UrbanOmniDetect-2](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2) at revision `be4f69a`, AGPL-3.0 |
| Export | Stock Ultralytics 8.4.61, static shape, batch 1. OpenVINO IR in FP16 (`half=True`); TensorRT FP16 engine |
| Quantization | NNCF post-training INT8, preset `mixed`, 300 calibration frames from 70 s to the end of the Karol video at 3 fps (no overlap with the timed segment); "float head" variant ignores every op under `__module.model.23.` |
| Data | Karol Majek's footbridge video, 26 s to 61 s, 1920x1080, first 220 frames; Shanghai check on Thompson Fang's video, 238 s to 260 s |
| Timing | Ultralytics per-stage `speed["inference"]` for each frame, so the model's forward pass only; preprocessing (letterbox) and postprocessing are excluded. 20 warmup frames discarded, median of 200. Square letterbox (`rect=False`) for every backend, so PyTorch sees the same 640x640 or 960x960 input as the static exports. Batch 1, score threshold 0.1 while timing |
| Sessions | s1 (4 to 5 Oct 2026) and s2 (5 Oct 2026): full sweeps, no cool-down. s3 (5 Oct 2026): featured rows, each started after the CPU package fell below 55 C or after 180 s. `cpu_c_start` in the logs is read after model load and warmup, so it does not show whether 55 C was reached. Shanghai: one session, no cool-down |
| Parity | Against the session-1 PyTorch FP32 predictions on the same frames; score 0.25 applied to both, class-aware Hungarian matching at IoU 0.5, corner error only where the reference corner confidence is 0.5 or more |

Limitations worth knowing before using these numbers elsewhere:

- The timing is the model only. A real pipeline also decodes video, letterboxes, decodes outputs and tracks; on the iGPU rows at 640, letterboxing alone took 1.5 to 2.3 ms per frame (`pre_ms` in `output/latency.jsonl`).
- Laptop iGPU timings drift with temperature. The min-max column is wide on several iGPU rows (M INT8 float head: 26.1 to 47.1 ms) for that reason; it is reported rather than hidden.
- The main sweep is one clip from one camera. The Shanghai run is one more camera, one session.

## Licences: what Apache-2.0 here does NOT cover

The Apache-2.0 licence in `LICENSE` covers the code written for this repository: `scripts/`, `stubs/`, and the files in `output/`. It does not cover anything these scripts download, import or produce from third-party material. Checked on 5 Oct 2026:

- **UrbanOmniDetect-2 weights: AGPL-3.0.** The Hugging Face model card for [mehmetkeremturkcan/UrbanOmniDetect-2](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2) declares `license: agpl-3.0` and states "Released under the GNU Affero General Public License v3.0".
- **UrbanOmniDetect code ([mkturkcan/urbanomnidetect](https://github.com/mkturkcan/urbanomnidetect)): AGPL-3.0 by its README.** At commit `caa8537` the repository has no `LICENSE` file (GitHub reports no detected licence); its README says "This project is released under the GNU Affero General Public License v3.0". This repository does not include or redistribute any of it. The stub in `stubs/v2/` is our own two-line subclass of an Ultralytics class.
- **Ultralytics: AGPL-3.0.** The [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) repository is AGPL-3.0, and the 8.4.61 wheel's metadata says `License: AGPL-3.0`. Every script here that loads a model imports it.
- **TensorRT: NVIDIA proprietary.** `tensorrt-cu13` 10.16.1.11, installed by `uv sync`, declares `License: Proprietary` and ships NVIDIA's own licence terms.
- **Footage: CC BY 3.0, not redistributed.** Both videos are marked on YouTube as "Creative Commons Attribution license (reuse allowed)" (read with yt-dlp 2026.08.19), which is YouTube's CC BY 3.0 option. This repository only documents how to fetch them.
  - Karol Majek, "4K Road traffic video for object detection and tracking - free download now!", <https://youtu.be/MNn9qKG2UFI>
  - Thompson Fang, "[2160p]記錄週末傍晚的上海延安東路-西藏中路路口 In Recording of Yanan Road East-Tibet Road Middle Crossing", <https://youtu.be/O0LyetdH4Lg>
- **open-image-models: MIT.** The [ankandrew/open-image-models](https://github.com/ankandrew/open-image-models) repository and the 0.6.0 wheel are MIT. The plate detector's ONNX file is downloaded at run time from that repository's GitHub release assets; I found no separate licence statement for the weights or their training data.
- **Training data.** The UrbanOmniView dataset is CC BY-NC 4.0 (its Hugging Face card), but it is the training set of the first UrbanOmniDetect, not of UrbanOmniDetect-2, and nothing here downloads it. The UrbanOmniDetect-2 card lists COCO, VisDrone, KITTI, DAIR-V2X, CDrone, MeshFleet and Objaverse as its training data; I did not verify how those datasets' terms bear on the weights.

**Running these scripts imports Ultralytics and loads AGPL-3.0 weights.** That is fine for study and benchmarking. Anyone who distributes a product built on them, or offers one to users over a network, must meet the AGPL-3.0 terms for Ultralytics and for UrbanOmniDetect-2 (or obtain other licences from their owners), whatever the licence of this repository.
