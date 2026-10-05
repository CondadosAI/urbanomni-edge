"""Time one exported UrbanOmniDetect-2 model on a clip and save its predictions.

Every backend goes through the same Ultralytics predictor with a square
letterbox (rect=False), so PyTorch sees the same 640x640 / 960x960 input as the
static-shape exports. Timing is Ultralytics' own per-stage profile
(preprocess / inference / postprocess, ms per frame); the first --warmup frames
are discarded.

    uv run python scripts/bench.py --model models/v2n_640_fp16.engine --device 0 \
        --imgsz 640 --clip data/clips/x_seg.mp4 --tag trt-fp16
"""
import json
import platform
import sys
import time
from pathlib import Path

import click
import cv2
import numpy as np

sys.path.insert(0, "stubs")
import v2.hybrid_v2  # noqa: E402,F401  (inference-only stub)
from ultralytics import YOLO  # noqa: E402


def sensors():
    """iGPU actual frequency (MHz) and CPU package temperature (C), or None."""
    out = {}
    for c in Path("/sys/class/drm").glob("card*/gt_act_freq_mhz"):
        out["igpu_mhz"] = int(c.read_text())
    for h in Path("/sys/class/hwmon").glob("hwmon*"):
        if (h / "name").read_text().strip() == "coretemp":
            out["cpu_c"] = int((h / "temp1_input").read_text()) / 1000
    return out


def read_frames(clip, n, stride):
    cap = cv2.VideoCapture(str(clip))
    frames, i = [], 0
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(f)
        i += 1
    cap.release()
    return frames


@click.command()
@click.option("--model", required=True)
@click.option("--device", required=True, help="cpu | 0 | intel:cpu | intel:gpu.0")
@click.option("--imgsz", default=640, type=int)
@click.option("--half", is_flag=True, help="PyTorch only: FP16 weights.")
@click.option("--clip", required=True, type=click.Path(exists=True))
@click.option("--frames", default=200, type=int)
@click.option("--stride", default=1, type=int)
@click.option("--warmup", default=20, type=int)
@click.option("--conf", default=0.1, type=float)
@click.option("--tag", required=True)
@click.option("--out", default="runs/bench", type=click.Path())
@click.option("--session", default="s1", help="Benchmark session id (separate boots/sittings).")
def main(model, device, imgsz, half, clip, frames, stride, warmup, conf, tag, out, session):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    imgs = read_frames(clip, frames + warmup, stride)
    m = YOLO(model, task="pose")
    kw = dict(imgsz=imgsz, conf=conf, device=device, rect=False, verbose=False, half=half,
              project=str(Path("runs/ul").resolve()), name="predict", exist_ok=True, save=False)

    for f in imgs[:warmup]:
        m.predict(f, **kw)

    speeds, preds, sens = [], [], []
    t0 = time.perf_counter()
    for f in imgs[warmup:]:
        r = m.predict(f, **kw)[0]
        sens.append(sensors())
        speeds.append([r.speed["preprocess"], r.speed["inference"], r.speed["postprocess"]])
        preds.append(
            dict(
                xyxy=r.boxes.xyxy.cpu().numpy(),
                cls=r.boxes.cls.cpu().numpy(),
                conf=r.boxes.conf.cpu().numpy(),
                kpts=r.keypoints.data.cpu().numpy(),
            )
        )
    wall = time.perf_counter() - t0
    s = np.array(speeds)
    row = dict(
        tag=tag,
        session=session,
        model=str(model),
        device=device,
        imgsz=imgsz,
        frames=len(s),
        warmup=warmup,
        pre_ms=float(np.median(s[:, 0])),
        inf_ms=float(np.median(s[:, 1])),
        inf_p10=float(np.percentile(s[:, 1], 10)),
        inf_p90=float(np.percentile(s[:, 1], 90)),
        post_ms=float(np.median(s[:, 2])),
        total_ms=float(np.median(s.sum(1))),
        wall_fps=len(s) / wall,
        igpu_mhz_median=float(np.median([x.get("igpu_mhz", np.nan) for x in sens])),
        cpu_c_max=float(np.max([x.get("cpu_c", np.nan) for x in sens])),
        cpu_c_start=sens[0].get("cpu_c") if sens else None,
        clip=str(clip),
        host=platform.node(),
        time=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    np.save(out / f"{tag}_{session}_preds.npy", np.array(preds, dtype=object), allow_pickle=True)
    with open(out / "results.jsonl", "a") as fh:
        fh.write(json.dumps(row) + "\n")
    print(json.dumps(row))


if __name__ == "__main__":
    main()
