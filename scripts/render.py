"""Render UrbanOmniDetect-2 cuboids on a clip, with the measured latency on screen.

Corner order (v2): 0-3 ground contact, 4-7 top. The ground face is filled so
the footprint reads at a glance; the HUD shows the backend and the rolling
median of the model's own inference time on this machine.

    uv run python scripts/render.py --model models/v2s_640_int8head_openvino_model \
        --device intel:gpu.0 --imgsz 640 --label "Intel Iris Xe iGPU · OpenVINO INT8" \
        --clip data/clips/karol_seg_1080.mp4 --out runs/render/s640_igpu_int8.mp4
"""
import subprocess
import sys
from collections import deque
from pathlib import Path

import click
import cv2
import numpy as np

sys.path.insert(0, "stubs")
import v2.hybrid_v2  # noqa: E402,F401
from ultralytics import YOLO  # noqa: E402

GROUND = [(0, 1), (1, 2), (2, 3), (3, 0)]
TOP = [(4, 5), (5, 6), (6, 7), (7, 4)]
VERT = [(0, 4), (1, 5), (2, 6), (3, 7)]
# BGR per COCO class: person, bicycle, car, motorcycle, bus, truck
COLORS = {0: (80, 220, 120), 1: (60, 200, 255), 2: (255, 170, 40), 3: (60, 200, 255), 5: (90, 90, 255), 7: (200, 110, 255)}


def dedupe(xyxy, conf, thr=0.9):
    """Indices to keep after class-agnostic suppression of near-identical boxes.

    The end-to-end head can report one vehicle twice under two classes (a bus
    as bus 0.33 and truck 0.29); keep the higher-scoring one.
    """
    keep = []
    for i in np.argsort(-conf):
        if all(box_iou(xyxy[i], xyxy[j]) < thr for j in keep):
            keep.append(i)
    return keep


def box_iou(a, b):
    tl, br = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    inter = np.prod(np.clip(br - tl, 0, None))
    return inter / (np.prod(a[2:] - a[:2]) + np.prod(b[2:] - b[:2]) - inter + 1e-9)


def hull_fill(corners, box):
    """Area of the corners' convex hull, clipped to the 2D box, over the box area.

    A sound cuboid's projection fills most of its box; a collapsed one (all
    corners bunched on one face) fills little of it.
    """
    x1, y1, x2, y2 = box
    pts = np.clip(corners, [x1, y1], [x2, y2]).astype(np.float32)
    area = cv2.contourArea(cv2.convexHull(pts))
    return area / max((x2 - x1) * (y2 - y1), 1.0)


def draw_cuboids(img, r, kconf, min_corners=6, min_fill=0.6):
    """Draw a cuboid when at least `min_corners` corners clear `kconf` and its
    hull fills at least `min_fill` of the 2D box; otherwise only the 2D box.

    A vehicle cut by the frame edge has corners outside the image whose
    confidence is low; requiring all eight drops every truncated vehicle.
    """
    overlay = img.copy()
    lines = []
    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    kpts = r.keypoints.data.cpu().numpy()
    boxes_only = []
    for i in dedupe(xyxy, conf):
        c, k = cls[i], kpts[i]
        if c not in COLORS or (k[:, 2] >= kconf).sum() < min_corners:
            continue
        if hull_fill(k[:, :2], xyxy[i]) < min_fill:
            boxes_only.append((xyxy[i].round().astype(int), COLORS[c]))
            continue
        p = k[:, :2].round().astype(np.int32)
        cv2.fillPoly(overlay, [p[:4]], COLORS[c])
        lines.append((p, COLORS[c]))
    img = cv2.addWeighted(overlay, 0.28, img, 0.72, 0)
    for (x1, y1, x2, y2), col in boxes_only:
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 1, cv2.LINE_AA)
    for p, col in lines:
        for a, b in VERT + TOP:
            cv2.line(img, tuple(p[a]), tuple(p[b]), col, 2, cv2.LINE_AA)
        for a, b in GROUND:
            cv2.line(img, tuple(p[a]), tuple(p[b]), (255, 255, 255), 2, cv2.LINE_AA)
    return img, len(lines)


def hud(img, label, model_name, ms, n):
    h, w = img.shape[:2]
    pad, lh = 18, 34
    rows = [label, f"{model_name}   inference {ms:.1f} ms ({1000 / ms:.0f} fps)", f"{n} cuboids"]
    box_w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0] for t in rows) + 2 * pad
    cv2.rectangle(img, (0, 0), (box_w, pad + lh * len(rows)), (20, 20, 20), -1)
    for i, t in enumerate(rows):
        cv2.putText(img, t, (pad, pad + 22 + i * lh), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


@click.command()
@click.option("--model", required=True)
@click.option("--device", required=True)
@click.option("--imgsz", default=640, type=int)
@click.option("--label", required=True, help="Hardware + runtime line for the HUD.")
@click.option("--model-name", default=None, help="Defaults to e.g. 'UrbanOmniDetect-2 S @640'.")
@click.option("--clip", required=True, type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
@click.option("--conf", default=0.25, type=float)
@click.option("--kconf", default=0.5, type=float)
@click.option("--width", default=1920, type=int, help="Output width.")
def main(model, device, imgsz, label, model_name, clip, out, conf, kconf, width):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if model_name is None:
        size = Path(model).name.split("_")[0].removeprefix("v2").upper()
        model_name = f"UrbanOmniDetect-2 {size} @{imgsz}"
    m = YOLO(model, task="pose")
    kw = dict(imgsz=imgsz, conf=conf, device=device, rect=False, verbose=False,
              project=str(Path("runs/ul").resolve()), name="predict", exist_ok=True, save=False)
    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    raw = out.with_suffix(".raw.mp4")
    writer = None
    window = deque(maxlen=30)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        r = m.predict(f, **kw)[0]
        window.append(r.speed["inference"])
        img, n = draw_cuboids(f, r, kconf)
        img = hud(img, label, model_name, float(np.median(window)), n)
        if img.shape[1] != width:
            img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1] / 2) * 2), interpolation=cv2.INTER_AREA)
        if writer is None:
            writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, img.shape[1::-1])
        writer.write(img)
    writer.release()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
    raw.unlink()
    print("RENDERED", out)


if __name__ == "__main__":
    main()
