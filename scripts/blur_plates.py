"""Blur licence plates in a clip before rendering demo videos.

Plates are detected on the 4K source in overlapping tiles (open-image-models'
YOLOv9 plate detector, MIT), mapped to the output resolution, padded, held for a
few frames either side to cover single-frame misses, and Gaussian-blurred.

    uv run python scripts/blur_plates.py --src data/clips/karol-road-4k_seg.mp4 \
        --out data/clips/karol_seg_1080_blur.mp4 --width 1920
"""
import subprocess
from pathlib import Path

import click
import cv2
import numpy as np
from open_image_models import create_detector


def tiles(w, h, nx, ny, overlap):
    tw, th = int(w / nx * (1 + overlap)), int(h / ny * (1 + overlap))
    for iy in range(ny):
        for ix in range(nx):
            x0 = min(int(ix * (w - tw) / max(nx - 1, 1)), w - tw)
            y0 = min(int(iy * (h - th) / max(ny - 1, 1)), h - th)
            yield x0, y0, tw, th


@click.command()
@click.option("--src", required=True, type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
@click.option("--width", default=1920, type=int)
@click.option("--model", default="yolo-v9-s-608-license-plate-end2end")
@click.option("--conf", default=0.2, type=float)
@click.option("--hold", default=2, type=int, help="Frames a box is kept after/before it is seen.")
@click.option("--pad", default=0.35, type=float, help="Fractional padding around each plate.")
def main(src, out, width, model, conf, hold, pad):
    det = create_detector(model, conf_thresh=conf, providers=["CPUExecutionProvider"])
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    boxes = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        s = width / w
        found = []
        for x0, y0, tw, th in tiles(w, h, 3, 2, 0.15):
            for d in det.predict(np.ascontiguousarray(f[y0:y0 + th, x0:x0 + tw])):
                b = d.bounding_box
                found.append(np.array([b.x1 + x0, b.y1 + y0, b.x2 + x0, b.y2 + y0], float) * s)
        boxes.append(found)
    cap.release()

    # Second pass: re-read, resize and blur, so no frame is held in memory.
    n = len(boxes)
    raw = Path(out).with_suffix(".raw.mp4")
    cap = cv2.VideoCapture(src)
    wr = None
    total = 0
    for i in range(n):
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        f = cv2.resize(f, (width, round(h * width / w / 2) * 2), interpolation=cv2.INTER_AREA)
        if wr is None:
            wr = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, f.shape[1::-1])
        for b in (b for j in range(max(0, i - hold), min(n, i + hold + 1)) for b in boxes[j]):
            x1, y1, x2, y2 = b
            px, py = (x2 - x1) * pad, (y2 - y1) * pad
            x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
            x2, y2 = int(min(f.shape[1], x2 + px)), int(min(f.shape[0], y2 + py))
            if x2 > x1 and y2 > y1:
                k = max(3, ((y2 - y1) // 2) | 1)
                f[y1:y2, x1:x2] = cv2.GaussianBlur(f[y1:y2, x1:x2], (k * 2 + 1, k * 2 + 1), 0)
        total += len(boxes[i])
        wr.write(f)
    wr.release()
    cap.release()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-crf", "16",
                    "-pix_fmt", "yuv420p", "-an", str(out)], check=True)
    raw.unlink()
    print(f"BLURRED {out}: {n} frames, {total} plate detections ({total / n:.1f}/frame)")


if __name__ == "__main__":
    main()
