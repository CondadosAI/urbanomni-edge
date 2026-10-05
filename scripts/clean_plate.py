"""Empty-road background for a fixed camera: the per-pixel temporal median.

Moving traffic occupies any given pixel for a minority of the frames, so the
median over a few minutes of footage is the road with no vehicles on it. A
second, cartoon-style version flattens the texture into a map-like look.

    uv run python scripts/clean_plate.py --clip data/clips/karol_calib_1080.mp4 \
        --out runs/plate/road.png --every 3
"""
from pathlib import Path

import click
import cv2
import numpy as np


def cartoon(img, k=7):
    """Flat, map-like rendering: mean-shift smoothing, then k colours, no drawn edges."""
    smooth = cv2.pyrMeanShiftFiltering(img, sp=15, sr=30)
    smooth = cv2.medianBlur(smooth, 5)
    data = smooth.reshape(-1, 3).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    return centers.astype(np.uint8)[labels.ravel()].reshape(img.shape)


def vehicle_masks(frames, model, device, pad):
    """Boolean (N, H, W): True where a detected road user covers the pixel."""
    import sys

    sys.path.insert(0, "stubs")
    import v2.hybrid_v2  # noqa: F401
    from ultralytics import YOLO

    m = YOLO(model, task="pose")
    masks = np.zeros((len(frames),) + frames[0].shape[:2], bool)
    for k, f in enumerate(frames):
        r = m.predict(f, imgsz=640, conf=0.15, device=device, rect=False, verbose=False,
                      project=str(Path("runs/ul").resolve()), name="predict", exist_ok=True, save=False)[0]
        for x1, y1, x2, y2 in r.boxes.xyxy.cpu().numpy():
            px, py = (x2 - x1) * pad, (y2 - y1) * pad
            masks[k, max(0, int(y1 - py)):int(y2 + py), max(0, int(x1 - px)):int(x2 + px)] = True
    print(f"masked {masks.mean() * 100:.1f}% of pixel-samples")
    return masks


def masked_median(stack, masks, rows=60):
    """Per-pixel median over the samples not covered by a vehicle; plain median where all are."""
    n, h, w, _ = stack.shape
    out = np.empty((h, w, 3), np.uint8)
    for y in range(0, h, rows):
        s = stack[:, y:y + rows].astype(np.float32)
        m = masks[:, y:y + rows]
        s[m] = np.nan
        med = np.nanmedian(s, axis=0)
        fallback = np.median(stack[:, y:y + rows], axis=0)
        out[y:y + rows] = np.where(np.isnan(med), fallback, med).astype(np.uint8)
    return out


@click.command()
@click.option("--clip", required=True, type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
@click.option("--every", default=3, type=int, help="Use one frame in N.")
@click.option("--max-frames", default=300, type=int)
@click.option("--mask-model", default=None, type=click.Path(exists=True),
              help="Detector used to mask vehicles out before the median (for queues that never clear).")
@click.option("--mask-device", default="intel:gpu.0")
@click.option("--mask-pad", default=0.1, type=float, help="Fractional padding around each masked box.")
def main(clip, out, every, max_frames, mask_model, mask_device, mask_pad):
    cap = cv2.VideoCapture(clip)
    frames, i = [], 0
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        if i % every == 0:
            frames.append(f)
        i += 1
    cap.release()
    stack = np.stack(frames)
    if mask_model is None:
        plate = np.median(stack, axis=0).astype(np.uint8)
    else:
        plate = masked_median(stack, vehicle_masks(frames, mask_model, mask_device, mask_pad))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), plate)
    cv2.imwrite(str(out.with_name(out.stem + "_cartoon.png")), cartoon(plate))
    print(f"PLATE {out} from {len(frames)} frames")


if __name__ == "__main__":
    main()
