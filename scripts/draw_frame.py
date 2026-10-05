"""Draw UrbanOmniDetect-2 cuboids on one frame of a clip (sanity check).

Corner order (v2): 0-3 ground contact, 4-7 top. Edges: ground ring, top ring,
four verticals.
"""
import sys

import click
import cv2
import numpy as np

sys.path.insert(0, "stubs")
import v2.hybrid_v2  # noqa: E402,F401
from ultralytics import YOLO  # noqa: E402

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
ROAD = {0, 1, 2, 3, 5, 7}


def draw(img, r, kconf=0.5):
    for b, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.keypoints.data.cpu().numpy()):
        if int(c) not in ROAD:
            continue
        if (k[:, 2] >= kconf).all():
            p = k[:, :2].astype(int)
            for a, bb in EDGES:
                col = (60, 200, 255) if a < 4 and bb < 4 else (255, 160, 40)
                cv2.line(img, tuple(p[a]), tuple(p[bb]), col, 2, cv2.LINE_AA)
        else:
            x1, y1, x2, y2 = b.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (120, 120, 120), 1)
    return img


@click.command()
@click.option("--model", required=True)
@click.option("--device", default="intel:cpu")
@click.option("--imgsz", default=960, type=int)
@click.option("--clip", required=True)
@click.option("--t", default=0.0, type=float, help="Seconds into the clip.")
@click.option("--conf", default=0.25, type=float)
@click.option("--out", required=True)
def main(model, device, imgsz, clip, t, conf, out):
    cap = cv2.VideoCapture(clip)
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, f = cap.read()
    assert ok
    m = YOLO(model, task="pose")
    r = m.predict(f, imgsz=imgsz, conf=conf, device=device, rect=False, verbose=False)[0]
    n_ok = int(((r.keypoints.data[:, :, 2] >= 0.5).all(1)).sum())
    print(f"dets={len(r.boxes)} cuboids_all_corners_conf>=0.5={n_ok} classes={sorted(set(r.boxes.cls.int().tolist()))}")
    cv2.imwrite(out, draw(f, r))


if __name__ == "__main__":
    main()
