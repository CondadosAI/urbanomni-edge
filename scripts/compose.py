"""Compose the demo video from a clip and UrbanOmniDetect-2 pseudo-labels.

Reads the per-frame geometry written by the upstream renderer's ``--labels``
(8 cuboid corners per track, the ground quad, its ground-plane footprint and
the image->ground homography H) and draws a dashboard:

  * camera view with the cuboids (top left),
  * an info strip with hardware, model and the measured detector latency
    (bottom left),
  * a minimap (right): an empty-road image of the same fixed camera warped
    through H onto the ground plane, with every footprint and a short trail per
    track on top. Warping the live frame instead would smear every vehicle away
    from the camera, since only the road surface lies on the ground plane.

    uv run python scripts/compose.py --clip data/clips/karol_cut_1080_blur.mp4 \
        --labels runs/labels/m640_cut.json.gz --out runs/render/demo_m640.mp4 \
        --line1 "Intel Iris Xe iGPU (laptop)  |  OpenVINO INT8" \
        --line2 "UrbanOmniDetect-2 M @640  |  detector 33 ms per frame (30 fps)"
"""
import gzip
import json
import subprocess
from collections import defaultdict, deque
from pathlib import Path

import click
import cv2
import numpy as np

W, H_OUT = 1920, 1080
CAM_W, CAM_H = 1440, 810
MAP_W = W - CAM_W
BG = (18, 18, 18)
WHITE = (245, 245, 245)
GREY = (150, 150, 150)
# BGR per COCO class
COLORS = {0: (80, 220, 120), 1: (60, 200, 255), 2: (255, 170, 40), 3: (60, 200, 255), 5: (90, 90, 255), 7: (200, 110, 255)}
NAMES = {0: "person", 1: "bike", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
PLURAL = {"person": "people", "bus": "buses"}
GROUND = [(0, 1), (1, 2), (2, 3), (3, 0)]
TOP = [(4, 5), (5, 6), (6, 7), (7, 4)]
VERT = [(0, 4), (1, 5), (2, 6), (3, 7)]


def ground_canvas(labels, margin, map_h, min_box_h):
    """Similarity transform ground -> minimap canvas covering the footprints of
    vehicles at least `min_box_h` px tall (far-field footprints stretch the map)."""
    pts = np.array([p for f in labels["frames"] for t in f["tracks"]
                    if t.get("footprint") and t["box"][3] - t["box"][1] >= min_box_h for p in t["footprint"]])
    lo, hi = np.percentile(pts, 0.5, axis=0), np.percentile(pts, 99.5, axis=0)
    span = hi - lo
    lo, hi = lo - span * margin, hi + span * margin
    s = min((MAP_W - 40) / (hi[0] - lo[0]), (map_h - 40) / (hi[1] - lo[1]))
    ox = (MAP_W - s * (hi[0] - lo[0])) / 2 - s * lo[0]
    oy = (map_h - s * (hi[1] - lo[1])) / 2 - s * lo[1]
    return np.array([[s, 0, ox], [0, s, oy], [0, 0, 1]], dtype=np.float64)


RED = (60, 60, 255)


def at_edge(t, w, h, margin):
    x1, y1, x2, y2 = t["box"]
    return margin > 0 and (x1 <= margin or y1 <= margin or x2 >= w - margin or y2 >= h - margin)


def draw_camera(img, tracks, hot=frozenset(), edge=0):
    overlay = img.copy()
    lines = []
    h, w = img.shape[:2]
    for t in tracks:
        if not t.get("kpts") or t["cls"] not in COLORS:
            continue
        if at_edge(t, w, h, edge):
            x1, y1, x2, y2 = (int(v) for v in t["box"])
            cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[t["cls"]], 1, cv2.LINE_AA)
            continue
        p = np.array(t["kpts"]).round().astype(np.int32)
        col = RED if t["id"] in hot else COLORS[t["cls"]]
        cv2.fillPoly(overlay, [p[:4]], col)
        lines.append((p, col))
    img = cv2.addWeighted(overlay, 0.28, img, 0.72, 0)
    for p, col in lines:
        for a, b in VERT + TOP:
            cv2.line(img, tuple(p[a]), tuple(p[b]), col, 2, cv2.LINE_AA)
        for a, b in GROUND:
            cv2.line(img, tuple(p[a]), tuple(p[b]), WHITE, 2, cv2.LINE_AA)
    return img, len(lines)


def map_background(plate, Hm, map_h, dim):
    """The empty road warped onto the ground plane once, dimmed so footprints read on top."""
    road = cv2.warpPerspective(plate, Hm, (MAP_W, map_h), flags=cv2.INTER_AREA, borderValue=BG)
    return cv2.addWeighted(road, 1 - dim, np.full_like(road, BG), dim, 0)


def draw_map(background, tracks, trails, hot=frozenset()):
    road = background.copy()
    for t in tracks:
        if not t.get("footprint") or t["cls"] not in COLORS:
            continue
        col = RED if t["id"] in hot else COLORS[t["cls"]]
        tr = trails[t["id"]]
        if len(tr) > 1:
            c = np.array(tr, np.int32).reshape(-1, 1, 2)
            cv2.polylines(road, [c], False, col, 2, cv2.LINE_AA)
        q = np.array(t["footprint_px"], np.int32)
        cv2.fillPoly(road, [q], col)
        cv2.polylines(road, [q], True, WHITE, 1, cv2.LINE_AA)
    return road


def text(img, s, org, scale, col=WHITE, thick=2):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


@click.command()
@click.option("--clip", required=True, type=click.Path(exists=True))
@click.option("--labels", required=True, type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
@click.option("--line1", required=True)
@click.option("--line2", required=True)
@click.option("--line3", default="3D cuboids from one camera, no calibration  |  top-down view via ground homography")
@click.option("--credit", default="Footage: Karol Majek, CC BY 3.0  |  Model: UrbanOmniDetect-2 (M. K. Turkcan), AGPL-3.0")
@click.option("--trail", default=45, type=int, help="Trail length in frames.")
@click.option("--margin", default=0.08, type=float)
@click.option("--plate", required=True, type=click.Path(exists=True),
              help="Empty-road image from the same fixed camera (scripts/clean_plate.py).")
@click.option("--dim", default=0.35, type=float, help="How much to darken the map background.")
@click.option("--conflicts", default=None, type=click.Path(exists=True),
              help="Output of scripts/conflicts.py; pairs under --ttc-max are drawn in red.")
@click.option("--ttc-max", default=2.0, type=float)
@click.option("--hold", default=10, type=int, help="Frames a flagged pair stays red after its last flag.")
@click.option("--edge", default=0, type=int,
              help="Vehicles whose box comes within this many px of the frame edge get a 2D box, no cuboid or footprint.")
@click.option("--map-min-box-h", default=0, type=float, help="Fit the minimap to vehicles at least this tall (px).")
def main(clip, labels, out, line1, line2, line3, credit, trail, margin, plate, dim, conflicts, ttc_max, hold,
         edge, map_min_box_h):
    lab = json.load(gzip.open(labels)) if labels.endswith(".gz") else json.load(open(labels))
    frames = {f["i"]: f for f in lab["frames"]}
    H = np.array(lab["frames"][0]["H"], np.float64)
    map_h = H_OUT - 60
    G2C = ground_canvas(lab, margin, map_h, map_min_box_h)
    Hm = G2C @ H  # image pixels -> minimap pixels
    background = map_background(cv2.imread(plate), Hm, map_h, dim)

    flags = {}
    if conflicts:
        cj = json.load(open(conflicts))
        for k, hits in cj["frames"].items():
            for h in hits:
                if h["ttc"] <= ttc_max:
                    flags.setdefault(int(k), []).append(h)
    shown = {}  # (a, b) -> (last frame flagged, ttc)

    cap = cv2.VideoCapture(clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_suffix(".raw.mp4")
    wr = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H_OUT))
    trails = defaultdict(lambda: deque(maxlen=trail))
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr = frames.get(i, {"tracks": []})
        tracks = fr["tracks"]
        live = set()
        for t in tracks:
            if t.get("footprint"):
                fp = np.array(t["footprint"], np.float64).reshape(-1, 1, 2)
                t["footprint_px"] = cv2.perspectiveTransform(fp, G2C).reshape(-1, 2)
                trails[t["id"]].append(t["footprint_px"].mean(0))
                live.add(t["id"])
        for tid in [k for k in trails if k not in live]:
            del trails[tid]

        for h in flags.get(i, []):
            shown[(h["a"], h["b"])] = (i, h["ttc"])
        live_ids = {t["id"] for t in tracks}
        active = {k: v for k, v in shown.items() if i - v[0] <= hold and k[0] in live_ids and k[1] in live_ids}
        hot = frozenset(x for k in active for x in k)

        if edge:
            fh, fw = f.shape[:2]
            tracks = [t for t in tracks if not (t.get("footprint") and at_edge(t, fw, fh, edge))] + \
                     [dict(t, footprint=None) for t in tracks if t.get("footprint") and at_edge(t, fw, fh, edge)]
        cam, n = draw_camera(f.copy(), tracks, hot, edge)
        byid = {t["id"]: t for t in tracks}
        for (a, b), (_, ttc) in active.items():
            pa = np.array(byid[a]["ground"]).mean(0).astype(int)
            pb = np.array(byid[b]["ground"]).mean(0).astype(int)
            cv2.line(cam, tuple(pa), tuple(pb), RED, 3, cv2.LINE_AA)
            mid = tuple(((pa + pb) // 2 + [12, -12]).tolist())
            label = f"TTC {ttc:.1f} s"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(cam, (mid[0] - 6, mid[1] - th - 8), (mid[0] + tw + 6, mid[1] + 8), RED, -1)
            text(cam, label, mid, 1.0)
        canvas = np.full((H_OUT, W, 3), BG, np.uint8)
        canvas[:CAM_H, :CAM_W] = cv2.resize(cam, (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
        mp = draw_map(background, tracks, trails, hot)
        for (a, b), (_, ttc) in active.items():
            if "footprint_px" in byid[a] and "footprint_px" in byid[b]:
                qa = byid[a]["footprint_px"].mean(0).astype(int)
                qb = byid[b]["footprint_px"].mean(0).astype(int)
                cv2.line(mp, tuple(qa), tuple(qb), RED, 2, cv2.LINE_AA)
        canvas[60:, CAM_W:] = mp
        cv2.line(canvas, (CAM_W, 0), (CAM_W, H_OUT), (60, 60, 60), 2)

        # minimap header
        text(canvas, "Top-down view", (CAM_W + 20, 40), 0.9)
        text(canvas, "camera", (CAM_W + MAP_W // 2 - 40, H_OUT - 14), 0.6, GREY, 1)
        # info strip
        y = CAM_H + 52
        text(canvas, line1, (30, y), 1.0)
        text(canvas, line2, (30, y + 48), 1.0)
        text(canvas, line3, (30, y + 96), 0.75, GREY)
        text(canvas, credit, (30, H_OUT - 24), 0.6, GREY, 1)
        counts = defaultdict(int)
        for t in tracks:
            if t.get("kpts") and t["cls"] in COLORS:
                counts[t["cls"]] += 1
        x = CAM_W - 30
        for c in sorted(counts, reverse=True):
            s = f"{counts[c]} {PLURAL.get(NAMES[c], NAMES[c] + 's') if counts[c] > 1 else NAMES[c]}"
            tw = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
            x -= tw
            text(canvas, s, (x, y), 0.8, COLORS[c])
            x -= 30
        wr.write(canvas)
        i += 1
    wr.release()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
    raw.unlink()
    print("COMPOSED", out, i, "frames")


if __name__ == "__main__":
    main()
