"""Time-to-collision between every pair of tracked vehicles, on the ground plane.

Each track's footprint (the 4 ground corners, already on the ground plane in the
pseudo-labels) moves with a velocity fitted by least squares over a short
centred window. For every pair present in a frame, both footprints are swept
forward at constant velocity and the first time they touch is the TTC.

TTC is a distance divided by a closing speed, so the unknown scale of the
calibration-free ground plane cancels: the result is in seconds whatever the
plane's units.

    uv run python scripts/conflicts.py --labels runs/labels/m640_cut.json.gz \
        --out runs/conflicts/m640_cut.json
"""
import gzip
import json
from collections import defaultdict
from pathlib import Path

import click
import numpy as np
from shapely.geometry import Polygon

VEHICLES = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck


def velocities(frames, half_window, fps):
    """Per (track, frame) ground velocity in plane units per second."""
    cents = defaultdict(dict)
    for f in frames:
        for t in f["tracks"]:
            if t.get("footprint") and t.get("observed", True):
                cents[t["id"]][f["i"]] = np.mean(t["footprint"], axis=0)
    vel = {}
    for tid, c in cents.items():
        idx = sorted(c)
        for i in idx:
            w = [j for j in range(i - half_window, i + half_window + 1) if j in c]
            if len(w) < half_window + 1:
                continue
            tt = (np.array(w) - i) / fps
            P = np.array([c[j] for j in w])
            vel[(tid, i)] = np.polyfit(tt, P, 1)[0]
    return vel


def inside(box, w, h, margin):
    x1, y1, x2, y2 = box
    return x1 > margin and y1 > margin and x2 < w - margin and y2 < h - margin


def first_contact(fa, va, fb, vb, horizon, step):
    """Earliest t in (0, horizon] at which the translated footprints intersect, else None."""
    A, B = np.asarray(fa), np.asarray(fb)
    if Polygon(A).intersects(Polygon(B)):
        return None  # already overlapping: a footprint artefact, not a conflict
    # cheap disc pre-check on the relative motion
    dp = B.mean(0) - A.mean(0)
    dv = vb - va
    ra = np.max(np.linalg.norm(A - A.mean(0), axis=1))
    rb = np.max(np.linalg.norm(B - B.mean(0), axis=1))
    a, b, c = dv @ dv, 2 * dp @ dv, dp @ dp - (ra + rb) ** 2
    if a < 1e-12 or b >= 0 or b * b - 4 * a * c < 0:
        return None
    t0 = max((-b - np.sqrt(b * b - 4 * a * c)) / (2 * a), step)
    if t0 > horizon:
        return None
    for t in np.arange(t0, horizon + 1e-9, step):
        if Polygon(A + va * t).intersects(Polygon(B + vb * t)):
            return float(t)
    return None


@click.command()
@click.option("--labels", required=True, type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
@click.option("--horizon", default=3.0, type=float, help="Seconds to look ahead.")
@click.option("--step", default=0.05, type=float)
@click.option("--half-window", default=6, type=int, help="Frames each side for the velocity fit.")
@click.option("--min-closing", default=0.25, type=float,
              help="Ignore pairs closing slower than this many median vehicle lengths per second.")
@click.option("--min-box-h", default=60, type=float,
              help="Ignore vehicles whose 2D box is shorter than this (px): far-field footprints are unreliable.")
@click.option("--opposite-cos", default=-0.7, type=float,
              help="Ignore pairs moving in opposite, near-parallel directions (separate carriageways).")
@click.option("--border", default=4, type=float,
              help="Ignore vehicles whose 2D box comes within this many px of the frame edge (truncated).")
@click.option("--frame-wh", default="1920x1080", help="Size of the processed frames (boxes of truncated vehicles extend past it).")
def main(labels, out, horizon, step, half_window, min_closing, min_box_h, opposite_cos, border, frame_wh):
    lab = json.load(gzip.open(labels)) if labels.endswith(".gz") else json.load(open(labels))
    fps = lab["source"]["fps"]
    frames = lab["frames"]
    vel = velocities(frames, half_window, fps)
    img_w, img_h = (int(v) for v in frame_wh.split("x"))

    lengths = [max(np.linalg.norm(np.subtract(t["footprint"][k], t["footprint"][(k + 1) % 4])) for k in range(4))
               for f in frames for t in f["tracks"] if t.get("footprint") and t["cls"] == 2]
    unit = float(np.median(lengths))  # one median car length in plane units

    per_frame, pairs = {}, defaultdict(list)
    for f in frames:
        i = f["i"]
        ts = [t for t in f["tracks"] if t["cls"] in VEHICLES and t.get("footprint") and (t["id"], i) in vel
              and t["box"][3] - t["box"][1] >= min_box_h and inside(t["box"], img_w, img_h, border)]
        hits = []
        for a in range(len(ts)):
            for b in range(a + 1, len(ts)):
                A, B = ts[a], ts[b]
                va, vb = vel[(A["id"], i)], vel[(B["id"], i)]
                na, nb = np.linalg.norm(va), np.linalg.norm(vb)
                if na > 1e-9 and nb > 1e-9 and (va @ vb) / (na * nb) < opposite_cos:
                    continue
                dp = np.mean(B["footprint"], 0) - np.mean(A["footprint"], 0)
                closing = -(dp @ (vb - va)) / (np.linalg.norm(dp) + 1e-9)
                if closing < min_closing * unit:
                    continue
                ttc = first_contact(A["footprint"], va, B["footprint"], vb, horizon, step)
                if ttc is not None:
                    hits.append({"a": A["id"], "b": B["id"], "ttc": round(ttc, 2),
                                 "closing_len_per_s": round(float(closing / unit), 2)})
                    pairs[(A["id"], B["id"])].append((i, ttc))
        per_frame[i] = hits

    events = []
    for (a, b), hs in pairs.items():
        i_min, t_min = min(hs, key=lambda x: x[1])
        events.append({"a": a, "b": b, "min_ttc": round(t_min, 2), "frame": i_min, "t": round(i_min / fps, 2),
                       "frames_flagged": len(hs)})
    events.sort(key=lambda e: e["min_ttc"])
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"fps": fps, "horizon_s": horizon, "unit_car_len": unit, "events": events,
               "frames": per_frame}, open(out, "w"))
    print(f"{len(events)} pairs with TTC <= {horizon} s; car length = {unit:.3f} plane units")
    for e in events[:15]:
        print(e)


if __name__ == "__main__":
    main()
