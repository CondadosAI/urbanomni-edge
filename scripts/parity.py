"""Compare a backend's saved predictions with a reference run (PyTorch FP32).

Detections are matched per frame by class and box IoU (Hungarian, IoU >= 0.5).
Corner error is the pixel distance between matched cuboid corners, counted only
where the reference is confident in that corner (kpt conf >= --kconf).

    uv run python scripts/parity.py --ref runs/bench/pt-cuda-n640_preds.npy \
        runs/bench/trt-fp16-n640_preds.npy runs/bench/ov-int8-n640_preds.npy
"""
import json
from pathlib import Path

import click
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a, b):
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    aa = np.prod(a[:, 2:] - a[:, :2], axis=1)
    ab = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def compare(ref, test, conf, kconf):
    n_ref = n_test = n_match = 0
    ious, corner_err = [], []
    for r, t in zip(ref, test):
        rk = r["conf"] >= conf
        tk = t["conf"] >= conf
        rb, rc, rp = r["xyxy"][rk], r["cls"][rk], r["kpts"][rk]
        tb, tc, tp = t["xyxy"][tk], t["cls"][tk], t["kpts"][tk]
        n_ref += len(rb)
        n_test += len(tb)
        if not len(rb) or not len(tb):
            continue
        m = iou(rb, tb) * (rc[:, None] == tc[None, :])
        ri, ti = linear_sum_assignment(-m)
        ok = m[ri, ti] >= 0.5
        ri, ti = ri[ok], ti[ok]
        n_match += len(ri)
        ious += m[ri, ti].tolist()
        for a, b in zip(ri, ti):
            vis = rp[a, :, 2] >= kconf
            d = np.linalg.norm(rp[a, vis, :2] - tp[b, vis, :2], axis=1)
            corner_err += d.tolist()
    ce = np.array(corner_err) if corner_err else np.array([np.nan])
    return dict(
        ref_dets=n_ref,
        test_dets=n_test,
        recall_vs_ref=n_match / max(n_ref, 1),
        precision_vs_ref=n_match / max(n_test, 1),
        box_iou_median=float(np.median(ious)) if ious else float("nan"),
        corner_px_median=float(np.median(ce)),
        corner_px_p90=float(np.percentile(ce, 90)),
        corners=int(len(corner_err)),
    )


@click.command()
@click.option("--ref", required=True, type=click.Path(exists=True))
@click.option("--conf", default=0.25, type=float, help="Score threshold applied to both runs.")
@click.option("--kconf", default=0.5, type=float)
@click.option("--out", default=None, type=click.Path())
@click.argument("tests", nargs=-1, type=click.Path(exists=True))
def main(ref, conf, kconf, out, tests):
    R = np.load(ref, allow_pickle=True)
    rows = []
    for t in tests:
        T = np.load(t, allow_pickle=True)
        assert len(R) == len(T), f"{t}: {len(T)} frames vs ref {len(R)}"
        row = dict(test=Path(t).name.removesuffix("_preds.npy"), ref=Path(ref).name.removesuffix("_preds.npy"))
        row |= compare(R, T, conf, kconf)
        rows.append(row)
        print(json.dumps(row))
    if out:
        with open(out, "a") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
