"""Post-training INT8 quantization of an OpenVINO IR with NNCF.

Calibration frames are sampled from a clip with the same preprocessing the
Ultralytics predictor applies (square letterbox, RGB, /255, NCHW). Use frames
that are NOT in the evaluation segment.

    uv run python scripts/quant_ov.py --ir models/v2n_640_fp16_openvino_model \
        --clip data/clips/x.mp4 --skip-start 60 --n 300 --imgsz 640
"""
import shutil
from pathlib import Path

import click
import cv2
import nncf
import numpy as np
import openvino as ov
from ultralytics.data.augment import LetterBox


def calib_frames(clip, n, skip_start, skip_end):
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    lo, hi = int(skip_start * fps), total - int(skip_end * fps)
    idx = np.linspace(lo, hi - 1, n).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return frames


@click.command()
@click.option("--ir", required=True, type=click.Path(exists=True))
@click.option("--clip", required=True, type=click.Path(exists=True), multiple=True)
@click.option("--n", default=300, type=int, help="Calibration frames per clip.")
@click.option("--skip-start", default=0.0, type=float, help="Seconds to skip at the start (eval segment).")
@click.option("--skip-end", default=0.0, type=float)
@click.option("--imgsz", default=640, type=int)
@click.option("--preset", default="mixed", type=click.Choice(["performance", "mixed"]))
@click.option("--ignore-head", is_flag=True, help="Keep the detection/pose head in float.")
def main(ir, clip, n, skip_start, skip_end, imgsz, preset, ignore_head):
    ir = Path(ir)
    xml = next(ir.glob("*.xml"))
    model = ov.Core().read_model(xml)
    lb = LetterBox((imgsz, imgsz), auto=False)

    frames = [f for c in clip for f in calib_frames(c, n, skip_start, skip_end)]
    print(f"calibration frames: {len(frames)}")

    def transform(f):
        x = lb(image=f)[..., ::-1].transpose(2, 0, 1)
        return np.ascontiguousarray(x, dtype=np.float32)[None] / 255.0

    kw = {}
    if ignore_head:
        # The Pose26 head is module 23 in every scale's yaml.
        head_ops = [
            op.get_friendly_name()
            for op in model.get_ops()
            if op.get_friendly_name().startswith("__module.model.23.")
        ]
        kw["ignored_scope"] = nncf.IgnoredScope(names=head_ops, validate=False)
        print(f"ignored head ops: {len(head_ops)}")

    q = nncf.quantize(
        model,
        nncf.Dataset(frames, transform),
        preset=nncf.QuantizationPreset.MIXED if preset == "mixed" else nncf.QuantizationPreset.PERFORMANCE,
        subset_size=len(frames),
        **kw,
    )
    name = ir.name.replace("_fp16_", "_int8head_" if ignore_head else "_int8_").replace("_fp32_", "_int8_")
    dst = ir.parent / name
    dst.mkdir(exist_ok=True)
    ov.save_model(q, dst / xml.name)
    shutil.copy(ir / "metadata.yaml", dst / "metadata.yaml")
    print("SAVED", dst)


if __name__ == "__main__":
    main()
