"""Export an UrbanOmniDetect-2 checkpoint with stock Ultralytics.

    uv run python scripts/export.py --size n --imgsz 640 --fmt openvino --half
"""
import shutil
import sys
from pathlib import Path

import click

sys.path.insert(0, "stubs")
import v2.hybrid_v2  # noqa: E402,F401  (inference-only stub)
from ultralytics import YOLO  # noqa: E402


@click.command()
@click.option("--size", default="n", type=click.Choice(list("nsmlx")))
@click.option("--imgsz", default=640, type=int)
@click.option("--fmt", default="openvino", type=click.Choice(["openvino", "engine", "onnx"]))
@click.option("--half", is_flag=True)
@click.option("--int8", is_flag=True)
@click.option("--data", default=None, help="Calibration dataset yaml for INT8.")
@click.option("--fraction", default=1.0, type=float)
def main(size, imgsz, fmt, half, int8, data, fraction):
    ckpt = Path(f"checkpoints/urbanomnidetect_v2_{size}_640.pt")
    m = YOLO(str(ckpt))
    kw = dict(format=fmt, imgsz=imgsz, half=half, int8=int8, dynamic=False, batch=1)
    if fmt == "engine":
        kw["device"] = 0
    if int8:
        kw["data"] = data
        kw["fraction"] = fraction
    out = Path(m.export(**kw))
    prec = "int8" if int8 else ("fp16" if half else "fp32")
    dst = Path("models") / f"v2{size}_{imgsz}_{prec}{'_openvino_model' if fmt == 'openvino' else out.suffix}"
    dst.parent.mkdir(exist_ok=True)
    if dst.exists():
        shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
    shutil.move(str(out), dst)
    print("EXPORTED", dst)


if __name__ == "__main__":
    main()
