"""Build the results table: latency per backend (median per session, min-max
across sessions) joined with corner parity against PyTorch FP32.

    uv run python scripts/summarize.py > output/results.md
"""
import json
from collections import defaultdict
from pathlib import Path

RUNTIME = {
    "pt-cuda-fp32": ("RTX 3060 Laptop", "PyTorch FP32"),
    "trt-fp16": ("RTX 3060 Laptop", "TensorRT FP16"),
    "ov-igpu-fp16": ("Iris Xe iGPU", "OpenVINO FP16"),
    "ov-igpu-int8": ("Iris Xe iGPU", "OpenVINO INT8"),
    "ov-igpu-int8head": ("Iris Xe iGPU", "OpenVINO INT8, float head"),
    "ov-cpu-fp32": ("i7-12700H CPU", "OpenVINO FP32"),
    "ov-cpu-int8": ("i7-12700H CPU", "OpenVINO INT8"),
    "ov-cpu-int8head": ("i7-12700H CPU", "OpenVINO INT8, float head"),
}


def split(tag):
    rt, rest = tag.rsplit("-", 1)
    return rt, rest[0], int(rest[1:])


def main():
    lat = defaultdict(list)
    for line in Path("runs/bench/results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["tag"].startswith("test-"):
            continue
        lat[r["tag"]].append(r)
    par = {}
    for f in ("runs/parity_s1.jsonl", "runs/parity_int8.jsonl"):
        if Path(f).exists():
            for line in Path(f).read_text().splitlines():
                p = json.loads(line)
                par[p["test"].rsplit("_", 1)[0]] = p

    print("| Size | Input | Hardware | Runtime | Model ms (median) | min-max over sessions | Sessions | Corner err. median / p90 (px) | Dets matched |")
    print("|---|--:|---|---|--:|--:|--:|--:|--:|")
    keys = sorted(lat, key=lambda t: (split(t)[2], "nsmlx".index(split(t)[1]), list(RUNTIME).index(split(t)[0])))
    for tag in keys:
        rt, size, imgsz = split(tag)
        rows = lat[tag]
        meds = [r["inf_ms"] for r in rows]
        hw, name = RUNTIME[rt]
        p = par.get(tag)
        pe = "reference" if rt == "pt-cuda-fp32" else (
            f"{p['corner_px_median']:.2f} / {p['corner_px_p90']:.2f}" if p else "n/a")
        pm = "reference" if rt == "pt-cuda-fp32" else (f"{p['recall_vs_ref']:.3f}" if p else "n/a")
        med = sorted(meds)[len(meds) // 2]
        print(f"| {size.upper()} | {imgsz} | {hw} | {name} | {med:.1f} | {min(meds):.1f}-{max(meds):.1f} | "
              f"{len(rows)} | {pe} | {pm} |")


if __name__ == "__main__":
    main()
