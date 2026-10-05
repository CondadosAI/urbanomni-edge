"""Copy the raw benchmark and parity logs from runs/ into output/ for publishing.

The only change to any row is that the machine's hostname ("host", written by
bench.py) is dropped. Rows whose tag starts with "test-" (smoke runs) are
skipped. Every other field keeps its value and its key order, and the script
checks that before writing.

    uv run python scripts/export_evidence.py
"""
import json
from pathlib import Path

# output file <- source files, concatenated in this order
EXPORTS = {
    "output/latency.jsonl": ["runs/bench/results.jsonl"],
    "output/latency_shanghai.jsonl": ["runs/bench_shanghai/results.jsonl"],
    "output/parity_footbridge.jsonl": ["runs/parity_s1.jsonl", "runs/parity_int8.jsonl"],
    "output/parity_shanghai.jsonl": ["runs/parity_shanghai.jsonl"],
}
DROP = ("host",)


def main():
    for dst, srcs in EXPORTS.items():
        out, skipped, dropped = [], 0, 0
        for src in srcs:
            for line in Path(src).read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get("tag", "")).startswith("test-"):
                    skipped += 1
                    continue
                clean = {k: v for k, v in row.items() if k not in DROP}
                dropped += len(row) - len(clean)
                assert all(json.dumps(clean[k]) == json.dumps(row[k]) for k in clean)
                for k in ("model", "clip"):
                    assert not str(clean.get(k, "")).startswith("/"), f"absolute path in {src}: {clean[k]}"
                out.append(json.dumps(clean))
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_text("\n".join(out) + "\n")
        print(f"{dst}: {len(out)} rows from {', '.join(srcs)}; {skipped} test rows skipped; {dropped} fields dropped")


if __name__ == "__main__":
    main()
