"""Normalize public benchmarks once; retain source revisions and row audit."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import requests
from dotenv import load_dotenv
from huggingface_hub import HfApi

from neuralsignal.datasets.sources.terminal_wrench import TerminalWrenchSource
from neuralsignal.datasets.sources.trace import TraceSource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["terminal-wrench", "trace"])
    parser.add_argument("--variant", choices=["original", "sanitized", "stripped"], default="original")
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_dotenv(".env")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.dataset == "trace":
        revision = args.revision or HfApi().dataset_info("PatronusAI/trace-dataset").sha
        source = TraceSource(revision)
    else:
        revision = args.revision
        if not revision:
            response = requests.get("https://api.github.com/repos/few-sh/terminal-wrench/commits/main", timeout=60)
            response.raise_for_status()
            revision = response.json()["sha"]
        source = TerminalWrenchSource("runs/datasets/terminal-wrench-cache", revision, args.variant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_suffix(".pending")
    counts, ids, tasks, models = Counter(), set(), set(), Counter()
    with pending.open("w", encoding="utf-8") as handle:
        for example in source.iter_examples():
            if example.id in ids:
                raise ValueError(f"Duplicate trajectory: {example.id}")
            ids.add(example.id)
            digest = hashlib.sha256(json.dumps([example.input, example.output], ensure_ascii=False).encode()).hexdigest()
            metadata = {**example.metadata, "revision": revision, "transcript_sha256": digest}
            counts[metadata["abnormal"]] += 1
            if "task_id" in metadata:
                tasks.add(metadata["task_id"])
                models[metadata["model"]] += 1
            # Flat metadata is consumed unchanged by the existing HF JSON source.
            handle.write(json.dumps({"id": example.id, "input": example.input, "output": example.output,
                                     "labels": [], **metadata}, ensure_ascii=False) + "\n")
            if len(ids) % 250 == 0:
                print(f"Normalized {len(ids)} trajectories", flush=True)
    expected = {0: 249, 1: 268} if args.dataset == "trace" else {0: 3568, 1: 3632}
    if dict(counts) != expected:
        raise ValueError(f"Counts differ from published corpus: {dict(counts)} != {expected}")
    pending.rename(args.output)
    audit = {"dataset": args.dataset, "revision": revision, "rows": len(ids),
             "class_counts": dict(counts), "tasks": len(tasks), "models": dict(models),
             "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
             **getattr(source, "audit", {})}
    args.output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
