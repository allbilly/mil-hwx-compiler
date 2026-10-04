#!/usr/bin/env python3
"""Compile the captured GPT-2 MIL graphs and ordinary weight blobs for H13G.

The input directory contains model.mil files and their BLOBFILE weight data.
The compiler does not need the reference HWX files or the replay package.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import re

from h13g_reference import container

ROOT = Path(__file__).resolve().parents[1]


def reference_labels(directory):
    """Read historical command-line labels only; no instruction/weight data."""
    labels = {}
    for path in directory.glob("*/model.hwx"):
        data = path.read_bytes()
        command = next(c for c in container(data)["commands"] if c["cmd"] == 8)
        start = command["offset"] + 8
        text = data[start:command["offset"] + command["size"]].split(b"\0")[0].decode()
        source = re.search(r"(?m)^\t-i (.+)$", text)
        output = re.search(r"(?m)^\t-o (.+)$", text)
        if not source or not output: raise ValueError(f"missing source/output labels in {path}")
        labels[path.parent.name] = dict(source=source[1], output=output[1])
    return labels


def compile_models(models, output, compiler, jobs=2, labels=None):
    sources = sorted(models.glob("*/model.mil"))
    if not sources:
        raise ValueError(f"no MIL models in {models}")
    output.mkdir(parents=True, exist_ok=True)

    def compile_one(source):
        name = source.parent.name.removesuffix("_loaded")
        destination = output / name
        command = [str(compiler), "--mil", str(source),
            "--model-root", str(source.parent), "--target", "H13G",
            "--output", str(destination)]
        if labels is not None:
            command += ["--source-label", labels[name]["source"], "--output-label", labels[name]["output"]]
        process = subprocess.run(command, capture_output=True, text=True)
        if process.returncode:
            raise RuntimeError(f"{name}: compiler exit {process.returncode}\n{process.stderr}")
        image = destination / "program-0.hwx"
        return dict(kernel=name, bundle=name, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    hwx_sha256=hashlib.sha256(image.read_bytes()).hexdigest(), bytes=image.stat().st_size)

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        records = list(executor.map(compile_one, sources))
    manifest = dict(formatVersion=1, target="H13G", kernels=records,
                    historical_metadata_labels=labels is not None)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, default=ROOT / "build/mil-hwxc")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--reference-labels", type=Path,
                        help="optional HWX directory supplying historical source/output labels for whole-file comparisons")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    labels = reference_labels(args.reference_labels) if args.reference_labels else None
    records = compile_models(args.models, args.output, args.compiler.resolve(), args.jobs, labels)
    print(f"Compiled {len(records)} H13G kernels into {args.output}")


if __name__ == "__main__":
    main()
