#!/usr/bin/env python3
"""Compare the native upstream H16G packer against original H13G GPT-2 HWX.

Run with the GPT-2 port's Python dependencies installed. No generated weights
are retained: temporary raw and packed matrices are deleted on exit.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def matrix_only(block, operation):
    """Remove bias and engine padding from the actual H13G reference block."""
    result = bytearray()
    cursor = 0
    inputs = operation["shape"][1]
    engine_bytes = sum(tile * (inputs + 1) * 2 for tile in operation["tiles"])
    for _ in range(16):
        for tile in operation["tiles"]:
            cursor += tile * 2
            size = inputs * tile * 2
            result.extend(block[cursor:cursor + size])
            cursor += size
        cursor += -engine_bytes % 64
    assert cursor == len(block)
    assert len(result) == operation["shape"][0] * inputs * 2
    return bytes(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--package", type=Path, default=Path.home() / "ane/gpt2")
    parser.add_argument("--dump", type=Path, default=Path.home() / "Desktop/GPT2-ANE-Dump")
    parser.add_argument("--report", type=Path, default=Path(__file__).with_name("results.json"))
    args = parser.parse_args()
    sys.path[:0] = [str(args.package), str(args.package / "tools")]
    import numpy as np
    from external_weights import find_weights, verify_weights, load_weights
    from derive_packing import references
    from packing import reconstruct, operation_bytes

    source = find_weights()
    assert source is not None
    verify_weights(source, args.package)
    weights = load_weights(source)
    names = json.loads((args.package / "package.json").read_text())["kernels"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.upstream, text=True).strip()
    report = dict(upstream="maderix/mil-hwx-compiler", commit=commit,
                  target_reference="H13G/base M1", target_upstream="H16G/M4",
                  native_upstream_packer=True, hardware_execution=False,
                  checkpoint_sha256=digest(source.read_bytes()),
                  own_payloads=[], matrices=[], compiler_probes=[])
    packer = args.upstream / "build/gpt2-pack-probe"
    compiler = args.upstream / "build/mil-hwxc"
    cache = {}
    with tempfile.TemporaryDirectory(prefix="gpt2-maderix-compare-") as directory:
        temporary = Path(directory)
        raw_path, packed_path = temporary / "raw.bin", temporary / "packed.bin"
        for name in names:
            meta = json.loads((args.package / "kernels" / name / "meta.json").read_text())
            reference = references(args.dump, name, meta)
            for field, actual in reference.items():
                generated = reconstruct(args.package, weights, meta["packing"][field])
                exact = generated == actual
                assert exact, f"our reconstruction failed: {name}/{field}"
                report["own_payloads"].append(dict(kernel=name, field=field, exact=exact,
                                                   sha256=digest(actual), size=len(actual)))
            for operation in meta["packing"]["weights"]["operations"]:
                if operation["kind"] != "matrix":
                    continue
                offset, size = operation["offset"], operation["size"]
                block = reference["weights"][offset:offset + size]
                assert operation_bytes(weights, operation) == block
                expected = matrix_only(block, operation)
                outputs, inputs = operation["shape"]
                matrix = weights.get(operation["matrix"], (outputs, inputs)).astype("<f2").tobytes()
                formats = [(0, "dense4x8")]
                if inputs == outputs and outputs % 128 == 0:
                    formats.append((2, "layout_conv"))
                for format_id, format_name in formats:
                    key = (operation["matrix"], format_id)
                    if key not in cache:
                        raw_path.write_bytes(matrix)
                        process = subprocess.run([str(packer), str(raw_path), str(packed_path),
                                                  str(inputs), str(outputs), str(format_id)],
                                                 capture_output=True, text=True)
                        assert process.returncode == 0, process.stderr
                        cache[key] = packed_path.read_bytes()
                    packed = cache[key]
                    assert len(packed) == len(expected)
                    actual_words = np.frombuffer(packed, dtype="<u2")
                    reference_words = np.frombuffer(expected, dtype="<u2")
                    different = np.flatnonzero(actual_words != reference_words)
                    report["matrices"].append(dict(
                        kernel=name, matrix=operation["matrix"], shape=[outputs, inputs],
                        h13g_tiles=operation["tiles"], upstream_format=format_name,
                        exact=packed == expected, compared_fp16_values=len(actual_words),
                        mismatched_fp16_values=len(different),
                        first_mismatch_fp16_index=int(different[0]) if len(different) else None,
                        upstream_sha256=digest(packed), reference_sha256=digest(expected)))
            # Use the real captured MIL and its original source weight bundle.
            process = subprocess.run([str(compiler), "--mil",
                str(args.package / "kernels" / name / "model.mil"), "--model-root",
                str(args.dump / "bundles" / (name + "_loaded")), "--target", "H16G",
                "--output", str(temporary / (name + "-compiled"))], capture_output=True, text=True)
            report["compiler_probes"].append(dict(kernel=name, target="H16G",
                exit_code=process.returncode, stdout=process.stdout.strip(), stderr=process.stderr.strip()))
        for target in ("H13G", "H16G"):
            process = subprocess.run([str(compiler), "--mil",
                str(args.upstream / "tests/fixtures/conv_relu.mil"), "--model-root",
                str(args.upstream / "tests/models/conv_relu"), "--target", target,
                "--output", str(temporary / (target + "-control"))], capture_output=True, text=True)
            report["compiler_probes"].append(dict(kernel="upstream_conv_relu_control", target=target,
                exit_code=process.returncode, stdout=process.stdout.strip(), stderr=process.stderr.strip()))
    report["summary"] = dict(
        own_reference_payloads=len(report["own_payloads"]),
        own_reference_payloads_exact=sum(x["exact"] for x in report["own_payloads"]),
        matrix_comparisons_by_format={f: dict(
            comparisons=sum(x["upstream_format"] == f for x in report["matrices"]),
            exact=sum(x["upstream_format"] == f and x["exact"] for x in report["matrices"]))
            for f in sorted({x["upstream_format"] for x in report["matrices"]})},
        gpt2_mil_graphs=len(names),
        gpt2_mil_graphs_compiled=sum(x["exit_code"] == 0 and x["kernel"] in names
                                   for x in report["compiler_probes"]),
        gpt2_compile_diagnostics=dict(Counter(x["stderr"].split(": error ")[-1]
            for x in report["compiler_probes"] if x["kernel"] in names)))
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
