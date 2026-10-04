#!/usr/bin/env python3
"""Verify the independent C++ H13G packer against the original GPT-2 HWX.

The port supplies HF tensor extraction, reference parsing, and captured recipes.
The C++ executable supplies every learned byte in all regenerated payloads.
No raw or compiled learned weights are retained by this verifier.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path.home() / "ane/gpt2")
    parser.add_argument("--dump", type=Path, default=Path.home() / "Desktop/GPT2-ANE-Dump")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--packer", type=Path, default=repository / "build/h13g-pack")
    parser.add_argument("--report", type=Path, default=repository / "h13g-validation.json")
    args = parser.parse_args()
    sys.path[:0] = [str(args.package), str(args.package / "tools")]
    import numpy as np
    from external_weights import find_weights, verify_weights, load_weights
    from derive_packing import references

    source = find_weights(args.weights)
    assert source is not None, "cached GPT-2 checkpoint required"
    verify_weights(source, args.package)
    weights = load_weights(source)
    names = json.loads((args.package / "package.json").read_text())["kernels"]
    report = dict(target="H13G/base M1", implementation="native C++ H13GConstantPacker",
                  hardware_execution=False, learned_weights_retained=False,
                  checkpoint_sha256=digest(source.read_bytes()) if source.is_file() else "verified Orion blobs",
                  payloads=[], operations=[], affine_precision_checks=[], cli_checks=[])
    cache = {}
    with tempfile.TemporaryDirectory(prefix="h13g-gpt2-validation-") as directory:
        temporary = Path(directory)
        a, b, output = temporary / "a.bin", temporary / "b.bin", temporary / "packed.bin"

        def invoke(command, first, second):
            a.write_bytes(first)
            b.write_bytes(second)
            process = subprocess.run([str(args.packer), command[0], str(a), str(b), str(output),
                                      *command[1:]], capture_output=True, text=True)
            assert process.returncode == 0, process.stderr
            return output.read_bytes()

        def native(operation):
            key = json.dumps({k: v for k, v in operation.items() if k not in ("offset", "size")}, sort_keys=True)
            if key in cache:
                return cache[key]
            if operation["kind"] == "matrix":
                outputs, inputs = operation["shape"]
                packed = invoke(["matrix", str(inputs), str(outputs),
                                 ",".join(map(str, operation["tiles"]))],
                    weights.get(operation["matrix"], (outputs, inputs)).astype("<f2").tobytes(),
                    weights.get(operation["bias"], (outputs,)).astype("<f2").tobytes())
            else:
                assert operation["kind"] == "affine"
                packed = invoke(["affine", operation["layout"], str(operation.get("scale", 1))],
                    weights.get(operation["gamma"], (768,)).astype("<f2").tobytes(),
                    weights.get(operation["beta"], (768,)).astype("<f2").tobytes())
            cache[key] = packed
            return packed

        for name in names:
            meta = json.loads((args.package / "kernels" / name / "meta.json").read_text())
            original = references(args.dump, name, meta)
            for field, recipe in meta["packing"].items():
                regenerated = bytearray(recipe["size"])
                if "template" in recipe:
                    template = (args.package / recipe["template"]).read_bytes()
                    assert len(template) == len(regenerated)
                    regenerated[:] = template
                for offset, hexadecimal in recipe.get("literals", []):
                    literal = bytes.fromhex(hexadecimal)
                    assert 0 <= offset <= len(regenerated) - len(literal)
                    regenerated[offset:offset + len(literal)] = literal
                for operation in recipe["operations"]:
                    offset, size = operation["offset"], operation["size"]
                    packed = native(operation)
                    assert len(packed) == size and 0 <= offset <= len(regenerated) - size
                    assert not any(regenerated[offset:offset + size]), "learned bytes remain in template"
                    exact = packed == original[field][offset:offset + size]
                    assert exact, f"C++ packing mismatch: {name}/{field}/{operation}"
                    regenerated[offset:offset + size] = packed
                    report["operations"].append(dict(kernel=name, field=field, kind=operation["kind"],
                        name=operation.get("matrix", operation.get("gamma")), exact=exact, size=size))
                exact = regenerated == original[field]
                assert exact and digest(regenerated) == recipe["sha256"], f"full payload differs: {name}/{field}"
                report["payloads"].append(dict(kernel=name, field=field, size=len(regenerated),
                                               sha256=digest(regenerated), exact=exact))

        # Exercise affine folding over every finite, nonzero binary16 gamma
        # bit pattern (trimmed to complete groups of 16 for engine-pair layout).
        bits = np.arange(65536, dtype="<u2")
        finite_nonzero = bits[((bits & 0x7c00) != 0x7c00) & ((bits & 0x7fff) != 0)]
        finite_nonzero = finite_nonzero[:len(finite_nonzero) // 16 * 16]
        g = finite_nonzero.view("<f2").astype(np.float32)
        beta = finite_nonzero[::-1].copy()
        bv = beta.view("<f2").astype(np.float32)
        for scale in (1, 2, 32):
            with np.errstate(over="ignore"):
                ratio = (bv / g * np.float32(scale)).astype("<f2")
            for layout in ("linear", "engine_pairs"):
                if layout == "linear":
                    expected = ratio.tobytes() + finite_nonzero.tobytes()
                else:
                    pairs = np.column_stack((finite_nonzero, ratio.view("<u2")))
                    expected = pairs.reshape(-1, 16, 2).transpose(1, 0, 2).astype("<u2").tobytes()
                actual = invoke(["affine", layout, str(scale)], finite_nonzero.tobytes(), beta.tobytes())
                assert actual == expected, f"affine precision mismatch: {layout}/{scale}"
                report["affine_precision_checks"].append(dict(layout=layout, scale=scale,
                                                              channels=len(g), exact=True))

        # A rejected CLI call must leave an existing output intact.
        a.write_bytes(bytes(256 * 3 * 2))
        b.write_bytes(bytes(256 * 2))
        for schedule in ("8,8", "16,", "32", "-16"):
            output.write_bytes(b"preserve existing output")
            process = subprocess.run([str(args.packer), "matrix", str(a), str(b), str(output),
                                      "3", "256", schedule], capture_output=True, text=True)
            assert process.returncode != 0 and output.read_bytes() == b"preserve existing output"
            report["cli_checks"].append(dict(schedule=schedule, rejected=True, output_preserved=True))

    report["summary"] = dict(kernels=len(names), payloads=len(report["payloads"]),
        payloads_exact=sum(x["exact"] for x in report["payloads"]),
        matrix_blocks_exact=sum(x["kind"] == "matrix" and x["exact"] for x in report["operations"]),
        affine_blocks_exact=sum(x["kind"] == "affine" and x["exact"] for x in report["operations"]),
        unique_packing_operations=len(cache), affine_precision_checks=len(report["affine_precision_checks"]),
        cli_invalid_input_checks=len(report["cli_checks"]))
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
