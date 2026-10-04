#!/usr/bin/env python3
"""Verify fresh MIL-to-H13G compilation against every complete reference HWX."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from compile_h13g_gpt2 import compile_models, reference_labels
from h13g_reference import read_object


def section_bytes(data, parsed, segment, name):
    section = next(s for s in parsed["sections"] if s["segment"] == segment and s["name"] == name)
    return data[section["offset"]:section["offset"] + section["size"]]


def invoke(compiler, model, output):
    return subprocess.run([str(compiler), "--mil", str(model / "model.mil"),
        "--model-root", str(model), "--target", "H13G", "--output", str(output)],
        capture_output=True, text=True)


def check_independence(dump, compiler):
    # Copy only MIL and raw model weights. Rename all SSA identifiers and the
    # function. The compiler must select the same schedule from typed edges.
    cases = ["prefill_final_ln_L-1", "decode_ffn_L0", "decode_ffn_L6",
             "decode_ffn_L10", "decode_proj_L0", "prefill_attn_L0"]
    checks = []
    with tempfile.TemporaryDirectory(prefix="h13g-compiler-independence-") as directory:
        temporary = Path(directory)
        for case in cases:
            source = dump / "bundles" / (case + "_loaded")
            model = temporary / case
            model.mkdir()
            shutil.copy2(source / "model.mil", model / "model.mil")
            shutil.copytree(source / "weights", model / "weights")
            mil = (model / "model.mil").read_text()
            identifiers = re.findall(r"(?:tensor<[^\n]*?>|fp16|int32|bool|string)\s+(\w+)\s*(?:=|\))", mil)
            # The function input has no '='. Include it explicitly.
            identifiers += re.findall(r"func\s+\w+<[^>]+>\([^\n]*?\]\>\s+(\w+)\)", mil)
            mapping = {name: f"value_{i}" for i, name in enumerate(dict.fromkeys(identifiers))}
            mapping["main"] = "renamed_function"
            def rename(match):
                # Named argument keys such as x= are syntax, not SSA values.
                prefix = mil[:match.start()].rstrip()
                if prefix and prefix[-1] in "(," and re.match(r"\s*=", mil[match.end():]):
                    return match[0]
                return mapping.get(match[0], match[0])
            renamed = re.sub(r"\b[A-Za-z_]\w*\b", rename, mil)
            # Rename the raw weight file too, preserving its valid blob headers.
            renamed = renamed.replace("weights/packed.bin", "weights/source-weights.bin")
            (model / "weights/packed.bin").rename(model / "weights/source-weights.bin")
            (model / "model.mil").write_text(renamed)
            output = temporary / (case + "-compiled")
            process = invoke(compiler, model, output)
            if process.returncode:
                raise AssertionError(f"renamed {case}: {process.stderr}")
            actual, ac = read_object(output / "program-0.hwx")
            original, oc = read_object(dump / "hwx" / case / "model.hwx")
            for segment, section in [("__TEXT", "__text"), ("__TEXT", "__const"), ("__KERN_0", "__kern_0")]:
                if section_bytes(actual, ac, segment, section) != section_bytes(original, oc, segment, section):
                    raise AssertionError(f"renamed {case}: {segment}/{section} changed")
            checks.append(dict(case=case, renamed_identifiers=True, renamed_function=True,
                               reference_payloads_exact=True))

        # Recompile a changed learned beta. Its emitted affine must reflect the
        # changed source value while the instruction schedule stays unchanged.
        model = temporary / "changed-weight"
        source = dump / "bundles/prefill_final_ln_L-1_loaded"
        shutil.copytree(source, model)
        mil = (model / "model.mil").read_text()
        chunk = int(re.search(r"lnf_beta = const.*offset=uint64\((\d+)\)", mil)[1])
        blob_path = model / "weights/packed.bin"
        blob = bytearray(blob_path.read_bytes())
        beta_offset = struct.unpack_from("<Q", blob, chunk + 16)[0]
        struct.pack_into("<e", blob, beta_offset, 0.75)
        blob_path.write_bytes(blob)
        output = temporary / "changed-weight-compiled"
        process = invoke(compiler, model, output)
        if process.returncode: raise AssertionError(process.stderr)
        actual, ac = read_object(output / "program-0.hwx")
        original, oc = read_object(dump / "hwx/prefill_final_ln_L-1/model.hwx")
        constants = section_bytes(actual, ac, "__TEXT", "__const")
        gamma = struct.unpack_from("<e", constants, 768 * 2)[0]
        fp32_ratio = struct.unpack("<f", struct.pack("<f", 0.75 / gamma))[0]
        expected = struct.pack("<e", fp32_ratio)
        if constants[:2] != expected or constants == section_bytes(original, oc, "__TEXT", "__const"):
            raise AssertionError("changed learned beta was not independently repacked")
        if section_bytes(actual, ac, "__TEXT", "__text") != section_bytes(original, oc, "__TEXT", "__text"):
            raise AssertionError("changing final LN beta changed the instruction schedule")
        checks.append(dict(case="changed-learned-beta", native_repacking=True))

        # Change a projection weight not present in the metadata alias catalog.
        # Verify its packed location and a new content-derived symbol, while
        # preserving the instruction stream selected from the typed graph.
        matrix_model = temporary / "changed-matrix"
        source = dump / "bundles/decode_proj_L0_loaded"
        matrix_model.mkdir()
        shutil.copy2(source / "model.mil", matrix_model / "model.mil")
        shutil.copytree(source / "weights", matrix_model / "weights")
        matrix_mil = (matrix_model / "model.mil").read_text()
        chunk = int(re.search(r"\bq_W = const.*offset=uint64\((\d+)\)", matrix_mil)[1])
        blob_path = matrix_model / "weights/packed.bin"
        blob = bytearray(blob_path.read_bytes())
        matrix_offset = struct.unpack_from("<Q", blob, chunk + 16)[0]
        struct.pack_into("<e", blob, matrix_offset, 0.75)
        blob_path.write_bytes(blob)
        destination = temporary / "changed-matrix-compiled"
        process = invoke(compiler, matrix_model, destination)
        if process.returncode: raise AssertionError(process.stderr)
        actual, ac = read_object(destination / "program-0.hwx")
        original, oc = read_object(dump / "hwx/decode_proj_L0/model.hwx")
        packed = section_bytes(actual, ac, "__KERN_0", "__kern_0")
        if packed[256 + 16 * 2:256 + 16 * 2 + 2] != struct.pack("<e", 0.75):
            raise AssertionError("changed matrix was not repacked from its source blob")
        if section_bytes(actual, ac, "__TEXT", "__text") != section_bytes(original, oc, "__TEXT", "__text"):
            raise AssertionError("changed projection weight changed the measured instruction schedule")
        digest = hashlib.sha256(packed[256:1181952]).hexdigest().upper()
        if ("K" + digest + "_ne_0").encode() not in actual:
            raise AssertionError("unknown coefficient data did not receive a fresh content-derived symbol")
        checks.append(dict(case="changed-learned-matrix", native_repacking=True,
                           new_content_derived_symbol=True, instructions_unchanged=True))

        # An unsupported literal, geometry or operand edge must fail instead
        # of silently emitting the measured program for a different graph.
        for label, changed in [
            ("epsilon", mil.replace("fp16(1e-05)", "fp16(0.1)")),
            ("geometry", mil.replace("768", "512")),
            ("operand-edge", mil.replace("mul(x=lnf_cent, y=lnf_cent)", "mul(x=x, y=lnf_cent)")),
        ]:
            (model / "model.mil").write_text(changed)
            destination = temporary / (label + "-rejected")
            process = invoke(compiler, model, destination)
            if process.returncode != 65 or "h13g.legalize.unsupported-graph" not in process.stderr or destination.exists():
                raise AssertionError(f"unsupported {label} was not rejected: {process.stderr}")
            checks.append(dict(case="unsupported-" + label, rejected=True))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=Path.home() / "Desktop/GPT2-ANE-Dump")
    parser.add_argument("--compiler", type=Path, default=ROOT / "build/mil-hwxc")
    parser.add_argument("--output", type=Path, default=ROOT / "build/h13g-gpt2")
    parser.add_argument("--report", type=Path, default=ROOT / "h13g-compiler-validation.json")
    args = parser.parse_args()
    records = compile_models(args.dump / "bundles", args.output, args.compiler.resolve(),
                             labels=reference_labels(args.dump / "hwx"))
    expected_names = {p.parent.name for p in (args.dump / "hwx").glob("*/model.hwx")}
    if len(records) != 49 or {r["kernel"] for r in records} != expected_names:
        raise AssertionError("compiler comparison does not cover all 49 captured graphs")
    for record in records:
        actual = (args.output / record["kernel"] / "program-0.hwx").read_bytes()
        expected = (args.dump / "hwx" / record["kernel"] / "model.hwx").read_bytes()
        if actual != expected:
            offset = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b),
                          min(len(actual), len(expected)))
            raise AssertionError(f"{record['kernel']}: full HWX differs at {offset:#x}; "
                                 f"lengths {len(actual)} / {len(expected)}")
        record["reference_sha256"] = hashlib.sha256(expected).hexdigest()
        record["whole_hwx_exact"] = True
    checks = check_independence(args.dump, args.compiler.resolve())
    report = dict(target="H13G/base M1", implementation="MIL parser, typed contracts, measured target fields, native packing and fresh object writer",
        hardware_execution=False, whole_hwx_exact=True, kernels=len(records),
        historical_metadata_labels=True, opaque_debug_symbol_aliases=True,
        records=records, independence_checks=checks,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "plugins/H13G").rglob("*")) if p.is_file()})
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(kernels=len(records), whole_hwx_exact=True,
                         independence_checks=len(checks), hardware_execution=False), indent=2))


if __name__ == "__main__":
    main()
