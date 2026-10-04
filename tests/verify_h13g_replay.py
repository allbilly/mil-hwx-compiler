#!/usr/bin/env python3
"""Compare native compilation with the 49 Asahi GPT-2 replay kernel payloads.

The portable replay package records hashes of the original macOS dump's
program, constants and coefficients. Only MIL and ordinary cached HF tensors
are given to mil-hwxc. Replay instructions/packed coefficients are never input
to the compiler. Raw BLOBFILEs live in a temporary directory and are removed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = {
    "ln1_g": "ln1_g", "ln1_beta": "ln1_b", "ln2_g": "ln2_g", "ln2_beta": "ln2_b",
    "q_W": "wq", "k_W": "wk", "v_W": "wv", "q_b": "bq", "k_b": "bk", "v_b": "bv",
    "proj_W": "wo", "proj_b": "bo", "ffn_fc_W": "wfc", "ffn_fc_b": "bfc",
    "ffn_proj_W": "wproj", "ffn_proj_b": "bproj", "lnf_g": "ln_f_g", "lnf_beta": "ln_f_b",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def make_blob(mil, name, weights, np):
    blob = bytearray(64)
    for line in mil.splitlines():
        if "BLOBFILE" not in line:
            continue
        field, dimensions = re.search(
            r"tensor<fp16, \[([^]]+)\]> (\w+) = const", line).group(2, 1)
        shape = tuple(int(d) for d in dimensions.split(","))
        offset = int(re.search(r"offset=uint64\((\d+)\)", line).group(1))
        if field == "attn_mask":
            value = np.triu(np.full((32, 32), -1e4, dtype="<f2"), 1)
        else:
            tensor = PARAMETERS[field]
            if not name.endswith("L-1"):
                tensor = f"layer{int(name.rsplit('_L', 1)[1])}/{tensor}"
            # The replay tensor uses [out,in] or [channels]; MIL adds unit axes.
            logical = tuple(d for d in shape if d != 1)
            value = weights.get(tensor, logical).astype("<f2")
        raw = value.tobytes()
        if len(raw) != 2 * int(np.prod(shape)):
            raise AssertionError(f"raw tensor size differs: {name}/{field}")
        end = offset + 64 + len(raw)
        if len(blob) < end:
            blob.extend(bytes(end - len(blob)))
        struct.pack_into("<IIQQ", blob, offset, 0xDEADBEEF, 1, len(raw), offset + 64)
        blob[offset + 64:end] = raw
    return blob


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--compiler", type=Path, default=ROOT / "build/mil-hwxc")
    parser.add_argument("--output", type=Path, default=ROOT / "build/h13g-replay-compiled")
    parser.add_argument("--report", type=Path, default=ROOT / "build/h13g-replay-validation.json")
    args = parser.parse_args()
    package = args.package.expanduser().resolve()
    sys.path.insert(0, str(package))
    import numpy as np
    from external_weights import find_weights, verify_weights, load_weights
    from hwx import parse_container, parse_tasks, relocate

    source = find_weights(args.weights)
    if source is None:
        raise ValueError("A cached GPT-2 checkpoint is required; no weights are downloaded.")
    verify_weights(source, package)
    weights = load_weights(source)
    names = json.loads((package / "package.json").read_text())["kernels"]
    if len(names) != 49 or len(set(names)) != 49:
        raise AssertionError("Reference package must contain all 49 captured kernels")
    checksums = json.loads((package / "checksums.json").read_text())
    report = dict(host=platform.platform(), target="H13G", kernels=[],
        hardware_execution=False, whole_hwx_comparison=False,
        reference="Asahi replay payload hashes exported from the macOS dump",
        compiler_sha256=sha(args.compiler.read_bytes()),
        raw_weights_retained=False)
    with tempfile.TemporaryDirectory(prefix="h13g-native-mil-") as temporary:
        models = Path(temporary)
        for name in names:
            for filename in ("model.mil", "meta.json"):
                relative = f"kernels/{name}/{filename}"
                if sha((package / relative).read_bytes()) != checksums[relative]:
                    raise AssertionError(f"Reference package changed: {relative}")
            meta = json.loads((package / "kernels" / name / "meta.json").read_text())
            mil = (package / "kernels" / name / "model.mil").read_text()
            model = models / name
            (model / "weights").mkdir(parents=True)
            (model / "model.mil").write_text(mil)
            (model / "weights/packed.bin").write_bytes(make_blob(mil, name, weights, np))
            output = args.output.resolve() / name
            process = subprocess.run([str(args.compiler.resolve()), "--mil", str(model / "model.mil"),
                "--model-root", str(model), "--target", "H13G", "--output", str(output)],
                capture_output=True, text=True)
            if process.returncode:
                raise RuntimeError(f"{name}: compiler exit {process.returncode}\n{process.stderr}")
            image = (output / "program-0.hwx").read_bytes()
            parsed = parse_container(image)
            text = next(s for s in parsed["segments"] if s["name"] == "__TEXT")
            kern = next(s for s in parsed["segments"] if s["name"] == "__KERN_0")
            const = next(s for s in parsed["sections"]
                if s["name"] == "__const" and s["segment"] == "__TEXT")
            program = image[text["fileoff"]:text["fileoff"] + text["filesize"]]
            tasks = parse_tasks(program, meta["td_size"], meta["td_count"])
            payloads = dict(
                program=relocate(program, tasks, {int(k): v for k, v in meta["bank_map"].items()}),
                weights=image[kern["fileoff"]:kern["fileoff"] + kern["filesize"]],
                constants=image[const["offset"]:const["offset"] + const["size"]])
            hashes = {}
            for field, payload in payloads.items():
                expected = meta["packing"][field]
                actual = sha(payload)
                if len(payload) != expected["size"] or actual != expected["sha256"]:
                    raise AssertionError(f"{name}/{field}: {actual} differs from {expected['sha256']}")
                hashes[field] = actual
            report["kernels"].append(dict(kernel=name, tasks=len(tasks), payloads=hashes, exact=True))
            print(f"{name}: program/constants/weights match ({len(tasks)} tasks)", flush=True)
    report.update(kernels_exact=len(report["kernels"]), payloads_exact=len(names) * 3,
                  tasks_exact=sum(record["tasks"] for record in report["kernels"]))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"H13G replay comparison: {len(names)}/49 kernels, {len(names) * 3}/147 payloads PASS")


if __name__ == "__main__":
    main()
