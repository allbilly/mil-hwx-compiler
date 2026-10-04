#!/usr/bin/env python3
"""Re-encode every captured H13G task from decoded fields with the native writer."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from h13g_reference import read_object


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=Path.home() / "Desktop/GPT2-ANE-Dump")
    parser.add_argument("--encoder", type=Path, default=ROOT / "build/h13g-encode-tasks")
    parser.add_argument("--report", type=Path, default=ROOT / "build/h13g-task-validation.json")
    args = parser.parse_args()
    records = []
    with tempfile.TemporaryDirectory(prefix="h13g-task-check-") as directory:
        fields = Path(directory) / "fields.txt"
        output = Path(directory) / "program.bin"
        for path in sorted((args.dump / "hwx").glob("*/model.hwx")):
            data, parsed = read_object(path)
            words = [len(parsed["tasks"])]
            for task in parsed["tasks"]:
                words.extend(task["controls"])
                if task["controls"][6] & (1 << 24): words.append(task["extended_control"])
                words.append(task["padding_after"])
                words.append(len(task["packets"]))
                for packet in task["packets"]:
                    words.extend([packet["address"], len(packet["values"]), *packet["values"]])
            fields.write_text("\n".join(f"{w:x}" for w in words) + "\n")
            process = subprocess.run([str(args.encoder), str(fields), str(output)],
                                     capture_output=True, text=True)
            if process.returncode:
                raise AssertionError(f"{path.parent.name}: {process.stderr}")
            text = next(s for s in parsed["sections"] if s["segment"] == "__TEXT" and s["name"] == "__text")
            expected = data[text["offset"]:text["offset"] + text["size"]]
            actual = output.read_bytes()
            if actual != expected:
                mismatch = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b),
                                min(len(actual), len(expected)))
                raise AssertionError(f"{path.parent.name}: encoded instruction mismatch at {mismatch:#x}")
            records.append(dict(kernel=path.parent.name, tasks=len(parsed["tasks"]),
                bytes=len(actual), sha256=hashlib.sha256(actual).hexdigest(), exact=True))
    if len(records) != 49:
        raise AssertionError(f"expected all 49 kernels, found {len(records)}")
    report = dict(kernels=len(records), tasks=sum(r["tasks"] for r in records),
                  byte_exact=True, hardware_execution=False, records=records)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
