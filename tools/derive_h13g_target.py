#!/usr/bin/env python3
"""Derive readable H13G target measurements from the captured GPT-2 graphs.

Preparation only. Production embeds decoded register fields, typed graph
contracts, relocations and object metadata; it reads no reference object or
instruction blob. Learned coefficients are never included in this catalog.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

from h13g_reference import read_object

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("norm768x32", "prefill_final_ln_L-1"),
    ("norm_expand_gelu_project_residual", "decode_ffn_L0"),
    ("norm_expand_gelu_project_residual_engine_affine", "decode_ffn_L6"),
    ("norm_parallel_project3", "decode_proj_L0"),
    ("norm_masked_attention12x32x64", "prefill_attn_L0"),
]


def named_string(text, names):
    # Source identifiers are metadata, never target selection keys. Longest
    # first prevents a one-letter input from replacing pieces of other names.
    import re
    for i, name in sorted(enumerate(names), key=lambda pair: -len(pair[1])):
        text = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])",
                      "${" + str(i) + "}", text)
        if text.startswith("TT" + name + "_"):
            text = "TT${" + str(i) + "}" + text[2 + len(name):]
    return text


def object_layout(data, parsed, names):
    commands = []
    for command in parsed["commands"]:
        start, size, cmd = command["offset"], command["size"], command["cmd"]
        b = data[start:start + size]
        record = dict(cmd=cmd)
        if cmd == 0x19:
            record.update(name=b[8:24].split(b"\0")[0].decode(),
                address=struct.unpack_from("<Q", b, 24)[0],
                allocation=struct.unpack_from("<Q", b, 32)[0],
                protection=list(struct.unpack_from("<2I", b, 56)),
                flags=struct.unpack_from("<I", b, 68)[0], sections=[])
            count = struct.unpack_from("<I", b, 64)[0]
            for i in range(count):
                s = b[72 + 80 * i:152 + 80 * i]
                values = struct.unpack_from("<QQ8I", s, 32)
                record["sections"].append(dict(name=s[:16].split(b"\0")[0].decode(),
                    segment=s[16:32].split(b"\0")[0].decode(),
                    address=values[0], size=values[1], alignment=values[3],
                    flags=values[6], reserved=list(values[7:])))
        elif cmd == 0x40:
            record.update(address=struct.unpack_from("<Q", b, 16)[0],
                          name=named_string(b[24:].split(b"\0")[0].decode(), names))
        elif cmd == 4:
            flavor = struct.unpack_from("<I", b, 8)[0]
            record["flavor"] = flavor
            if flavor == 1:
                record["fields"] = [[i, struct.unpack_from("<I", b, i)[0]]
                    for i in range(0x820, 0x888, 4) if struct.unpack_from("<I", b, i)[0]]
            elif flavor == 3:
                string, short = struct.unpack_from("<I", b, 0x20)[0], struct.unpack_from("<I", b, 0x7c)[0]
                symbol = b[string:b.find(b"\0", string)].decode()
                short_name = b[short:b.find(b"\0", short)].decode()
                record.update(value=names.index(short_name) if names else -1, output=symbol.endswith("@output"),
                              index=struct.unpack_from("<I", b, 0x14)[0])
            else:
                raise ValueError(f"unknown thread flavor {flavor}")
        elif cmd == 8:
            import re
            text = b[8:].split(b"\0")[0].decode()
            text = re.sub(r"(?m)^\t-i .+$", "\t-i ${source-label}", text)
            text = re.sub(r"(?m)^\t-o .+$", "\t-o ${output-label}", text)
            record["text"] = text
        elif cmd == 0x31:
            record["owner"] = b[8:24].split(b"\0")[0].decode()
            length = struct.unpack_from("<Q", b, 32)[0]
            payload = b[40:40 + length]
            if record["owner"] == "src model info":
                record["text"] = payload.decode()
            elif record["owner"] == "rt.version":
                record["version"] = struct.unpack("<I", payload)[0]
            else:
                raise ValueError("unknown object note")
        elif cmd != 2:
            raise ValueError(f"unknown command {cmd:#x}")
        commands.append(record)
    symtab = next(c for c in parsed["commands"] if c["cmd"] == 2)
    symoff, count, stroff, _ = struct.unpack_from("<4I", data, symtab["offset"] + 8)
    symbols = []
    for i in range(count):
        string, kind, section, description, value = struct.unpack_from("<IBBHQ", data, symoff + 16 * i)
        name = data[stroff + string:data.find(b"\0", stroff + string)].decode()
        symbols.append(dict(name=named_string(name, names), kind=kind, section=section,
                            description=description, value=value))
    text = next(s for s in parsed["sections"] if s["segment"] == "__TEXT" and s["name"] == "__text")
    relocations = [list(struct.unpack_from("<2I", data, text["relocation_offset"] + 8 * i))
                   for i in range(text["relocation_count"])]
    return dict(commands=commands, symbols=symbols, relocations=relocations,
                bars=parsed["thread"]["bars"])


def kernel_groups(layout, coefficient_section):
    import re
    groups = {}
    for symbol in layout["symbols"]:
        if symbol["kind"] != 0xf or not re.match(r"K[0-9A-F]{64}(?:_ne_\d+)?$", symbol["name"]):
            continue
        prefix = re.sub(r"_ne_\d+$", "", symbol["name"])
        offset = symbol["value"] - coefficient_section["address"]
        groups[prefix] = min(offset, groups.get(prefix, offset))
    ordered = sorted(groups.items(), key=lambda item: item[1])
    return [dict(prefix=prefix, offset=offset,
                 size=(ordered[i + 1][1] if i + 1 < len(ordered) else coefficient_section["size"]) - offset)
            for i, (prefix, offset) in enumerate(ordered)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", type=Path, default=Path.home() / "Desktop/GPT2-ANE-Dump")
    parser.add_argument("--inspector", type=Path, default=ROOT / "build/h13g-inspect")
    parser.add_argument("--output", type=Path, default=ROOT / "plugins/H13G/Encoding/H13GTargetData.inc")
    args = parser.parse_args()
    catalog = dict(format=1, packets=[], contracts={}, plans={}, symbol_aliases={})
    packet_ids = {}
    receipts = []
    for key, name in CASES:
        data, parsed = read_object(args.dump / "hwx" / name / "model.hwx")
        inspection = json.loads(subprocess.check_output([str(args.inspector),
            str(args.dump / "bundles" / (name + "_loaded") / "model.mil")], text=True))
        names = inspection["names"]
        contract_key = key.removesuffix("_engine_affine")
        catalog["contracts"][contract_key] = inspection["contract"]
        task_rows = []
        for task in parsed["tasks"]:
            controls = task["controls"].copy()
            controls[0] &= ~((1 << 25) | 0xffff)
            controls[1] &= ~(0x1ff << 16)
            controls[7] = 0
            packet_indices = []
            for packet in task["packets"]:
                signature = json.dumps(packet, sort_keys=True)
                if signature not in packet_ids:
                    packet_ids[signature] = len(catalog["packets"])
                    catalog["packets"].append(packet)
                packet_indices.append(packet_ids[signature])
            task_rows.append(dict(controls=controls, extended=task["extended_control"],
                padding=task["padding_after"], packets=packet_indices))
        weights = next(s for s in parsed["sections"] if s["segment"] == "__KERN_0")
        lut = list(struct.unpack_from("<128H", data, weights["offset"]))
        plan = dict(contract=contract_key, tasks=task_rows, lut=lut,
                    object=object_layout(data, parsed, names), gamma=1, beta=2)
        plan["kernel_groups"] = kernel_groups(plan["object"], weights)
        if "expand" in key:
            engine = key.endswith("_engine_affine")
            affine_size = 3072 if engine else 0
            fc_offset = 256 + affine_size
            fc_size = 4724736
            gelu_offset = fc_offset + fc_size
            proj_offset = gelu_offset + 2048
            plan.update(matrices=[dict(weight=6, bias=7, inputs=768, outputs=3072,
                tiles=[32] * 6, offset=fc_offset), dict(weight=12, bias=13,
                inputs=3072, outputs=768, tiles=[16] * 3, offset=proj_offset)],
                gelu=dict(offset=gelu_offset,
                    values=list(struct.unpack_from("<64H", data, weights["offset"] + gelu_offset))))
            if engine: plan["engine_affine"] = True
        elif "parallel" in key:
            plan["matrices"] = [dict(weight=w, bias=b, inputs=768, outputs=768,
                tiles=[16] * 3, offset=256 + i * 1181696)
                for i, (w, b) in enumerate([(6, 7), (10, 11), (8, 9)])]
        elif "attention" in key:
            plan["matrices"] = [dict(weight=w, bias=b, inputs=768, outputs=768,
                tiles=tiles, offset=offset) for w, b, tiles, offset in
                [(10, 11, [16] * 3, 256), (8, 9, [16] * 3, 1181952),
                 (6, 7, [32, 16], 2363648), (17, 18, [16] * 3, 3546880)]]
            plan.update(mask=15, softmax_lut_offset=3545344)
        else:
            plan["matrices"] = []
        catalog["plans"][key] = plan
        receipts.append(dict(plan=key, source=name, source_sha256=hashlib.sha256(data).hexdigest(),
                             tasks=len(task_rows)))
    # Apple's opaque coefficient symbol IDs are debug metadata. Keep a
    # compatibility alias from the independently packed content's SHA-256;
    # this lookup is never used for legality, scheduling or instructions.
    # Unknown coefficients get a new content-derived symbol automatically.
    for path in sorted((args.dump / "hwx").glob("*/model.hwx")):
        name = path.parent.name
        key = "norm768x32" if "final_ln" in name else "norm_parallel_project3" if "decode_proj" in name else \
            "norm_masked_attention12x32x64" if "prefill_attn" in name else "norm_expand_gelu_project_residual"
        data, parsed = read_object(path)
        if "ffn" in name and len(parsed["tasks"]) == 20: key += "_engine_affine"
        weights = next(s for s in parsed["sections"] if s["segment"] == "__KERN_0")
        layout = object_layout(data, parsed, [])
        groups = kernel_groups(layout, weights)
        for group in groups:
            start = weights["offset"] + group["offset"]
            digest = hashlib.sha256(data[start:start + group["size"]]).hexdigest()
            alias_key = f"{key}:{group['offset']}:{group['size']}:{digest}"
            existing = catalog["symbol_aliases"].setdefault(alias_key, group["prefix"])
            if existing != group["prefix"]: raise ValueError("ambiguous coefficient symbol alias")
    raw = json.dumps(catalog, sort_keys=True, separators=(",", ":"))
    args.output.write_text("// Decoded target measurements; generated by tools/derive_h13g_target.py.\n"
        "// Contains no learned weights or binary program/container templates.\n"
        'static const char kH13GTargetData[] = R"H13GDATA(' + raw + ')H13GDATA";\n')
    (ROOT / "build/h13g-target-derivation.json").write_text(json.dumps(receipts, indent=2) + "\n")
    print(f"Derived {len(catalog['plans'])} plans, {len(catalog['packets'])} unique register packets, "
          f"{len(catalog['symbol_aliases'])} debug symbol aliases")


if __name__ == "__main__":
    main()
