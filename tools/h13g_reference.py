"""Read captured H13G objects for target derivation and independent checks.

This module is a developer tool. It is never imported by the compiler.
"""
import struct


def container(data):
    if len(data) < 32:
        raise ValueError("truncated HWX header")
    header = struct.unpack_from("<8I", data)
    if header[:3] != (0xBEEFFACE, 128, 4):
        raise ValueError("expected H13G HWX")
    end = 32 + header[5]
    if end > len(data):
        raise ValueError("truncated load commands")
    commands, sections, segments = [], [], []
    offset = 32
    thread = None
    for _ in range(header[4]):
        cmd, size = struct.unpack_from("<2I", data, offset)
        if size < 8 or offset + size > end:
            raise ValueError("invalid load command")
        commands.append(dict(cmd=cmd, offset=offset, size=size))
        if cmd == 0x19:
            segment_name = data[offset + 8:offset + 24].split(b"\0")[0].decode()
            address, allocation, file_offset, file_size = struct.unpack_from("<4Q", data, offset + 24)
            segment = dict(name=segment_name, address=address, allocation=allocation,
                           offset=file_offset, size=file_size)
            segments.append(segment)
            count = struct.unpack_from("<I", data, offset + 64)[0]
            for i in range(count):
                pos = offset + 72 + 80 * i
                name = data[pos:pos + 16].split(b"\0")[0].decode()
                values = struct.unpack_from("<QQ8I", data, pos + 32)
                sections.append(dict(name=name, segment=segment_name, address=values[0],
                    size=values[1], offset=values[2], alignment=values[3],
                    relocation_offset=values[4], relocation_count=values[5],
                    flags=values[6], reserved=list(values[7:])))
        elif cmd == 4 and struct.unpack_from("<I", data, offset + 8)[0] == 1:
            bars = list(struct.unpack_from("<32Q", data, offset + 16))
            entry, size_minus_one, count = struct.unpack_from("<QII", data, offset + 0x810)
            thread = dict(bars=bars, entry=entry, first_task_size=(size_minus_one + 1) * 4,
                          task_count=count)
        offset += size
    if offset != end or thread is None:
        raise ValueError("invalid H13G command table")
    return dict(header=header, commands=commands, sections=sections,
                segments=segments, thread=thread)


def tasks(program, first_size, count):
    result = []
    offset, size = 0, first_size
    for i in range(count):
        if size < 40 or size % 4 or offset + size > len(program):
            raise ValueError("invalid captured task bounds")
        controls = list(struct.unpack_from("<10I", program, offset))
        if controls[0] & 0xffff != i:
            raise ValueError("nonsequential captured task ID")
        packets = []
        extended = struct.unpack_from("<I", program, offset + 40)[0] if controls[6] & (1 << 24) else 0
        cursor = offset + (44 if controls[6] & (1 << 24) else 40)
        while cursor < offset + size:
            word = struct.unpack_from("<I", program, cursor)[0]
            cursor += 4
            if not word:
                raise ValueError("unexpected zero packet inside task")
            length, address = (word >> 26) + 1, word & 0x3ffffff
            values = list(struct.unpack_from(f"<{length}I", program, cursor))
            cursor += length * 4
            packets.append(dict(address=address, values=values))
        if cursor != offset + size:
            raise ValueError("captured packet crosses task boundary")
        result.append(dict(offset=offset, size=size, controls=controls,
                           extended_control=extended,
                           padding_after=controls[7] - ((offset + size + 255) // 256 * 256)
                               if controls[7] else 0, packets=packets))
        if i + 1 == count:
            if controls[7] or not controls[0] & (1 << 25):
                raise ValueError("invalid final captured task")
        else:
            offset = controls[7]
            size = (((controls[1] >> 16) & 0x1ff) + 1) * 4
    return result


def read_object(path):
    data = path.read_bytes()
    parsed = container(data)
    text = next(s for s in parsed["sections"] if s["segment"] == "__TEXT" and s["name"] == "__text")
    program = data[text["offset"]:text["offset"] + text["size"]]
    parsed["tasks"] = tasks(program, parsed["thread"]["first_task_size"], parsed["thread"]["task_count"])
    return data, parsed
