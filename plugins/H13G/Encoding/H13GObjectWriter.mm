#import "H13GObjectWriter.h"
#import "HWXImage.h"

#import "ANEMachO.h"

static NSUInteger alignTo(NSUInteger value, NSUInteger alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

static void word(NSMutableData *data, NSUInteger offset, uint32_t value) {
    [data replaceBytesInRange:NSMakeRange(offset, 4) withBytes:&value];
}

static void quad(NSMutableData *data, NSUInteger offset, uint64_t value) {
    [data replaceBytesInRange:NSMakeRange(offset, 8) withBytes:&value];
}

static void string(NSMutableData *data, NSUInteger offset, NSString *value) {
    NSData *bytes = [value dataUsingEncoding:NSUTF8StringEncoding];
    [data replaceBytesInRange:NSMakeRange(offset, bytes.length) withBytes:bytes.bytes];
}

static void name(char result[16], NSString *value) {
    memset(result, 0, 16);
    memcpy(result, value.UTF8String, strnlen(value.UTF8String, 15));
}

static NSString *render(NSString *format, NSArray<ANEGraphValue *> *values) {
    NSMutableString *text = [format mutableCopy];
    for (NSUInteger i = 0; i < values.count; ++i) {
        NSString *key = [NSString stringWithFormat:@"${%lu}", (unsigned long)i];
        [text replaceOccurrencesOfString:key withString:values[i].name
                                 options:0 range:NSMakeRange(0, text.length)];
    }
    return text;
}

static NSData *tensorDescriptor(NSDictionary *record,
                               NSArray<ANEGraphValue *> *values) {
    ANEGraphValue *value = values[[record[@"value"] unsignedIntegerValue]];
    NSString *shortName = value.name;
    NSString *symbol = [record[@"output"] boolValue]
        ? [shortName stringByAppendingString:@"@output"] : shortName;
    NSData *symbolBytes = [symbol dataUsingEncoding:NSUTF8StringEncoding];
    NSData *shortBytes = [shortName dataUsingEncoding:NSUTF8StringEncoding];
    NSUInteger shortOffset = 0xd3d + symbolBytes.length + 1;
    NSUInteger length = MAX((NSUInteger)0xd48, alignTo(shortOffset + shortBytes.length + 1, 8));
    NSMutableData *data = [NSMutableData dataWithLength:length];
    word(data, 0, LC_THREAD); word(data, 4, (uint32_t)length);
    word(data, 8, 3); word(data, 0xc, 0x34a); word(data, 0x10, 3);
    word(data, 0x14, [record[@"index"] unsignedIntValue]);
    word(data, 0x18, 0xd38); word(data, 0x20, 0xd3d); word(data, 0x24, 5);
    for (NSUInteger i = 0; i < 4; ++i)
        word(data, 0x28 + i * 4, value.type.shape[i].unsignedIntValue);
    word(data, 0x38, 1);
    NSUInteger row = alignTo(value.type.shape[3].unsignedIntegerValue * 2, 64);
    NSUInteger plane = value.type.shape[2].unsignedIntegerValue * row;
    NSUInteger batch = value.type.shape[1].unsignedIntegerValue * plane;
    NSUInteger total = value.type.shape[0].unsignedIntegerValue * batch;
    quad(data, 0x50, batch); quad(data, 0x58, plane); quad(data, 0x60, row);
    quad(data, 0x68, 2); quad(data, 0x70, total); word(data, 0x78, 1);
    word(data, 0x7c, (uint32_t)shortOffset); quad(data, 0x80, total);
    string(data, 0xd38, @"main"); string(data, 0xd3d, symbol);
    string(data, shortOffset, shortName);
    return data;
}

@implementation H13GObjectWriter
+ (NSData *)buildObjectWithLayout:(NSDictionary *)layout
                           values:(NSArray<ANEGraphValue *> *)values
                     objectLabels:(NSDictionary<NSString *, NSString *> *)objectLabels
                    symbolAliases:(NSDictionary<NSString *, NSString *> *)symbolAliases
                   taskDescriptor:(NSData *)tasks
                    firstTaskSize:(NSUInteger)firstTaskSize
                        taskCount:(NSUInteger)taskCount
                   constantRegion:(NSData *)constants
                coefficientRegion:(NSData *)coefficients
                            error:(NSError **)error {
    NSUInteger constantOffset = alignTo(tasks.length, 64);
    NSUInteger textAllocation = alignTo(constantOffset + constants.length, 0x4000);
    NSUInteger coefficientAllocation = alignTo(coefficients.length, 0x4000);
    NSMutableArray<NSData *> *commands = [NSMutableArray array];
    NSUInteger textSectionIndex = 0, sectionCount = 0;
    for (NSDictionary *record in layout[@"commands"]) {
        uint32_t cmd = [record[@"cmd"] unsignedIntValue];
        NSMutableData *data = nil;
        if (cmd == LC_SEGMENT_64) {
            NSArray *sections = record[@"sections"];
            data = [NSMutableData dataWithLength:72 + 80 * sections.count];
            struct segment_command_64 segment = {};
            segment.cmd = cmd; segment.cmdsize = (uint32_t)data.length;
            name(segment.segname, record[@"name"]);
            segment.vmaddr = [record[@"address"] unsignedLongLongValue];
            segment.vmsize = [record[@"allocation"] unsignedLongLongValue];
            segment.maxprot = [record[@"protection"][0] intValue];
            segment.initprot = [record[@"protection"][1] intValue];
            segment.nsects = (uint32_t)sections.count;
            segment.flags = [record[@"flags"] unsignedIntValue];
            if ([record[@"name"] isEqualToString:@"__TEXT"])
                segment.vmsize = segment.filesize = textAllocation;
            if ([record[@"name"] isEqualToString:@"__KERN_0"])
                segment.vmsize = segment.filesize = coefficientAllocation;
            [data replaceBytesInRange:NSMakeRange(0, 72) withBytes:&segment];
            for (NSUInteger i = 0; i < sections.count; ++i) {
                NSDictionary *s = sections[i];
                struct section_64 section = {};
                name(section.sectname, s[@"name"]); name(section.segname, s[@"segment"]);
                section.addr = [s[@"address"] unsignedLongLongValue];
                section.size = [s[@"size"] unsignedLongLongValue];
                section.align = [s[@"alignment"] unsignedIntValue];
                section.flags = [s[@"flags"] unsignedIntValue];
                section.reserved1 = [s[@"reserved"][0] unsignedIntValue];
                section.reserved2 = [s[@"reserved"][1] unsignedIntValue];
                section.reserved3 = [s[@"reserved"][2] unsignedIntValue];
                ++sectionCount;
                if ([s[@"segment"] isEqualToString:@"__TEXT"]) {
                    if ([s[@"name"] isEqualToString:@"__text"]) {
                        section.size = tasks.length;
                        section.nreloc = (uint32_t)[layout[@"relocations"] count];
                        textSectionIndex = sectionCount;
                    } else section.size = constants.length;
                }
                if ([s[@"segment"] isEqualToString:@"__KERN_0"])
                    section.size = coefficients.length;
                [data replaceBytesInRange:NSMakeRange(72 + 80 * i, 80) withBytes:&section];
            }
        } else if (cmd == 0x40) {
            data = [NSMutableData dataWithLength:32];
            word(data, 0, cmd); word(data, 4, 32); quad(data, 8, 24);
            quad(data, 16, [record[@"address"] unsignedLongLongValue]);
            NSData *bytes = [render(record[@"name"], values) dataUsingEncoding:NSUTF8StringEncoding];
            [data replaceBytesInRange:NSMakeRange(24, MIN((NSUInteger)8, bytes.length))
                           withBytes:bytes.bytes];
        } else if (cmd == LC_THREAD) {
            if ([record[@"flavor"] unsignedIntegerValue] == 3) {
                data = [tensorDescriptor(record, values) mutableCopy];
            } else {
                data = [NSMutableData dataWithLength:0x890];
                word(data, 0, cmd); word(data, 4, 0x890);
                word(data, 8, 1); word(data, 12, 0x21e);
                NSArray *bars = layout[@"bars"];
                for (NSUInteger i = 0; i < 32; ++i)
                    quad(data, 16 + i * 8, [bars[i] unsignedLongLongValue]);
                quad(data, 0x810, [bars[0] unsignedLongLongValue]);
                word(data, 0x818, (uint32_t)(firstTaskSize / 4 - 1));
                word(data, 0x81c, (uint32_t)taskCount);
                for (NSArray *field in record[@"fields"])
                    word(data, [field[0] unsignedIntegerValue], [field[1] unsignedIntValue]);
                string(data, 0x888, @"main");
            }
        } else if (cmd == 8) {
            NSMutableString *text = [record[@"text"] mutableCopy];
            for (NSString *key in objectLabels)
                [text replaceOccurrencesOfString:[NSString stringWithFormat:@"${%@}", key]
                    withString:objectLabels[key] options:0 range:NSMakeRange(0, text.length)];
            NSData *bytes = [text dataUsingEncoding:NSUTF8StringEncoding];
            data = [NSMutableData dataWithLength:alignTo(8 + bytes.length + 1, 8)];
            word(data, 0, cmd); word(data, 4, (uint32_t)data.length);
            string(data, 8, text);
        } else if (cmd == 0x31) {
            NSData *payload = nil;
            if (record[@"text"]) payload = [record[@"text"] dataUsingEncoding:NSUTF8StringEncoding];
            else {
                uint32_t version = [record[@"version"] unsignedIntValue];
                payload = [NSData dataWithBytes:&version length:4];
            }
            data = [NSMutableData dataWithLength:alignTo(40 + payload.length, 8)];
            word(data, 0, cmd); word(data, 4, (uint32_t)data.length);
            string(data, 8, record[@"owner"]); quad(data, 24, 40);
            quad(data, 32, payload.length);
            [data replaceBytesInRange:NSMakeRange(40, payload.length) withBytes:payload.bytes];
        } else if (cmd == LC_SYMTAB) {
            data = [NSMutableData dataWithLength:24];
            word(data, 0, cmd); word(data, 4, 24);
        }
        if (!data) return nil;
        [commands addObject:data];
    }
    NSMutableData *symbols = [NSMutableData data];
    NSMutableData *strings = [NSMutableData dataWithLength:1];
    for (NSDictionary *record in layout[@"symbols"]) {
        struct nlist_64 entry = {};
        entry.n_un.n_strx = (uint32_t)strings.length;
        entry.n_type = [record[@"kind"] unsignedCharValue];
        entry.n_sect = [record[@"section"] unsignedCharValue];
        entry.n_desc = [record[@"description"] unsignedShortValue];
        entry.n_value = [record[@"value"] unsignedLongLongValue];
        NSString *symbolName = render(record[@"name"], values);
        for (NSString *prefix in symbolAliases)
            if ([symbolName hasPrefix:prefix]) {
                symbolName = [symbolAliases[prefix] stringByAppendingString:
                    [symbolName substringFromIndex:prefix.length]];
                break;
            }
        [symbols appendBytes:&entry length:sizeof(entry)];
        [strings appendData:[symbolName dataUsingEncoding:NSUTF8StringEncoding]];
        uint8_t zero = 0;
        [strings appendBytes:&zero length:1];
    }
    NSUInteger loadBytes = 0;
    for (NSData *command in commands) loadBytes += command.length;
    NSUInteger symbolOffset = 32 + loadBytes;
    NSUInteger stringOffset = symbolOffset + symbols.length;
    NSUInteger relocationOffset = alignTo(stringOffset + strings.length, 8);
    NSUInteger relocationBytes = [layout[@"relocations"] count] * 8;
    NSUInteger textOffset = alignTo(relocationOffset + relocationBytes, 0x4000);
    NSUInteger coefficientOffset = textOffset + textAllocation;
    NSMutableData *image = [NSMutableData dataWithLength:coefficientOffset + coefficientAllocation];
    struct mach_header_64 header = {};
    header.magic = 0xBEEFFACE; header.cputype = 128; header.cpusubtype = 4;
    header.filetype = MH_EXECUTE; header.ncmds = (uint32_t)commands.count;
    header.sizeofcmds = (uint32_t)loadBytes; header.flags = 0x200000;
    [image replaceBytesInRange:NSMakeRange(0, 32) withBytes:&header];
    NSUInteger commandOffset = 32, currentSection = 0;
    for (NSData *command in commands) {
        NSMutableData *data = [command mutableCopy];
        uint32_t cmd = 0;
        memcpy(&cmd, data.bytes, 4);
        if (cmd == LC_SEGMENT_64) {
            struct segment_command_64 *segment = (struct segment_command_64 *)data.mutableBytes;
            BOOL text = strcmp(segment->segname, "__TEXT") == 0;
            BOOL coefficientsSegment = strcmp(segment->segname, "__KERN_0") == 0;
            if (text) segment->fileoff = textOffset;
            if (coefficientsSegment) segment->fileoff = coefficientOffset;
            auto *sections = (struct section_64 *)((uint8_t *)data.mutableBytes + 72);
            for (NSUInteger i = 0; i < segment->nsects; ++i) {
                ++currentSection;
                if (text) {
                    sections[i].offset = (uint32_t)(textOffset + (i == 0 ? 0 : constantOffset));
                    if (currentSection == textSectionIndex)
                        sections[i].reloff = (uint32_t)relocationOffset;
                }
                if (coefficientsSegment) sections[i].offset = (uint32_t)coefficientOffset;
            }
        } else if (cmd == LC_SYMTAB) {
            word(data, 8, (uint32_t)symbolOffset); word(data, 12, (uint32_t)(symbols.length / 16));
            word(data, 16, (uint32_t)stringOffset); word(data, 20, (uint32_t)strings.length);
        }
        [image replaceBytesInRange:NSMakeRange(commandOffset, data.length) withBytes:data.bytes];
        commandOffset += data.length;
    }
    [image replaceBytesInRange:NSMakeRange(symbolOffset, symbols.length) withBytes:symbols.bytes];
    [image replaceBytesInRange:NSMakeRange(stringOffset, strings.length) withBytes:strings.bytes];
    for (NSArray *relocation in layout[@"relocations"]) {
        word(image, relocationOffset, [relocation[0] unsignedIntValue]);
        word(image, relocationOffset + 4, [relocation[1] unsignedIntValue]);
        relocationOffset += 8;
    }
    [image replaceBytesInRange:NSMakeRange(textOffset, tasks.length) withBytes:tasks.bytes];
    [image replaceBytesInRange:NSMakeRange(textOffset + constantOffset, constants.length) withBytes:constants.bytes];
    [image replaceBytesInRange:NSMakeRange(coefficientOffset, coefficients.length) withBytes:coefficients.bytes];
    return [HWXImage imageWithData:image error:error] ? image : nil;
}
@end
