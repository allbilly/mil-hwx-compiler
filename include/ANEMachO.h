#pragma once

// HWX uses the Mach-O wire layout even when generated on Linux. These are
// fixed-width format records, not interfaces to the host's executable loader.
#ifdef __APPLE__
#include <mach-o/loader.h>
#include <mach-o/nlist.h>
#else
#include <stdint.h>

enum {
    MH_EXECUTE = 2,
    LC_SYMTAB = 2,
    LC_THREAD = 4,
    LC_SEGMENT_64 = 0x19,
    N_EXT = 1,
    N_TYPE = 0x0e,
    N_SECT = 0x0e,
};

struct mach_header_64 {
    uint32_t magic;
    int32_t cputype, cpusubtype;
    uint32_t filetype, ncmds, sizeofcmds, flags, reserved;
};
struct load_command { uint32_t cmd, cmdsize; };
struct segment_command_64 {
    uint32_t cmd, cmdsize;
    char segname[16];
    uint64_t vmaddr, vmsize, fileoff, filesize;
    int32_t maxprot, initprot;
    uint32_t nsects, flags;
};
struct section_64 {
    char sectname[16], segname[16];
    uint64_t addr, size;
    uint32_t offset, align, reloff, nreloc, flags;
    uint32_t reserved1, reserved2, reserved3;
};
struct symtab_command {
    uint32_t cmd, cmdsize, symoff, nsyms, stroff, strsize;
};
struct nlist_64 {
    union { uint32_t n_strx; } n_un;
    uint8_t n_type, n_sect;
    uint16_t n_desc;
    uint64_t n_value;
};
#endif

static_assert(sizeof(mach_header_64) == 32, "HWX header layout");
static_assert(sizeof(load_command) == 8, "HWX load command layout");
static_assert(sizeof(segment_command_64) == 72, "HWX segment layout");
static_assert(sizeof(section_64) == 80, "HWX section layout");
static_assert(sizeof(symtab_command) == 24, "HWX symbol table layout");
static_assert(sizeof(nlist_64) == 16, "HWX symbol layout");
