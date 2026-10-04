# MIL-to-HWX compiler

This repository contains a research compiler for the H16G Apple Neural Engine
in the M4, with a measured H13G/base-M1 backend for the captured GPT-2 graphs.
It reads textual MIL and ordinary model weight blobs and writes new HWX
objects without calling Apple's compiler.

The project is a canary for the compiler pipeline recovered in *Inside the M4
Apple Neural Engine*, Part 4b. It shows which parts of that pipeline are
understood well enough to reproduce in code and verify on hardware.

## Asahi Linux compiler

The complete H13G/H16G compiler and its software tests run natively on Linux
arm64. Apple Foundation is replaced by GNUstep Base with libobjc2; Linux hashes
use OpenSSL, HWX uses portable Mach-O format records, and one-time target-table
initialization uses C++ `std::call_once`.

Install build prerequisites on Fedora Asahi Remix:

```sh
sudo dnf install clang make git cmake python3 python3-pip \
  pkgconf-pkg-config libffi-devel libicu-devel libxml2-devel \
  libcurl-devel openssl-devel
bash tools/bootstrap-linux.sh
make test -j4
make build/mil-hwxc
```

The bootstrap installs pinned GNUstep/libobjc2 releases under `.deps/gnustep`
without root access. The compiler has an embedded library search path, so no
shell setup or `LD_LIBRARY_PATH` is needed to run it. A custom modern GNUstep
installation can be selected with `GNUSTEP_PREFIX` and `GNUSTEP_CONFIG`.
GNUstep packages built against GCC's legacy Objective-C runtime do not support
the ARC and weak references used here. `make clean` preserves `.deps/`.

The Asahi verification compiled all 49 captured GPT-2 graphs from MIL and the
cached HF checkpoint. All 147 program/constant/coefficient payloads matched
the reference hashes in the existing Asahi GPT-2 replay package, covering
1,574 tasks. See the [verification receipt](docs/VERIFICATION.md#asahi-linux-compiler-verification-2026-10-04)
for the comparison method and recorded results.

Repeat that comparison without needing the original macOS dump:

```sh
python3 -m venv --system-site-packages .deps/verify-env
.deps/verify-env/bin/pip install numpy safetensors
make verify-h13g-replay PYTHON=.deps/verify-env/bin/python \
  H13G_PACKAGE="$HOME/allbilly_ane/gpt2"
```

This requires the existing replay package and its matching cached GPT-2
checkpoint. The verifier creates temporary raw model blobs, removes them
after compilation, and writes compiled bundles and a validation report under
`build/`. The compiler does not read replay instructions or packed weights.

The IOSurface/private Apple runtime and its contract test remain macOS-only.
Linux `make test` runs the compiler tests and explicitly reports that runtime
test as unavailable. This port supplies compilation for the separately
verified Asahi inference path; it does not replace the Linux ANE driver.

## Local M1 / H13G GPT-2 compiler

`mil-hwxc --target H13G` compiles all 49 graphs captured in
`~/Desktop/GPT2-ANE-Dump`. The native backend emits task headers, register
packets, links, alignment, coefficients, object commands, symbols and
relocations. It does not read a reference instruction stream or HWX template
when compiling. The compiler input is MIL plus the raw FP16 BLOBFILE tensors.

Build and compile the complete captured graph set:

```sh
make build/mil-hwxc
python3 tools/compile_h13g_gpt2.py \
  --models ~/Desktop/GPT2-ANE-Dump/bundles \
  --output build/h13g-gpt2
```

Each kernel gets a normal compiler bundle containing `program-0.hwx` and a
binding manifest. The top-level manifest lists all kernel bundles and hashes.
You can compile one graph directly:

```sh
./build/mil-hwxc \
  --mil ~/Desktop/GPT2-ANE-Dump/bundles/decode_ffn_L6_loaded/model.mil \
  --model-root ~/Desktop/GPT2-ANE-Dump/bundles/decode_ffn_L6_loaded \
  --target H13G --output build/h13g-ffn
```

Run the full reference comparison:

```sh
make verify-h13g-gpt2
# For another reference directory:
make verify-h13g-gpt2 H13G_DUMP="$HOME/Desktop/GPT2-ANE-Dump"
```

`h13g-task-validation.json` records exact reconstruction of all 1,574 tasks.
`h13g-compiler-validation.json` records 49/49 complete HWX files matching the
originals and eleven independence/rejection checks. These include renamed
functions, SSA identifiers and weight files across all six instruction
variants, changed learned coefficients, and rejection of unsupported
constants, geometry and operand edges.

Whole-file equality includes historical debug metadata. The verifier passes
the original source/output labels through `--source-label` and
`--output-label`. Ordinary compilation records the current paths; its task,
constant and coefficient sections still match the captured payloads.
The target also contains compatibility aliases for Apple's opaque coefficient
symbol IDs, indexed by the SHA-256 of freshly packed coefficient data.
Aliases affect symbols only. New coefficients receive new content-derived
symbols. Source labels and symbol hashes never select instructions.

The supported H13G coverage is deliberately explicit: FP16 `[1,768,1,32]`
LayerNorm; LayerNorm plus three 768-channel projections; LayerNorm, 768-to-3072
expansion, tanh GELU, projection and residual; and masked attention with 12
heads, sequence 32 and head dimension 64. Structural contracts check every
operation, type, literal and operand edge, in the captured declaration order.
Function and SSA names and weight-file paths are not legality keys. Other
graphs receive `h13g.legalize.unsupported-graph`.

The backend uses five measured fused graph schedules and decoded register
fields in `H13GTargetData.inc`. The FFN engine-affine schedule computes its
bias scale from the smallest nonzero FP32 `beta/gamma`, selecting a power of
two that reaches `2^-16`. This reproduces the layer-6 scale 32 and layer-10
scale 2 without selecting a schedule by layer number. These observations are
validated for the captured checkpoint and are not general H13G coverage.
`tools/derive_h13g_target.py` can reproduce the target catalog from the original
captures after `make build/h13g-inspect`; derivation is a developer step.

Hardware execution of these compiler outputs has not been rerun. The existing
runtime in this repository continues to accept H16G bundles. The separate
`~/ane/gpt2` package supplies the GPT-2 inference runner and H13G replay path;
this change supplies independent compilation of its captured ANE graph set.

### Standalone H13G weight packing

This checkout adds a standalone C++ packer for the captured base-M1 GPT-2
layouts in `plugins/H13G/Encoding/H13GConstantPacker.{h,cpp}`. It generates
coefficient bytes from ordinary little-endian FP16 matrices and biases. It
does not call Apple's compiler or read dumped coefficient bytes during packing.
The upstream compiler's H16G route remains available alongside the new H13G
compiler route. The packer can also be built independently of both routes.

Build and test the packer without Foundation or other macOS frameworks:

```sh
make test-h13g
./build/h13g-pack --help
```

The target uses C++17 and Clang/GCC `_Float16` support. Its unit tests pass on
macOS/base M1 and Linux arm64 (Fedora Asahi Remix).

Inputs are headerless FP16 data. Matrices have logical shape `[output,input]`.
Provide the tile schedule selected by the captured compiler:

```sh
# 768 input/output channels, a captured decode projection schedule.
./build/h13g-pack matrix W.fp16 bias.fp16 packed.bin 768 768 16,16,16

# LayerNorm affine coefficients, captured linear layout.
./build/h13g-pack affine gamma.fp16 beta.fp16 affine.bin linear 1

# The captured layer-6 FFN uses engine pairs and bias scaling by 32.
./build/h13g-pack affine gamma.fp16 beta.fp16 affine.bin engine_pairs 32
```

There are 16 engines. Each scheduled stripe stores its bias values followed by
the matrix transposed to `[input,stripe-output]`. Each engine block ends with
zero padding to 64 bytes. The captured GPT-2 schedules are:

| Operation | Stripe widths per engine |
| --- | --- |
| Decode Q/K/V and projection/down matrices | `16,16,16` |
| FFN expansion | `32,32,32,32,32,32` |
| Prefill Q | `32,16` |
| Other prefill attention matrices | `16,16,16` |

Affine packing computes `beta/gamma*scale` in FP32 from already FP16-rounded
inputs, then rounds to FP16. Linear layout stores `[ratio,gamma]`. Engine-pair
layout stores `(gamma,ratio)` pairs for channels `engine + 16*group`; the
captured FFN layers 6 and 10 use scales 32 and 2. Shapes alone do not select
these schedules or affine strategies. This is a verified GPT-2 layout packer,
not a general ANE compiler.

To reproduce the reference comparison, install the Python dependencies from
`~/ane/gpt2/requirements.txt`, keep the original dump available, and run:

```sh
python3 tests/verify_h13g_gpt2.py \
  --package ~/ane/gpt2 \
  --dump ~/Desktop/GPT2-ANE-Dump
```

The verifier reads the existing HF cache and captured recipes. Every learned
byte comes from the native C++ packer. It compares 132 matrix/bias blocks,
94 affine blocks, and all 147 complete payloads across 49 original HWX files.
All matched exactly in `h13g-validation.json`. It also checks affine rounding
against NumPy over 63,472 finite nonzero gamma bit patterns in each of six
layout/scale combinations, plus malformed CLI inputs. Temporary weights are
deleted; the checkout does not retain regenerated model weights.

`results.json`, `software-tests.log`, `compare.py`, and `pack_probe.mm` retain
the earlier comparison with the upstream H16G packer. Run
`make build/gpt2-pack-probe` before repeating `python3 compare.py --upstream "$PWD"`.
H13G packing is also included in `make test` on both macOS and Linux. This
repository's macOS runtime does not execute models on Asahi.

## Quickstart

### macOS requirements

- An Apple silicon Mac. Hardware results in this repository were measured on an
  M4 (`Mac16,10`).
- macOS and the Xcode Command Line Tools, which provide `clang++`, `make`, the
  Foundation SDK, and the IOSurface SDK.
- macOS 26.3 build `25D125` for the recorded H16G hardware results. Other
  releases may use different private interfaces or descriptor layouts.
- Administrator access for hardware tests. Compilation and software tests do
  not require `sudo`.

The macOS build has no third-party package dependencies. Linux uses the open
dependencies listed above.

Install the command-line tools if needed:

```sh
xcode-select --install
```

Clone the repository and run the software test suite:

```sh
git clone https://github.com/allbilly/mil-hwx-compiler.git
cd mil-hwx-compiler
make test -j4
```

Build only the compiler:

```sh
make build/mil-hwxc -j4
```

Compile the included Conv1x1 and ReLU fixture:

```sh
./build/mil-hwxc \
  --mil tests/fixtures/conv_relu.mil \
  --model-root tests/models/conv_relu \
  --target H16G \
  --output build/conv-bundle
```

The output directory contains one or more `program-N.hwx` files and a
`manifest.json` file. The manifest records dispatch order, tensor bindings,
physical strides, and shared intermediate surfaces.

To compile and run the Chunked DeltaNet block on an M4:

```sh
sudo -v
bash tests/run_chunked_deltanet_hardware.sh
```

The script builds the compiler and runner, emits 58 HWX programs, provisions
them in the macOS `aned` cache, runs two numerical cases, and prints a warm
latency measurement. The script exits with a failure if compilation,
provisioning, execution, or either output comparison fails.

To compile and run the FP16 attention graph:

```sh
sudo -v
bash tests/run_online_reduction_hardware.sh
```

This emits three programs and checks all 16,384 output elements against the
CPU reference. To compare it with the output of Apple's compiler on the same
MIL, inputs, and synchronous runtime path:

```sh
ANE_BENCHMARK_WARMUP=50 \
ANE_BENCHMARK_ITERATIONS=5000 \
ANE_BENCHMARK_BATCHES=5 \
bash tests/run_fa2_ab_hardware.sh
```

## Scope

The repository provides:

- A compiler path from textual MIL to fresh H16G HWX objects.
- Hardware tests with independent CPU references for every supported family.
- Encoders for the decoded HWX container and Task Descriptor fields.
- A small runtime that loads provisioned objects and submits them through
  `_ANEClient`.

Coverage is limited to the operation and shape rows listed below. The target
tables were measured on one M4 and one macOS build. This compiler is not a Core
ML or MLX replacement, and the private runtime is unsuitable for App Store
software.

## Compiler pipeline

```text
MIL source
  -> lexer and parser
  -> typed SSA graph
  -> normalization and decomposition
  -> structural fusion
  -> H16G legality and numeric-mode selection
  -> tiling, liveness, SRAM, and DMA planning
  -> H16G task encoding and program composition
  -> HWX object and binding manifest
```

The production path does not load an Apple-compiled HWX file, choose code from
a fixture name, or patch an existing container. `HWXObjectWriter` (H16G) and
`H13GObjectWriter` create objects from compiler data structures. The compiler reparses the completed
object before accepting it.

Multi-operation graphs are partitioned according to target capability tables.
Compatible adjacent tasks can share one HWX program. Other tasks remain
separate programs connected through manifest-managed IOSurfaces. Unsupported
operations, shapes, data types, and axes fail compilation. A transition that
cannot be composed stays on the standalone program path.

Composition has two forms. Simple elementwise operations can be folded into a
field of an adjacent task. Longer chains use operation-specific SRAM input and
output forms so an intermediate remains inside one program. The planner selects
these forms from operation, shape, data type, bridge state, and value lifetime.
It does not select them from a model or function name.

## Coverage and measured results

### Supported operation families

| Family | Measured H16G coverage |
|---|---|
| Conv1x1 | 16 FP16 channel and spatial geometries, with optional ReLU |
| Regular convolution | C64/C128, S32/S64, K3/K5 |
| Depthwise convolution | C64/C128/C256/C512, S64, K3 |
| Matmul | Square N128/N256/N512 and tiled multiples of 128 through N4096 |
| Binary ALU | Add at N128 through N2048; multiply at N128/N512; max/min at N512 |
| Unary and LUT | ReLU, sigmoid, tanh, GELU, SiLU, exp, log, sqrt, rsqrt, reciprocal |
| Reduction | Sum, mean, and max over measured channel, height, and width axes |
| Layout | S2D and D2S B4/B8, including 64-byte physical row padding |
| Fused layout | S2D, Conv1x1, and D2S at natural C8/C16/C24/C32 |
| W8A8 | Four-layer C64/S64 Conv1x1 chain with packed middle blocks |
| Attention | H4/S64/D64 decomposition; unmasked FP16 S128/D128 forward graph in three programs |
| State updates | Four-step N128 affine scan; FP16 Chunked DeltaNet block at C128/D128 |

The Chunked DeltaNet fixture uses ordinary matmul, add, multiply, and exp
operations. The caller supplies normalized Q and K tensors, the transposed K
layout, and the fixed triangular matrices. No DeltaNet operation name reaches
the planner or H16G encoders.

### M4 multi-operation results

| Graph | HWX programs | CPU-reference result | Research latency | Apple compiler | ANE power capture |
|---|---:|---|---:|---:|---|
| FP16 attention, S128/D128 | 3 | 16,384/16,384 elements passed; maximum error `0` | `369.458 us` | `127.208 us` | 61/140 active; 601.9 mW active average; 681 mW peak |
| Matmul, reshape, GELU, N256 | 1 | Maximum error `0.00610352` | `66.94 us` | `76.60 us` | 8/60 active; 36-134 mW; GPU 0 mW |
| Four-step FP16 affine scan, N128 | 8 | Every stage within 1 FP16 ULP; final output within 3 ULP | Not measured | `InvalidMILProgram` | 60/60 active; 37-152 mW |
| Chunked DeltaNet block, C128/D128 | 58 | Output relative L2 `0.004214`; final-state relative L2 `0.004443`; maximum error `3.69e-05` | `22728.792 us` | `InvalidMILProgram` | 73/80 active; 19-403 mW; 86.34 mW average; GPU 0 mW |

The attention graph is split at its two external-surface boundaries. Program 0
runs matmul and scale. Program 1 runs reduce-max, subtraction, exponentiation,
reduce-sum, reciprocal, and multiply while keeping its intermediate values in
SRAM. Program 2 runs the final matmul.

The attention A/B used 50 warmups and five alternating batches of 5,000
evaluations per compiler, for 25,000 measured samples each. Both implementations
used the same MIL, inputs, IOSurfaces, QoS, and synchronous completion boundary.
The research compiler took 2.986 times the Apple compiler median. A separate
research-only profile measured median submission times of 114.458, 110.583,
and 113.146 microseconds for the three programs. The median complete chain was
369.417 microseconds without profiling instrumentation.

The matmul-GELU values are medians from two earlier runs. Each run used 20
warmups and five alternating batches of 2,000 evaluations per compiler.

The Chunked DeltaNet result used 10 warmups and five batches of 50 evaluations.
Its 58-program schedule is a correctness result and currently carries
substantial dispatch and intermediate-surface cost. Apple's compiler rejected
the same 14-input MIL program, so an equivalent latency comparison is not
available.

Powermetrics sampled the system ANE and GPU rails every 100 ms while the
hardware tests ran. The attention power capture ran only research-generated
programs. These values confirm ANE activity during the tests. They are
system-wide estimates and do not measure per-process energy.

The current attention result is recorded in a
[benchmark and profile receipt](docs/evidence/fa2-three-program-m4-2026-09-03.txt)
and a [powermetrics screenshot](docs/evidence/fa2-three-program-powermetrics.png).
Earlier compiler A/B results are in the
[original baseline](docs/evidence/compiler-ab-m4-2026-09-02.txt) and the
[first optimization receipt](docs/evidence/compiler-ab-m4-2026-09-02-optimized.txt).
The current Chunked DeltaNet run is recorded in
[docs/evidence/chunked-deltanet-m4-2026-09-03.txt](docs/evidence/chunked-deltanet-m4-2026-09-03.txt).

## Hardware tests

Each hardware script compiles its MIL fixture, provisions the emitted object,
runs it on the M4 ANE, and checks the output. Useful entry points include:

```sh
bash tests/run_m4_hardware.sh
bash tests/run_matmul_hardware.sh
bash tests/run_unary_hardware.sh
bash tests/run_reduce_hardware.sh
bash tests/run_layout_hardware.sh
bash tests/run_online_reduction_hardware.sh
bash tests/run_online_reduction_fallback_hardware.sh
bash tests/run_affine_scan_hardware.sh
bash tests/run_matmul_gelu_hardware.sh
bash tests/run_chunked_deltanet_hardware.sh
bash tests/run_compiler_ab_hardware.sh
```

Stock macOS loads these objects from
`/Library/Caches/com.apple.aned/<build>/InMemoryModelCache/<executable>/`.
The scripts use `sudo -n` to create that cache entry and install the generated
HWX file. Run `sudo -v` first if the current shell has no valid credential
timestamp. The software test suite writes only to `build/`.

## Runtime boundary

`ANEProvisionedRuntime` creates IOSurfaces from the binding manifest and calls
the private `AppleNeuralEngine.framework` runtime. The manifest keeps logical
tensor sizes separate from physical row, plane, batch, and allocation sizes.
This is required for narrow tensors whose rows are padded to 64 bytes.

The compiler does not provide a kernel-driver path or bypass the normal `aned`
cache requirement. Private interfaces and accepted HWX layouts can change with
macOS releases.

## Provenance

The HWX container, Task Descriptor fields, and compiler stages were recovered
by compiling author-written MIL with Apple's compiler, comparing generated
objects, tracing compiler execution, and testing edited objects on hardware.
No Apple source code was available or used, and no Apple binary code is
distributed here.

Some `plugins/H16G/Encoding/*EncoderData.inc` files contain measured Task
Descriptor words for operation and geometry rows whose field grammar is still
incomplete. They are indexed target measurements rather than copied program
containers. Tests store hashes and decoded field values, not Apple-generated
HWX files.

See [DISCLAIMER.md](DISCLAIMER.md) for the full scope and private-API notes.

## License

The original code and documentation in this repository are available under the
[MIT License](LICENSE).
