# H13G GPT-2 reference compiler

The local backend compiles the 49 captured GPT-2 MIL graphs into fresh H13G
objects. The comparison target is `/Users/yeren/Desktop/GPT2-ANE-Dump`.
The requested `~/Desktop/ane/gpt2 dump` path was absent in this workspace;
the existing dump README and manifest identify the available capture.

## Compiler boundary

The frontend uses the repository's lexer, parser, graph importer and verifier.
H13G then checks a name-independent typed SSA contract against measured target
coverage, loads the declared FP16 tensors with `ANEBlobResolver`, selects
affine precision handling, packs coefficients, emits tasks, and writes the
complete object and binding manifest. The H16G lowering route is unchanged.

The current contracts cover four graph structures: channel LayerNorm, parallel
projections, residual feed-forward, and masked attention. Feed-forward has two
measured schedules; the engine-affine schedule parameterizes its compensation
shift, giving the two captured non-unit scales. Declaration order is part of
these contracts. Equivalent operation reorderings and additional geometries
require extending target coverage.

`H13GTargetData.inc` contains decoded register packets, scheduling controls,
object format measurements, fixed activation lookup tables, and graph
contracts. It contains no learned tensors or binary instruction/container
templates. This is a compiler with measured fused graph schedules; it does
not claim a general H13G instruction selector or scheduler.

## Task format observations

Each descriptor has a 40-byte header. Header word 6 bit 24 adds a 44th header
byte. Consecutive-register packets encode `(count-1)<<26 | address`, followed
by 1 through 64 32-bit values. Descriptors begin on 256-byte boundaries. Some
attention transitions reserve additional 256-byte slots. Newly emitted chains
compute sequential task IDs, NextPtr, NextSize and end-of-network markers.
The native decoder validates each completed chain before object writing.

The independent field comparison covers all 1,574 captured tasks, including
the 504-, 628- and 632-byte descriptor forms and attention extended headers.

## Affine precision observations

All affine inputs are already FP16 rounded. Folding computes `beta/gamma` in
FP32. Ordinary graphs use the linear `[ratio,gamma]` constant layout. For
feed-forward graphs, a nonzero folded bias below `2^-16` chooses engine-pair
coefficients with the smallest power-of-two bias scale that reaches that
threshold. The neural-engine AccBias compensation shift is `32-log2(scale)`.
The captured layer-6 and layer-10 coefficients select 32 and 2 respectively.
This policy is inferred from the capture and verified across all 24 captured
feed-forward graphs. It is not hardware validation for arbitrary new weights.

## Metadata fidelity

Instruction, constant and coefficient sections match with ordinary compilation
of the captured inputs. Complete byte equality additionally requires the
historical source/output command-line labels. Those labels are explicit CLI
inputs used only in the object metadata record. The verification tool reads
these two labels from the reference metadata; the production compiler reads
no reference file.

Apple's coefficient symbol naming algorithm has not been recovered. A target
metadata alias catalog maps hashes of independently packed coefficient groups
to their recorded symbol IDs. It contains hashes and names, never learned
coefficient bytes. No alias participates in graph matching, precision policy,
task selection or instruction encoding. Unknown groups use their current
content hash as a new symbol ID. Fixed Apple compiler/version records are
retained as measured compatibility metadata; the bundle manifest records the
actual local compilation passes.

## Validation and remaining scope

`tests/verify_h13g_compiler.py` compiles all 49 original model sources, then
compares every byte of every complete HWX against the capture. Its additional
checks compile isolated copies with renamed SSA identifiers, function and
weight filenames; repack changed learned beta and matrix coefficients; emit a
new content-derived symbol for an unknown matrix; and reject changed epsilon,
geometry and operand topology. Generated learned coefficients live only in
ignored build output or temporary validation directories.

The repository software suite also passes, including the existing H16G CLI,
staged compiler, runtime contract and production-route checks. Hardware
execution and token generation through these new compiler bundles have not
been rerun. `ANEProvisionedRuntime` is still H16G-only; the external GPT-2
package provides an H13G replay runner. These are distinct from the verified
MIL-to-HWX compilation result.
