# Design: Add -O3 and -flto=thin to ORT build

Date: 2026-03-27

## Goal

Optimize `libonnxruntime.so` by upgrading the compiler optimization level from
`-O2` (CMake `Release` default) to `-O3`, and enabling ThinLTO (`-flto=thin`)
for cross-translation-unit inlining and dead-code elimination at link time.

## Scope

- **Applies unconditionally** to all `cpu_tuning` values (current: `neoverse-n1`; future values too).
- Affects all C and C++ translation units compiled by Clang. MLAS assembly
  (`.S`) kernels override their own per-file `-march` via
  `set_source_files_properties` and are not affected by global `CMAKE_C_FLAGS` /
  `CMAKE_CXX_FLAGS`.

## Changes — `scripts/build_target.sh`

### 1. Base `CMAKE_EXTRA_DEFINES` block (unconditional)

Add two new entries for compiler flags and extend the three linker flag entries:

```bash
CMAKE_EXTRA_DEFINES=(
    "onnxruntime_BUILD_SHARED_LIB=ON"
    "CMAKE_C_COMPILER=clang"
    "CMAKE_CXX_COMPILER=clang++"
    "CMAKE_C_FLAGS=-O3 -flto=thin"
    "CMAKE_CXX_FLAGS=-O3 -flto=thin"
    "CMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld -flto=thin"
    "CMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld -flto=thin"
    "CMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld -flto=thin"
    "CMAKE_C_COMPILER_LAUNCHER=ccache"
    "CMAKE_CXX_COMPILER_LAUNCHER=ccache"
)
```

`-flto=thin` must appear in both compile and link flags; lld has native
ThinLTO support so no additional linker plugin is required.

### 2. neoverse-n1 branch (CPU-tuning-specific flags)

Appends its `-march`/`-mtune` on top of the base flags, as today. The
`CMAKE_CXX_FLAGS` / `CMAKE_C_FLAGS` entries in this branch must include the
base `-O3 -flto=thin` so that the append does not lose them:

```bash
"CMAKE_CXX_FLAGS=-O3 -flto=thin -march=armv8.2-a+dotprod+fp16+rcpc -mtune=neoverse-n1"
"CMAKE_C_FLAGS=-O3 -flto=thin -march=armv8.2-a+dotprod+fp16+rcpc -mtune=neoverse-n1"
```

(The base-block entries are overwritten by CMake when the same variable is set
twice via `--cmake_extra_defines`; the neoverse-n1 entries are the ones that
take effect for that tuning target, so they must carry the full flag set.)

## Why `-O3` and `-flto=thin`

- **`-O3`** enables additional loop unrolling, auto-vectorization, and
  inter-procedural optimizations that `-O2` skips. For a shared library used
  in a tight inference loop this is directly beneficial.
- **`-flto=thin`** (ThinLTO) performs lightweight whole-program analysis at
  link time: cross-TU inlining of hot call sites, dead-symbol stripping, and
  devirtualization. "Thin" means each TU gets a summary bitcode file; the
  linker merges only the relevant summaries rather than the full IR, so link
  time is far lower than full LTO while capturing most of the benefit.
- lld is already required (`-fuse-ld=lld`), so no toolchain change is needed.
- ccache is already in use; ThinLTO bitcode objects are cached normally.

## What is NOT changed

- `--config Release` on `build.py` — kept as-is.
- MLAS assembly kernel per-file flags — unaffected.
- `--no_sve`, `--disable_rtti`, `--disable_ml_ops` — unchanged.
- `builds/release.yaml`, `emit_matrix.py`, `validate_manifest.py` — no changes needed.
- Tests — no new tests required; the change is a build-flag tweak with no
  observable API or behavioral change to test.

## Risk

Low. Both flags are standard Clang + lld features available in any recent
toolchain. The Lambda container (`amazonlinux:2023`) ships Clang ≥ 15 and
lld ≥ 15, both of which fully support ThinLTO on AArch64. The only downside
is slightly longer build time (ThinLTO link phase), acceptable for a CI build.
