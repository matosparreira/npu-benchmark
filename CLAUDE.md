# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A minimal OpenVINO benchmark harness for testing AI inference acceleration on Linux, including Intel NPU (VPU) devices. Two entry points:

- **`test_npu.py`** — quick smoke test. Downloads a small MNIST ONNX model, compiles it for the single device in `target_device`, and runs a 500-iteration inference benchmark reporting average latency.
- **`benchmark_resnet.py`** — realistic CPU-vs-NPU comparison. Downloads ResNet-50, then benchmarks every available target in `DEVICES` two ways: synchronous batch-1 (latency) and `AsyncInferQueue` pipeline-full (throughput).

Both auto-download their ONNX model on first run (and reuse it after); the models are git-ignored.

Code comments and console output are in Portuguese.

## Environment

This is a NixOS environment. The native AI framework libraries are provided via `shell.nix`, and a Python virtualenv (`.venv`, Python 3.13) holds the Python packages (`openvino` 2026.2.0, `numpy` 2.4.6).

`shell.nix` exists specifically to set `LD_LIBRARY_PATH` so that the prebuilt OpenVINO wheels can find native deps (`stdenv.cc.cc.lib`, `zlib`, `glib`, `libglvnd`, `level-zero`). OpenVINO will fail to import outside the nix-shell because of missing shared libraries.

`level-zero` (the `libze_loader.so` loader) is required for OpenVINO to even *detect* the NPU. Without it on `LD_LIBRARY_PATH`, `core.available_devices` shows only `['CPU']` even though the hardware and `intel_vpu` kernel driver are present.

## Commands

```bash
# Enter the nix dev shell (sets LD_LIBRARY_PATH for native libs)
nix-shell

# Inside the shell, activate the venv and run a benchmark
source .venv/bin/activate
python test_npu.py          # quick MNIST smoke test (single device)
python benchmark_resnet.py  # ResNet-50 CPU-vs-NPU comparison
```

There are no tests, linters, or build steps — the two `*.py` scripts are the only entry points.

## Key detail

The benchmark target device is set via `target_device` in `test_npu.py:23` (currently `"NPU"`). `"CPU"`, `"NPU"`, and `"GPU"` are all valid targets (see NPU support below for what makes NPU work). The script prints `core.available_devices` at startup and exits early if the requested device isn't present — check that output to see what OpenVINO actually detected.

## NPU support (Meteor Lake / arch 3720, "Intel AI Boost")

The NPU **works end-to-end** (compiles and runs, ~0.29 ms/inference on the MNIST model), but only because of two fixes in `shell.nix`. Both are required:

1. **`level-zero`** — provides `libze_loader.so`, needed for OpenVINO to *detect* the NPU at all (see Environment section above).
2. **`intel-npu-compiler.nix`** — provides `libnpu_driver_compiler.so`, needed to *compile* models for the NPU.

### Why `intel-npu-compiler.nix` exists

The nixpkgs `intel-npu-driver` package (currently 1.28.0) ships only the runtime driver (`libze_intel_npu.so`) and **omits the Compiler-in-Driver library** (`libnpu_driver_compiler.so`) — building it from source would pull in the whole OpenVINO-based NPU compiler. Without that library, `core.compile_model(..., "NPU")` fails with:

```
L0 pfnCreate2 result: ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
```

and `NPU_COMPILER_VERSION` reads `0`. This is **not** an OpenVINO version issue — every cp313 wheel (2025.0.0 → 2026.2.0) fails identically without the compiler, and the legacy in-plugin compiler (`NPU_COMPILER_TYPE=MLIR`) no longer ships. Do **not** try to fix it by changing the `openvino` pin.

`intel-npu-compiler.nix` packages Intel's *prebuilt* `libnpu_driver_compiler.so`, extracted from the `intel-driver-compiler-npu` `.deb` inside the official `linux-npu-driver` release tarball, and `shell.nix` puts it on `LD_LIBRARY_PATH` so the runtime driver can `dlopen` it. When it is loaded, `NPU_COMPILER_VERSION` becomes non-zero (e.g. `458778`).

**Maintenance:** the version in `intel-npu-compiler.nix` (1.28.0) **must match the installed `intel-npu-driver` package**. If the system driver is bumped, update this file's `version` and `hash` to the matching `linux-npu-driver` release, or NPU compilation may break again.

### NPU vs CPU performance — depends entirely on the model

Measured here: on **MNIST** the CPU is ~3.6x *faster* (the model is too small — per-inference dispatch overhead dominates and the NPU never does real work). On **ResNet-50** the NPU is ~5.8x lower latency and ~6x higher throughput (real compute, where the NPU's dedicated MAC arrays win). So a CPU-faster result on a trivial model is expected, not a regression — use `benchmark_resnet.py` to see the NPU's actual advantage.
