# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A minimal OpenVINO benchmark harness for testing AI inference acceleration on Linux, including Intel NPU (VPU) devices. Three entry points:

- **`test_npu.py`** — quick smoke test. Downloads a small MNIST ONNX model, compiles it for the single device in `target_device`, and runs a 500-iteration inference benchmark reporting average latency.
- **`benchmark_resnet.py`** — realistic CPU/GPU/NPU comparison. Downloads ResNet-50, then benchmarks every available target in `DEVICES` (`CPU`, `GPU`, `NPU`) two ways: synchronous batch-1 (latency) and `AsyncInferQueue` pipeline-full (throughput).
- **`benchmark_vgg16.py`** — heavier CPU-vs-NPU comparison. Identical methodology to `benchmark_resnet.py` but downloads VGG-16 (~528MB, same `1×3×224×224` input, ~3.5× the compute), to stress the NPU with a denser convolutional workload.
- **`benchmark_vgg19.py`** — heaviest CPU-vs-NPU comparison. Identical methodology again but downloads VGG-19 (`vgg19-bn-7`, ~574MB, same `1×3×224×224` input, ~19.6 GFLOPs / ~1.26× VGG-16), the densest convolutional workload in the set.
- **`benchmark_power.py`** — energy comparison. Runs sustained ResNet-50 load on each available device and reads Intel RAPL to report package power (W), energy per inference (mJ), and efficiency (inf/s per W) — the performance-per-watt axis the other benchmarks don't measure. Needs RAPL read access (see "Measuring power" below).

The four download-based scripts auto-download their ONNX model on first run (and reuse it after); the models are git-ignored. `benchmark_power.py` reuses the ResNet-50 download.

Code comments and console output are in Portuguese.

## Environment

This is a NixOS environment. The native AI framework libraries are provided via `shell.nix`, and a Python virtualenv (`.venv`, Python 3.13) holds the Python packages (`openvino` 2026.2.0, `numpy` 2.4.6).

`shell.nix` exists specifically to set `LD_LIBRARY_PATH` so that the prebuilt OpenVINO wheels can find native deps (`stdenv.cc.cc.lib`, `zlib`, `glib`, `libglvnd`, `level-zero`). OpenVINO will fail to import outside the nix-shell because of missing shared libraries.

`level-zero` (the `libze_loader.so` loader) is required for OpenVINO to even *detect* the NPU. Without it on `LD_LIBRARY_PATH`, `core.available_devices` shows only `['CPU']` even though the hardware and `intel_vpu` kernel driver are present.

## Commands

```bash
# Enter the nix dev shell (sets LD_LIBRARY_PATH for native libs)
nix-shell;source .venv/bin/activate;python benchmark_resnet.py

# Inside the shell, activate the venv and run a benchmark
source .venv/bin/activate
python test_npu.py          # quick MNIST smoke test (single device)
python benchmark_resnet.py  # ResNet-50 CPU-vs-NPU comparison
python benchmark_vgg16.py   # VGG-16 (heavier) CPU-vs-NPU comparison
python benchmark_vgg19.py   # VGG-19 (heaviest) CPU-vs-NPU comparison

# Power draw per device (needs RAPL read access, see "Measuring power" below)
sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj  # once, reverts on reboot
python benchmark_power.py   # energy / efficiency per device (CPU/GPU/NPU)
```

There are no tests, linters, or build steps — the five `*.py` scripts are the only entry points.

## Key detail

The benchmark target device is set via `target_device` in `test_npu.py:23` (currently `"NPU"`). `"CPU"`, `"NPU"`, and `"GPU"` are all valid targets (see NPU support below for what makes NPU work). The script prints `core.available_devices` at startup and exits early if the requested device isn't present — check that output to see what OpenVINO actually detected.

## NPU support (Meteor Lake / arch 3720, "Intel AI Boost")

The NPU **works end-to-end** (compiles and runs, ~0.32 ms/inference on the MNIST model), but only because of two fixes in `shell.nix`. Both are required:

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

Measured here: on **MNIST** the CPU is ~3.2x *faster* (the model is too small — per-inference dispatch overhead dominates and the NPU never does real work). On **ResNet-50** the NPU is ~5.9x lower latency and ~6.1x higher throughput (real compute, where the NPU's dedicated MAC arrays win). On **VGG-16** the NPU is ~4.8x lower latency (13.45 ms vs 64.32 ms) and ~5x higher throughput (73 vs 15 inf/s) — a much heavier model, so absolute NPU latency is higher than ResNet-50's while the ~5x advantage holds. On **VGG-19** (the heaviest, ~19.6 GFLOPs) the NPU is ~5.1x lower latency (15.69 ms vs 80.25 ms) and ~6.4x higher throughput (64 vs 10 inf/s) — highest absolute NPU latency of the set, but its densest convolutions give the NPU its best throughput advantage. So a CPU-faster result on a trivial model is expected, not a regression — use `benchmark_resnet.py`, `benchmark_vgg16.py`, or `benchmark_vgg19.py` to see the NPU's actual advantage.

### GPU vs NPU — the GPU wins wall-clock, the NPU wins watts

When the iGPU is enabled (see "GPU support" below), the three comparison benchmarks run all of `CPU`, `GPU`, `NPU`. Measured 3-way (latency ms / throughput inf/s): ResNet-50 — CPU 20.43/48, **GPU 2.88/332**, NPU 3.56/290; VGG-16 — CPU 65.06/11, **GPU 8.00/113**, NPU 13.63/74; VGG-19 — CPU 79.76/9, **GPU 12.11/81**, NPU 15.69/64. The **GPU is fastest on every real model** (~6.6–8.1x lower latency than CPU; the NPU ~4.8–5.7x), because the Arc iGPU has far more raw compute. The NPU's real advantage is **performance-per-watt** — see below.

### Power / energy per device (`benchmark_power.py`)

`benchmark_power.py` measures it. ResNet-50, sustained load, Intel RAPL (idle package 9.28 W): CPU 48 inf/s @ 28.02 W (584.7 mJ/inf, 1.71 inf/s/W); GPU 333 inf/s @ 27.99 W (83.9 mJ/inf, 11.91 inf/s/W); **NPU 294 inf/s @ 22.87 W (77.8 mJ/inf, 12.85 inf/s/W)**. So although the GPU wins throughput, the **NPU is the most efficient** — ~88% of the GPU's throughput at ~82% of the power, the lowest energy per inference, and the best inf/s-per-watt. The CPU is ~7.5x worse energy-per-inference. This is the trade-off the NPU exists for.

## Measuring power

RAPL energy lives at `/sys/class/powercap/intel-rapl:0/energy_uj` (microjoules, cumulative, wraps at `max_energy_range_uj`). Since the CVE-2020-8694 mitigation it is **root-only**, so `benchmark_power.py` exits with an instruction unless you grant read access once per boot:

```bash
sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj
```

**Key limitation:** RAPL only exposes the `package-0` domain (plus `core`/`uncore` subdomains) — CPU, GPU, and NPU all share one package energy counter; there is **no per-accelerator RAPL domain**. So the script measures *whole-package* power while the workload runs on each device and subtracts idle. That's a fair relative comparison (energy-per-inference for the same task per device), but it does not attribute exact watts to each silicon block. `sudo` is interactive (needs a TTY), so the chmod must be run by the user in a real terminal, not via OpenVINO's process.

## GPU support (Intel Arc iGPU, Meteor Lake)

The integrated GPU works end-to-end (compiles and runs, fastest of the three), but only once the OpenCL stack is on `LD_LIBRARY_PATH` — same class of problem as the NPU, one level up. OpenVINO's GPU plugin talks **OpenCL**; without the loader and Intel ICD, `core.available_devices` shows only `['CPU', 'NPU']`. Two pieces, both in `shell.nix`:

1. **`ocl-icd`** — the OpenCL loader (`libOpenCL.so`), added to `LD_LIBRARY_PATH`.
2. **`intel-compute-runtime`** (NEO) — the Intel OpenCL ICD (`libigdrcl.so`). The loader discovers it via the `OCL_ICD_VENDORS` env var, which `shell.nix` sets to the package's `etc/OpenCL/vendors` directory (contains `intel-neo.icd`).

This nixpkgs package ships only the OpenCL path (`libigdrcl.so`); it does **not** include the GPU's Level Zero driver (`libze_intel_gpu.so` — only headers), but the OpenVINO GPU plugin uses OpenCL, so detection works. `/dev/dri/renderD128` must be accessible (be in the `render` group). With both libraries loaded, `available_devices` becomes `['CPU', 'GPU', 'NPU']` and the GPU reports as `Intel(R) Arc(TM) Graphics (iGPU)`.
