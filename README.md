# npu-benchmark

A minimal [OpenVINO](https://docs.openvino.ai/) benchmark harness for testing AI
inference acceleration on Linux — CPU, GPU, and **Intel NPU** (Neural Processing
Unit / VPU). It compiles ONNX models for a target device and measures latency and
throughput, with a focus on getting the Intel NPU working under NixOS.

> Code comments and console output are in Portuguese; this README and `CLAUDE.md`
> are in English.

## Scripts

| Script | Purpose |
| --- | --- |
| `test_npu.py` | Quick smoke test. Downloads a small MNIST model and runs a 500-iteration latency benchmark on the single device set in `target_device`. |
| `benchmark_resnet.py` | Realistic CPU-vs-NPU comparison. Downloads ResNet-50 and benchmarks every available device two ways: synchronous batch-1 (latency) and `AsyncInferQueue` pipeline-full (throughput). |
| `benchmark_vgg16.py` | Heavier CPU-vs-NPU comparison. Same methodology as `benchmark_resnet.py` but downloads VGG-16 (~528MB, same `1×3×224×224` input, ~3.5× the compute) to stress the NPU with a denser convolutional workload. |
| `benchmark_vgg19.py` | Heaviest CPU-vs-NPU comparison. Same methodology again but downloads VGG-19 (`vgg19-bn-7`, ~574MB, same `1×3×224×224` input, ~19.6 GFLOPs / ~1.26× VGG-16) — the densest convolutional workload in the set. |

All scripts auto-download their ONNX model on first run and reuse it afterwards.
The models are git-ignored.

## Requirements

- NixOS (the dev shell pins native library paths via `shell.nix`)
- A Python virtualenv at `.venv` (Python 3.13) with `openvino` 2026.2.0 and `numpy`
- For NPU: Intel NPU hardware (e.g. Meteor Lake "Intel AI Boost") with the
  `intel_vpu` kernel driver loaded

## Usage

```bash
# Enter the nix dev shell (sets LD_LIBRARY_PATH for OpenVINO's native deps,
# the Level Zero loader, and the Intel NPU compiler)
nix-shell

# Activate the venv and run a benchmark
source .venv/bin/activate
python test_npu.py          # MNIST smoke test (single device)
python benchmark_resnet.py  # ResNet-50 CPU-vs-NPU comparison
python benchmark_vgg16.py   # VGG-16 (heavier) CPU-vs-NPU comparison
python benchmark_vgg19.py   # VGG-19 (heaviest) CPU-vs-NPU comparison
```

`test_npu.py` targets the device named in `target_device` (currently `"NPU"`;
`"CPU"` and `"GPU"` also work). Both scripts print `core.available_devices` at
startup so you can see what OpenVINO actually detected.

## Results

Measured on Intel Meteor Lake (NPU arch 3720, "Intel AI Boost"):

**MNIST-12** (tiny model — dominated by per-inference dispatch overhead):

| Device | Avg latency | Throughput |
| --- | ---: | ---: |
| CPU | 0.10 ms | ~9,660 inf/s |
| NPU | 0.32 ms | ~3,130 inf/s |

The three comparison benchmarks run all available targets — CPU, the integrated
Arc **GPU**, and the **NPU** — latency (sync, batch-1) and throughput (async):

**ResNet-50** (realistic model — real compute):

| Device | Latency (sync, batch-1) | Throughput (async) | vs CPU |
| --- | ---: | ---: | ---: |
| CPU | 20.43 ms | ~48 inf/s | — |
| GPU | **2.88 ms** | **~332 inf/s** | ~7.1× lat / ~7.0× thr |
| NPU | 3.56 ms | ~290 inf/s | ~5.7× lat / ~6.1× thr |

**VGG-16** (heavier model — ~15.5 GFLOPs):

| Device | Latency (sync, batch-1) | Throughput (async) | vs CPU |
| --- | ---: | ---: | ---: |
| CPU | 65.06 ms | ~11 inf/s | — |
| GPU | **8.00 ms** | **~113 inf/s** | ~8.1× lat / ~10.1× thr |
| NPU | 13.63 ms | ~74 inf/s | ~4.8× lat / ~6.6× thr |

**VGG-19** (heaviest model — ~19.6 GFLOPs):

| Device | Latency (sync, batch-1) | Throughput (async) | vs CPU |
| --- | ---: | ---: | ---: |
| CPU | 79.76 ms | ~9 inf/s | — |
| GPU | **12.11 ms** | **~81 inf/s** | ~6.6× lat / ~9.2× thr |
| NPU | 15.69 ms | ~64 inf/s | ~5.1× lat / ~7.2× thr |

Two lessons here:

- **The CPU loses badly on any real model.** On the tiny MNIST model the CPU
  *wins* because per-inference dispatch overhead dominates and the accelerators
  never do real work — that flip is expected, not a regression. On ResNet-50 /
  VGG-16 / VGG-19 both accelerators pull 5–8× ahead.
- **The GPU is fastest in raw latency and throughput; the NPU trails closely.**
  The Arc iGPU has far more raw compute, so it wins the wall-clock race. The
  NPU's real edge is **power efficiency** (it does this at a fraction of the
  GPU's/CPU's watts), which this harness does not measure — so "GPU faster than
  NPU" here is about throughput-per-second, not performance-per-watt.

## The NixOS NPU fix

Getting the NPU to actually *compile* models under NixOS required two pieces,
both wired into `shell.nix`:

1. **Level Zero loader** (`level-zero` → `libze_loader.so`) — without it on
   `LD_LIBRARY_PATH`, OpenVINO reports only `['CPU']` even though the hardware
   and `intel_vpu` driver are present.
2. **Intel NPU Compiler-in-Driver** (`intel-npu-compiler.nix` →
   `libnpu_driver_compiler.so`) — the nixpkgs `intel-npu-driver` package ships
   only the runtime driver and **omits the compiler**. Without it,
   `compile_model(..., "NPU")` fails with
   `L0 pfnCreate2 result: ZE_RESULT_ERROR_UNSUPPORTED_FEATURE` and
   `NPU_COMPILER_VERSION` reads `0`.

`intel-npu-compiler.nix` packages Intel's *prebuilt* compiler, extracted from the
official `linux-npu-driver` release. Its version **must match** the installed
`intel-npu-driver` package (currently 1.28.0).

## The NixOS GPU fix

Enabling the integrated Arc GPU is the same kind of problem, one level up: the
OpenVINO GPU plugin talks **OpenCL**, and the loader + Intel ICD must be on the
path or the GPU is never detected (`available_devices` shows only CPU/NPU). Two
pieces, both wired into `shell.nix`:

1. **OpenCL loader** (`ocl-icd` → `libOpenCL.so`) — added to `LD_LIBRARY_PATH`.
2. **Intel Compute Runtime / NEO** (`intel-compute-runtime` → `libigdrcl.so`) —
   the OpenCL ICD for Intel GPUs. The loader finds it via the
   `OCL_ICD_VENDORS` env var, which `shell.nix` points at the package's
   `etc/OpenCL/vendors` directory.

With both in place OpenVINO reports `['CPU', 'GPU', 'NPU']` and the GPU
(`Intel(R) Arc(TM) Graphics (iGPU)`) compiles and runs models. `/dev/dri/renderD128`
must also be accessible (be in the `render` group).

See [`CLAUDE.md`](CLAUDE.md) for the full diagnosis and maintenance notes.
