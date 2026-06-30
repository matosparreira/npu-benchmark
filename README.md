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

**ResNet-50** (realistic model — real compute):

| Device | Latency (sync, batch-1) | Throughput (async) |
| --- | ---: | ---: |
| CPU | 20.5 ms | ~48 inf/s |
| NPU | **3.5 ms** | **~292 inf/s** |

→ On ResNet-50 the NPU is **~5.9× lower latency** and **~6.1× higher throughput**.

**VGG-16** (heavier model — ~3.5× the compute):

| Device | Latency (sync, batch-1) | Throughput (async) |
| --- | ---: | ---: |
| CPU | 64.32 ms | ~15 inf/s |
| NPU | **13.45 ms** | **~73 inf/s** |

→ On VGG-16 the NPU is **~4.8× lower latency** and **~5× higher throughput**.
Absolute NPU latency is higher than ResNet-50's (the model is heavier), but the
~5× advantage over the CPU holds.

**VGG-19** (heaviest model — ~19.6 GFLOPs):

| Device | Latency (sync, batch-1) | Throughput (async) |
| --- | ---: | ---: |
| CPU | 80.25 ms | ~10 inf/s |
| NPU | **15.69 ms** | **~64 inf/s** |

→ On VGG-19 the NPU is **~5.1× lower latency** and **~6.4× higher throughput**.
The heaviest model in the set: NPU latency is the highest measured (15.69 ms),
yet the dense convolutions give the NPU its best throughput advantage of all.

The flip is the whole lesson: on a trivial model the CPU wins because dispatch
overhead dominates and the NPU never does real work; on a real network the NPU's
dedicated MAC arrays pull far ahead — at a fraction of the CPU's power draw. A
CPU-faster result on MNIST is expected, not a regression.

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

See [`CLAUDE.md`](CLAUDE.md) for the full diagnosis and maintenance notes.
