{ pkgs ? import <nixpkgs> {} }:

let
  # Prebuilt Intel NPU Compiler-in-Driver, missing from nixpkgs intel-npu-driver.
  # Without it OpenVINO NPU compilation fails (pfnCreate2 UNSUPPORTED_FEATURE).
  npuCompiler = pkgs.callPackage ./intel-npu-compiler.nix { };
  # Intel Compute Runtime (NEO): provides the OpenCL ICD (libigdrcl.so) that
  # OpenVINO's GPU plugin needs to even detect the integrated Arc GPU.
  computeRuntime = pkgs.intel-compute-runtime;
in
pkgs.mkShell {
  name = "npu-test-env";

  buildInputs = with pkgs; [
    python3
    python3Packages.pip
    python3Packages.virtualenv
    # Dependências nativas necessárias para frameworks de IA no Linux
    stdenv.cc.cc.lib
    zlib
    glib
    libglvnd
    level-zero  # Loader (libze_loader.so) required by OpenVINO's NPU plugin
    npuCompiler # libnpu_driver_compiler.so, required to compile models for NPU
    ocl-icd     # OpenCL loader (libOpenCL.so), used by OpenVINO's GPU plugin
    computeRuntime # Intel NEO OpenCL ICD (libigdrcl.so), to detect the iGPU
  ];

  shellHook = ''
    # Garante que as bibliotecas nativas fiquem visíveis para os pacotes do Python
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.glib}/lib:${pkgs.libglvnd}/lib:${pkgs.level-zero}/lib:${npuCompiler}/lib:${pkgs.ocl-icd}/lib:$LD_LIBRARY_PATH"

    # Aponta o loader OpenCL ao ICD da Intel (NEO) para que a GPU seja detetada.
    export OCL_ICD_VENDORS="${computeRuntime}/etc/OpenCL/vendors"

    echo "=== Ambiente NixOS para NPU Ativo ==="
    echo "Execute os comandos para configurar o ambiente Python virtual."
  '';
}
