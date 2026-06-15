{ pkgs ? import <nixpkgs> {} }:

let
  # Prebuilt Intel NPU Compiler-in-Driver, missing from nixpkgs intel-npu-driver.
  # Without it OpenVINO NPU compilation fails (pfnCreate2 UNSUPPORTED_FEATURE).
  npuCompiler = pkgs.callPackage ./intel-npu-compiler.nix { };
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
  ];

  shellHook = ''
    # Garante que as bibliotecas nativas fiquem visíveis para os pacotes do Python
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.glib}/lib:${pkgs.libglvnd}/lib:${pkgs.level-zero}/lib:${npuCompiler}/lib:$LD_LIBRARY_PATH"
    
    echo "=== Ambiente NixOS para NPU Ativo ==="
    echo "Execute os comandos para configurar o ambiente Python virtual."
  '';
}
