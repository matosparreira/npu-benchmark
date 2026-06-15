# Intel NPU "Compiler-in-Driver" (libnpu_driver_compiler.so).
#
# The nixpkgs `intel-npu-driver` package builds only the runtime UMD
# (libze_intel_npu.so) and omits the compiler, because building it from source
# pulls in the whole OpenVINO-based NPU compiler. Without this library,
# OpenVINO NPU compilation fails with:
#     L0 pfnCreate2 result: ZE_RESULT_ERROR_UNSUPPORTED_FEATURE
# and NPU_COMPILER_VERSION reports 0.
#
# This derivation drops in Intel's prebuilt compiler, extracted from the
# `intel-driver-compiler-npu` .deb inside the official linux-npu-driver release
# tarball. The version MUST match the installed intel-npu-driver (1.28.0).
{
  lib,
  stdenvNoCC,
  stdenv,
  fetchurl,
  dpkg,
  autoPatchelfHook,
  onetbb,
  zlib,
  zstd,
}:

stdenvNoCC.mkDerivation rec {
  pname = "intel-npu-compiler";
  version = "1.28.0";

  src = fetchurl {
    url = "https://github.com/intel/linux-npu-driver/releases/download/v${version}/linux-npu-driver-v${version}.20251218-20347000698-ubuntu2404.tar.gz";
    hash = "sha256-CcyiJ9fxh5wKN4XWOSP2xjj+ogaTPWMfvnJiN8CNA8I=";
  };

  nativeBuildInputs = [
    dpkg
    autoPatchelfHook
  ];

  # Runtime deps of the compiler blob (from `objdump -p ... | grep NEEDED`).
  buildInputs = [
    stdenv.cc.cc.lib # libstdc++.so.6 / libgcc_s.so.1
    onetbb # libtbb.so.12
    zlib # libz.so.1
    zstd # libzstd.so.1
  ];

  dontConfigure = true;
  dontBuild = true;

  unpackPhase = ''
    runHook preUnpack
    tar xzf "$src"
    dpkg-deb -x intel-driver-compiler-npu_*.deb deb
    runHook postUnpack
  '';

  installPhase = ''
    runHook preInstall
    install -Dm555 deb/usr/lib/x86_64-linux-gnu/libnpu_driver_compiler.so \
      "$out/lib/libnpu_driver_compiler.so"
    runHook postInstall
  '';

  meta = {
    description = "Intel NPU Compiler-in-Driver (prebuilt), missing from nixpkgs intel-npu-driver";
    homepage = "https://github.com/intel/linux-npu-driver";
    platforms = [ "x86_64-linux" ];
    license = lib.licenses.mit; # same as nixpkgs intel-npu-driver (upstream is MIT)
  };
}
