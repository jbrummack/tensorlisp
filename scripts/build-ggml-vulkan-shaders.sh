#!/usr/bin/env bash
# Compiles every ggml Vulkan shader variant to SPIR-V with glslc, the way
# vendor/ggml/src/ggml-vulkan/CMakeLists.txt does: probe glslc for the GLSL
# extensions it supports (feature-tests/*.comp), build upstream's
# vulkan-shaders-gen, and run it once per .comp source (it enumerates that
# source's variants and writes <variant>.spv).
#
#   scripts/build-ggml-vulkan-shaders.sh [out-dir]      (default target/ggml-vulkan)
#
# Needs glslc (Vulkan SDK; GLSLC=/path to override) and a C++17 compiler
# (MSVC via vcvars64.bat on Windows, c++ elsewhere). Output: <out>/spv/*.spv
# and <out>/gen-flags.txt (the feature macros that were enabled).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SHADERS="$ROOT/vendor/ggml/src/ggml-vulkan/vulkan-shaders"
OUT="${1:-$ROOT/target/ggml-vulkan}"
JOBS="${JOBS:-$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4) )}"

find_glslc() {
    if [ -n "${GLSLC:-}" ]; then echo "$GLSLC"; return; fi
    if command -v glslc >/dev/null 2>&1; then command -v glslc; return; fi
    if [ -n "${VULKAN_SDK:-}" ] && [ -x "$VULKAN_SDK/Bin/glslc.exe" ]; then echo "$VULKAN_SDK/Bin/glslc.exe"; return; fi
    for d in /c/VulkanSDK/*/Bin; do [ -x "$d/glslc.exe" ] && echo "$d/glslc.exe" && return; done
    echo "glslc not found (install the Vulkan SDK or set GLSLC)" >&2
    exit 1
}
GLSLC=$(find_glslc)
echo "glslc: $GLSLC ($("$GLSLC" --version | head -1))"

mkdir -p "$OUT/spv" "$OUT/dep"

# Which optional GLSL extensions does this glslc know? (CMake: ggml_vk_test_glslc_extension)
FLAGS=()
probe() { # name, test file, macro
    if "$GLSLC" -o - -fshader-stage=compute --target-env=vulkan1.3 "$SHADERS/feature-tests/$2" 2>&1 >/dev/null | grep -q "extension not supported"; then
        echo "  $1: not supported by glslc"
    else
        echo "  $1: supported"
        FLAGS+=("$3")
    fi
}
probe coopmat coopmat.comp GGML_VULKAN_COOPMAT_GLSLC_SUPPORT
probe coopmat2 coopmat2.comp GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT
probe coopmat2_decode_vector coopmat2_decode_vector.comp GGML_VULKAN_COOPMAT2_DECODE_VECTOR_GLSLC_SUPPORT
probe integer_dot integer_dot.comp GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT
probe bfloat16 bfloat16.comp GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT
probe float_e2m1 float_e2m1.comp GGML_VULKAN_FLOAT_E2M1_GLSLC_SUPPORT
probe float_e4m3 float_e4m3.comp GGML_VULKAN_FLOAT_E4M3_GLSLC_SUPPORT
printf '%s\n' "${FLAGS[@]}" >"$OUT/gen-flags.txt"

# Build the generator.
GEN="$OUT/vulkan-shaders-gen"
case "$(uname -s)" in
MINGW* | MSYS* | CYGWIN*)
    GEN="$GEN.exe"
    VCVARS=$(ls "/c/Program Files/Microsoft Visual Studio"/*/*/VC/Auxiliary/Build/vcvars64.bat 2>/dev/null | head -1 || true)
    [ -n "$VCVARS" ] || { echo "vcvars64.bat not found (install Visual Studio Build Tools)" >&2; exit 1; }
    DEFS=""
    for f in "${FLAGS[@]}"; do DEFS="$DEFS /D$f"; done
    W() { cygpath -w "$1"; }
    BAT="$OUT/build-gen.bat"
    {
        echo "@echo off"
        echo "set \"PATH=%PATH%;C:\Program Files (x86)\Microsoft Visual Studio\Installer\""
        echo "call \"$(W "$VCVARS")\" >nul"
        echo "cd /d \"$(W "$OUT")\""
        echo "cl /nologo /std:c++17 /EHsc /O2 $DEFS \"$(W "$SHADERS/vulkan-shaders-gen.cpp")\" /Fe:\"$(W "$GEN")\" >nul"
    } >"$BAT"
    MSYS2_ARG_CONV_EXCL='*' cmd.exe /c "$(W "$BAT")"
    ;;
*)
    DEFS=()
    for f in "${FLAGS[@]}"; do DEFS+=("-D$f"); done
    "${CXX:-c++}" -std=c++17 -O2 "${DEFS[@]}" "$SHADERS/vulkan-shaders-gen.cpp" -o "$GEN" -pthread
    ;;
esac
[ -x "$GEN" ] || { echo "failed to build $GEN" >&2; exit 1; }

# Like CMake: one run for the header, then one run per source file for its
# arrays (<out>/gen/<file>.comp.cpp). <out>/gen is what ggml-sys compiles.
GENDIR="$OUT/gen"
mkdir -p "$GENDIR"
"$GEN" --output-dir "$OUT/spv" --target-hpp "$GENDIR/ggml-vulkan-shaders.hpp"
echo "compiling shaders with $JOBS jobs into $OUT/spv ..."
ls "$SHADERS"/*.comp | xargs -P "$JOBS" -I{} bash -c '
    f="$1"; name=$(basename "$f")
    if ! "'"$GEN"'" --glslc "'"$GLSLC"'" --source "$f" --output-dir "'"$OUT"'/spv"             --target-hpp "'"$GENDIR"'/ggml-vulkan-shaders.hpp" --target-cpp "'"$GENDIR"'/$name.cpp" >"'"$OUT"'/dep/$name.log" 2>&1; then
        echo "FAILED: $name (see '"$OUT"'/dep/$name.log)"; exit 255
    fi' _ {}

echo "$(ls "$OUT/spv" | wc -l) SPIR-V modules in $OUT/spv; $(ls "$GENDIR"/*.cpp | wc -l) source files + header in $GENDIR"
printf '%s
' "${FLAGS[@]}" >"$GENDIR/feature-defines.txt"
