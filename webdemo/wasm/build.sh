#!/bin/bash
# Build pocket_tts_wasm.mjs/.wasm against the ORT WASM static library.
# Prereq: the ORT static lib built under ../../../ort-wasm (see CMakeLists).
set -euo pipefail
cd "$(dirname "$0")"

ORT_ROOT="${ORT_WASM_ROOT:-$(pwd)/../../../ort-wasm}"
EMSDK="$ORT_ROOT/cmake/external/emsdk"

if [ ! -f "$EMSDK/emsdk_env.sh" ]; then
    echo "emsdk not found at $EMSDK" >&2
    exit 1
fi
# ORT's build already installed/activated a pinned emsdk version.
saved_em_cache="${EM_CACHE:-}"
source "$EMSDK/emsdk_env.sh" >/dev/null 2>&1
if [ -n "$saved_em_cache" ]; then export EM_CACHE="$saved_em_cache"; fi

emcmake cmake -B build -DCMAKE_BUILD_TYPE=Release -DORT_WASM_ROOT="$ORT_ROOT" "$@"
cmake --build build -j"${PTT_BUILD_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 2)}"

mkdir -p ../vendor/ptt
cp build/pocket_tts_wasm.mjs build/pocket_tts_wasm.wasm build/pocket_tts_wasm_growth.mjs build/pocket_tts_wasm_growth.wasm ../vendor/ptt/
ls -la ../vendor/ptt/
echo "done"
