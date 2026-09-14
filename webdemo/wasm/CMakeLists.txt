# Emscripten build of PocketTTS for the webdemo.
#
#   ./build.sh          (activates ORT's bundled emsdk, configures, builds)
#
# Links the ONNX Runtime WASM static library (v1.27.0, threads + SIMD)
# built from ../../../ort-wasm via:
#   ./build.sh --config Release --build_wasm_static_lib \
#              --enable_wasm_threads --enable_wasm_simd \
#              --disable_wasm_exception_catching \
#              --enable_wasm_api_exception_catching \
#              --skip_tests --parallel

cmake_minimum_required(VERSION 3.28)
project(pocket_tts_wasm LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT EMSCRIPTEN)
    message(FATAL_ERROR "This project must be configured with emcmake")
endif()

set(ORT_WASM_ROOT "${CMAKE_SOURCE_DIR}/../../../ort-wasm" CACHE PATH "onnxruntime source root")
set(ORT_WASM_LIB "" CACHE FILEPATH "libonnxruntime_webassembly.a path")
if(NOT ORT_WASM_LIB)
    file(GLOB_RECURSE _ort_libs "${ORT_WASM_ROOT}/build/*/Release/libonnxruntime_webassembly.a")
    list(GET _ort_libs 0 ORT_WASM_LIB)
endif()
if(NOT EXISTS ${ORT_WASM_LIB})
    message(FATAL_ERROR "ORT wasm static lib not found (looked under ${ORT_WASM_ROOT}/build). Build it first.")
endif()
message(STATUS "ORT wasm lib: ${ORT_WASM_LIB}")

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(dr_libs
    GIT_REPOSITORY https://github.com/mackron/dr_libs.git
    GIT_TAG cd99e2cca5ccb48f8d000f1cd844424dcdbbaf59 # dr_libs pinned
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(dr_libs)

add_executable(pocket-tts-wasm ${CMAKE_SOURCE_DIR}/../../src/pocket_tts.cpp)
target_compile_definitions(pocket-tts-wasm PRIVATE PTT_SHARED_LIB)
target_include_directories(pocket-tts-wasm PRIVATE
    ${ORT_WASM_ROOT}/include/onnxruntime/core/session
    ${CMAKE_SOURCE_DIR}/../../include
    ${dr_libs_SOURCE_DIR}
)
# Match the ORT static lib's build mode: classic emscripten exceptions enabled.
target_compile_options(pocket-tts-wasm PRIVATE -O3 -pthread -msimd128 -sDISABLE_EXCEPTION_CATCHING=0)

set(EXPORTED_FUNCS "_ptt_set_emotion,_ptt_set_seed,_ptt_load_soura_vectors,_ptt_prepare_voice,_ptt_create,_ptt_create_ex,_ptt_create_profiled,_ptt_load_encoder,_ptt_encode_voice,_ptt_decode,_ptt_decoder_reset,_ptt_destroy,_ptt_warmup,_ptt_set_temperature,_ptt_set_soften_commas,_ptt_set_max_chunk,_ptt_stream_start,_ptt_stream_read,_ptt_stream_poll,_ptt_stream_stop,_ptt_latents_start,_ptt_latents_poll,_ptt_latents_stop,_ptt_latents_end,_ptt_stream_end,_ptt_free_audio,_ptt_set_profiling,_ptt_configure_pool,_ptt_debug_bench,_ptt_print_profile,_malloc,_free")
set(RUNTIME_METHODS "ccall,cwrap,FS,UTF8ToString,stringToUTF8,lengthBytesUTF8,HEAPF32,HEAPU8,HEAPU32")

target_link_options(pocket-tts-wasm PRIVATE
    -O3 -pthread
    "-sDISABLE_EXCEPTION_CATCHING=0"
    "-sMODULARIZE=1"
    "-sEXPORT_ES6=1"
    "-sEXPORT_NAME=createPocketTTS"
    "-sENVIRONMENT=worker,node"
    "SHELL:-s PTHREAD_POOL_SIZE='Math.min((navigator.hardwareConcurrency||8)+6,16)'"
    "-sPTHREAD_POOL_SIZE_STRICT=0"
    "-sINITIAL_MEMORY=805306368"
    "-sALLOW_MEMORY_GROWTH=0"
    "-sMAXIMUM_MEMORY=805306368"
    "-sSTACK_SIZE=5242880"
    "-sDEFAULT_PTHREAD_STACK_SIZE=2097152"
    "-sEXPORTED_FUNCTIONS=${EXPORTED_FUNCS}"
    "-sEXPORTED_RUNTIME_METHODS=${RUNTIME_METHODS}"
    "-sWASM_BIGINT=1"
)

target_link_libraries(pocket-tts-wasm PRIVATE
    ${ORT_WASM_LIB}
)

set_target_properties(pocket-tts-wasm PROPERTIES
    OUTPUT_NAME "pocket_tts_wasm"
    SUFFIX ".mjs"
)

# iOS/low-memory variant: growable memory (fixed 1GB SharedArrayBuffer can be
# refused by mobile Safari). ~10% slower on desktop; the worker falls back to
# this module when the fixed-memory one fails to instantiate.
add_executable(pocket-tts-wasm-growth ${CMAKE_SOURCE_DIR}/../../src/pocket_tts.cpp)
target_compile_definitions(pocket-tts-wasm-growth PRIVATE PTT_SHARED_LIB)
target_include_directories(pocket-tts-wasm-growth PRIVATE
    ${ORT_WASM_ROOT}/include/onnxruntime/core/session
    ${CMAKE_SOURCE_DIR}/../../include
    ${dr_libs_SOURCE_DIR}
)
target_compile_options(pocket-tts-wasm-growth PRIVATE -O3 -pthread -msimd128 -sDISABLE_EXCEPTION_CATCHING=0)
target_link_options(pocket-tts-wasm-growth PRIVATE
    -O3 -pthread
    "-sDISABLE_EXCEPTION_CATCHING=0"
    "-sMODULARIZE=1"
    "-sEXPORT_ES6=1"
    "-sEXPORT_NAME=createPocketTTS"
    "-sENVIRONMENT=worker,node"
    "SHELL:-s PTHREAD_POOL_SIZE='Math.min((navigator.hardwareConcurrency||8)+6,16)'"
    "-sPTHREAD_POOL_SIZE_STRICT=0"
    "-sINITIAL_MEMORY=268435456"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sMAXIMUM_MEMORY=1610612736"
    "-sSTACK_SIZE=5242880"
    "-sDEFAULT_PTHREAD_STACK_SIZE=2097152"
    "-sEXPORTED_FUNCTIONS=${EXPORTED_FUNCS}"
    "-sEXPORTED_RUNTIME_METHODS=${RUNTIME_METHODS}"
    "-sWASM_BIGINT=1"
)
target_link_libraries(pocket-tts-wasm-growth PRIVATE ${ORT_WASM_LIB})
set_target_properties(pocket-tts-wasm-growth PROPERTIES
    OUTPUT_NAME "pocket_tts_wasm_growth"
    SUFFIX ".mjs"
)
