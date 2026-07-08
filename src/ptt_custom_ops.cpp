// Shared library exposing the PocketTTS custom ORT ops for python-side
// verification:
//
//   so = onnxruntime.SessionOptions()
//   so.register_custom_ops_library("libptt_custom_ops.dylib")
//
// Lets rewrite scripts compare original vs custom-op models on identical
// inputs without going through the CLI (whose chunking is timing-dependent).

// Route all ORT calls through the host runtime's api table (set in
// RegisterCustomOps) instead of a linked copy of onnxruntime — the host is
// typically the python wheel, which may be a different build.
#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>

#ifdef __APPLE__
  #include <Accelerate/Accelerate.h>
#endif
#if defined(__aarch64__) || defined(__ARM_NEON)
  #include <arm_neon.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "pocket_tts_custom_attention.hpp"

extern "C" __attribute__((visibility("default")))
OrtStatus* RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api_base) {
    const OrtApi* api = api_base->GetApi(ORT_API_VERSION);
    Ort::InitApi(api);
#if PTT_HAVE_CUSTOM_OPS
    try {
        return api->AddCustomOpDomain(options, custom_attention_domain());
    } catch (const std::exception& e) {
        return api->CreateStatus(ORT_RUNTIME_EXCEPTION, e.what());
    }
#else
    return api->CreateStatus(ORT_NOT_IMPLEMENTED, "pockettts custom ops need GCC/Clang (MSVC unsupported)");
#endif
}
