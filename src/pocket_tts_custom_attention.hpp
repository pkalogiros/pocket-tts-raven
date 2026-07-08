#pragma once

// Custom ORT ops for PocketTTS. Three backends:
//   __APPLE__      — Accelerate (cblas/vDSP, AMX-backed); call sequences are
//                    preserved exactly so native output stays bit-identical.
//   __EMSCRIPTEN__ — portable WASM-SIMD kernels (wasm_simd128 intrinsics).
//   other GCC/Clang — portable kernels via compiler vector extensions;
//                    compile to NEON on ARM and SSE/AVX on x86-64. Only the
//                    attention ops pay off here — the conv ops need
//                    Accelerate-class GEMM, so kAccelConvAvailable stays
//                    Apple-only and other platforms keep MLAS for the decoder.
//   (MSVC has no vector extensions: no custom ops; use clang-cl on Windows.)
// Define PTT_FORCE_PORTABLE to benchmark the portable backend on macOS.
#if defined(__APPLE__) || defined(__EMSCRIPTEN__) || defined(__GNUC__) || defined(__clang__)
#define PTT_HAVE_CUSTOM_OPS 1

#ifdef __EMSCRIPTEN__
#include <wasm_simd128.h>
#endif

// ── BLAS/vector abstraction ─────────────────────────────────────────────────
#if defined(__APPLE__) && !defined(PTT_FORCE_PORTABLE)

static inline void ptt_gemm_nn(int M, int N, int K, float alpha,
                               const float* A, int lda, const float* B, int ldb,
                               float beta, float* C, int ldc) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha,
                A, lda, B, ldb, beta, C, ldc);
}
static inline void ptt_gemm_nt(int M, int N, int K, float alpha,
                               const float* A, int lda, const float* B, int ldb,
                               float beta, float* C, int ldc) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, alpha,
                A, lda, B, ldb, beta, C, ldc);
}
static inline void ptt_vadd(float* a, const float* b, int n) {
    vDSP_vadd(a, 1, b, 1, a, 1, static_cast<vDSP_Length>(n));
}
static inline float ptt_maxv(const float* a, int n) {
    float m = -std::numeric_limits<float>::infinity();
    vDSP_maxv(a, 1, &m, static_cast<vDSP_Length>(n));
    return m;
}
static inline void ptt_vsadd(float* a, float sc, int n) {
    vDSP_vsadd(a, 1, &sc, a, 1, static_cast<vDSP_Length>(n));
}
static inline void ptt_expv(float* a, int n) {
    vvexpf(a, a, &n);
}
static inline float ptt_sum(const float* a, int n) {
    float s = 0.0f;
    vDSP_sve(a, 1, &s, static_cast<vDSP_Length>(n));
    return s;
}
static inline void ptt_scale(float* a, float sc, int n) {
    vDSP_vsmul(a, 1, &sc, a, 1, static_cast<vDSP_Length>(n));
}
static inline void ptt_elu_inplace(float* a, float* tmp, int n) {
    const float lo = -std::numeric_limits<float>::max();
    const float hi = std::numeric_limits<float>::max();
    const float zero = 0.0f, neg_one = -1.0f;
    vDSP_vclip(a, 1, &lo, &zero, tmp, 1, static_cast<vDSP_Length>(n));
    vvexpf(tmp, tmp, &n);
    vDSP_vsadd(tmp, 1, &neg_one, tmp, 1, static_cast<vDSP_Length>(n));
    vDSP_vclip(a, 1, &zero, &hi, a, 1, static_cast<vDSP_Length>(n));
    vDSP_vadd(a, 1, tmp, 1, a, 1, static_cast<vDSP_Length>(n));
}

#elif defined(__EMSCRIPTEN__)  // portable SIMD128 kernels

// C[M,N] = alpha*A[M,K]@B[K,N] + beta*C — saxpy ordering (m,k,n-vectorized)
static inline void ptt_gemm_nn(int M, int N, int K, float alpha,
                               const float* A, int lda, const float* B, int ldb,
                               float beta, float* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        float* c = C + size_t(m) * ldc;
        if (beta == 0.0f) std::fill(c, c + N, 0.0f);
        else if (beta != 1.0f) for (int n = 0; n < N; ++n) c[n] *= beta;
        const float* a_row = A + size_t(m) * lda;
        for (int k = 0; k < K; ++k) {
            const float a = alpha * a_row[k];
            if (a == 0.0f) continue;
            const float* b = B + size_t(k) * ldb;
            v128_t va = wasm_f32x4_splat(a);
            int n = 0;
            for (; n + 8 <= N; n += 8) {
                v128_t c0 = wasm_v128_load(c + n);
                v128_t c1 = wasm_v128_load(c + n + 4);
                c0 = wasm_f32x4_add(c0, wasm_f32x4_mul(va, wasm_v128_load(b + n)));
                c1 = wasm_f32x4_add(c1, wasm_f32x4_mul(va, wasm_v128_load(b + n + 4)));
                wasm_v128_store(c + n, c0);
                wasm_v128_store(c + n + 4, c1);
            }
            for (; n < N; ++n) c[n] += a * b[n];
        }
    }
}

// C[M,N] = alpha*A[M,K]@B[N,K]^T + beta*C — dot-product ordering
static inline void ptt_gemm_nt(int M, int N, int K, float alpha,
                               const float* A, int lda, const float* B, int ldb,
                               float beta, float* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        const float* a_row = A + size_t(m) * lda;
        float* c = C + size_t(m) * ldc;
        for (int n = 0; n < N; ++n) {
            const float* b_row = B + size_t(n) * ldb;
            v128_t s0 = wasm_f32x4_splat(0.0f), s1 = wasm_f32x4_splat(0.0f);
            int k = 0;
            for (; k + 8 <= K; k += 8) {
                s0 = wasm_f32x4_add(s0, wasm_f32x4_mul(wasm_v128_load(a_row + k),
                                                       wasm_v128_load(b_row + k)));
                s1 = wasm_f32x4_add(s1, wasm_f32x4_mul(wasm_v128_load(a_row + k + 4),
                                                       wasm_v128_load(b_row + k + 4)));
            }
            v128_t s = wasm_f32x4_add(s0, s1);
            float dot = wasm_f32x4_extract_lane(s, 0) + wasm_f32x4_extract_lane(s, 1) +
                        wasm_f32x4_extract_lane(s, 2) + wasm_f32x4_extract_lane(s, 3);
            for (; k < K; ++k) dot += a_row[k] * b_row[k];
            c[n] = alpha * dot + (beta == 0.0f ? 0.0f : beta * c[n]);
        }
    }
}

static inline void ptt_vadd(float* a, const float* b, int n) {
    for (int i = 0; i < n; ++i) a[i] += b[i];
}
static inline float ptt_maxv(const float* a, int n) {
    float m = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) if (a[i] > m) m = a[i];
    return m;
}
static inline void ptt_vsadd(float* a, float sc, int n) {
    for (int i = 0; i < n; ++i) a[i] += sc;
}
static inline void ptt_expv(float* a, int n) {
    for (int i = 0; i < n; ++i) a[i] = expf(a[i]);
}
static inline float ptt_sum(const float* a, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i];
    return s;
}
static inline void ptt_scale(float* a, float sc, int n) {
    for (int i = 0; i < n; ++i) a[i] *= sc;
}
static inline void ptt_elu_inplace(float* a, float*, int n) {
    for (int i = 0; i < n; ++i) a[i] = a[i] > 0.0f ? a[i] : expf(a[i]) - 1.0f;
}

#else  // portable: GCC/Clang vector extensions (NEON/SSE/AVX via codegen)

typedef float ptt_v4f __attribute__((vector_size(16)));

static inline ptt_v4f ptt_v4_load(const float* p) {
    ptt_v4f v; std::memcpy(&v, p, sizeof(v)); return v;
}
static inline void ptt_v4_store(float* p, ptt_v4f v) {
    std::memcpy(p, &v, sizeof(v));
}

// C[M,N] = alpha*A[M,K]@B[K,N] + beta*C — saxpy ordering (m,k,n-vectorized)
static inline void ptt_gemm_nn(int M, int N, int K, float alpha,
                               const float* A, int lda, const float* B, int ldb,
                               float beta, float* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        float* c = C + size_t(m) * ldc;
        if (beta == 0.0f) std::fill(c, c + N, 0.0f);
        else if (beta != 1.0f) for (int n = 0; n < N; ++n) c[n] *= beta;
        const float* a_row = A + size_t(m) * lda;
        for (int k = 0; k < K; ++k) {
            const float a = alpha * a_row[k];
            if (a == 0.0f) continue;
            const float* b = B + size_t(k) * ldb;
            const ptt_v4f va = {a, a, a, a};
            int n = 0;
            for (; n + 8 <= N; n += 8) {
                ptt_v4f c0 = ptt_v4_load(c + n);
                ptt_v4f c1 = ptt_v4_load(c + n + 4);
                c0 += va * ptt_v4_load(b + n);
                c1 += va * ptt_v4_load(b + n + 4);
                ptt_v4_store(c + n, c0);
                ptt_v4_store(c + n + 4, c1);
            }
            for (; n < N; ++n) c[n] += a * b[n];
        }
    }
}

// C[M,N] = alpha*A[M,K]@B[N,K]^T + beta*C — dot-product ordering
static inline void ptt_gemm_nt(int M, int N, int K, float alpha,
                               const float* A, int lda, const float* B, int ldb,
                               float beta, float* C, int ldc) {
    for (int m = 0; m < M; ++m) {
        const float* a_row = A + size_t(m) * lda;
        float* c = C + size_t(m) * ldc;
        for (int n = 0; n < N; ++n) {
            const float* b_row = B + size_t(n) * ldb;
            ptt_v4f s0 = {0,0,0,0}, s1 = {0,0,0,0};
            int k = 0;
            for (; k + 8 <= K; k += 8) {
                s0 += ptt_v4_load(a_row + k) * ptt_v4_load(b_row + k);
                s1 += ptt_v4_load(a_row + k + 4) * ptt_v4_load(b_row + k + 4);
            }
            const ptt_v4f sv = s0 + s1;
            float dot = sv[0] + sv[1] + sv[2] + sv[3];
            for (; k < K; ++k) dot += a_row[k] * b_row[k];
            c[n] = alpha * dot + (beta == 0.0f ? 0.0f : beta * c[n]);
        }
    }
}

static inline void ptt_vadd(float* a, const float* b, int n) {
    for (int i = 0; i < n; ++i) a[i] += b[i];
}
static inline float ptt_maxv(const float* a, int n) {
    float m = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < n; ++i) if (a[i] > m) m = a[i];
    return m;
}
static inline void ptt_vsadd(float* a, float sc, int n) {
    for (int i = 0; i < n; ++i) a[i] += sc;
}
static inline void ptt_expv(float* a, int n) {
    for (int i = 0; i < n; ++i) a[i] = expf(a[i]);
}
static inline float ptt_sum(const float* a, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; ++i) s += a[i];
    return s;
}
static inline void ptt_scale(float* a, float sc, int n) {
    for (int i = 0; i < n; ++i) a[i] *= sc;
}
static inline void ptt_elu_inplace(float* a, float*, int n) {
    for (int i = 0; i < n; ++i) a[i] = a[i] > 0.0f ? a[i] : expf(a[i]) - 1.0f;
}

#endif

struct AttentionSmallShape {
    std::array<int64_t, 5> d{};
    size_t rank = 0;
};

class AttentionTailKernel {
public:
    AttentionTailKernel(const OrtApi&, const OrtKernelInfo*) {
        scores_.reserve(32768);
    }

    void Compute(OrtKernelContext* raw_context) {
        Ort::KernelContext ctx(raw_context);
        auto q_value = ctx.GetInput(0);
        auto state_value = ctx.GetInput(1);
        auto new_k_value = ctx.GetInput(2);
        auto new_v_value = ctx.GetInput(3);
        auto step_value = ctx.GetInput(4);
        auto scale_value = ctx.GetInput(5);
        auto mask_value = ctx.GetInput(6);

        const AttentionSmallShape q_shape = tensor_shape(q_value);
        const AttentionSmallShape state_shape = tensor_shape(state_value);
        const AttentionSmallShape new_k_shape = tensor_shape(new_k_value);
        const AttentionSmallShape mask_shape = tensor_shape(mask_value);
        if (q_shape.rank != 4 || state_shape.rank != 5 || new_k_shape.rank != 5) {
            throw std::runtime_error("AttentionTail expects q [1,H,Q,D], state [2,1,C,H,D], new K/V [1,1,Q,H,D]");
        }

        const int64_t heads = q_shape.d[1];
        const int64_t query = q_shape.d[2];
        const int64_t dim = q_shape.d[3];
        const int64_t capacity = state_shape.d[2];
        const int64_t new_len = new_k_shape.d[2];
        const int64_t old_len = step_value.GetTensorData<int64_t>()[0];
        if (q_shape.d[0] != 1 || state_shape.d[0] != 2 || state_shape.d[1] != 1 ||
            state_shape.d[3] != heads || state_shape.d[4] != dim ||
            new_k_shape.d[0] != 1 || new_k_shape.d[1] != 1 ||
            new_k_shape.d[2] != query || new_k_shape.d[3] != heads ||
            new_k_shape.d[4] != dim || old_len < 0 || old_len > capacity) {
            throw std::runtime_error("AttentionTail input shape mismatch");
        }

        const float* q = q_value.GetTensorData<float>();
        const float* state = state_value.GetTensorData<float>();
        const float* new_k = new_k_value.GetTensorData<float>();
        const float* new_v = new_v_value.GetTensorData<float>();
        const float k_scale = scale_value.GetTensorData<float>()[0];
        const float* mask = mask_value.GetTensorData<float>();

        const std::array<int64_t, 3> out_shape = {1, query, heads * dim};
        auto out_value = ctx.GetOutput(0, out_shape.data(), out_shape.size());
        float* out = out_value.GetTensorMutableData<float>();

        compute(q, state, new_k, new_v, k_scale, mask, mask_shape,
                out, heads, query, dim, old_len, new_len, capacity);
    }

private:
    static AttentionSmallShape tensor_shape(const Ort::ConstValue& value) {
        auto info = value.GetTensorTypeAndShapeInfo();
        AttentionSmallShape shape;
        shape.rank = info.GetDimensionsCount();
        if (shape.rank > shape.d.size()) {
            throw std::runtime_error("AttentionTail unsupported rank");
        }
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
        info.GetDimensions(shape.d.data(), shape.rank);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
        return shape;
    }

    void compute(const float* q, const float* state,
                 const float* new_k, const float* new_v, float k_scale,
                 const float* mask, const AttentionSmallShape& mask_shape,
                 float* out, int64_t heads, int64_t query, int64_t dim,
                 int64_t old_len, int64_t new_len, int64_t capacity) {
#if defined(__aarch64__) || defined(__ARM_NEON)
        if (query == 1 && dim == 64) {
            compute_ar_neon(q, state, new_k, new_v, k_scale, mask, mask_shape,
                            out, heads, old_len, new_len, capacity);
            return;
        }
#endif

        const int64_t time = old_len + new_len;
        scores_.resize(static_cast<size_t>(query * time));
        const int lda_cache = static_cast<int>(heads * dim);
        const int query_i = static_cast<int>(query);
        const int time_i = static_cast<int>(time);
        const int dim_i = static_cast<int>(dim);
        const int old_i = static_cast<int>(old_len);
        const int new_i = static_cast<int>(new_len);
        const int64_t value_offset = capacity * heads * dim;

        for (int64_t h = 0; h < heads; ++h) {
            const float* q_head = q + h * query * dim;
            const float* old_k_head = state + h * dim;
            const float* old_v_head = state + value_offset + h * dim;
            const float* new_k_head = new_k + h * dim;
            const float* new_v_head = new_v + h * dim;
            float* out_head = out + h * dim;

            if (old_len > 0) {
                ptt_gemm_nt(query_i, old_i, dim_i, k_scale,
                            q_head, dim_i, old_k_head, lda_cache,
                            0.0f, scores_.data(), time_i);
            }
            if (new_len > 0) {
                ptt_gemm_nt(query_i, new_i, dim_i, k_scale,
                            q_head, dim_i, new_k_head, lda_cache,
                            0.0f, scores_.data() + old_len, time_i);
            }

            for (int64_t qi = 0; qi < query; ++qi) {
                softmax_row(scores_.data() + qi * time, mask, mask_shape, qi, h, time);
            }

            if (old_len > 0) {
                ptt_gemm_nn(query_i, dim_i, old_i, 1.0f,
                            scores_.data(), time_i, old_v_head, lda_cache,
                            0.0f, out_head, lda_cache);
            }
            if (new_len > 0) {
                ptt_gemm_nn(query_i, dim_i, new_i, 1.0f,
                            scores_.data() + old_len, time_i, new_v_head, lda_cache,
                            old_len > 0 ? 1.0f : 0.0f, out_head, lda_cache);
            }
        }
    }

#if defined(__aarch64__) || defined(__ARM_NEON)
    static float dot64_neon(const float* a, const float* b) {
        float32x4_t s0 = vdupq_n_f32(0.0f);
        float32x4_t s1 = vdupq_n_f32(0.0f);
        float32x4_t s2 = vdupq_n_f32(0.0f);
        float32x4_t s3 = vdupq_n_f32(0.0f);
        for (int d = 0; d < 64; d += 16) {
            s0 = vfmaq_f32(s0, vld1q_f32(a + d),      vld1q_f32(b + d));
            s1 = vfmaq_f32(s1, vld1q_f32(a + d + 4),  vld1q_f32(b + d + 4));
            s2 = vfmaq_f32(s2, vld1q_f32(a + d + 8),  vld1q_f32(b + d + 8));
            s3 = vfmaq_f32(s3, vld1q_f32(a + d + 12), vld1q_f32(b + d + 12));
        }
        return vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    }

    void compute_ar_neon(const float* q, const float* state,
                         const float* new_k, const float* new_v, float k_scale,
                         const float* mask, const AttentionSmallShape& mask_shape,
                         float* out, int64_t heads, int64_t old_len,
                         int64_t new_len, int64_t capacity) {
        const int64_t time = old_len + new_len;
        scores_.resize(static_cast<size_t>(heads * time));
        const int64_t stride_t = heads * 64;
        const int64_t value_offset = capacity * stride_t;

        for (int64_t h = 0; h < heads; ++h) {
            const float* q_head = q + h * 64;
            float* row = scores_.data() + h * time;
            for (int64_t t = 0; t < old_len; ++t) {
                row[t] = dot64_neon(q_head, state + t * stride_t + h * 64) * k_scale;
            }
            for (int64_t t = 0; t < new_len; ++t) {
                row[old_len + t] = dot64_neon(q_head, new_k + t * stride_t + h * 64) * k_scale;
            }
            softmax_row(row, mask, mask_shape, 0, h, time);

            float32x4_t acc[16];
            for (auto& v : acc) v = vdupq_n_f32(0.0f);

            auto accumulate = [&](const float* v_head, float weight) {
                const float32x4_t wt = vdupq_n_f32(weight);
                for (int i = 0; i < 16; ++i) {
                    acc[i] = vfmaq_f32(acc[i], wt, vld1q_f32(v_head + i * 4));
                }
            };

            for (int64_t t = 0; t < old_len; ++t) {
                accumulate(state + value_offset + t * stride_t + h * 64, row[t]);
            }
            for (int64_t t = 0; t < new_len; ++t) {
                accumulate(new_v + t * stride_t + h * 64, row[old_len + t]);
            }

            float* out_head = out + h * 64;
            for (int i = 0; i < 16; ++i) {
                vst1q_f32(out_head + i * 4, acc[i]);
            }
        }
    }
#endif

    static void softmax_row(float* row, const float* mask, const AttentionSmallShape& mask_shape,
                            int64_t qi, int64_t h, int64_t time) {
        const int n = static_cast<int>(time);
        if (mask_shape.rank == 1) {
            ptt_vadd(row, mask, n);
        } else if (mask_shape.rank == 2) {
            const int64_t q_index = mask_shape.d[0] == 1 ? 0 : qi;
            ptt_vadd(row, mask + q_index * mask_shape.d[1], n);
        } else if (mask_shape.rank == 4 && mask_shape.d[1] == 1) {
            const int64_t q_index = mask_shape.d[2] == 1 ? 0 : qi;
            ptt_vadd(row, mask + q_index * mask_shape.d[3], n);
        } else {
            for (int64_t t = 0; t < time; ++t) row[t] += mask_at(mask, mask_shape, qi, h, t);
        }

        const float max_score = ptt_maxv(row, n);
        ptt_vsadd(row, -max_score, n);
        ptt_expv(row, n);
        ptt_scale(row, 1.0f / ptt_sum(row, n), n);
    }

    static float mask_at(const float* mask, const AttentionSmallShape& shape,
                         int64_t qi, int64_t h, int64_t t) {
        if (shape.rank == 1) return mask[t];
        if (shape.rank == 2) {
            const int64_t q_index = shape.d[0] == 1 ? 0 : qi;
            return mask[q_index * shape.d[1] + t];
        }
        if (shape.rank == 4) {
            const int64_t h_index = shape.d[1] == 1 ? 0 : h;
            const int64_t q_index = shape.d[2] == 1 ? 0 : qi;
            return mask[((h_index * shape.d[2] + q_index) * shape.d[3]) + t];
        }
        throw std::runtime_error("AttentionTail unsupported mask shape");
    }

    std::vector<float> scores_;
};

class DecoderAttentionTailKernel {
public:
    DecoderAttentionTailKernel(const OrtApi&, const OrtKernelInfo*) {
        scores_.reserve(16384);
        update_scores_.reserve(256);
    }

    void Compute(OrtKernelContext* raw_context) {
        Ort::KernelContext ctx(raw_context);
        auto q_value = ctx.GetInput(0);
        auto state_value = ctx.GetInput(1);
        auto new_k_value = ctx.GetInput(2);
        auto new_v_value = ctx.GetInput(3);
        auto index_value = ctx.GetInput(4);
        auto scale_value = ctx.GetInput(5);
        auto mask_value = ctx.GetInput(6);

        const AttentionSmallShape q_shape = tensor_shape(q_value);
        const AttentionSmallShape state_shape = tensor_shape(state_value);
        const AttentionSmallShape new_k_shape = tensor_shape(new_k_value);
        const AttentionSmallShape new_v_shape = tensor_shape(new_v_value);
        const AttentionSmallShape index_shape = tensor_shape(index_value);
        const AttentionSmallShape mask_shape = tensor_shape(mask_value);
        if (q_shape.rank != 4 || state_shape.rank != 5 || new_k_shape.rank != 4 ||
            new_v_shape.rank != 4 || index_shape.rank != 4) {
            throw std::runtime_error("DecoderAttentionTail expects q/new K/V [1,H,Q,D], state [2,1,H,C,D]");
        }

        const int64_t heads = q_shape.d[1];
        const int64_t query = q_shape.d[2];
        const int64_t dim = q_shape.d[3];
        const int64_t capacity = state_shape.d[3];
        if (q_shape.d[0] != 1 || state_shape.d[0] != 2 || state_shape.d[1] != 1 ||
            state_shape.d[2] != heads || state_shape.d[4] != dim ||
            new_k_shape.d[0] != 1 || new_k_shape.d[1] != heads ||
            new_k_shape.d[2] != query || new_k_shape.d[3] != dim ||
            new_v_shape.d != new_k_shape.d || index_shape.d != new_k_shape.d ||
            dim <= 0 || query <= 0 || capacity <= 0) {
            throw std::runtime_error("DecoderAttentionTail input shape mismatch");
        }

        const float* q = q_value.GetTensorData<float>();
        const float* state = state_value.GetTensorData<float>();
        const float* new_k = new_k_value.GetTensorData<float>();
        const float* new_v = new_v_value.GetTensorData<float>();
        const int64_t* indices = index_value.GetTensorData<int64_t>();
        const float k_scale = scale_value.GetTensorData<float>()[0];
        const float* mask = mask_value.GetTensorData<float>();

        const std::array<int64_t, 3> out_shape = {1, query, heads * dim};
        auto out_value = ctx.GetOutput(0, out_shape.data(), out_shape.size());
        float* out = out_value.GetTensorMutableData<float>();

        compute(q, state, new_k, new_v, indices, k_scale, mask, mask_shape,
                out, heads, query, dim, capacity);
    }

private:
    static AttentionSmallShape tensor_shape(const Ort::ConstValue& value) {
        auto info = value.GetTensorTypeAndShapeInfo();
        AttentionSmallShape shape;
        shape.rank = info.GetDimensionsCount();
        if (shape.rank > shape.d.size()) {
            throw std::runtime_error("DecoderAttentionTail unsupported rank");
        }
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
        info.GetDimensions(shape.d.data(), shape.rank);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
        return shape;
    }

    void compute(const float* q, const float* state,
                 const float* new_k, const float* new_v, const int64_t* indices,
                 float k_scale, const float* mask, const AttentionSmallShape& mask_shape,
                 float* out, int64_t heads, int64_t query, int64_t dim,
                 int64_t capacity) {
        scores_.resize(static_cast<size_t>(query * capacity));
        update_scores_.resize(static_cast<size_t>(query * query));

        const int q_i = static_cast<int>(query);
        const int c_i = static_cast<int>(capacity);
        const int d_i = static_cast<int>(dim);
        const int64_t head_cache_stride = capacity * dim;
        const int64_t value_offset = heads * head_cache_stride;

        for (int64_t h = 0; h < heads; ++h) {
            const float* q_head = q + h * query * dim;
            const float* k_head = state + h * head_cache_stride;
            const float* v_head = state + value_offset + h * head_cache_stride;
            const float* new_k_head = new_k + h * query * dim;
            const float* new_v_head = new_v + h * query * dim;

            ptt_gemm_nt(q_i, c_i, d_i, k_scale,
                        q_head, d_i, k_head, d_i,
                        0.0f, scores_.data(), c_i);

            ptt_gemm_nt(q_i, q_i, d_i, k_scale,
                        q_head, d_i, new_k_head, d_i,
                        0.0f, update_scores_.data(), q_i);

            for (int64_t qi = 0; qi < query; ++qi) {
                float* row = scores_.data() + qi * capacity;
                for (int64_t j = 0; j < query; ++j) {
                    const int64_t pos = update_pos(indices, h, j, query, dim, capacity);
                    row[pos] = update_scores_[static_cast<size_t>(qi * query + j)];
                }
                softmax_row(row, mask, mask_shape, qi, h, capacity);
            }

            float* out_head = out + h * dim;
            ptt_gemm_nn(q_i, d_i, c_i, 1.0f,
                        scores_.data(), c_i, v_head, d_i,
                        0.0f, out_head, static_cast<int>(heads * dim));

            for (int64_t qi = 0; qi < query; ++qi) {
                float* dst = out + qi * heads * dim + h * dim;
                const float* row = scores_.data() + qi * capacity;
                for (int64_t j = 0; j < query; ++j) {
                    const int64_t pos = update_pos(indices, h, j, query, dim, capacity);
                    const float weight = row[pos];
                    const float* old_v = v_head + pos * dim;
                    const float* repl_v = new_v_head + j * dim;
                    for (int64_t d = 0; d < dim; ++d) {
                        dst[d] += weight * (repl_v[d] - old_v[d]);
                    }
                }
            }
        }
    }

    static int64_t update_pos(const int64_t* indices, int64_t h, int64_t q,
                              int64_t query, int64_t dim, int64_t capacity) {
        const int64_t pos = indices[(h * query + q) * dim];
        if (pos < 0 || pos >= capacity) {
            throw std::runtime_error("DecoderAttentionTail update index out of range");
        }
        return pos;
    }

    static void softmax_row(float* row, const float* mask, const AttentionSmallShape& mask_shape,
                            int64_t qi, int64_t h, int64_t capacity) {
        const int n = static_cast<int>(capacity);
        if (mask_shape.rank == 1) {
            ptt_vadd(row, mask, n);
        } else if (mask_shape.rank == 2) {
            const int64_t q_index = mask_shape.d[0] == 1 ? 0 : qi;
            ptt_vadd(row, mask + q_index * mask_shape.d[1], n);
        } else if (mask_shape.rank == 4 && mask_shape.d[1] == 1) {
            const int64_t q_index = mask_shape.d[2] == 1 ? 0 : qi;
            ptt_vadd(row, mask + q_index * mask_shape.d[3], n);
        } else {
            for (int64_t t = 0; t < capacity; ++t) row[t] += mask_at(mask, mask_shape, qi, h, t);
        }

        const float max_score = ptt_maxv(row, n);
        if (!std::isfinite(max_score)) {
            std::fill(row, row + capacity, 0.0f);
            return;
        }
        ptt_vsadd(row, -max_score, n);
        ptt_expv(row, n);
        const float denom = ptt_sum(row, n);
        if (!std::isfinite(denom) || denom <= 0.0f) {
            std::fill(row, row + capacity, 0.0f);
            return;
        }
        ptt_scale(row, 1.0f / denom, n);
    }

    static float mask_at(const float* mask, const AttentionSmallShape& shape,
                         int64_t qi, int64_t h, int64_t t) {
        if (shape.rank == 1) return mask[t];
        if (shape.rank == 2) {
            const int64_t q_index = shape.d[0] == 1 ? 0 : qi;
            return mask[q_index * shape.d[1] + t];
        }
        if (shape.rank == 4) {
            const int64_t h_index = shape.d[1] == 1 ? 0 : h;
            const int64_t q_index = shape.d[2] == 1 ? 0 : qi;
            return mask[((h_index * shape.d[2] + q_index) * shape.d[3]) + t];
        }
        throw std::runtime_error("DecoderAttentionTail unsupported mask shape");
    }

    std::vector<float> scores_;
    std::vector<float> update_scores_;
};

class DecoderConvTransposeOverlapKernel {
public:
    DecoderConvTransposeOverlapKernel(const OrtApi&, const OrtKernelInfo*) {
        x_packed_.reserve(131072);
        first_.reserve(131072);
        tail_.reserve(131072);
    }

    void Compute(OrtKernelContext* raw_context) {
        Ort::KernelContext ctx(raw_context);
        auto x_value = ctx.GetInput(0);
        auto w_first_value = ctx.GetInput(1);
        auto w_tail_value = ctx.GetInput(2);
        auto bias_value = ctx.GetInput(3);
        auto state_value = ctx.GetInput(4);

        const AttentionSmallShape x_shape = tensor_shape(x_value);
        const AttentionSmallShape w_first_shape = tensor_shape(w_first_value);
        const AttentionSmallShape w_tail_shape = tensor_shape(w_tail_value);
        const AttentionSmallShape bias_shape = tensor_shape(bias_value);
        const AttentionSmallShape state_shape = tensor_shape(state_value);
        if (x_shape.rank != 3 || w_first_shape.rank != 3 || w_tail_shape.rank != 3 ||
            state_shape.rank != 3 || bias_shape.rank != 1) {
            throw std::runtime_error("DecoderConvTransposeOverlap expects x [1,C,T], weights [S,C,O], bias [O], state [1,O,S]");
        }

        const int64_t cin = x_shape.d[1];
        const int64_t time = x_shape.d[2];
        const int64_t stride = state_shape.d[2];
        const int64_t cout = state_shape.d[1];
        if (x_shape.d[0] != 1 || state_shape.d[0] != 1 ||
            w_first_shape.d[0] != stride || w_tail_shape.d[0] != stride ||
            w_first_shape.d[1] != cin || w_tail_shape.d[1] != cin ||
            w_first_shape.d[2] != cout || w_tail_shape.d[2] != cout ||
            bias_shape.d[0] != cout || time <= 0 || stride <= 0 || cin <= 0 || cout <= 0) {
            throw std::runtime_error("DecoderConvTransposeOverlap input shape mismatch");
        }

        const int64_t out_len = time * stride;
        const std::array<int64_t, 3> out_shape = {1, cout, out_len};
        const std::array<int64_t, 3> state_out_shape = {1, cout, stride};
        auto out_value = ctx.GetOutput(0, out_shape.data(), out_shape.size());
        auto state_out_value = ctx.GetOutput(1, state_out_shape.data(), state_out_shape.size());

        compute(x_value.GetTensorData<float>(),
                w_first_value.GetTensorData<float>(),
                w_tail_value.GetTensorData<float>(),
                bias_value.GetTensorData<float>(),
                state_value.GetTensorData<float>(),
                out_value.GetTensorMutableData<float>(),
                state_out_value.GetTensorMutableData<float>(),
                cin, time, stride, cout);
    }

private:
    static AttentionSmallShape tensor_shape(const Ort::ConstValue& value) {
        auto info = value.GetTensorTypeAndShapeInfo();
        AttentionSmallShape shape;
        shape.rank = info.GetDimensionsCount();
        if (shape.rank > shape.d.size()) {
            throw std::runtime_error("DecoderConvTransposeOverlap unsupported rank");
        }
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
        info.GetDimensions(shape.d.data(), shape.rank);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
        return shape;
    }

    void compute(const float* x, const float* w_first, const float* w_tail,
                 const float* bias, const float* state, float* out, float* out_state,
                 int64_t cin, int64_t time, int64_t stride, int64_t cout) {
        const size_t x_count = static_cast<size_t>(time * cin);
        const size_t tmp_count = static_cast<size_t>(time * cout);
        x_packed_.resize(x_count);
        first_.resize(tmp_count);
        tail_.resize(tmp_count);

        for (int64_t t = 0; t < time; ++t) {
            float* dst = x_packed_.data() + t * cin;
            for (int64_t c = 0; c < cin; ++c) {
                dst[c] = x[c * time + t];
            }
        }

        const int time_i = static_cast<int>(time);
        const int cin_i = static_cast<int>(cin);
        const int cout_i = static_cast<int>(cout);
        const int64_t out_len = time * stride;
        const int64_t phase_stride = cin * cout;

        for (int64_t r = 0; r < stride; ++r) {
            const float* wf = w_first + r * phase_stride;
            const float* wt = w_tail + r * phase_stride;

            ptt_gemm_nn(time_i, cout_i, cin_i, 1.0f,
                        x_packed_.data(), cin_i, wf, cout_i,
                        0.0f, first_.data(), cout_i);
            ptt_gemm_nn(time_i, cout_i, cin_i, 1.0f,
                        x_packed_.data(), cin_i, wt, cout_i,
                        0.0f, tail_.data(), cout_i);

            for (int64_t c = 0; c < cout; ++c) {
                float* out_c = out + c * out_len + r;
                const float b = bias[c];
                out_c[0] = b + first_[c] + state[c * stride + r];
                for (int64_t t = 1; t < time; ++t) {
                    out_c[t * stride] = b + first_[t * cout + c] + tail_[(t - 1) * cout + c];
                }
                out_state[c * stride + r] = tail_[(time - 1) * cout + c];
            }
        }
    }

    std::vector<float> x_packed_;
    std::vector<float> first_;
    std::vector<float> tail_;
};

// ── AccelConv ────────────────────────────────────────────────────────────────
// Stride-1 / dilation-1 / group-1 Conv1d as K accumulated sgemm calls through
// Accelerate. Weights are pre-packed offline to [K, Cout, Cin] so each kernel
// tap is a contiguous [Cout, Cin] matrix; the input tap is a [Cin, Tout]
// submatrix of x addressed with ldb = Tin (no im2col, no repacking at runtime).
// Optionally fuses the following Elu (alpha=1).

class AccelConvKernel {
public:
    AccelConvKernel(const OrtApi&, const OrtKernelInfo*, bool fuse_elu)
        : fuse_elu_(fuse_elu) {
        elu_tmp_.reserve(32768);
    }

    void Compute(OrtKernelContext* raw_context) {
        Ort::KernelContext ctx(raw_context);
        auto x_value = ctx.GetInput(0);
        auto w_value = ctx.GetInput(1);
        auto b_value = ctx.GetInput(2);

        const AttentionSmallShape x_shape = tensor_shape(x_value);
        const AttentionSmallShape w_shape = tensor_shape(w_value);
        const AttentionSmallShape b_shape = tensor_shape(b_value);
        if (x_shape.rank != 3 || w_shape.rank != 3 || b_shape.rank != 1) {
            throw std::runtime_error("AccelConv expects x [1,Cin,Tin], w [K,Cout,Cin], bias [Cout]");
        }

        const int64_t cin = x_shape.d[1];
        const int64_t tin = x_shape.d[2];
        const int64_t kernel = w_shape.d[0];
        const int64_t cout = w_shape.d[1];
        const int64_t tout = tin - kernel + 1;
        if (x_shape.d[0] != 1 || w_shape.d[2] != cin || b_shape.d[0] != cout ||
            cin <= 0 || cout <= 0 || kernel <= 0 || tout <= 0) {
            throw std::runtime_error("AccelConv input shape mismatch");
        }

        const float* x = x_value.GetTensorData<float>();
        const float* w = w_value.GetTensorData<float>();
        const float* bias = b_value.GetTensorData<float>();

        const std::array<int64_t, 3> out_shape = {1, cout, tout};
        auto out_value = ctx.GetOutput(0, out_shape.data(), out_shape.size());
        float* out = out_value.GetTensorMutableData<float>();

        for (int64_t c = 0; c < cout; ++c) {
            std::fill(out + c * tout, out + (c + 1) * tout, bias[c]);
        }

        const int cout_i = static_cast<int>(cout);
        const int tout_i = static_cast<int>(tout);
        const int cin_i = static_cast<int>(cin);
        const int tin_i = static_cast<int>(tin);
        for (int64_t k = 0; k < kernel; ++k) {
            ptt_gemm_nn(cout_i, tout_i, cin_i, 1.0f,
                        w + k * cout * cin, cin_i,
                        x + k, tin_i,
                        1.0f, out, tout_i);
        }

        if (fuse_elu_) {
            // elu(x) = max(x,0) + exp(min(x,0)) - 1, alpha = 1
            const int64_t n = cout * tout;
            elu_tmp_.resize(static_cast<size_t>(n));
            ptt_elu_inplace(out, elu_tmp_.data(), static_cast<int>(n));
        }
    }

private:
    static AttentionSmallShape tensor_shape(const Ort::ConstValue& value) {
        auto info = value.GetTensorTypeAndShapeInfo();
        AttentionSmallShape shape;
        shape.rank = info.GetDimensionsCount();
        if (shape.rank > shape.d.size()) {
            throw std::runtime_error("AccelConv unsupported rank");
        }
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
        info.GetDimensions(shape.d.data(), shape.rank);
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
        return shape;
    }

    bool fuse_elu_;
    std::vector<float> elu_tmp_;
};

struct AttentionTailOp : Ort::CustomOpBase<AttentionTailOp, AttentionTailKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new AttentionTailKernel(api, info);
    }
    const char* GetName() const { return "AttentionTail"; }
    const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
    size_t GetInputTypeCount() const { return 7; }
    ONNXTensorElementDataType GetInputType(size_t index) const {
        return index == 4 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
};

struct DecoderAttentionTailOp : Ort::CustomOpBase<DecoderAttentionTailOp, DecoderAttentionTailKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new DecoderAttentionTailKernel(api, info);
    }
    const char* GetName() const { return "DecoderAttentionTail"; }
    const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
    size_t GetInputTypeCount() const { return 7; }
    ONNXTensorElementDataType GetInputType(size_t index) const {
        return index == 4 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
};

struct DecoderConvTransposeOverlapOp : Ort::CustomOpBase<DecoderConvTransposeOverlapOp, DecoderConvTransposeOverlapKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new DecoderConvTransposeOverlapKernel(api, info);
    }
    const char* GetName() const { return "DecoderConvTransposeOverlap"; }
    const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
    size_t GetInputTypeCount() const { return 5; }
    ONNXTensorElementDataType GetInputType(size_t) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    size_t GetOutputTypeCount() const { return 2; }
    ONNXTensorElementDataType GetOutputType(size_t) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
};

struct AccelConvOp : Ort::CustomOpBase<AccelConvOp, AccelConvKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new AccelConvKernel(api, info, false);
    }
    const char* GetName() const { return "AccelConv"; }
    const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
    size_t GetInputTypeCount() const { return 3; }
    ONNXTensorElementDataType GetInputType(size_t) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
};

struct AccelConvEluOp : Ort::CustomOpBase<AccelConvEluOp, AccelConvKernel> {
    void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
        return new AccelConvKernel(api, info, true);
    }
    const char* GetName() const { return "AccelConvElu"; }
    const char* GetExecutionProviderType() const { return "CPUExecutionProvider"; }
    size_t GetInputTypeCount() const { return 3; }
    ONNXTensorElementDataType GetInputType(size_t) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetOutputType(size_t) const {
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
};

static Ort::CustomOpDomain& custom_attention_domain() {
    static AttentionTailOp op;
    static DecoderAttentionTailOp decoder_op;
    static DecoderConvTransposeOverlapOp decoder_convtr_op;
    static AccelConvOp accel_conv_op;
    static AccelConvEluOp accel_conv_elu_op;
    static Ort::CustomOpDomain* domain = [] {
        auto* d = new Ort::CustomOpDomain("pockettts");
        d->Add(&op);
        d->Add(&decoder_op);
        d->Add(&decoder_convtr_op);
        d->Add(&accel_conv_op);
        d->Add(&accel_conv_elu_op);
        return d;
    }();
    return *domain;
}

static void register_custom_attention_ops(Ort::SessionOptions& opts) {
    opts.Add(custom_attention_domain());
}

// Attention ops pay off wherever they compile; the conv ops need
// Accelerate-class GEMM throughput (AMX) to beat ORT's MLAS convs, so the
// engine only selects the conv-fused decoder models on Apple.
static constexpr bool kCustomAttentionAvailable = true;
#if defined(__APPLE__) && !defined(PTT_FORCE_PORTABLE)
static constexpr bool kAccelConvAvailable = true;
#else
static constexpr bool kAccelConvAvailable = false;
#endif

#else

#define PTT_HAVE_CUSTOM_OPS 0

static void register_custom_attention_ops(Ort::SessionOptions&) {}

static constexpr bool kCustomAttentionAvailable = false;
static constexpr bool kAccelConvAvailable = false;

#endif
