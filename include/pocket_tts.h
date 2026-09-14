/* PocketTTS C API — embed faster-than-realtime TTS + voice cloning.
 *
 * Build the shared library (see CMakeLists.txt) and link, or compile
 * src/pocket_tts.cpp into your project with PTT_SHARED_LIB defined.
 *
 * Audio is always mono float32 PCM at 24000 Hz.
 * All functions are thread-compatible but not thread-safe per handle:
 * drive one engine handle from one thread (streams spawn their own).
 *
 * Minimal usage:
 *
 *   void* tts = ptt_create("models", "voices", "models/tokenizer.model",
 *                          "int8", 0.7f, 1, 0);
 *   void* s = ptt_stream_start(tts, "Hello there.", "example.wav");
 *   float* chunk; int n;
 *   while (ptt_stream_read(s, &chunk, &n) == 1) {   // blocks between chunks
 *       play(chunk, n);
 *       ptt_free_audio(chunk);
 *   }
 *   ptt_stream_end(s);
 *   ptt_destroy(tts);
 *
 * Voice cloning is implicit: pass any .wav/.mp3 in voices_dir as the voice
 * argument — the embedding is computed on first use and cached on disk
 * (voices_dir/.cache). 6–15 seconds of clean speech works best.
 */
#ifndef POCKET_TTS_H
#define POCKET_TTS_H

#ifdef __cplusplus
extern "C" {
#endif

/* ── Engine lifecycle ──────────────────────────────────────────────────── */

/* Create an engine. Returns NULL on failure (details on stderr).
 *   precision:   "int8" (recommended) or "fp32"
 *   temperature: sampling temperature (0 = deterministic, ~0.7 typical)
 *   lsd_steps:   flow-matching steps (1 = fastest, uses the merged model)
 *   num_threads: total thread budget (0 = half the cores)               */
void* ptt_create(const char* models_dir, const char* voices_dir,
                 const char* tokenizer_path, const char* precision,
                 float temperature, int lsd_steps, int num_threads);

/* ptt_create with a flags word for special engine modes:
 *   bit 0: decoder_only  — loads just the Mimi decoder (see ptt_decode)
 *   bit 1: defer_encoder — skip the voice encoder until ptt_load_encoder
 *   bit 2: encoder_only  — loads just the voice encoder (ptt_encode_voice)
 *   bit 3: soura — opt-in steering graphs and models/soura_vectors.npy
 *   bit 4: conditioning_only — AR/text only, prepare caches from existing .emb;
 *          implies defer_encoder. No audio generation/decoding on this handle.
 * decoder_only, encoder_only and conditioning_only are mutually exclusive.
 * soura is compatible with normal/deferred and conditioning-only handles. */
void* ptt_create_ex(const char* models_dir, const char* voices_dir,
                    const char* tokenizer_path, const char* precision,
                    float temperature, int lsd_steps, int num_threads,
                    int flags);

/* ptt_create that also writes ONNX Runtime per-op profile JSON files
 * into profile_dir (flushed when the engine is destroyed).             */
void* ptt_create_profiled(const char* models_dir, const char* voices_dir,
                          const char* tokenizer_path, const char* precision,
                          float temperature, int lsd_steps, int num_threads,
                          const char* profile_dir);

void ptt_destroy(void* handle);

/* Run a dummy generation to warm caches/kernels. Returns elapsed ms.   */
double ptt_warmup(void* handle);

/* ── Runtime configuration ─────────────────────────────────────────────── */

void ptt_set_temperature(void* handle, float temperature);

/* soften=1 (default) replaces , ; : with spaces before synthesis;
 * soften=0 lets the model voice its own clause pauses (better for prose). */
void ptt_set_soften_commas(void* handle, int soften);

/* Max latent frames per decoder batch (default 15; 80ms audio each).   */
void ptt_set_max_chunk(void* handle, int frames);

/* Optional controls: 0 success, -1 invalid input/disabled feature.
 * Call only when no stream is active. Neutral or zero intensity bypasses steering.
 * intensity: finite [0,1.2]. Labels: neutral, angry, disgust, fear, happy, sad.
 * Seed: integer [0,2^53-1], passed as double for JS Number compatibility.
 * RNG is module-global: do not interleave independently seeded handles.
 * Vector replacement validates before changing the current set, then resets neutral.
 * C callers must reset emotion for each new request; omitted controls are not implicit. */
int ptt_set_emotion(void* handle, const char* emotion, double intensity);
int ptt_set_seed(void* handle, double seed);
int ptt_load_soura_vectors(void* handle, const char* path);
/* Build/validate .emb and .kv without generating audio. A conditioning-only
 * handle requires a pre-encoded .emb in voices/.cache. In WASM, run in a
 * disposable worker so conditioning memory is reclaimed on worker termination. */
int ptt_prepare_voice(void* handle, const char* voice);

/* Must be called BEFORE the first ptt_create* on WASM builds: size of the
 * shared ORT thread pool and whether idle workers spin-wait (spin=1).   */
void ptt_configure_pool(int threads, int spin);

/* ── Streaming synthesis ───────────────────────────────────────────────── */

/* Start synthesizing text with a voice (filename inside voices_dir, or a
 * name whose .emb is already cached). Returns a stream context or NULL. */
void* ptt_stream_start(void* handle, const char* text, const char* voice);

/* Blocking read: 1 = wrote a chunk (free with ptt_free_audio),
 * 0 = stream finished. Waits for the next chunk when none is ready.    */
int ptt_stream_read(void* stream_ctx, float** out_samples, int* out_len);

/* Non-blocking variant: additionally returns -1 when nothing is
 * available yet. Use this when the calling thread must not block
 * (e.g. the Emscripten runtime thread).                                 */
int ptt_stream_poll(void* stream_ctx, float** out_samples, int* out_len);

/* Request a graceful abort; generation stops within one decode batch.
 * Does not join or free — follow with ptt_stream_end.                   */
void ptt_stream_stop(void* stream_ctx);

/* Join and free the stream context (aborts first if still running).    */
void ptt_stream_end(void* stream_ctx);

void ptt_free_audio(float* samples);

/* ── Latent streaming (split-pipeline embedders) ───────────────────────
 * AR side only: emits raw 32-float latent frames instead of audio, for
 * consumers that run the Mimi decoder elsewhere (see ptt_decode).       */

void* ptt_latents_start(void* handle, const char* text, const char* voice);

/* 1 = wrote a frame into out_frame[32] (or a boundary marker: flags
 * bit0 = chunk boundary, bit1 = terminal sentence, bit2 = last),
 * 0 = finished, -1 = nothing available yet (non-blocking).              */
int ptt_latents_poll(void* latents_ctx, float* out_frame, int* out_flags);

void ptt_latents_stop(void* latents_ctx);
void ptt_latents_end(void* latents_ctx);

/* Synchronous decode for decoder_only engines: latents[frames*32] in,
 * malloc'd samples out (free with ptt_free_audio). Decoder state
 * persists across calls until ptt_decoder_reset. Returns 0 / -1.        */
int ptt_decode(void* handle, const float* latents, int frames,
               float** out_samples, int* out_len);
void ptt_decoder_reset(void* handle);

/* ── Voice encoding ────────────────────────────────────────────────────── */

/* defer_encoder engines: load the encoder session once the model file
 * exists in models_dir. Returns 0 / -1.                                 */
int ptt_load_encoder(void* handle);

/* Encode a voice file to its cached .emb (voices_dir/.cache/<name>.emb).
 * Works on encoder_only engines. Returns 0 / -1.                        */
int ptt_encode_voice(void* handle, const char* voice);

/* ── Diagnostics ───────────────────────────────────────────────────────── */

void ptt_set_profiling(int enabled);
void ptt_print_profile(void);

/* Isolated component timing: which 0 = decoder chunk of `frames`,
 * 1 = AR step. Returns average ms per run, or -1.                       */
double ptt_debug_bench(void* handle, int which, int frames, int iters);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* POCKET_TTS_H */
