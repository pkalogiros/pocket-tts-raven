// Compile with the same includes/libraries as pocket_tts.cpp; do not use NDEBUG.
// Usage: steering-test MODELS VOICES VOICE [neutral-latents.bin]
// PTT_TEST_BASELINE builds this against an unmodified checkout to compare defaults.
#define PTT_SHARED_LIB
#include "../src/pocket_tts.cpp"
#include <cassert>

int main(int argc, char** argv) {
    if (argc < 4) return 2;
    pocket_tts::Config cfg;
    cfg.models_dir = argv[1]; cfg.voices_dir = argv[2];
    cfg.tokenizer_path = cfg.models_dir + "/tokenizer.model";
    cfg.num_threads_ar = 3; cfg.num_threads_dec = 2; cfg.num_threads_full = 2;
    cfg.temperature = .6f;
    auto collect = [&](pocket_tts::PocketTTS& tts) {
        pocket_tts::rng::seed(31);
        std::vector<float> frames;
        bool ok = tts.stream_latents("There you are. It is good to see you again.", argv[3],
            [&](const float* frame, int) {
                if (frame) frames.insert(frames.end(), frame, frame + 32);
                return true;
            });
        assert(ok && !frames.empty());
        assert(std::all_of(frames.begin(), frames.end(), [](float f) { return std::isfinite(f); }));
        return frames;
    };
    std::vector<float> original;
    {
        pocket_tts::PocketTTS tts(cfg);
        original = collect(tts);
#ifndef PTT_TEST_BASELINE
        bool rejected = false;
        try { tts.set_emotion("angry", .8); } catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
#endif
    }
    if (argc > 4) {
        std::ofstream out(argv[4], std::ios::binary);
        out.write(reinterpret_cast<const char*>(original.data()), original.size() * sizeof(float));
    }
#ifndef PTT_TEST_BASELINE
    cfg.soura = true;
    pocket_tts::PocketTTS tts(cfg);
    auto neutral = collect(tts);
    assert(neutral.size() == original.size());
    float worst = 0;
    for (size_t i = 0; i < neutral.size(); ++i) worst = std::max(worst, std::abs(neutral[i] - original[i]));
    assert(worst < 1e-4f);
    tts.set_emotion("angry", .8);
    auto angry = collect(tts);
    assert(angry != neutral);
    assert(angry == collect(tts));
    tts.set_emotion("neutral", 0);
    assert(neutral == collect(tts));
    tts.set_emotion("angry", 0);
    assert(neutral == collect(tts));
    tts.set_emotion("happy", .8);
    tts.stream_latents("This is a longer sentence to interrupt.", argv[3], [](const float*, int) { return false; });
    tts.set_emotion("neutral", 0);
    assert(neutral == collect(tts));
    for (const auto* label : {"angry", "disgust", "fear", "happy", "sad"}) {
        tts.set_emotion(label, .8);
        collect(tts);
    }
    // Controls inside text/nested fields must not be treated as request controls.
    assert(pocket_tts::json_get_number("{\"text\":\"seed\",\"nested\":{\"seed\":4},\"seed\":31}", "seed", 0) == 31);
    for (const auto* raw : {"NaN", "Infinity", "\"0.8\"", "+1", ".8", "01", "0x1", "1.", "1e", "null", "true", "[]"}) {
        bool rejected = false;
        try { pocket_tts::json_get_number(std::string("{\"intensity\":") + raw + "}", "intensity", 0); }
        catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected);
    }
    std::cout << "PASS: neutral equivalence (max error " << worst
              << "), repeatability, steering, disabled rejection, zero bypass, cancellation, six labels, numeric validation\n";
#else
    std::cout << "PASS: original default latents recorded\n";
#endif
}
