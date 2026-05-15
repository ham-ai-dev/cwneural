#pragma once
// Morse decoder — Otsu threshold + Viterbi-style decoding
// C++ port of deepspan's batch_decode_w1aw.py decode pipeline

#include <string>
#include <vector>
#include <map>

class MorseDecoder {
public:
    MorseDecoder();

    // Decode from envelope signal + sample rate
    std::string decode(const float* envelope, int len, float sample_rate);

    float get_wpm() const { return wpm_; }
    float get_dit_ms() const { return dit_ms_; }
    float get_dah_ms() const { return dah_ms_; }

private:
    // Otsu's method for optimal binary threshold
    float otsu_threshold(const float* env, int len);

    // Extract ON/OFF keying runs
    void keying_runs(const float* env, int len, float sample_rate, float threshold,
                     std::vector<float>& on_durs, std::vector<float>& off_durs);

    // Estimate dit/dah durations from bimodal clustering
    void estimate_dit_dah(const std::vector<float>& on_durs,
                          float& dit_ms, float& dah_ms, float& split_val, float& wpm);

    // Estimate Farnsworth gaps from off_durs
    void estimate_gaps(const std::vector<float>& off_durs,
                       float dit_ms, float& letter_gap, float& word_gap);

    // Viterbi-style Morse decode
    std::string viterbi_decode(const std::vector<float>& on_durs,
                               const std::vector<float>& off_durs,
                               float dit_ms, float dah_ms);

    // Morse lookup table
    static const std::map<std::string, char>& morse_table();

    float wpm_ = 0.0f;
    float dit_ms_ = 0.0f;
    float dah_ms_ = 0.0f;
    float letter_gap_ = 0.0f;
    float word_gap_ = 0.0f;
};
