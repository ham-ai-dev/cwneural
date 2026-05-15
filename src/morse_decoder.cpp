#include "morse_decoder.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>

MorseDecoder::MorseDecoder() {}

const std::map<std::string, char>& MorseDecoder::morse_table() {
    static const std::map<std::string, char> table = {
        {".-", 'A'}, {"-...", 'B'}, {"-.-.", 'C'}, {"-..", 'D'}, {".", 'E'},
        {"..-.", 'F'}, {"--.", 'G'}, {"....", 'H'}, {"..", 'I'}, {".---", 'J'},
        {"-.-", 'K'}, {".-..", 'L'}, {"--", 'M'}, {"-.", 'N'}, {"---", 'O'},
        {".--.", 'P'}, {"--.-", 'Q'}, {".-.", 'R'}, {"...", 'S'}, {"-", 'T'},
        {"..-", 'U'}, {"...-", 'V'}, {".--", 'W'}, {"-..-", 'X'}, {"-.--", 'Y'},
        {"--..", 'Z'},
        {".----", '1'}, {"..---", '2'}, {"...--", '3'}, {"....-", '4'},
        {".....", '5'}, {"-....", '6'}, {"--...", '7'}, {"---..", '8'},
        {"----.", '9'}, {"-----", '0'},
        {".-.-.-", '.'}, {"--..--", ','}, {"..--..", '?'}, {".----.", '\''},
        {"-..-.", '/'}, {"-....-", '-'}, {"-...-", '='},
    };
    return table;
}

float MorseDecoder::otsu_threshold(const float* env, int len) {
    // Collect valid (non-zero) values
    std::vector<float> valid;
    valid.reserve(len);
    for (int i = 0; i < len; i++) {
        if (env[i] > 0.0f) valid.push_back(env[i]);
    }
    if (valid.empty()) return 0.0f;

    // Trim bottom 10th percentile — these are noise-floor samples that bias
    // Otsu toward a too-low threshold on weak signals.
    std::sort(valid.begin(), valid.end());
    size_t trim_lo = valid.size() / 10;
    float min_v = valid[trim_lo];
    float max_v = valid.back();
    if (max_v - min_v < 1e-10f) return max_v * 0.5f;

    // Build histogram on trimmed range
    constexpr int NBINS = 256;
    std::vector<int> hist(NBINS, 0);
    for (float v : valid) {
        if (v < min_v) continue;
        int bin = std::clamp(static_cast<int>((v - min_v) / (max_v - min_v) * (NBINS - 1)), 0, NBINS - 1);
        hist[bin]++;
    }

    float total = 0.0f;
    for (int c : hist) total += c;
    float sum_total = 0.0f;
    for (int i = 0; i < NBINS; i++) {
        float mid = min_v + (max_v - min_v) * (i + 0.5f) / NBINS;
        sum_total += hist[i] * mid;
    }

    float best_thresh = 0.0f;
    float best_var = 0.0f;
    float sum_bg = 0.0f;
    float weight_bg = 0.0f;

    for (int i = 0; i < NBINS; i++) {
        float mid = min_v + (max_v - min_v) * (i + 0.5f) / NBINS;
        weight_bg += hist[i];
        if (weight_bg == 0) continue;
        float weight_fg = total - weight_bg;
        if (weight_fg == 0) break;
        sum_bg += hist[i] * mid;
        float mean_bg = sum_bg / weight_bg;
        float mean_fg = (sum_total - sum_bg) / weight_fg;
        float var = weight_bg * weight_fg * (mean_bg - mean_fg) * (mean_bg - mean_fg);
        if (var > best_var) {
            best_var = var;
            best_thresh = mid;
        }
    }
    return best_thresh;
}

void MorseDecoder::keying_runs(const float* env, int len, float sample_rate,
                                float threshold,
                                std::vector<float>& on_durs,
                                std::vector<float>& off_durs) {
    on_durs.clear();
    off_durs.clear();

    float high_t = threshold * 1.1f;
    float low_t = threshold * 0.7f;
    float ms_per_sample = 1000.0f / sample_rate;

    bool in_tone = false;
    int run_start = 0;

    for (int i = 0; i < len; i++) {
        if (!in_tone && env[i] > high_t) {
            off_durs.push_back((i - run_start) * ms_per_sample);
            run_start = i;
            in_tone = true;
        } else if (in_tone && env[i] < low_t) {
            on_durs.push_back((i - run_start) * ms_per_sample);
            run_start = i;
            in_tone = false;
        }
    }
    // Flush
    float final_dur = (len - run_start) * ms_per_sample;
    if (in_tone) on_durs.push_back(final_dur);

    // Remove first off-duration (before first tone)
    if (!off_durs.empty()) off_durs.erase(off_durs.begin());
}

void MorseDecoder::estimate_dit_dah(const std::vector<float>& on_durs,
                                     float& dit_ms, float& dah_ms,
                                     float& split_val, float& wpm) {
    // Use tracked WPM to compute minimum valid element duration.
    // At the tracked WPM, a dit = 1200/WPM ms. Reject anything < 40% of that.
    float min_element_ms = 5.0f;
    if (wpm_ > 0) min_element_ms = std::max(5.0f, 1200.0f / wpm_ * 0.40f);

    if (on_durs.size() < 3) {
        // Not enough data — hold tracked WPM if available
        if (wpm_ > 0) {
            dit_ms = 1200.0f / wpm_;
            dah_ms = dit_ms * 3.0f;
            split_val = dit_ms * 2.0f;
            wpm = wpm_;
        } else {
            dit_ms = 60.0f; dah_ms = 180.0f; split_val = 120.0f; wpm = 20.0f;
        }
        return;
    }

    // Filter noise spikes using adaptive minimum
    std::vector<float> sorted;
    for (float d : on_durs) {
        if (d >= min_element_ms) sorted.push_back(d);
    }
    if (sorted.size() < 3) {
        if (wpm_ > 0) {
            dit_ms = 1200.0f / wpm_;
            dah_ms = dit_ms * 3.0f;
            split_val = dit_ms * 2.0f;
            wpm = wpm_;
        } else {
            dit_ms = 60.0f; dah_ms = 180.0f; split_val = 120.0f; wpm = 20.0f;
        }
        return;
    }

    std::sort(sorted.begin(), sorted.end());

    // Find largest gap
    float max_gap = 0.0f;
    int split_idx = 0;
    for (size_t i = 0; i < sorted.size() - 1; i++) {
        float gap = sorted[i + 1] - sorted[i];
        if (gap > max_gap) {
            max_gap = gap;
            split_idx = i;
        }
    }
    split_val = (sorted[split_idx] + sorted[split_idx + 1]) / 2.0f;

    // Compute medians
    std::vector<float> dits, dahs;
    for (float d : on_durs) {
        if (d < 5.0f) continue;
        if (d < split_val) dits.push_back(d);
        else dahs.push_back(d);
    }

    if (!dits.empty()) {
        std::sort(dits.begin(), dits.end());
        dit_ms = dits[dits.size() / 2];
    } else {
        dit_ms = 60.0f;
    }

    if (!dahs.empty()) {
        std::sort(dahs.begin(), dahs.end());
        dah_ms = dahs[dahs.size() / 2];
    } else {
        dah_ms = dit_ms * 3.0f;
    }

    float new_wpm = (dit_ms > 0) ? 1200.0f / dit_ms : 20.0f;

    // WPM exponential moving average — smooth across decode windows.
    // Only update if the new estimate is within 50% of the tracked value
    // (prevents garbage windows from resetting the tracker).
    if (wpm_ <= 0.0f) {
        wpm_ = new_wpm;  // first estimate — accept directly
    } else if (new_wpm > wpm_ * 0.5f && new_wpm < wpm_ * 2.0f) {
        wpm_ = 0.85f * wpm_ + 0.15f * new_wpm;  // slow adaptation
    }
    // else: new estimate is wildly different — ignore it, hold previous

    wpm = wpm_;
}

void MorseDecoder::estimate_gaps(const std::vector<float>& off_durs,
                                 float dit_ms, float& letter_gap, float& word_gap) {
    if (off_durs.empty()) {
        letter_gap = dit_ms * 3.0f;
        word_gap = dit_ms * 7.0f;
        return;
    }

    std::vector<float> sorted;
    for (float d : off_durs) {
        if (d >= 5.0f) sorted.push_back(d);
    }
    if (sorted.empty()) {
        letter_gap = dit_ms * 3.0f;
        word_gap = dit_ms * 7.0f;
        return;
    }

    std::sort(sorted.begin(), sorted.end());

    // We expect up to 3 clusters: element gap, letter gap, word gap.
    // To find letter gap and word gap splits, look at the largest jumps in the sorted array.
    // Since element gaps are much more common, they will dominate the lower end.
    
    // Default fallback
    letter_gap = dit_ms * 3.0f;
    word_gap = dit_ms * 7.0f;

    // A simple clustering approach:
    // Let's find gaps > dit_ms * 2.0. These are letter or word gaps.
    std::vector<float> large_gaps;
    for (float d : off_durs) {
        if (d > dit_ms * 2.0f) {
            large_gaps.push_back(d);
        }
    }

    if (large_gaps.empty()) return;

    std::vector<float> sorted_large = large_gaps;
    std::sort(sorted_large.begin(), sorted_large.end());

    // Filter outliers: only keep gaps that have at least one close neighbor
    // (within 20% of their value) to represent a cluster.
    std::vector<float> valid_large;
    for (size_t i = 0; i < sorted_large.size(); i++) {
        bool has_neighbor = false;
        if (i > 0 && sorted_large[i] - sorted_large[i-1] < sorted_large[i] * 0.2f) has_neighbor = true;
        if (i + 1 < sorted_large.size() && sorted_large[i+1] - sorted_large[i] < sorted_large[i] * 0.2f) has_neighbor = true;
        
        if (has_neighbor) {
            valid_large.push_back(sorted_large[i]);
        }
    }

    if (valid_large.empty()) valid_large = sorted_large; // Fallback

    // Now find the largest jump in valid_large
    float max_jump = 0.0f;
    int jump_idx = -1;
    for (size_t i = 0; i + 1 < valid_large.size(); i++) {
        float jump = valid_large[i+1] - valid_large[i];
        if (jump > max_jump) {
            max_jump = jump;
            jump_idx = i;
        }
    }

    if (jump_idx != -1 && valid_large[jump_idx+1] > valid_large[jump_idx] * 1.5f) {
        letter_gap = valid_large.front() * 0.75f;
        if (letter_gap < dit_ms * 2.0f) letter_gap = dit_ms * 2.0f;
        word_gap = (valid_large[jump_idx] + valid_large[jump_idx+1]) / 2.0f;
    } else if (letter_gap_ > 0.0f && word_gap_ > 0.0f) {
        // Fallback to cached values if clustering fails (e.g. short 10s window)
        letter_gap = letter_gap_;
        word_gap = word_gap_;
    } else {
        float median_large = valid_large[valid_large.size() / 2];
        if (median_large > dit_ms * 8.0f) {
            word_gap = median_large * 0.75f;
            letter_gap = dit_ms * 2.5f;
        } else {
            letter_gap = std::max(dit_ms * 2.0f, valid_large.front() * 0.75f);
            word_gap = median_large * 1.5f; 
        }
    }

    // Cache the values for next time
    letter_gap_ = letter_gap;
    word_gap_ = word_gap;
}

std::string MorseDecoder::viterbi_decode(const std::vector<float>& on_durs,
                                          const std::vector<float>& off_durs,
                                          float dit_ms, float dah_ms) {
    float split = (dit_ms + dah_ms) / 2.0f;
    float letter_gap, word_gap;
    estimate_gaps(off_durs, dit_ms, letter_gap, word_gap);

    const auto& table = morse_table();
    std::string decoded;
    std::string current_symbol;

    for (size_t i = 0; i < on_durs.size(); i++) {
        if (on_durs[i] < 5.0f) continue;

        if (on_durs[i] < split)
            current_symbol += '.';
        else
            current_symbol += '-';

        if (i < off_durs.size()) {
            if (off_durs[i] >= word_gap) {
                auto it = table.find(current_symbol);
                decoded += (it != table.end()) ? it->second : '?';
                decoded += ' ';
                current_symbol.clear();
            } else if (off_durs[i] >= letter_gap) {
                auto it = table.find(current_symbol);
                decoded += (it != table.end()) ? it->second : '?';
                current_symbol.clear();
            }
        } else {
            auto it = table.find(current_symbol);
            decoded += (it != table.end()) ? it->second : '?';
        }
    }

    return decoded;
}

std::string MorseDecoder::decode(const float* envelope, int len, float sample_rate) {
    float thresh = otsu_threshold(envelope, len);
    if (thresh <= 0.0f) return "";

    std::vector<float> on_durs, off_durs;
    keying_runs(envelope, len, sample_rate, thresh, on_durs, off_durs);

    if (on_durs.size() < 3) return "";

    float split_val;
    estimate_dit_dah(on_durs, dit_ms_, dah_ms_, split_val, wpm_);

    return viterbi_decode(on_durs, off_durs, dit_ms_, dah_ms_);
}
