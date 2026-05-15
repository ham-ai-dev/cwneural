#include "dsp.hpp"
#include <algorithm>
#include <numeric>

NeuralDsp::NeuralDsp(uint32_t sdr_sample_rate, double center_freq,
                     double target_freq, float output_sample_rate)
    : sdr_rate_(sdr_sample_rate),
      center_freq_(center_freq),
      target_freq_(target_freq),
      output_rate_(output_sample_rate)
{
    freq_offset_ = target_freq_ - center_freq_;
    decimation_factor_ = std::max(1, static_cast<int>(sdr_rate_ / output_rate_));
    output_rate_ = static_cast<float>(sdr_rate_) / decimation_factor_;

    // Design FIR lowpass filter for anti-aliasing before decimation
    int num_taps = std::min(201, decimation_factor_ * 4 + 1);
    if (num_taps % 2 == 0) num_taps++;
    design_lpf(num_taps, 1.0f / decimation_factor_);

    fir_delay_.resize(fir_taps_.size(), {0.0f, 0.0f});
    chunk_buf_.reserve(CHUNK_SIZE);
}

void NeuralDsp::design_lpf(int num_taps, float cutoff) {
    fir_taps_.resize(num_taps);
    int mid = num_taps / 2;
    float sum = 0.0f;
    for (int i = 0; i < num_taps; i++) {
        float x = static_cast<float>(i - mid);
        float sinc = (std::fabs(x) < 1e-10f)
            ? 2.0f * cutoff
            : std::sin(2.0f * M_PI * cutoff * x) / (M_PI * x);
        // Blackman window
        float win = 0.42f - 0.50f * std::cos(2.0f * M_PI * i / num_taps)
                          + 0.08f * std::cos(4.0f * M_PI * i / num_taps);
        fir_taps_[i] = sinc * win;
        sum += fir_taps_[i];
    }
    // Normalize
    for (auto& t : fir_taps_) t /= sum;
}

bool NeuralDsp::process_iq(const std::complex<float>* samples, int count,
                            std::vector<std::complex<float>>& chunk_out) {
    bool chunk_ready = false;

    for (int i = 0; i < count; i++) {
        // Frequency shift to baseband
        float cos_v = std::cos(phase_acc_);
        float sin_v = std::sin(phase_acc_);
        std::complex<float> nco(cos_v, -sin_v);
        std::complex<float> shifted = samples[i] * nco;

        phase_acc_ += 2.0 * M_PI * freq_offset_ / sdr_rate_;
        if (phase_acc_ > 2.0 * M_PI) phase_acc_ -= 2.0 * M_PI;
        if (phase_acc_ < -2.0 * M_PI) phase_acc_ += 2.0 * M_PI;

        // FIR filter
        fir_delay_[fir_idx_] = shifted;
        fir_idx_ = (fir_idx_ + 1) % fir_taps_.size();

        // Decimate
        if (++dec_counter_ >= decimation_factor_) {
            dec_counter_ = 0;

            // Compute filtered output
            std::complex<float> out(0.0f, 0.0f);
            int idx = fir_idx_;
            for (size_t j = 0; j < fir_taps_.size(); j++) {
                out += fir_taps_[j] * fir_delay_[idx];
                idx = (idx + 1) % fir_taps_.size();
            }

            chunk_buf_.push_back(out);

            if (static_cast<int>(chunk_buf_.size()) >= CHUNK_SIZE) {
                chunk_out = std::move(chunk_buf_);
                chunk_buf_.clear();
                chunk_buf_.reserve(CHUNK_SIZE);
                chunk_ready = true;
            }
        }
    }
    return chunk_ready;
}

std::vector<float> NeuralDsp::extract_envelope(const std::complex<float>* baseband,
                                                int len, float sample_rate) {
    // Step 1: Magnitude envelope
    std::vector<float> envelope(len);
    for (int i = 0; i < len; i++) {
        envelope[i] = std::abs(baseband[i]);
    }

    // Step 2: Low-pass filter at 60 Hz (matches deepspan's Python pipeline)
    // This preserves CW keying up to 60 WPM while rejecting carrier hash
    float lp_cutoff = 60.0f;
    int lp_taps_len = 61;
    std::vector<float> lp_taps(lp_taps_len);
    float fc = lp_cutoff / (sample_rate / 2.0f);
    int mid = lp_taps_len / 2;
    float tap_sum = 0.0f;
    for (int i = 0; i < lp_taps_len; i++) {
        float x = static_cast<float>(i - mid);
        float sinc = (std::fabs(x) < 1e-10f)
            ? 2.0f * fc
            : std::sin(2.0f * M_PI * fc * x) / (M_PI * x);
        // Hamming window
        float win = 0.54f - 0.46f * std::cos(2.0f * M_PI * i / (lp_taps_len - 1));
        lp_taps[i] = sinc * win;
        tap_sum += lp_taps[i];
    }
    for (auto& t : lp_taps) t /= tap_sum;

    // Apply FIR LP filter
    std::vector<float> filtered(len);
    for (int i = 0; i < len; i++) {
        float sum = 0.0f;
        for (int j = 0; j < lp_taps_len; j++) {
            int idx = i - j;
            if (idx >= 0) sum += lp_taps[j] * envelope[idx];
        }
        filtered[i] = sum;
    }

    // Step 3: Median filter to remove impulse noise (8ms window, matches deepspan)
    int med_size = std::max(3, static_cast<int>(sample_rate * 0.008f));
    if (med_size % 2 == 0) med_size++;
    int half = med_size / 2;

    std::vector<float> smoothed(len);
    std::vector<float> window;
    for (int i = 0; i < len; i++) {
        window.clear();
        for (int j = std::max(0, i - half); j <= std::min(len - 1, i + half); j++) {
            window.push_back(filtered[j]);
        }
        std::sort(window.begin(), window.end());
        smoothed[i] = window[window.size() / 2];
    }

    return smoothed;
}

void NeuralDsp::set_target_freq(double freq_hz) {
    target_freq_ = freq_hz;
    freq_offset_ = target_freq_ - center_freq_;
}
