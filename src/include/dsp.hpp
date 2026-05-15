#pragma once
// DSP frontend — frequency shift, FIR decimate, envelope extraction
// Replaces cwdaemon's Goertzel + fldigi bandpass pipeline.

#include <complex>
#include <vector>
#include <cstdint>
#include <cmath>

class NeuralDsp {
public:
    NeuralDsp(uint32_t sdr_sample_rate, double center_freq, double target_freq,
              float output_sample_rate = 4000.0f);

    // Feed raw IQ samples from HackRF. Internally frequency-shifts and decimates.
    // When a full 2048-sample chunk is ready, returns true and fills `out`.
    bool process_iq(const std::complex<float>* samples, int count,
                    std::vector<std::complex<float>>& chunk_out);

    // Extract CW envelope from a baseband chunk
    static std::vector<float> extract_envelope(const std::complex<float>* baseband,
                                                int len, float sample_rate);

    void set_target_freq(double freq_hz);
    double get_target_freq() const { return target_freq_; }
    double get_tracked_offset() const { return freq_offset_; }
    float get_output_rate() const { return output_rate_; }

    // Search for the actual carrier peak within search_bw_hz of target.
    // Call this on the first few seconds of accumulated IQ to lock on.
    // Returns the adjusted freq_offset_ used from now on.
    double auto_track_carrier(const std::complex<float>* samples, int count,
                              double search_bw_hz = 10000.0);

private:
    void design_lpf(int num_taps, float cutoff);

    uint32_t sdr_rate_;
    double center_freq_;
    double target_freq_;
    float output_rate_;
    int decimation_factor_;

    // NCO state
    double phase_acc_ = 0.0;
    double freq_offset_ = 0.0;
    bool carrier_locked_ = false;

    // FIR lowpass filter
    std::vector<float> fir_taps_;
    std::vector<std::complex<float>> fir_delay_;
    int fir_idx_ = 0;

    // Decimation counter
    int dec_counter_ = 0;

    // Accumulation buffer for 2048-sample chunks
    std::vector<std::complex<float>> chunk_buf_;
    static constexpr int CHUNK_SIZE = 2048;
};
