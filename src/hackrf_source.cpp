#include "hackrf_source.hpp"
#include <iostream>
#include <cstring>

HackRFSource::HackRFSource(std::shared_ptr<RingBuffer<std::complex<float>>> ring_buf)
    : ring_buf_(ring_buf) {}

HackRFSource::~HackRFSource() {
    stop();
}

bool HackRFSource::init(double center_freq_hz, uint32_t sample_rate,
                         uint32_t lna_gain, uint32_t vga_gain) {
    center_freq_ = center_freq_hz;
    sample_rate_ = sample_rate;
    lna_gain_ = lna_gain;
    vga_gain_ = vga_gain;

    int result = hackrf_init();
    if (result != HACKRF_SUCCESS) {
        std::cerr << "hackrf_init() failed: " << hackrf_error_name(static_cast<hackrf_error>(result)) << std::endl;
        return false;
    }

    result = hackrf_open(&device_);
    if (result != HACKRF_SUCCESS) {
        std::cerr << "hackrf_open() failed: " << hackrf_error_name(static_cast<hackrf_error>(result)) << std::endl;
        return false;
    }

    hackrf_set_sample_rate(device_, sample_rate_);
    hackrf_set_freq(device_, static_cast<uint64_t>(center_freq_));
    hackrf_set_lna_gain(device_, lna_gain_);
    hackrf_set_vga_gain(device_, vga_gain_);

    std::cerr << "HackRF initialized: " << center_freq_ / 1e6 << " MHz, "
              << sample_rate_ / 1e6 << " Msps, LNA=" << lna_gain_
              << "dB, VGA=" << vga_gain_ << "dB" << std::endl;
    return true;
}

int HackRFSource::rx_callback(hackrf_transfer* transfer) {
    auto* self = static_cast<HackRFSource*>(transfer->rx_ctx);
    if (!self->running_.load()) return -1;

    // HackRF sends int8 I/Q pairs
    int8_t* buf = reinterpret_cast<int8_t*>(transfer->buffer);
    int num_samples = transfer->valid_length / 2;

    for (int i = 0; i < num_samples; i++) {
        float I = static_cast<float>(buf[2 * i]) / 128.0f;
        float Q = static_cast<float>(buf[2 * i + 1]) / 128.0f;
        self->ring_buf_->push(std::complex<float>(I, Q));
    }
    return 0;
}

bool HackRFSource::start() {
    if (!device_) return false;
    running_ = true;
    int result = hackrf_start_rx(device_, rx_callback, this);
    if (result != HACKRF_SUCCESS) {
        std::cerr << "hackrf_start_rx() failed: " << hackrf_error_name(static_cast<hackrf_error>(result)) << std::endl;
        running_ = false;
        return false;
    }
    std::cerr << "HackRF RX started" << std::endl;
    return true;
}

void HackRFSource::stop() {
    running_ = false;
    if (device_) {
        hackrf_stop_rx(device_);
        hackrf_close(device_);
        device_ = nullptr;
    }
    hackrf_exit();
}

void HackRFSource::set_frequency(double freq_hz) {
    center_freq_ = freq_hz;
    if (device_)
        hackrf_set_freq(device_, static_cast<uint64_t>(freq_hz));
}

void HackRFSource::set_lna_gain(uint32_t gain_db) {
    lna_gain_ = gain_db;
    if (device_)
        hackrf_set_lna_gain(device_, gain_db);
}

void HackRFSource::set_vga_gain(uint32_t gain_db) {
    vga_gain_ = gain_db;
    if (device_)
        hackrf_set_vga_gain(device_, gain_db);
}
