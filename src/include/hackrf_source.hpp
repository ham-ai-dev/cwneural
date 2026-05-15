#pragma once
// HackRF SDR input source — replaces cwdaemon's AudioCapture
// Reads raw IQ samples from HackRF One via libhackrf C API.

#include <complex>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <functional>
#include <libhackrf/hackrf.h>

// Simple lock-free-ish ring buffer for complex IQ samples
template<typename T>
class RingBuffer {
public:
    explicit RingBuffer(size_t capacity)
        : buf_(capacity), head_(0), tail_(0), capacity_(capacity) {}

    bool push(const T& item) {
        size_t next = (head_ + 1) % capacity_;
        if (next == tail_.load(std::memory_order_acquire))
            return false; // full
        buf_[head_] = item;
        head_ = next;
        return true;
    }

    bool pop(T& item) {
        if (tail_.load(std::memory_order_acquire) == head_)
            return false; // empty
        item = buf_[tail_];
        tail_.store((tail_ + 1) % capacity_, std::memory_order_release);
        return true;
    }

    size_t size() const {
        size_t h = head_;
        size_t t = tail_.load(std::memory_order_acquire);
        return (h >= t) ? (h - t) : (capacity_ - t + h);
    }

private:
    std::vector<T> buf_;
    size_t head_;
    std::atomic<size_t> tail_;
    size_t capacity_;
};

class HackRFSource {
public:
    HackRFSource(std::shared_ptr<RingBuffer<std::complex<float>>> ring_buf);
    ~HackRFSource();

    bool init(double center_freq_hz, uint32_t sample_rate,
              uint32_t lna_gain, uint32_t vga_gain);
    bool start();
    void stop();

    // Live tuning from TUI
    void set_frequency(double freq_hz);
    void set_lna_gain(uint32_t gain_db);
    void set_vga_gain(uint32_t gain_db);

    double get_frequency() const { return center_freq_; }
    uint32_t get_sample_rate() const { return sample_rate_; }
    bool is_running() const { return running_.load(); }

private:
    static int rx_callback(hackrf_transfer* transfer);

    hackrf_device* device_ = nullptr;
    std::shared_ptr<RingBuffer<std::complex<float>>> ring_buf_;
    double center_freq_ = 6965000.0;
    uint32_t sample_rate_ = 2000000;
    uint32_t lna_gain_ = 16;
    uint32_t vga_gain_ = 20;
    std::atomic<bool> running_{false};
};
