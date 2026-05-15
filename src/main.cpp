#include <iostream>
#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <csignal>
#include <complex>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>

#include "hackrf_source.hpp"
#include "dsp.hpp"
#include "cnn_classifier.hpp"
#include "morse_decoder.hpp"
#include "tui.hpp"

std::atomic<bool> global_running{true};

void signal_handler(int) { global_running = false; }

// =========================================================================
// Benchmark accuracy comparison (identical output format to cwdaemon)
// =========================================================================
static void print_benchmark_results(const std::string& decoded, const std::string& expected_file,
                                    float final_wpm, float cnn_conf) {
    std::ifstream ef(expected_file);
    if (!ef.is_open()) {
        std::cerr << "ERROR: Cannot open expected file: " << expected_file << std::endl;
        return;
    }

    std::string expected;
    std::string line;
    while (std::getline(ef, line)) {
        if (line.substr(0, 9) == "EXPECTED:") {
            expected = line.substr(9);
            if (!expected.empty() && expected[0] == ' ') expected = expected.substr(1);
            break;
        }
    }
    if (expected.empty()) {
        ef.clear(); ef.seekg(0);
        std::ostringstream ss;
        ss << ef.rdbuf();
        expected = ss.str();
        while (!expected.empty() && (expected.back() == '\n' || expected.back() == '\r'))
            expected.pop_back();
    }

    // Levenshtein distance
    int n = decoded.size(), m = expected.size();
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));
    for (int i = 0; i <= n; i++) dp[i][0] = i;
    for (int j = 0; j <= m; j++) dp[0][j] = j;
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++) {
            int cost = (decoded[i-1] == expected[j-1]) ? 0 : 1;
            dp[i][j] = std::min({dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost});
        }
    int edit_distance = dp[n][m];
    int max_len = std::max(n, m);
    float accuracy = (max_len > 0) ? 100.0f * (1.0f - (float)edit_distance / max_len) : 100.0f;

    // Word accuracy
    auto split_words = [](const std::string& s) {
        std::vector<std::string> words;
        std::istringstream iss(s);
        std::string w;
        while (iss >> w) words.push_back(w);
        return words;
    };
    auto dec_words = split_words(decoded);
    auto exp_words = split_words(expected);
    int correct_words = 0, total_words = exp_words.size();
    for (size_t i = 0; i < std::min(dec_words.size(), exp_words.size()); i++)
        if (dec_words[i] == exp_words[i]) correct_words++;
    float word_accuracy = (total_words > 0) ? 100.0f * correct_words / total_words : 0.0f;

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║              CWNEURAL BENCHMARK RESULTS                     ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║ DECODED:  " << decoded.substr(0, 50)
              << std::string(std::max(0, 50 - (int)decoded.size()), ' ') << "║" << std::endl;
    std::cout << "║ EXPECTED: " << expected.substr(0, 50)
              << std::string(std::max(0, 50 - (int)expected.size()), ' ') << "║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════════════════╣" << std::endl;

    char acc_buf[64], wacc_buf[64], conf_buf[64];
    snprintf(acc_buf, sizeof(acc_buf), "%.1f%%", accuracy);
    snprintf(wacc_buf, sizeof(wacc_buf), "%.1f%%", word_accuracy);
    snprintf(conf_buf, sizeof(conf_buf), "%.2f", cnn_conf);

    std::cout << "║ Character Accuracy:  " << acc_buf
              << " (" << (max_len - edit_distance) << "/" << max_len << " correct)"
              << std::string(std::max(0, 25 - (int)strlen(acc_buf)), ' ') << "║" << std::endl;
    std::cout << "║ Word Accuracy:       " << wacc_buf
              << " (" << correct_words << "/" << total_words << " words)"
              << std::string(std::max(0, 27 - (int)strlen(wacc_buf)), ' ') << "║" << std::endl;
    std::cout << "║ CNN Confidence:      " << conf_buf
              << std::string(std::max(0, 38 - (int)strlen(conf_buf)), ' ') << "║" << std::endl;
    std::cout << "║ Final WPM:           " << (int)final_wpm
              << std::string(std::max(0, 38 - (int)std::to_string((int)final_wpm).size()), ' ') << "║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
}

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]" << std::endl;
    std::cerr << "  --tui                    Launch terminal UI" << std::endl;
    std::cerr << "  --freq <Hz>              Target CW frequency (default: 7035030)" << std::endl;
    std::cerr << "  --center-freq <Hz>       SDR center frequency (default: 6965000)" << std::endl;
    std::cerr << "  --lna-gain <dB>          HackRF LNA gain (default: 16)" << std::endl;
    std::cerr << "  --vga-gain <dB>          HackRF VGA gain (default: 20)" << std::endl;
    std::cerr << "  --model <path>           ONNX model path (default: models/crnn_finetuned.onnx)" << std::endl;
    std::cerr << "  --iq <file.raw>          Decode from raw IQ file (benchmark mode)" << std::endl;
    std::cerr << "  --expected <file.txt>    Compare against expected text" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Benchmark example:" << std::endl;
    std::cerr << "  " << prog << " --iq capture.raw --freq 7035030 --expected decode.txt" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Live decode:" << std::endl;
    std::cerr << "  " << prog << " --tui --freq 7035030" << std::endl;
}

int main(int argc, char** argv) {
    bool tui_mode = false;
    double target_freq = 7035030.0;
    double center_freq = 6965000.0;
    uint32_t lna_gain = 16;
    uint32_t vga_gain = 20;
    uint32_t sample_rate = 2000000;
    std::string model_path = "models/crnn_finetuned.onnx";
    std::string iq_file = "";
    std::string expected_file = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--tui") tui_mode = true;
        else if (arg == "--freq" && i+1 < argc) target_freq = std::stod(argv[++i]);
        else if (arg == "--center-freq" && i+1 < argc) center_freq = std::stod(argv[++i]);
        else if (arg == "--lna-gain" && i+1 < argc) lna_gain = std::stoi(argv[++i]);
        else if (arg == "--vga-gain" && i+1 < argc) vga_gain = std::stoi(argv[++i]);
        else if (arg == "--model" && i+1 < argc) model_path = argv[++i];
        else if (arg == "--iq" && i+1 < argc) iq_file = argv[++i];
        else if (arg == "--expected" && i+1 < argc) expected_file = argv[++i];
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
    }

    bool benchmark_mode = !iq_file.empty();

    // Auto-calculate center frequency if not explicitly set
    // Offset the target by 100kHz to avoid HackRF DC spike
    bool center_freq_explicit = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--center-freq") { center_freq_explicit = true; break; }
    }
    if (!center_freq_explicit) {
        center_freq = target_freq - 100000.0;  // 100kHz offset to dodge DC
        std::cerr << "Auto center freq: " << center_freq / 1e6 << " MHz "
                  << "(target " << target_freq / 1e6 << " MHz, offset +"
                  << (target_freq - center_freq) / 1e3 << " kHz)" << std::endl;
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Load CNN model
    CNNClassifier classifier(model_path);
    MorseDecoder morse;

    if (benchmark_mode) {
        // ============================================================
        // Benchmark mode: read raw IQ file, classify + decode
        // ============================================================
        std::cerr << "Benchmark mode: " << iq_file << " → " << target_freq << " Hz" << std::endl;

        std::ifstream f(iq_file, std::ios::binary | std::ios::ate);
        if (!f.is_open()) {
            std::cerr << "ERROR: Cannot open " << iq_file << std::endl;
            return 1;
        }
        size_t file_size = f.tellg();
        f.seekg(0);
        std::vector<int8_t> raw(file_size);
        f.read(reinterpret_cast<char*>(raw.data()), file_size);

        // Convert int8 IQ to complex float
        size_t num_samples = file_size / 2;
        std::vector<std::complex<float>> iq(num_samples);
        for (size_t i = 0; i < num_samples; i++) {
            iq[i] = std::complex<float>(raw[2*i] / 128.0f, raw[2*i+1] / 128.0f);
        }

        std::cerr << "  Loaded " << num_samples << " samples ("
                  << num_samples / (float)sample_rate << "s)" << std::endl;

        // DSP: freq shift + decimate — accumulate all baseband
        NeuralDsp dsp(sample_rate, center_freq, target_freq);

        std::vector<std::complex<float>> all_baseband;
        float last_cnn_conf = 0.0f;
        int cw_chunks = 0, total_chunks = 0;

        // Process all IQ through DSP, accumulate baseband chunks
        int batch_size = 65536;
        for (size_t offset = 0; offset < num_samples; offset += batch_size) {
            int count = std::min(batch_size, static_cast<int>(num_samples - offset));
            std::vector<std::complex<float>> chunk;

            if (dsp.process_iq(iq.data() + offset, count, chunk)) {
                total_chunks++;
                // CNN classify each chunk
                auto result = classifier.classify(chunk.data(), chunk.size());
                last_cnn_conf = std::max(last_cnn_conf, result.cw_confidence);

                if (result.is_cw) {
                    cw_chunks++;
                    // Accumulate CW-classified baseband for bulk decode
                    all_baseband.insert(all_baseband.end(), chunk.begin(), chunk.end());
                }
            }
        }

        std::cerr << "  CNN: " << cw_chunks << "/" << total_chunks
                  << " chunks classified as CW (conf=" << last_cnn_conf << ")" << std::endl;

        // Decode the entire accumulated baseband at once (proper timing recovery)
        std::string full_decode;
        if (!all_baseband.empty()) {
            auto envelope = NeuralDsp::extract_envelope(
                all_baseband.data(), all_baseband.size(), dsp.get_output_rate());
            full_decode = morse.decode(
                envelope.data(), envelope.size(), dsp.get_output_rate());
        }

        // Trim
        while (!full_decode.empty() && full_decode.back() == ' ')
            full_decode.pop_back();

        if (!expected_file.empty()) {
            print_benchmark_results(full_decode, expected_file,
                                     morse.get_wpm(), last_cnn_conf);
        } else {
            std::cout << "\nDecoded: " << full_decode << std::endl;
            std::cout << "WPM: " << morse.get_wpm() << std::endl;
        }
        return 0;
    }

    // ============================================================
    // Live mode: HackRF → CNN → Morse decode
    // ============================================================
    auto ring_buf = std::make_shared<RingBuffer<std::complex<float>>>(1 << 20);
    HackRFSource hackrf(ring_buf);

    if (!hackrf.init(center_freq, sample_rate, lna_gain, vga_gain)) {
        std::cerr << "Failed to initialize HackRF" << std::endl;
        return 1;
    }

    NeuralDsp dsp(sample_rate, center_freq, target_freq);

    // Wire TUI config changes
    Tui::set_config_change_callback([&](const std::string& key, const std::string& value) {
        if (key == "frequency") {
            try {
                double f = std::stod(value);
                dsp.set_target_freq(f);
                Tui::update_sdr_info(f, true);
                std::cerr << "SDR: Target freq set to " << f / 1e6 << " MHz" << std::endl;
            } catch (...) {}
        } else if (key == "lna_gain") {
            try {
                uint32_t g = std::stoi(value);
                hackrf.set_lna_gain(g);
                std::cerr << "SDR: LNA gain set to " << g << " dB" << std::endl;
            } catch (...) {}
        } else if (key == "vga_gain") {
            try {
                uint32_t g = std::stoi(value);
                hackrf.set_vga_gain(g);
                std::cerr << "SDR: VGA gain set to " << g << " dB" << std::endl;
            } catch (...) {}
        }
    });

    // Initialize TUI with CLI values so fields show correctly at launch
    Tui::set_initial_config(target_freq, lna_gain, vga_gain, model_path);
    Tui::update_sdr_info(target_freq, true);

    // Start HackRF
    if (!hackrf.start()) return 1;

    // Decoder thread — accumulates baseband into a sliding window
    std::thread decoder_thread([&]() {
        std::vector<std::complex<float>> iq_batch(8192);
        std::vector<std::complex<float>> chunk;

        // --- Phase 1: Carrier lock (first 2 seconds of raw IQ) ---
        // Accumulate raw IQ before decimation to FFT-search for the actual carrier
        std::vector<std::complex<float>> carrier_lock_buf;
        carrier_lock_buf.reserve(static_cast<int>(sample_rate * 2));
        bool carrier_locked = false;

        while (global_running && !carrier_locked) {
            int got = 0;
            for (int i = 0; i < 8192; i++) {
                if (ring_buf->pop(iq_batch[i])) got++;
                else break;
            }
            if (got == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            carrier_lock_buf.insert(carrier_lock_buf.end(), iq_batch.begin(), iq_batch.begin() + got);
            // Feed through DSP so it stays in sync
            dsp.process_iq(iq_batch.data(), got, chunk);

            if (static_cast<int>(carrier_lock_buf.size()) >= static_cast<int>(sample_rate * 2)) {
                double locked_offset = dsp.auto_track_carrier(carrier_lock_buf.data(), carrier_lock_buf.size(), 10000.0);
                double locked_freq = center_freq + locked_offset;
                std::cerr << "[Lock] Carrier found at " << locked_freq / 1e6 << " MHz"
                          << " (offset " << locked_offset - (target_freq - center_freq) << " Hz from target)\n";
                Tui::update_sdr_info(locked_freq, true);
                carrier_locked = true;
            }
        }

        // --- Phase 2: Decode with locked carrier ---
        // Accumulate up to 60 seconds of baseband.
        // Decode the entire buffer every 1 second to eliminate boundary corruption.
        float out_rate = dsp.get_output_rate();
        int max_window = static_cast<int>(out_rate * 60.0f);   // 60s max
        int advance_step = static_cast<int>(out_rate * 1.0f);  // update every 1s
        
        std::vector<std::complex<float>> accum_buf;
        accum_buf.reserve(max_window + advance_step);

        int samples_since_decode = 0;
        std::string last_stdout_text_;  // tracks what was last printed to stdout

        while (global_running) {
            // Drain ring buffer in batches
            int got = 0;
            for (int i = 0; i < 8192; i++) {
                if (ring_buf->pop(iq_batch[i])) got++;
                else break;
            }
            if (got == 0) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
            }

            if (dsp.process_iq(iq_batch.data(), got, chunk)) {
                // CNN classify each 2048-sample chunk
                auto result = classifier.classify(chunk.data(), chunk.size());

                // Track CNN confidence with slow EMA
                static float cnn_conf_ema = 0.5f;
                cnn_conf_ema = 0.9f * cnn_conf_ema + 0.1f * result.cw_confidence;

                Tui::update_metrics(morse.get_wpm(), result.cw_confidence * 100.0f,
                                    result.cw_confidence, result.class_name);

                // Always accumulate to preserve timing gaps (silence is meaningful).
                // But if a chunk is very clearly NOT CW (confidence < 0.2 AND EMA also
                // low), replace with silence rather than noise to protect the Viterbi
                // timing estimator from garbage signal segments.
                if (!result.is_cw && result.cw_confidence < 0.2f && cnn_conf_ema < 0.4f) {
                    // Insert silent chunk (zeros) to preserve the time gap without noise
                    std::vector<std::complex<float>> silence(chunk.size(), {0.0f, 0.0f});
                    accum_buf.insert(accum_buf.end(), silence.begin(), silence.end());
                } else {
                    accum_buf.insert(accum_buf.end(), chunk.begin(), chunk.end());
                }
                samples_since_decode += chunk.size();

                // Decode every 1 second
                if (samples_since_decode >= advance_step) {
                    samples_since_decode = 0;
                    
                    auto envelope = NeuralDsp::extract_envelope(
                        accum_buf.data(), accum_buf.size(), out_rate);
                    std::string text = morse.decode(
                        envelope.data(), envelope.size(), out_rate);

                    if (!text.empty() && text.size() >= 2) {
                        if (tui_mode) {
                            // TUI: always show the rolling 60s window as-is
                            Tui::set_decoded_text(text);
                        } else {
                            bool window_full = (static_cast<int>(accum_buf.size()) >= max_window);

                            if (last_stdout_text_.empty()) {
                                // First decode — print everything on one line
                                std::cout << text << std::flush;
                                last_stdout_text_ = text;
                            } else if (!window_full) {
                                // Window still growing: text always starts with previous text.
                                // Only print the newly added suffix.
                                if (text.size() > last_stdout_text_.size() &&
                                    text.substr(0, last_stdout_text_.size()) == last_stdout_text_) {
                                    std::cout << text.substr(last_stdout_text_.size()) << std::flush;
                                    last_stdout_text_ = text;
                                } else if (text != last_stdout_text_) {
                                    // Re-estimation changed earlier chars — print corrected line
                                    std::cout << "\n" << text << std::flush;
                                    last_stdout_text_ = text;
                                }
                            } else {
                                // Window is at max (60s). Each 1s update drops chars from the
                                // front and adds new ones to the back.
                                // Emit a newline per second so text streams forward cleanly.
                                if (text != last_stdout_text_) {
                                    // Find what's new at the end
                                    size_t common = 0;
                                    size_t min_len = std::min(text.size(), last_stdout_text_.size());
                                    for (size_t i = 0; i < min_len; i++) {
                                        if (text[i] == last_stdout_text_[i]) common = i + 1;
                                        else break;
                                    }
                                    // If most of the text changed (window slid, old chars dropped),
                                    // start a fresh line
                                    if (common < last_stdout_text_.size() / 2) {
                                        std::cout << "\n" << text << std::flush;
                                    } else {
                                        // Just append the new tail
                                        std::cout << text.substr(common) << std::flush;
                                    }
                                    last_stdout_text_ = text;
                                }
                            }
                        }
                    }

                    // Keep only up to 60 seconds
                    if (static_cast<int>(accum_buf.size()) > max_window) {
                        int excess = accum_buf.size() - max_window;
                        accum_buf.erase(accum_buf.begin(), accum_buf.begin() + excess);
                    }
                }
            }
        }
    });

    if (tui_mode) {
        Tui tui;
        tui.run();
        global_running = false;
    } else {
        std::cerr << "cwneural: Live decode on " << target_freq / 1e6
                  << " MHz (Ctrl+C to stop)" << std::endl;
        while (global_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    decoder_thread.join();
    hackrf.stop();

    return 0;
}
