#include "tui.hpp"
#include <ftxui/dom/elements.hpp>
#include <ftxui/component/component_options.hpp>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <thread>
#include <atomic>

using namespace ftxui;

std::string Tui::decoded_text_ = "";
float Tui::current_wpm_ = 0.0f;
float Tui::current_snr_ = 0.0f;
float Tui::cnn_confidence_ = 0.0f;
std::string Tui::cnn_class_ = "";
double Tui::sdr_frequency_ = 0.0;
bool Tui::sdr_locked_ = false;
std::mutex Tui::tui_mutex_;
Tui::ConfigChangeCallback Tui::config_change_cb_;

// Initial config defaults (overridden by set_initial_config)
std::string Tui::init_freq_str_ = "7035030";
std::string Tui::init_lna_str_ = "16";
std::string Tui::init_vga_str_ = "20";
std::string Tui::init_model_path_ = "models/crnn_finetuned.onnx";

static std::string freq_to_band(double freq_hz) {
    double mhz = freq_hz / 1e6;
    if (mhz >= 1.8 && mhz <= 2.0)   return "160m";
    if (mhz >= 3.5 && mhz <= 4.0)   return "80m";
    if (mhz >= 5.3 && mhz <= 5.4)   return "60m";
    if (mhz >= 7.0 && mhz <= 7.3)   return "40m";
    if (mhz >= 10.1 && mhz <= 10.15) return "30m";
    if (mhz >= 14.0 && mhz <= 14.35) return "20m";
    if (mhz >= 18.068 && mhz <= 18.168) return "17m";
    if (mhz >= 21.0 && mhz <= 21.45) return "15m";
    if (mhz >= 24.89 && mhz <= 24.99) return "12m";
    if (mhz >= 28.0 && mhz <= 29.7) return "10m";
    if (mhz >= 50.0 && mhz <= 54.0) return "6m";
    return "";
}

static std::string format_freq_mhz(double freq_hz) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.6f", freq_hz / 1e6);
    return std::string(buf);
}

Tui::Tui() : screen_(ScreenInteractive::Fullscreen()) {}

void Tui::set_initial_config(double freq_hz, uint32_t lna_gain, uint32_t vga_gain,
                              const std::string& model_path) {
    init_freq_str_ = std::to_string(static_cast<long long>(freq_hz));
    init_lna_str_ = std::to_string(lna_gain);
    init_vga_str_ = std::to_string(vga_gain);
    init_model_path_ = model_path;
}

void Tui::add_decoded_char(const std::string& s) {
    std::lock_guard<std::mutex> lock(tui_mutex_);
    decoded_text_ += s;
    if (decoded_text_.length() > 2000) {
        decoded_text_ = decoded_text_.substr(decoded_text_.length() - 2000);
    }
}

void Tui::set_decoded_text(const std::string& s) {
    std::lock_guard<std::mutex> lock(tui_mutex_);
    decoded_text_ = s;
}

void Tui::update_metrics(float wpm, float snr, float cnn_conf,
                          const std::string& cnn_class) {
    std::lock_guard<std::mutex> lock(tui_mutex_);
    current_wpm_ = wpm;
    current_snr_ = snr;
    cnn_confidence_ = cnn_conf;
    cnn_class_ = cnn_class;
}

void Tui::update_sdr_info(double freq_hz, bool locked) {
    std::lock_guard<std::mutex> lock(tui_mutex_);
    sdr_frequency_ = freq_hz;
    sdr_locked_ = locked;
}

void Tui::set_config_change_callback(ConfigChangeCallback cb) {
    config_change_cb_ = cb;
}

void Tui::run() {
    // SDR config fields — initialized from CLI args via set_initial_config()
    std::string freq_str = init_freq_str_;
    std::string lna_gain_str = init_lna_str_;
    std::string vga_gain_str = init_vga_str_;
    std::string model_path_str = init_model_path_;

    // Track the "active" (applied) values so we can show what's live
    std::string active_freq = freq_str;
    std::string active_lna = lna_gain_str;
    std::string active_vga = vga_gain_str;

    // Tab selection
    int tab_selected = 0;
    std::vector<std::string> tab_labels = {" SDR ", " CNN ", " Decode "};

    // Status
    std::string status_msg = "Listening on " + freq_str + " Hz";

    // SDR tab components
    auto freq_input = Input(&freq_str, "Frequency Hz");
    auto lna_input = Input(&lna_gain_str, "LNA dB");
    auto vga_input = Input(&vga_gain_str, "VGA dB");

    auto sdr_container = Container::Vertical({
        freq_input,
        lna_input,
        vga_input,
    });

    // CNN tab (read-only display, no interactive components needed)
    auto cnn_container = Container::Vertical({});

    // Tab switching
    auto tab_toggle = Toggle(&tab_labels, &tab_selected);
    auto tab_content = Container::Tab({
        sdr_container,
        cnn_container,
        cnn_container, // placeholder for Decode tab
    }, &tab_selected);

    // =====================================================================
    // Buttons — Apply sends changes live to HackRF + DSP
    // =====================================================================
    auto apply_btn = Button("⚡ Apply", [&] {
        if (config_change_cb_) {
            // Send each changed parameter
            bool changed = false;
            if (freq_str != active_freq) {
                config_change_cb_("frequency", freq_str);
                active_freq = freq_str;
                changed = true;
            }
            if (lna_gain_str != active_lna) {
                config_change_cb_("lna_gain", lna_gain_str);
                active_lna = lna_gain_str;
                changed = true;
            }
            if (vga_gain_str != active_vga) {
                config_change_cb_("vga_gain", vga_gain_str);
                active_vga = vga_gain_str;
                changed = true;
            }

            if (changed) {
                // Format frequency for display
                try {
                    double f = std::stod(freq_str);
                    status_msg = "Applied: " + format_freq_mhz(f) + " MHz";
                } catch (...) {
                    status_msg = "Applied settings";
                }
            } else {
                status_msg = "No changes to apply";
            }
        } else {
            status_msg = "ERROR: No config callback";
        }
    });

    auto clear_btn = Button("Clear", [&] {
        std::lock_guard<std::mutex> lock(tui_mutex_);
        decoded_text_.clear();
        status_msg = "Decode buffer cleared";
    });

    auto save_btn = Button("Save", [&] {
        std::lock_guard<std::mutex> lock(tui_mutex_);
        FILE* f = fopen("/tmp/cwneural_decode.txt", "w");
        if (f) {
            fprintf(f, "%s\n", decoded_text_.c_str());
            fclose(f);
            status_msg = "Saved to /tmp/cwneural_decode.txt";
            
            // Try to copy to clipboard via xclip if available
            std::string cmd = "printf '%s' \"" + decoded_text_ + "\" | xclip -selection clipboard 2>/dev/null || " +
                              "printf '%s' \"" + decoded_text_ + "\" | wl-copy 2>/dev/null";
            if (system(cmd.c_str()) == 0) {
                status_msg += " (Copied to clipboard!)";
            }
        } else {
            status_msg = "Failed to open /tmp/cwneural_decode.txt";
        }
    });

    auto buttons = Container::Horizontal({apply_btn, clear_btn, save_btn});

    auto main_container = Container::Vertical({
        tab_toggle,
        tab_content,
        buttons,
    });

    // =====================================================================
    // Renderer
    // =====================================================================
    auto renderer = Renderer(main_container, [&] {
        std::lock_guard<std::mutex> lock(tui_mutex_);

        // Decoded text panel
        auto text_panel = window(
            text(" ▶ Decoded CW Text (Neural) ") | bold,
            paragraph(decoded_text_) | flex
        ) | flex;

        // Metrics bar (matches cwdaemon's format)
        float snr_pct = std::clamp(current_snr_ / 100.0f, 0.0f, 1.0f);
        Color snr_color = (snr_pct > 0.5f) ? Color::Green
                        : (snr_pct > 0.2f) ? Color::Yellow
                        : Color::Red;

        std::string cnn_str = " CNN:" + cnn_class_;
        if (cnn_confidence_ > 0.0f) {
            char buf[16];
            snprintf(buf, sizeof(buf), "(%.2f)", cnn_confidence_);
            cnn_str += buf;
        }

        auto metrics_bar = hbox({
            text(" WPM: ") | bold,
            text(std::to_string(static_cast<int>(current_wpm_))) | bold | color(Color::Cyan),
            separator(),
            text(" SNR: ") | bold,
            gauge(snr_pct) | color(snr_color) | size(WIDTH, EQUAL, 15),
            text(" " + std::to_string(static_cast<int>(current_snr_)) + " "),
            separator(),
            text(cnn_str) | bold | color(cnn_confidence_ > 0.7f ? Color::Green : Color::Yellow),
            separator(),
            // Show the active frequency (what's actually tuned)
            (sdr_frequency_ > 0
                ? hbox({
                    text(" " + format_freq_mhz(sdr_frequency_) + " ") | bold | color(Color::Magenta),
                    text(freq_to_band(sdr_frequency_)) | bold | color(Color::Yellow),
                  })
                : text(" No SDR ") | dim),
            filler(),
            text(status_msg) | dim,
        }) | border;

        // Config panel
        Element tab_body;
        switch (tab_selected) {
            case 0: { // SDR
                // Show whether each field has been modified from the active value
                bool freq_dirty = (freq_str != active_freq);
                bool lna_dirty = (lna_gain_str != active_lna);
                bool vga_dirty = (vga_gain_str != active_vga);

                tab_body = vbox({
                    hbox({text("Target Freq (Hz): ") | bold,
                          freq_input->Render() | size(WIDTH, EQUAL, 12),
                          freq_dirty ? text(" *") | color(Color::Yellow) : text("")}),
                    hbox({text("LNA Gain (dB):    ") | bold,
                          lna_input->Render() | size(WIDTH, EQUAL, 6),
                          lna_dirty ? text(" *") | color(Color::Yellow) : text("")}),
                    hbox({text("VGA Gain (dB):    ") | bold,
                          vga_input->Render() | size(WIDTH, EQUAL, 6),
                          vga_dirty ? text(" *") | color(Color::Yellow) : text("")}),
                    separator(),
                    // Active status
                    hbox({
                        text("● Active: ") | bold | color(Color::Green),
                        text(format_freq_mhz(sdr_frequency_) + " MHz") | color(Color::Green),
                    }),
                    hbox({
                        text("  Band:   ") | dim,
                        text(freq_to_band(sdr_frequency_)) | bold | color(Color::Yellow),
                    }),
                    hbox({
                        text("  LNA:    ") | dim,
                        text(active_lna + " dB") | dim,
                        text("  VGA: ") | dim,
                        text(active_vga + " dB") | dim,
                    }),
                    separator(),
                    (freq_dirty || lna_dirty || vga_dirty)
                        ? text("Press Apply to update ▲") | color(Color::Yellow)
                        : text("Settings are current ✓") | color(Color::Green) | dim,
                });
                break;
            }
            case 1: { // CNN
                char conf_buf[16];
                snprintf(conf_buf, sizeof(conf_buf), "%.2f", cnn_confidence_);
                tab_body = vbox({
                    hbox({text("Model: ") | bold, text(model_path_str) | dim}),
                    separator(),
                    hbox({text("Class:      ") | bold,
                          text(cnn_class_) | bold | color(Color::Green)}),
                    hbox({text("CW Conf:    ") | bold,
                          text(conf_buf) | bold | color(cnn_confidence_ > 0.7f ? Color::Green : Color::Red)}),
                });
                break;
            }
            case 2: { // Decode
                char wpm_buf[16];
                snprintf(wpm_buf, sizeof(wpm_buf), "%.0f", current_wpm_);
                tab_body = vbox({
                    hbox({text("WPM:     ") | bold, text(wpm_buf) | color(Color::Cyan)}),
                    hbox({text("Decoder: ") | bold, text("Viterbi") | dim}),
                    separator(),
                    hbox({text("Chars:   ") | bold,
                          text(std::to_string(decoded_text_.size())) | dim}),
                });
                break;
            }
        }

        auto config_panel = window(
            text(" ⚙ Configuration ") | bold,
            vbox({
                tab_toggle->Render() | center,
                separator(),
                tab_body | flex,
                separator(),
                buttons->Render() | center,
            })
        ) | size(WIDTH, EQUAL, 40);

        return vbox({
            hbox({
                text_panel,
                config_panel,
            }) | flex,
            metrics_bar,
        });
    });

    // Periodic refresh (100ms)
    std::atomic<bool> ui_running{true};
    std::thread ui_thread([&] {
        while (ui_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            screen_.PostEvent(Event::Custom);
        }
    });

    screen_.Loop(renderer);

    ui_running = false;
    if (ui_thread.joinable()) ui_thread.join();
}
