#pragma once
// TUI — FTXUI terminal UI mirroring cwdaemon's layout
// Replaces Audio tab with SDR tab, adds CNN tab

#include <string>
#include <cstdint>
#include <mutex>
#include <functional>
#include <ftxui/component/component.hpp>
#include <ftxui/component/screen_interactive.hpp>

class Tui {
public:
    Tui();
    void run();

    // Set initial config values from CLI args (call before run())
    static void set_initial_config(double freq_hz, uint32_t lna_gain, uint32_t vga_gain,
                                   const std::string& model_path);

    static void add_decoded_char(const std::string& s);
    static void set_decoded_text(const std::string& s);
    static void update_metrics(float wpm, float snr, float cnn_conf = 0.0f,
                               const std::string& cnn_class = "");
    static void update_sdr_info(double freq_hz, bool locked);

    // Callback for live parameter changes
    using ConfigChangeCallback = std::function<void(const std::string& key, const std::string& value)>;
    static void set_config_change_callback(ConfigChangeCallback cb);

private:
    ftxui::ScreenInteractive screen_;

    // Global state for rendering
    static std::string decoded_text_;
    static float current_wpm_;
    static float current_snr_;
    static float cnn_confidence_;
    static std::string cnn_class_;
    static double sdr_frequency_;
    static bool sdr_locked_;
    static std::mutex tui_mutex_;
    static ConfigChangeCallback config_change_cb_;

    // Initial config from CLI
    static std::string init_freq_str_;
    static std::string init_lna_str_;
    static std::string init_vga_str_;
    static std::string init_model_path_;
};
