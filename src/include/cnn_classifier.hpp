#pragma once
// CNN Classifier — wraps ONNX Runtime for CW signal classification
// Runs the same model as deepspan's crnn_finetuned.pt (exported to .onnx)

#include <string>
#include <vector>
#include <complex>
#include <array>
#include <onnxruntime_cxx_api.h>

struct ClassifyResult {
    bool is_cw = false;
    float cw_confidence = 0.0f;
    std::string class_name;
    std::array<float, 8> probabilities{}; // CW,FT8,RTTY,SSB,PSK,QRSS,CARRIER,NOISE
};

class CNNClassifier {
public:
    explicit CNNClassifier(const std::string& model_path);

    // Classify a 2048-sample IQ chunk
    ClassifyResult classify(const std::complex<float>* samples, int len);

    bool is_loaded() const { return loaded_; }

    static constexpr int NUM_CLASSES = 8;
    static const char* CLASS_NAMES[NUM_CLASSES];

private:
    Ort::Env env_;
    Ort::SessionOptions session_opts_;
    std::unique_ptr<Ort::Session> session_;
    bool loaded_ = false;

    Ort::AllocatorWithDefaultOptions allocator_;
};
