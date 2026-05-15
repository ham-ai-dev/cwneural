#include "cnn_classifier.hpp"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <cmath>

const char* CNNClassifier::CLASS_NAMES[NUM_CLASSES] = {
    "CW", "FT8_FT4", "RTTY", "SSB", "PSK", "QRSS", "CARRIER", "NOISE"
};

CNNClassifier::CNNClassifier(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "cwneural")
{
    session_opts_.SetIntraOpNumThreads(1);
    session_opts_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    try {
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_opts_);
        loaded_ = true;
        std::cerr << "CNN model loaded: " << model_path << std::endl;
    } catch (const Ort::Exception& e) {
        std::cerr << "Failed to load CNN model: " << e.what() << std::endl;
        loaded_ = false;
    }
}

ClassifyResult CNNClassifier::classify(const std::complex<float>* samples, int len) {
    ClassifyResult result;
    if (!loaded_ || !session_) {
        result.class_name = "ERROR";
        return result;
    }

    // Prepare input: shape [1, 2, 2048] — split complex into I and Q channels
    std::vector<float> input_data(2 * len);
    for (int i = 0; i < len; i++) {
        input_data[i]       = samples[i].real();  // Channel 0: I
        input_data[len + i] = samples[i].imag();  // Channel 1: Q
    }

    // Create input tensor
    std::array<int64_t, 3> input_shape = {1, 2, static_cast<int64_t>(len)};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size()
    );

    // Run inference
    const char* input_names[] = {"iq"};
    const char* output_names[] = {"logits"};

    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1
    );

    // Parse output logits
    float* logits = output_tensors[0].GetTensorMutableData<float>();

    // Softmax
    float max_logit = *std::max_element(logits, logits + NUM_CLASSES);
    float sum_exp = 0.0f;
    for (int i = 0; i < NUM_CLASSES; i++) {
        result.probabilities[i] = std::exp(logits[i] - max_logit);
        sum_exp += result.probabilities[i];
    }
    for (int i = 0; i < NUM_CLASSES; i++) {
        result.probabilities[i] /= sum_exp;
    }

    // Find argmax
    int best_class = 0;
    float best_prob = 0.0f;
    for (int i = 0; i < NUM_CLASSES; i++) {
        if (result.probabilities[i] > best_prob) {
            best_prob = result.probabilities[i];
            best_class = i;
        }
    }

    result.class_name = CLASS_NAMES[best_class];
    result.is_cw = (best_class == 0); // CW is class 0
    result.cw_confidence = result.probabilities[0];

    return result;
}
