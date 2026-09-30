#pragma once

#include "intellibranch/common.hpp"
#include "intellibranch/tokenizer.hpp"
#include <string>
#include <vector>
#include <memory>
#include <tuple>
#include <span>

namespace intellibranch {

class InferenceModel {
public:
    InferenceModel(
        Header header,
        std::vector<std::string> labels,
        std::vector<std::string> vocab,
        std::vector<MergeRule> merge_rules,
        Weights weights
    );

    // Forward pass returning allocated probabilities
    std::vector<float> forward(const std::vector<uint32_t>& token_ids, float temperature = 0.0f) const;

    // Stack-only top-2 predictions and entropy
    StaticInferenceResult predict_slots(const std::vector<uint32_t>& token_ids, float temperature = 0.0f) const;

    // Fast top prediction on token IDs
    std::pair<std::string, double> predict_tokens(const std::vector<uint32_t>& token_ids) const;

    // Detailed prediction from text (returns result, unk_ratio, and error_status)
    std::pair<StaticInferenceResult, double> predict_detailed(const std::string& text) const;

    // End-to-end text prediction with safety guards and OOV penalty
    std::pair<std::string, double> predict(const std::string& text) const;

    // Zero-allocation feature extraction into external buffers
    void predict_features(
        const std::vector<uint32_t>& token_ids,
        float* out_pooled,
        float* out_logits
    ) const;

    const Header& header() const { return header_; }
    const std::vector<std::string>& labels() const { return labels_; }
    const std::vector<std::string>& vocab() const { return vocab_; }
    const std::vector<MergeRule>& merge_rules() const { return merge_rules_; }
    const Weights& weights() const { return weights_; }
    Weights& mutable_weights() { return weights_; }
    const BPETokenizer& tokenizer() const { return *tokenizer_; }
    std::shared_ptr<BPETokenizer> tokenizer_ptr() const { return tokenizer_; }

    float temperature() const { return temperature_; }
    void set_temperature(float temp) { temperature_ = temp; }

private:
    void forward_internal(
        const uint32_t* tokens,
        size_t seq_len,
        float temperature,
        float* pooled,
        float* hidden,
        float* logits,
        float* probs
    ) const;

    Header header_;
    std::vector<std::string> labels_;
    std::vector<std::string> vocab_;
    std::vector<MergeRule> merge_rules_;
    Weights weights_;
    float temperature_ = 1.0f;
    std::shared_ptr<BPETokenizer> tokenizer_;
};

} // namespace intellibranch
