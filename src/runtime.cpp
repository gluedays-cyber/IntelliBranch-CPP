#include "intellibranch/runtime.hpp"
#include "intellibranch/ops.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace intellibranch {

InferenceModel::InferenceModel(
    Header header,
    std::vector<std::string> labels,
    std::vector<std::string> vocab,
    std::vector<MergeRule> merge_rules,
    Weights weights
) : header_(header),
    labels_(std::move(labels)),
    vocab_(std::move(vocab)),
    merge_rules_(std::move(merge_rules)),
    weights_(std::move(weights)),
    temperature_(1.0f) {
    tokenizer_ = std::make_shared<BPETokenizer>(vocab_, merge_rules_);
}

void InferenceModel::forward_internal(
    const uint32_t* tokens,
    size_t seq_len,
    float temp,
    float* pooled,
    float* hidden,
    float* logits,
    float* probs
) const {
    if (seq_len == 0) {
        throw EmptyInputException();
    }

    size_t emb_dim = header_.embedding_dim;
    size_t hidden_dim = header_.hidden_dim;
    size_t num_classes = header_.num_classes;

    // 1. Mean Pooling with Positional Encoding
    const float* pos_ptr = (!weights_.positional.empty()) ? weights_.positional.data() : nullptr;
    size_t max_pos = (emb_dim > 0 && pos_ptr != nullptr) ? weights_.positional.size() / emb_dim : 0;

    mean_pooling_with_pos(
        tokens, seq_len,
        weights_.embedding.data(), header_.vocab_size,
        pos_ptr, max_pos,
        emb_dim, pooled
    );

    // 2. Layer 1 Linear: pooled [1 x emb_dim] * W1 [emb_dim x hidden_dim] + B1
    matmul_vec_add(pooled, weights_.w1.data(), weights_.b1.data(), emb_dim, hidden_dim, hidden);

    // 3. GELU Activation In-Place
    gelu_in_place(hidden, hidden_dim);

    // 4. Layer 2 Linear: hidden [1 x hidden_dim] * W2 [hidden_dim x num_classes] + B2
    matmul_vec_add(hidden, weights_.w2.data(), weights_.b2.data(), hidden_dim, num_classes, logits);

    // 5. Softmax with Temperature Scaling
    float final_temp = (temp <= 0.0f) ? temperature_ : temp;
    softmax(logits, num_classes, final_temp, probs);
}

std::vector<float> InferenceModel::forward(const std::vector<uint32_t>& token_ids, float temp) const {
    std::vector<float> pooled(header_.embedding_dim);
    std::vector<float> hidden(header_.hidden_dim);
    std::vector<float> logits(header_.num_classes);
    std::vector<float> probs(header_.num_classes);

    forward_internal(
        token_ids.data(), token_ids.size(), temp,
        pooled.data(), hidden.data(), logits.data(), probs.data()
    );

    return probs;
}

StaticInferenceResult InferenceModel::predict_slots(const std::vector<uint32_t>& token_ids, float temp) const {
    std::vector<float> pooled(header_.embedding_dim);
    std::vector<float> hidden(header_.hidden_dim);
    std::vector<float> logits(header_.num_classes);
    std::vector<float> probs(header_.num_classes);

    forward_internal(
        token_ids.data(), token_ids.size(), temp,
        pooled.data(), hidden.data(), logits.data(), probs.data()
    );

    int16_t top1_idx = -1;
    int16_t top2_idx = -1;
    float top1_prob = -1.0f;
    float top2_prob = -1.0f;

    for (size_t i = 0; i < header_.num_classes; ++i) {
        float p = probs[i];
        int16_t idx = static_cast<int16_t>(i);
        if (p > top1_prob) {
            top2_prob = top1_prob;
            top2_idx = top1_idx;
            top1_prob = p;
            top1_idx = idx;
        } else if (p > top2_prob) {
            top2_prob = p;
            top2_idx = idx;
        }
    }

    StaticInferenceResult res;
    if (top1_idx >= 0) {
        res.primary = MatchSlot{top1_idx, top1_prob};
        res.total = 1;
    }
    if (top2_idx >= 0) {
        res.secondary = MatchSlot{top2_idx, top2_prob};
        res.total = 2;
    }
    res.entropy = compute_entropy(probs.data(), header_.num_classes);
    return res;
}

std::pair<std::string, double> InferenceModel::predict_tokens(const std::vector<uint32_t>& token_ids) const {
    auto res = predict_slots(token_ids, temperature_);
    if (res.total == 0 || res.primary.index < 0 || static_cast<size_t>(res.primary.index) >= labels_.size()) {
        throw IntelliBranchException("predicted class index exceeds label count");
    }
    return {labels_[res.primary.index], static_cast<double>(res.primary.confidence)};
}

std::pair<StaticInferenceResult, double> InferenceModel::predict_detailed(const std::string& text) const {
    if (!is_valid_utf8(text)) {
        throw EmptyInputException();
    }
    std::string safe_text = text;
    if (safe_text.size() > MAX_INPUT_BYTES) {
        safe_text = truncate_to_rune_boundary(safe_text, MAX_INPUT_BYTES);
    }

    auto token_ids = tokenizer_->encode(safe_text);
    if (token_ids.empty()) {
        throw EmptyInputException();
    }
    if (token_ids.size() > MAX_SEQUENCE_TOKENS) {
        token_ids.resize(MAX_SEQUENCE_TOKENS);
    }

    auto res = predict_slots(token_ids, temperature_);

    double unk_ratio = 0.0;
    if (tokenizer_->has_unk() && !token_ids.empty()) {
        size_t unk_count = 0;
        uint32_t unk_id = tokenizer_->unk_id();
        for (uint32_t id : token_ids) {
            if (id == unk_id) {
                unk_count++;
            }
        }
        unk_ratio = static_cast<double>(unk_count) / static_cast<double>(token_ids.size());
        float decay = static_cast<float>(1.0 - unk_ratio);
        res.primary.confidence *= decay;
        res.secondary.confidence *= decay;
    }

    return {res, unk_ratio};
}

std::pair<std::string, double> InferenceModel::predict(const std::string& text) const {
    if (!is_valid_utf8(text)) {
        throw EmptyInputException();
    }
    std::string safe_text = text;
    if (safe_text.size() > MAX_INPUT_BYTES) {
        safe_text = truncate_to_rune_boundary(safe_text, MAX_INPUT_BYTES);
    }

    auto token_ids = tokenizer_->encode(safe_text);
    if (token_ids.empty()) {
        throw EmptyInputException();
    }
    if (token_ids.size() > MAX_SEQUENCE_TOKENS) {
        token_ids.resize(MAX_SEQUENCE_TOKENS);
    }

    auto [label, score] = predict_tokens(token_ids);

    if (tokenizer_->has_unk() && !token_ids.empty()) {
        size_t unk_count = 0;
        uint32_t unk_id = tokenizer_->unk_id();
        for (uint32_t id : token_ids) {
            if (id == unk_id) {
                unk_count++;
            }
        }
        double unk_ratio = static_cast<double>(unk_count) / static_cast<double>(token_ids.size());
        score *= (1.0 - unk_ratio);
    }

    return {label, score};
}

void InferenceModel::predict_features(
    const std::vector<uint32_t>& token_ids,
    float* out_pooled,
    float* out_logits
) const {
    std::vector<float> hidden(header_.hidden_dim);
    std::vector<float> probs(header_.num_classes);

    forward_internal(
        token_ids.data(), token_ids.size(), temperature_,
        out_pooled, hidden.data(), out_logits, probs.data()
    );
}

} // namespace intellibranch
