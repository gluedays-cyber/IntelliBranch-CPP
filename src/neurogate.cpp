#include "intellibranch/neurogate.hpp"
#include "intellibranch/binary.hpp"
#include "intellibranch/ops.hpp"
#include <mutex>
#include <bit>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <iostream>

namespace intellibranch {

namespace {

std::string trim_and_lower_str(const std::string& str) {
    if (str.empty()) return "";
    size_t start = 0;
    while (start < str.size() && (std::isspace(static_cast<unsigned char>(str[start])) != 0)) start++;
    if (start == str.size()) return "";
    size_t end = str.size() - 1;
    while (end > start && (std::isspace(static_cast<unsigned char>(str[end])) != 0)) end--;
    std::string res;
    res.reserve(end - start + 1);
    for (size_t i = start; i <= end; ++i) {
        res.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(str[i]))));
    }
    return res;
}

std::string make_pipeline_key(const std::string& primary, const std::string& secondary) {
    return primary + "->" + secondary;
}

} // namespace

GateRouteBuilder::GateRouteBuilder(NeuroGate* gate, int class_index, std::string label)
    : gate_(gate), class_index_(class_index), label_(std::move(label)) {}

GateRouteBuilder& GateRouteBuilder::with_anchor(float weight, const std::vector<std::string>& keywords) {
    std::unique_lock<std::shared_mutex> lock(gate_->mu_);

    uint64_t mask = 0;
    std::vector<std::string> clean_keywords;
    for (const auto& kw_raw : keywords) {
        std::string kw = trim_and_lower_str(kw_raw);
        if (kw.empty()) continue;
        clean_keywords.push_back(kw);

        uint64_t m = 0;
        auto it = gate_->anchor_dict_.find(kw);
        if (it != gate_->anchor_dict_.end()) {
            m = it->second;
        } else {
            if (gate_->anchor_dict_.size() < 64) {
                m = 1ULL << gate_->anchor_dict_.size();
                gate_->anchor_dict_[kw] = m;
            }
        }
        mask |= m;
    }

    auto model = gate_->model_;
    if (model && model->tokenizer_ptr()) {
        for (const auto& kw : clean_keywords) {
            auto toks = model->tokenizer().encode(kw);
            for (uint32_t tid : toks) {
                gate_->anchor_token_map_[tid] |= mask;
            }
        }
    }

    gate_->anchor_rules_.push_back(AnchorRule{
        .class_index = class_index_,
        .mask = mask,
        .weight = weight,
        .keywords = std::move(clean_keywords),
    });

    return *this;
}

GateRouteBuilder GateRouteBuilder::bind(const std::string& label, RouteAction handler) {
    return gate_->bind(label, std::move(handler));
}

std::shared_ptr<NeuroGate> NeuroGate::create(const std::string& model_path) {
    auto model = load_binary_model(model_path);
    return std::make_shared<NeuroGate>(model);
}

std::shared_ptr<NeuroGate> NeuroGate::create_with_model(std::shared_ptr<InferenceModel> model) {
    return std::make_shared<NeuroGate>(std::move(model));
}

NeuroGate::NeuroGate(std::shared_ptr<InferenceModel> model)
    : model_(std::move(model)), class_count_(0), min_cosine_sim_(0.25f) {
    policy_ = default_dispatch_policy();
    fallback_ = [](void*) {};

    if (model_) {
        for (size_t i = 0; i < model_->labels().size() && i < MAX_GATE_CLASSES; ++i) {
            labels_[i] = model_->labels()[i];
            label_to_index_[labels_[i]] = static_cast<int>(i);
            class_count_++;
        }
        compute_baseline_centroid(*model_);
    }
}

void NeuroGate::compute_baseline_centroid(const InferenceModel& model) {
    size_t emb_dim = model.header().embedding_dim;
    if (emb_dim > MAX_GATE_EMB_DIM || emb_dim == 0 || model.weights().embedding.empty()) {
        return;
    }

    std::array<double, MAX_GATE_EMB_DIM> sum{};
    size_t valid_tokens = 0;
    const auto& vocab = model.vocab();

    for (size_t tok_id = 0; tok_id < vocab.size(); ++tok_id) {
        if (vocab[tok_id] == "[PAD]" || vocab[tok_id] == "[UNK]") {
            continue;
        }
        size_t offset = tok_id * emb_dim;
        if (offset + emb_dim <= model.weights().embedding.size()) {
            for (size_t d = 0; d < emb_dim; ++d) {
                sum[d] += static_cast<double>(model.weights().embedding[offset + d]);
            }
            valid_tokens++;
        }
    }

    if (valid_tokens > 0) {
        double inv = 1.0 / static_cast<double>(valid_tokens);
        std::array<float, MAX_GATE_EMB_DIM> raw{};
        for (size_t d = 0; d < emb_dim; ++d) {
            raw[d] = static_cast<float>(sum[d] * inv);
        }
        l2_normalize(raw.data(), emb_dim, domain_centroid_.data());
        has_centroid_ = true;
    }
}

NeuroGate& NeuroGate::calibrate_domain_centroid(const std::vector<DataSample>& samples) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    if (!model_ || samples.empty()) return *this;

    size_t emb_dim = std::min(static_cast<size_t>(model_->header().embedding_dim), MAX_GATE_EMB_DIM);
    std::array<double, MAX_GATE_EMB_DIM> sum{};
    size_t valid_count = 0;

    std::array<float, MAX_GATE_EMB_DIM> pooled{};
    std::array<float, MAX_GATE_CLASSES> dummy_logits{};

    for (const auto& s : samples) {
        auto tokens = model_->tokenizer().encode(s.text);
        if (tokens.empty()) continue;
        try {
            model_->predict_features(tokens, pooled.data(), dummy_logits.data());
            for (size_t d = 0; d < emb_dim; ++d) {
                sum[d] += static_cast<double>(pooled[d]);
            }
            valid_count++;
        } catch (...) {}
    }

    if (valid_count > 0) {
        double inv = 1.0 / static_cast<double>(valid_count);
        std::array<float, MAX_GATE_EMB_DIM> raw{};
        for (size_t d = 0; d < emb_dim; ++d) {
            raw[d] = static_cast<float>(sum[d] * inv);
        }
        l2_normalize(raw.data(), emb_dim, domain_centroid_.data());
        has_centroid_ = true;
    }
    return *this;
}

NeuroGate& NeuroGate::set_domain_boundary(const std::vector<float>& centroid, float min_cosine) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    size_t dim = std::min(centroid.size(), MAX_GATE_EMB_DIM);
    std::copy_n(centroid.begin(), dim, domain_centroid_.begin());
    l2_normalize(domain_centroid_.data(), dim, domain_centroid_.data());
    has_centroid_ = true;
    min_cosine_sim_ = min_cosine;
    return *this;
}

NeuroGate& NeuroGate::set_min_cosine_sim(float threshold) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    min_cosine_sim_ = threshold;
    return *this;
}

NeuroGate& NeuroGate::set_policy(DispatchPolicy policy) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    policy_ = policy;
    return *this;
}

GateRouteBuilder NeuroGate::bind(const std::string& label, RouteAction handler) {
    std::unique_lock<std::shared_mutex> lock(mu_);

    int idx = -1;
    auto it = label_to_index_.find(label);
    if (it != label_to_index_.end()) {
        idx = it->second;
    } else {
        if (class_count_ >= MAX_GATE_CLASSES) {
            throw IntelliBranchException("neurogate: registered classes exceed MAX_GATE_CLASSES");
        }
        idx = static_cast<int>(class_count_);
        labels_[idx] = label;
        label_to_index_[label] = idx;
        class_count_++;
    }
    routes_[idx] = std::move(handler);

    return GateRouteBuilder(this, idx, label);
}

NeuroGate& NeuroGate::bind_pipeline(const std::string& primary, const std::string& secondary, PipelineAction handler) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    pipelines_[make_pipeline_key(primary, secondary)] = std::move(handler);
    return *this;
}

NeuroGate& NeuroGate::ambiguous(AmbiguousAction handler) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    ambiguous_ = std::move(handler);
    return *this;
}

NeuroGate& NeuroGate::fallback(RouteAction handler) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    fallback_ = std::move(handler);
    return *this;
}

void NeuroGate::swap_model(std::shared_ptr<InferenceModel> new_model) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    model_ = std::move(new_model);
}

std::shared_ptr<InferenceModel> NeuroGate::model() const {
    std::shared_lock<std::shared_mutex> lock(mu_);
    return model_;
}

GateTrace NeuroGate::inspect(const std::string& text) const {
    auto start = std::chrono::steady_clock::now();
    std::shared_lock<std::shared_mutex> lock(mu_);

    if (!model_) {
        return GateTrace{
            .input_text = text,
            .is_fallback = true,
            .fallback_reason = "model not loaded",
            .latency_micros = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count(),
        };
    }

    std::string safe_text = text;
    if (safe_text.size() > MAX_INPUT_BYTES) {
        safe_text = truncate_to_rune_boundary(safe_text, MAX_INPUT_BYTES);
    }

    auto tokens = model_->tokenizer().encode(safe_text);
    if (tokens.empty()) {
        return GateTrace{
            .input_text = text,
            .is_fallback = true,
            .fallback_reason = "empty input tokens",
            .latency_micros = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count(),
        };
    }

    if (tokens.size() > MAX_SEQUENCE_TOKENS) {
        tokens.resize(MAX_SEQUENCE_TOKENS);
    }

    std::vector<std::string> subwords(tokens.size());
    size_t unk_count = 0;
    bool has_unk = model_->tokenizer().has_unk();
    uint32_t unk_id = model_->tokenizer().unk_id();

    for (size_t i = 0; i < tokens.size(); ++i) {
        uint32_t id = tokens[i];
        if (has_unk && id == unk_id) {
            unk_count++;
        }
        if (id < model_->vocab().size()) {
            subwords[i] = model_->vocab()[id];
        }
    }

    double unk_ratio = tokens.empty() ? 0.0 : static_cast<double>(unk_count) / static_cast<double>(tokens.size());
    size_t emb_dim = std::min(static_cast<size_t>(model_->header().embedding_dim), MAX_GATE_EMB_DIM);
    size_t num_classes = class_count_;

    std::array<float, MAX_GATE_EMB_DIM> pooled_stack{};
    std::array<float, MAX_GATE_CLASSES> raw_logits_stack{};
    std::array<float, MAX_GATE_EMB_DIM> norm_pooled{};

    model_->predict_features(tokens, pooled_stack.data(), raw_logits_stack.data());

    // Head 1: Geometric L2 Cosine Guard
    l2_normalize(pooled_stack.data(), emb_dim, norm_pooled.data());
    float cosine_sim = 1.0f;
    bool is_ood = false;
    if (has_centroid_) {
        cosine_sim = dot_product(norm_pooled.data(), domain_centroid_.data(), emb_dim);
        if (cosine_sim < min_cosine_sim_) {
            is_ood = true;
        }
    }

    // Head 2: Anchor Bitmask & Symbolic Soft-Bias
    uint64_t text_bitmask = 0;
    std::vector<std::string> triggered_anchors;
    std::string lower_text = trim_and_lower_str(text);

    for (const auto& [kw, mask] : anchor_dict_) {
        if (lower_text.find(kw) != std::string::npos) {
            text_bitmask |= mask;
            triggered_anchors.push_back(kw);
        }
    }

    std::array<float, MAX_GATE_CLASSES> adjusted_logits{};
    std::copy_n(raw_logits_stack.begin(), num_classes, adjusted_logits.begin());

    for (const auto& rule : anchor_rules_) {
        uint64_t matched_bits = text_bitmask & rule.mask;
        if (matched_bits != 0) {
            float count = static_cast<float>(std::popcount(matched_bits));
            adjusted_logits[rule.class_index] += rule.weight * count;
        }
    }

    // Head 3: Softmax, Margin, Shannon Entropy
    std::array<float, MAX_GATE_CLASSES> probs{};
    softmax(adjusted_logits.data(), num_classes, model_->temperature(), probs.data());

    int best_idx = 0;
    int second_idx = (num_classes >= 2) ? 1 : 0;
    float best_score = -1.0f, second_score = -1.0f;
    std::map<std::string, float> prob_map;

    for (size_t i = 0; i < num_classes; ++i) {
        const auto& lbl = labels_[i];
        float p = probs[i];
        prob_map[lbl] = p;
        if (p > best_score) {
            second_score = best_score;
            second_idx = best_idx;
            best_score = p;
            best_idx = static_cast<int>(i);
        } else if (p > second_score) {
            second_score = p;
            second_idx = static_cast<int>(i);
        }
    }

    double cal_conf = static_cast<double>(best_score) * (1.0 - unk_ratio);
    double cal_sec = static_cast<double>(second_score) * (1.0 - unk_ratio);
    double margin = cal_conf - cal_sec;
    double entropy = compute_entropy(probs.data(), num_classes);

    std::string best_label = labels_[best_idx];
    std::string second_label = (num_classes >= 2) ? labels_[second_idx] : "";

    // LogSumExp & Free Energy
    float max_logit = adjusted_logits[0];
    for (size_t i = 1; i < num_classes; ++i) {
        if (adjusted_logits[i] > max_logit) max_logit = adjusted_logits[i];
    }
    double sum_exp = 0.0;
    for (size_t i = 0; i < num_classes; ++i) {
        sum_exp += std::exp(static_cast<double>(adjusted_logits[i] - max_logit));
    }
    double log_sum_exp = static_cast<double>(max_logit) + std::log(sum_exp);
    double free_energy = -log_sum_exp;

    std::string ood_reason;
    if (has_centroid_ && cosine_sim < min_cosine_sim_) {
        is_ood = true;
        ood_reason = "cosine similarity below domain threshold (OOD)";
    } else if (policy_.min_log_sum_exp > 0.0 && log_sum_exp < policy_.min_log_sum_exp) {
        is_ood = true;
        ood_reason = "free energy below in-distribution threshold (OOD)";
    } else if (entropy > policy_.max_entropy) {
        is_ood = true;
        ood_reason = "prediction entropy exceeds limit (OOD)";
    } else if (unk_ratio >= 0.5) {
        is_ood = true;
        ood_reason = "excessive unknown tokens (>= 0.50)";
    }

    int64_t latency = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();

    GateTrace trace{
        .input_text = text,
        .token_ids = std::move(tokens),
        .subwords = std::move(subwords),
        .unknown_token_ratio = unk_ratio,
        .cosine_similarity = cosine_sim,
        .is_ood = is_ood,
        .anchor_bitmask = text_bitmask,
        .triggered_anchors = std::move(triggered_anchors),
        .class_probabilities = std::move(prob_map),
        .predicted_label = best_label,
        .secondary_label = second_label,
        .confidence = cal_conf,
        .margin = margin,
        .entropy = entropy,
        .log_sum_exp = log_sum_exp,
        .free_energy = free_energy,
        .threshold = policy_.high_threshold,
        .latency_micros = latency,
    };

    if (is_ood) {
        trace.is_fallback = true;
        trace.fallback_reason = std::move(ood_reason);
    } else if (trace.confidence < policy_.low_threshold) {
        trace.is_fallback = true;
        trace.fallback_reason = "confidence below low threshold";
    } else {
        if (!second_label.empty()) {
            std::string pipe_key = make_pipeline_key(best_label, second_label);
            bool has_pipeline = pipelines_.contains(pipe_key);
            bool pri_anchor = false, sec_anchor = false;
            for (const auto& rule : anchor_rules_) {
                if ((text_bitmask & rule.mask) != 0) {
                    if (rule.class_index == best_idx) pri_anchor = true;
                    if (rule.class_index == second_idx) sec_anchor = true;
                }
            }
            if (cal_sec >= policy_.pipeline_threshold || (pri_anchor && sec_anchor && has_pipeline)) {
                trace.is_pipeline = true;
            }
        }
        if (trace.confidence < policy_.high_threshold || margin < policy_.margin_cutoff) {
            trace.is_ambiguous = true;
        }
        if (!routes_[best_idx]) {
            trace.is_fallback = true;
            trace.fallback_reason = "label has no bound route handler";
        }
    }

    return trace;
}

NeuroGate::GateEvaluation NeuroGate::evaluate_fast(const std::string& text) const {
    if (!model_) return GateEvaluation{.is_fallback = true};

    std::string safe_text = text;
    if (safe_text.size() > MAX_INPUT_BYTES) {
        safe_text = truncate_to_rune_boundary(safe_text, MAX_INPUT_BYTES);
    }

    auto tokens = model_->tokenizer().encode(safe_text);
    if (tokens.empty()) return GateEvaluation{.is_fallback = true};
    if (tokens.size() > MAX_SEQUENCE_TOKENS) {
        tokens.resize(MAX_SEQUENCE_TOKENS);
    }

    size_t unk_count = 0;
    bool has_unk = model_->tokenizer().has_unk();
    uint32_t unk_id = model_->tokenizer().unk_id();

    for (uint32_t id : tokens) {
        if (has_unk && id == unk_id) unk_count++;
    }
    double unk_ratio = static_cast<double>(unk_count) / static_cast<double>(tokens.size());

    size_t emb_dim = std::min(static_cast<size_t>(model_->header().embedding_dim), MAX_GATE_EMB_DIM);
    size_t num_classes = class_count_;

    std::array<float, MAX_GATE_EMB_DIM> pooled_stack{};
    std::array<float, MAX_GATE_CLASSES> raw_logits_stack{};
    std::array<float, MAX_GATE_EMB_DIM> norm_pooled{};

    model_->predict_features(tokens, pooled_stack.data(), raw_logits_stack.data());

    // Head 1
    l2_normalize(pooled_stack.data(), emb_dim, norm_pooled.data());
    if (has_centroid_) {
        float cosine_sim = dot_product(norm_pooled.data(), domain_centroid_.data(), emb_dim);
        if (cosine_sim < min_cosine_sim_) {
            return GateEvaluation{.is_fallback = true};
        }
    }

    // Head 2: Anchor Bitmask & Symbolic Bias via string search
    uint64_t text_bitmask = 0;
    std::string lower_text = trim_and_lower_str(text);
    for (const auto& [kw, mask] : anchor_dict_) {
        if (lower_text.find(kw) != std::string::npos) {
            text_bitmask |= mask;
        }
    }

    std::array<float, MAX_GATE_CLASSES> adjusted_logits{};
    std::copy_n(raw_logits_stack.begin(), num_classes, adjusted_logits.begin());

    for (const auto& rule : anchor_rules_) {
        uint64_t matched_bits = text_bitmask & rule.mask;
        if (matched_bits != 0) {
            float count = static_cast<float>(std::popcount(matched_bits));
            adjusted_logits[rule.class_index] += rule.weight * count;
        }
    }

    // Head 3
    std::array<float, MAX_GATE_CLASSES> probs{};
    softmax(adjusted_logits.data(), num_classes, model_->temperature(), probs.data());

    int best_idx = 0, second_idx = (num_classes >= 2) ? 1 : 0;
    float best_score = -1.0f, second_score = -1.0f;

    for (size_t i = 0; i < num_classes; ++i) {
        float p = probs[i];
        if (p > best_score) {
            second_score = best_score;
            second_idx = best_idx;
            best_score = p;
            best_idx = static_cast<int>(i);
        } else if (p > second_score) {
            second_score = p;
            second_idx = static_cast<int>(i);
        }
    }

    double cal_conf = static_cast<double>(best_score) * (1.0 - unk_ratio);
    double cal_sec = static_cast<double>(second_score) * (1.0 - unk_ratio);
    double margin = cal_conf - cal_sec;
    double entropy = compute_entropy(probs.data(), num_classes);

    float max_logit = adjusted_logits[0];
    for (size_t i = 1; i < num_classes; ++i) {
        if (adjusted_logits[i] > max_logit) max_logit = adjusted_logits[i];
    }
    double sum_exp = 0.0;
    for (size_t i = 0; i < num_classes; ++i) {
        sum_exp += std::exp(static_cast<double>(adjusted_logits[i] - max_logit));
    }
    double log_sum_exp = static_cast<double>(max_logit) + std::log(sum_exp);

    if ((policy_.min_log_sum_exp > 0.0 && log_sum_exp < policy_.min_log_sum_exp) ||
        unk_ratio >= 0.5 ||
        cal_conf < policy_.low_threshold ||
        entropy > policy_.max_entropy) {
        return GateEvaluation{.primary_idx = best_idx, .secondary_idx = second_idx, .is_fallback = true};
    }

    GateEvaluation eval{
        .primary_idx = best_idx,
        .secondary_idx = second_idx,
    };

    if (second_idx >= 0 && second_idx < static_cast<int>(num_classes)) {
        std::string pipe_key = make_pipeline_key(labels_[best_idx], labels_[second_idx]);
        bool has_pipeline = pipelines_.contains(pipe_key);
        bool pri_anchor = false, sec_anchor = false;
        for (const auto& rule : anchor_rules_) {
            if ((text_bitmask & rule.mask) != 0) {
                if (rule.class_index == best_idx) pri_anchor = true;
                if (rule.class_index == second_idx) sec_anchor = true;
            }
        }
        if (cal_sec >= policy_.pipeline_threshold || (pri_anchor && sec_anchor && has_pipeline)) {
            eval.is_pipeline = true;
        }
    }

    if (cal_conf < policy_.high_threshold || margin < policy_.margin_cutoff) {
        eval.is_ambiguous = true;
    }

    if (!routes_[best_idx]) {
        eval.is_fallback = true;
    }

    return eval;
}

NeuroGate::GateEvaluation NeuroGate::evaluate_fast_tokens(const std::vector<uint32_t>& tokens) const {
    if (!model_ || tokens.empty()) return GateEvaluation{.is_fallback = true};

    size_t unk_count = 0;
    bool has_unk = model_->tokenizer().has_unk();
    uint32_t unk_id = model_->tokenizer().unk_id();

    for (uint32_t id : tokens) {
        if (has_unk && id == unk_id) unk_count++;
    }
    double unk_ratio = static_cast<double>(unk_count) / static_cast<double>(tokens.size());

    size_t emb_dim = std::min(static_cast<size_t>(model_->header().embedding_dim), MAX_GATE_EMB_DIM);
    size_t num_classes = class_count_;

    std::array<float, MAX_GATE_EMB_DIM> pooled_stack{};
    std::array<float, MAX_GATE_CLASSES> raw_logits_stack{};
    std::array<float, MAX_GATE_EMB_DIM> norm_pooled{};

    model_->predict_features(tokens, pooled_stack.data(), raw_logits_stack.data());

    // Head 1
    l2_normalize(pooled_stack.data(), emb_dim, norm_pooled.data());
    if (has_centroid_) {
        float cosine_sim = dot_product(norm_pooled.data(), domain_centroid_.data(), emb_dim);
        if (cosine_sim < min_cosine_sim_) {
            return GateEvaluation{.is_fallback = true};
        }
    }

    // Head 2
    uint64_t text_bitmask = 0;
    for (uint32_t id : tokens) {
        auto it = anchor_token_map_.find(id);
        if (it != anchor_token_map_.end()) {
            text_bitmask |= it->second;
        }
    }

    std::array<float, MAX_GATE_CLASSES> adjusted_logits{};
    std::copy_n(raw_logits_stack.begin(), num_classes, adjusted_logits.begin());

    for (const auto& rule : anchor_rules_) {
        uint64_t matched_bits = text_bitmask & rule.mask;
        if (matched_bits != 0) {
            float count = static_cast<float>(std::popcount(matched_bits));
            adjusted_logits[rule.class_index] += rule.weight * count;
        }
    }

    // Head 3
    std::array<float, MAX_GATE_CLASSES> probs{};
    softmax(adjusted_logits.data(), num_classes, model_->temperature(), probs.data());

    int best_idx = 0, second_idx = (num_classes >= 2) ? 1 : 0;
    float best_score = -1.0f, second_score = -1.0f;

    for (size_t i = 0; i < num_classes; ++i) {
        float p = probs[i];
        if (p > best_score) {
            second_score = best_score;
            second_idx = best_idx;
            best_score = p;
            best_idx = static_cast<int>(i);
        } else if (p > second_score) {
            second_score = p;
            second_idx = static_cast<int>(i);
        }
    }

    double cal_conf = static_cast<double>(best_score) * (1.0 - unk_ratio);
    double cal_sec = static_cast<double>(second_score) * (1.0 - unk_ratio);
    double margin = cal_conf - cal_sec;
    double entropy = compute_entropy(probs.data(), num_classes);

    float max_logit = adjusted_logits[0];
    for (size_t i = 1; i < num_classes; ++i) {
        if (adjusted_logits[i] > max_logit) max_logit = adjusted_logits[i];
    }
    double sum_exp = 0.0;
    for (size_t i = 0; i < num_classes; ++i) {
        sum_exp += std::exp(static_cast<double>(adjusted_logits[i] - max_logit));
    }
    double log_sum_exp = static_cast<double>(max_logit) + std::log(sum_exp);

    if ((policy_.min_log_sum_exp > 0.0 && log_sum_exp < policy_.min_log_sum_exp) ||
        unk_ratio >= 0.5 ||
        cal_conf < policy_.low_threshold ||
        entropy > policy_.max_entropy) {
        return GateEvaluation{.primary_idx = best_idx, .secondary_idx = second_idx, .is_fallback = true};
    }

    GateEvaluation eval{
        .primary_idx = best_idx,
        .secondary_idx = second_idx,
    };

    if (second_idx >= 0 && second_idx < static_cast<int>(num_classes)) {
        std::string pipe_key = make_pipeline_key(labels_[best_idx], labels_[second_idx]);
        bool has_pipeline = pipelines_.contains(pipe_key);
        bool pri_anchor = false, sec_anchor = false;
        for (const auto& rule : anchor_rules_) {
            if ((text_bitmask & rule.mask) != 0) {
                if (rule.class_index == best_idx) pri_anchor = true;
                if (rule.class_index == second_idx) sec_anchor = true;
            }
        }
        if (cal_sec >= policy_.pipeline_threshold || (pri_anchor && sec_anchor && has_pipeline)) {
            eval.is_pipeline = true;
        }
    }

    if (cal_conf < policy_.high_threshold || margin < policy_.margin_cutoff) {
        eval.is_ambiguous = true;
    }

    if (!routes_[best_idx]) {
        eval.is_fallback = true;
    }

    return eval;
}

void NeuroGate::filter(const std::string& text, void* payload) {
    std::shared_lock<std::shared_mutex> lock(mu_);
    auto eval = evaluate_fast(text);

    if (eval.is_fallback) {
        if (fallback_) fallback_(payload);
        return;
    }
    if (eval.is_ambiguous) {
        if (ambiguous_) {
            std::string s_label = (eval.secondary_idx >= 0 && eval.secondary_idx < static_cast<int>(class_count_))
                                  ? labels_[eval.secondary_idx] : "";
            ambiguous_(labels_[eval.primary_idx], s_label, payload);
            return;
        }
        if (fallback_) fallback_(payload);
        return;
    }
    if (eval.primary_idx < 0 || eval.primary_idx >= static_cast<int>(class_count_) || !routes_[eval.primary_idx]) {
        if (fallback_) fallback_(payload);
        return;
    }
    routes_[eval.primary_idx](payload);
}

void NeuroGate::filter_pipeline(const std::string& text, void* payload) {
    std::shared_lock<std::shared_mutex> lock(mu_);
    auto eval = evaluate_fast(text);

    if (eval.is_fallback) {
        if (fallback_) fallback_(payload);
        return;
    }

    std::string p_label = labels_[eval.primary_idx];
    std::string s_label = (eval.secondary_idx >= 0 && eval.secondary_idx < static_cast<int>(class_count_))
                          ? labels_[eval.secondary_idx] : "";

    if (eval.is_pipeline && !s_label.empty()) {
        std::string pipe_key = make_pipeline_key(p_label, s_label);
        auto it = pipelines_.find(pipe_key);
        if (it != pipelines_.end()) {
            it->second(p_label, s_label, payload);
            return;
        }
    }

    if (eval.is_ambiguous) {
        if (ambiguous_) {
            ambiguous_(p_label, s_label, payload);
            return;
        }
        if (fallback_) fallback_(payload);
        return;
    }

    if (eval.primary_idx < 0 || eval.primary_idx >= static_cast<int>(class_count_) || !routes_[eval.primary_idx]) {
        if (fallback_) fallback_(payload);
        return;
    }
    routes_[eval.primary_idx](payload);
}

void NeuroGate::filter_tokens(const std::vector<uint32_t>& tokens, void* payload) {
    std::shared_lock<std::shared_mutex> lock(mu_);
    auto eval = evaluate_fast_tokens(tokens);

    if (eval.is_fallback) {
        if (fallback_) fallback_(payload);
        return;
    }
    if (eval.is_ambiguous) {
        if (ambiguous_) {
            std::string s_label = (eval.secondary_idx >= 0 && eval.secondary_idx < static_cast<int>(class_count_))
                                  ? labels_[eval.secondary_idx] : "";
            ambiguous_(labels_[eval.primary_idx], s_label, payload);
            return;
        }
        if (fallback_) fallback_(payload);
        return;
    }
    if (eval.primary_idx < 0 || eval.primary_idx >= static_cast<int>(class_count_) || !routes_[eval.primary_idx]) {
        if (fallback_) fallback_(payload);
        return;
    }
    routes_[eval.primary_idx](payload);
}

} // namespace intellibranch
