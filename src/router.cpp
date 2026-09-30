#include "intellibranch/router.hpp"
#include "intellibranch/binary.hpp"
#include "intellibranch/ops.hpp"
#include <mutex>
#include <chrono>

namespace intellibranch {

std::string Router::pipeline_key(const std::string& primary, const std::string& secondary) {
    return primary + "->" + secondary;
}

std::shared_ptr<Router> Router::create(const std::string& model_path, double default_threshold) {
    auto model = load_binary_model(model_path);
    return std::make_shared<Router>(model, default_threshold);
}

Router::Router(std::shared_ptr<InferenceModel> model, double default_threshold)
    : model_(std::move(model)), threshold_(default_threshold) {
    policy_ = default_dispatch_policy();
    if (default_threshold > 0.0) {
        policy_.high_threshold = default_threshold;
        policy_.low_threshold = default_threshold * 0.6;
    }
    fallback_ = [](void*) {};
    telemetry_ = std::make_unique<TelemetryRingBuffer>(1024);
}

void Router::reload(const std::string& model_path) {
    auto new_model = load_binary_model(model_path);
    std::unique_lock<std::shared_mutex> lock(mu_);
    model_ = std::move(new_model);
}

void Router::swap_model(std::shared_ptr<InferenceModel> new_model) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    model_ = std::move(new_model);
}

std::shared_ptr<InferenceModel> Router::model() const {
    std::shared_lock<std::shared_mutex> lock(mu_);
    return model_;
}

Router& Router::enable_telemetry(size_t capacity) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    telemetry_ = std::make_unique<TelemetryRingBuffer>(capacity);
    return *this;
}

std::vector<TelemetryEvent> Router::drain_telemetry() {
    std::shared_lock<std::shared_mutex> lock(mu_);
    if (!telemetry_) {
        return {};
    }
    return telemetry_->drain();
}

void Router::record_telemetry(
    const std::string& text,
    const std::string& primary,
    const std::string& secondary,
    double conf,
    double entropy,
    bool is_ambiguous,
    bool is_pipeline,
    bool is_fallback
) {
    if (telemetry_ && (is_ambiguous || is_pipeline || is_fallback)) {
        telemetry_->push(TelemetryEvent{
            .input_text = text,
            .predicted_label = primary,
            .secondary_label = secondary,
            .confidence = conf,
            .entropy = entropy,
            .is_ambiguous = is_ambiguous,
            .is_pipeline = is_pipeline,
            .is_fallback = is_fallback,
        });
    }
}

Router& Router::set_policy(DispatchPolicy policy) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    policy_ = policy;
    return *this;
}

Router& Router::bind(const std::string& label, RouteAction action) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    routes_[label] = std::move(action);
    return *this;
}

Router& Router::bind_pipeline(const std::string& primary, const std::string& secondary, PipelineAction action) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    pipelines_[pipeline_key(primary, secondary)] = std::move(action);
    return *this;
}

Router& Router::default_pipeline(PipelineAction action) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    default_pipeline_ = std::move(action);
    return *this;
}

Router& Router::ambiguous(AmbiguousAction action) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    ambiguous_ = std::move(action);
    return *this;
}

Router& Router::fallback(RouteAction action) {
    std::unique_lock<std::shared_mutex> lock(mu_);
    fallback_ = std::move(action);
    return *this;
}

void Router::dispatch(const std::string& text, void* payload, std::stop_token stop_token) {
    if (stop_token.stop_requested()) {
        throw IntelliBranchException("context canceled");
    }

    std::shared_lock<std::shared_mutex> lock(mu_);

    if (stop_token.stop_requested()) {
        throw IntelliBranchException("context canceled");
    }

    if (!model_) {
        record_telemetry(text, "", "", 0, 0, false, false, true);
        if (fallback_) fallback_(payload);
        return;
    }

    StaticInferenceResult res;
    double unk_ratio = 0.0;
    try {
        auto detailed = model_->predict_detailed(text);
        res = detailed.first;
        unk_ratio = detailed.second;
    } catch (...) {
        record_telemetry(text, "", "", 0, 0, false, false, true);
        if (fallback_) fallback_(payload);
        return;
    }

    if (res.total == 0 || res.primary.index < 0 || static_cast<size_t>(res.primary.index) >= model_->labels().size()) {
        record_telemetry(text, "", "", 0, 0, false, false, true);
        if (fallback_) fallback_(payload);
        return;
    }

    const std::string& primary_label = model_->labels()[res.primary.index];
    double primary_conf = res.primary.confidence;

    std::string secondary_label;
    double secondary_conf = 0.0;
    if (res.total >= 2 && res.secondary.index >= 0 && static_cast<size_t>(res.secondary.index) < model_->labels().size()) {
        secondary_label = model_->labels()[res.secondary.index];
        secondary_conf = res.secondary.confidence;
    }

    double margin = primary_conf - secondary_conf;
    double entropy = res.entropy;

    // 1. Fallback Isolation: Low confidence, excessive UNKs, or high entropy (OOD)
    if (primary_conf < policy_.low_threshold || unk_ratio >= 0.5 || entropy > policy_.max_entropy) {
        record_telemetry(text, primary_label, secondary_label, primary_conf, entropy, false, false, true);
        if (fallback_) fallback_(payload);
        return;
    }

    // 2. Ambiguous Route: Borderline confidence or narrow margin
    bool is_ambiguous = (primary_conf < policy_.high_threshold || margin < policy_.margin_cutoff);
    if (is_ambiguous) {
        record_telemetry(text, primary_label, secondary_label, primary_conf, entropy, true, false, false);
        if (ambiguous_) {
            ambiguous_(primary_label, secondary_label, payload);
            return;
        }
        if (fallback_) fallback_(payload);
        return;
    }

    // 3. Definite Route
    auto it = routes_.find(primary_label);
    if (it == routes_.end()) {
        record_telemetry(text, primary_label, secondary_label, primary_conf, entropy, false, false, true);
        if (fallback_) fallback_(payload);
        return;
    }

    if (stop_token.stop_requested()) {
        throw IntelliBranchException("context canceled");
    }

    it->second(payload);
}

void Router::dispatch_pipeline(const std::string& text, void* payload, std::stop_token stop_token) {
    if (stop_token.stop_requested()) {
        throw IntelliBranchException("context canceled");
    }

    std::shared_lock<std::shared_mutex> lock(mu_);

    if (stop_token.stop_requested()) {
        throw IntelliBranchException("context canceled");
    }

    if (!model_) {
        record_telemetry(text, "", "", 0, 0, false, false, true);
        if (fallback_) fallback_(payload);
        return;
    }

    StaticInferenceResult res;
    double unk_ratio = 0.0;
    try {
        auto detailed = model_->predict_detailed(text);
        res = detailed.first;
        unk_ratio = detailed.second;
    } catch (...) {
        record_telemetry(text, "", "", 0, 0, false, false, true);
        if (fallback_) fallback_(payload);
        return;
    }

    if (res.total == 0 || res.primary.index < 0 || static_cast<size_t>(res.primary.index) >= model_->labels().size()) {
        record_telemetry(text, "", "", 0, 0, false, false, true);
        if (fallback_) fallback_(payload);
        return;
    }

    const std::string& primary_label = model_->labels()[res.primary.index];
    double primary_conf = res.primary.confidence;

    std::string secondary_label;
    double secondary_conf = 0.0;
    if (res.total >= 2 && res.secondary.index >= 0 && static_cast<size_t>(res.secondary.index) < model_->labels().size()) {
        secondary_label = model_->labels()[res.secondary.index];
        secondary_conf = res.secondary.confidence;
    }

    double margin = primary_conf - secondary_conf;
    double entropy = res.entropy;

    // 1. Fallback Isolation
    if (primary_conf < policy_.low_threshold || unk_ratio >= 0.5 || entropy > policy_.max_entropy) {
        record_telemetry(text, primary_label, secondary_label, primary_conf, entropy, false, false, true);
        if (fallback_) fallback_(payload);
        return;
    }

    // 2. Multi-Intent Pipeline Dispatch
    if (!secondary_label.empty() && secondary_conf >= policy_.pipeline_threshold) {
        record_telemetry(text, primary_label, secondary_label, primary_conf, entropy, false, true, false);
        std::string key = pipeline_key(primary_label, secondary_label);
        auto it = pipelines_.find(key);
        if (it != pipelines_.end()) {
            it->second(primary_label, secondary_label, payload);
            return;
        }
        if (default_pipeline_) {
            default_pipeline_(primary_label, secondary_label, payload);
            return;
        }
    }

    // 3. Fallback to standard 3-tier routing
    bool is_ambiguous = (primary_conf < policy_.high_threshold || margin < policy_.margin_cutoff);
    if (is_ambiguous) {
        record_telemetry(text, primary_label, secondary_label, primary_conf, entropy, true, false, false);
        if (ambiguous_) {
            ambiguous_(primary_label, secondary_label, payload);
            return;
        }
        if (fallback_) fallback_(payload);
        return;
    }

    auto it = routes_.find(primary_label);
    if (it == routes_.end()) {
        record_telemetry(text, primary_label, secondary_label, primary_conf, entropy, false, false, true);
        if (fallback_) fallback_(payload);
        return;
    }

    it->second(payload);
}

RouteTrace Router::inspect(const std::string& text) const {
    auto start = std::chrono::steady_clock::now();
    std::shared_lock<std::shared_mutex> lock(mu_);

    if (!model_) {
        return RouteTrace{
            .input_text = text,
            .threshold = threshold_,
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
        return RouteTrace{
            .input_text = text,
            .threshold = threshold_,
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
    auto probs = model_->forward(tokens, model_->temperature());

    std::map<std::string, float> prob_map;
    std::string best_label, second_label;
    float best_score = -1.0f, second_score = -1.0f;

    for (size_t i = 0; i < probs.size(); ++i) {
        if (i < model_->labels().size()) {
            const auto& lbl = model_->labels()[i];
            float p = probs[i];
            prob_map[lbl] = p;
            if (p > best_score) {
                second_score = best_score;
                second_label = best_label;
                best_score = p;
                best_label = lbl;
            } else if (p > second_score) {
                second_score = p;
                second_label = lbl;
            }
        }
    }

    double cal_conf = static_cast<double>(best_score) * (1.0 - unk_ratio);
    double cal_sec = static_cast<double>(second_score) * (1.0 - unk_ratio);
    double margin = cal_conf - cal_sec;
    double entropy = compute_entropy(probs.data(), probs.size());

    int64_t latency = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count();

    RouteTrace trace{
        .input_text = text,
        .token_ids = std::move(tokens),
        .subwords = std::move(subwords),
        .unknown_token_ratio = unk_ratio,
        .class_probabilities = std::move(prob_map),
        .predicted_label = best_label,
        .secondary_label = second_label,
        .confidence = cal_conf,
        .margin = margin,
        .entropy = entropy,
        .threshold = policy_.high_threshold,
        .latency_micros = latency,
    };

    if (trace.confidence < policy_.low_threshold) {
        trace.is_fallback = true;
        trace.fallback_reason = "confidence below low threshold";
    } else if (unk_ratio >= 0.5) {
        trace.is_fallback = true;
        trace.fallback_reason = "excessive unknown tokens";
    } else if (entropy > policy_.max_entropy) {
        trace.is_fallback = true;
        trace.fallback_reason = "prediction entropy exceeds limit (OOD)";
    } else {
        if (!second_label.empty() && cal_sec >= policy_.pipeline_threshold) {
            trace.is_pipeline = true;
        }
        if (trace.confidence < policy_.high_threshold || margin < policy_.margin_cutoff) {
            trace.is_ambiguous = true;
        }
        if (!routes_.contains(best_label)) {
            trace.is_fallback = true;
            trace.fallback_reason = "label has no bound route handler";
        }
    }

    return trace;
}

std::pair<RouteTrace, bool> Router::dispatch_with_trace(const std::string& text, void* payload) {
    auto trace = inspect(text);

    std::shared_lock<std::shared_mutex> lock(mu_);
    record_telemetry(text, trace.predicted_label, trace.secondary_label, trace.confidence, trace.entropy, trace.is_ambiguous, trace.is_pipeline, trace.is_fallback);

    if (trace.is_fallback) {
        if (fallback_) fallback_(payload);
        return {trace, false};
    }
    if (trace.is_pipeline) {
        std::string key = pipeline_key(trace.predicted_label, trace.secondary_label);
        auto it = pipelines_.find(key);
        if (it != pipelines_.end()) {
            it->second(trace.predicted_label, trace.secondary_label, payload);
            return {trace, true};
        }
        if (default_pipeline_) {
            default_pipeline_(trace.predicted_label, trace.secondary_label, payload);
            return {trace, true};
        }
    }
    if (trace.is_ambiguous && ambiguous_) {
        ambiguous_(trace.predicted_label, trace.secondary_label, payload);
        return {trace, true};
    }

    auto it = routes_.find(trace.predicted_label);
    if (it != routes_.end()) {
        it->second(payload);
        return {trace, true};
    }

    if (fallback_) fallback_(payload);
    return {trace, false};
}

} // namespace intellibranch
