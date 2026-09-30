#pragma once

#include "intellibranch/common.hpp"
#include "intellibranch/runtime.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <shared_mutex>

namespace intellibranch {

class NeuroGate;

class GateRouteBuilder {
public:
    GateRouteBuilder(NeuroGate* gate, int class_index, std::string label);

    GateRouteBuilder& with_anchor(float weight, const std::vector<std::string>& keywords);

    template<typename... Args>
    requires (sizeof...(Args) > 0 && (std::is_convertible_v<Args, std::string> && ...))
    GateRouteBuilder& with_anchor(float weight, Args&&... args) {
        std::vector<std::string> kws = {std::string(std::forward<Args>(args))...};
        return with_anchor(weight, kws);
    }

    // Alias for Go WithAnchor
    template<typename... Args>
    GateRouteBuilder& WithAnchor(float weight, Args&&... args) {
        return with_anchor(weight, std::forward<Args>(args)...);
    }

    GateRouteBuilder bind(const std::string& label, RouteAction handler);
    GateRouteBuilder Bind(const std::string& label, RouteAction handler) {
        return bind(label, std::move(handler));
    }

private:
    NeuroGate* gate_;
    int class_index_;
    std::string label_;
};

class NeuroGate {
public:
    static std::shared_ptr<NeuroGate> create(const std::string& model_path);
    static std::shared_ptr<NeuroGate> create_with_model(std::shared_ptr<InferenceModel> model);

    explicit NeuroGate(std::shared_ptr<InferenceModel> model);

    NeuroGate& calibrate_domain_centroid(const std::vector<DataSample>& samples);
    NeuroGate& CalibrateDomainCentroid(const std::vector<DataSample>& samples) {
        return calibrate_domain_centroid(samples);
    }

    NeuroGate& set_domain_boundary(const std::vector<float>& centroid, float min_cosine);
    NeuroGate& SetDomainBoundary(const std::vector<float>& centroid, float min_cosine) {
        return set_domain_boundary(centroid, min_cosine);
    }

    NeuroGate& set_min_cosine_sim(float threshold);
    NeuroGate& SetMinCosineSim(float threshold) {
        return set_min_cosine_sim(threshold);
    }

    NeuroGate& set_policy(DispatchPolicy policy);
    NeuroGate& SetPolicy(DispatchPolicy policy) {
        return set_policy(policy);
    }

    GateRouteBuilder bind(const std::string& label, RouteAction handler);
    GateRouteBuilder Bind(const std::string& label, RouteAction handler) {
        return bind(label, std::move(handler));
    }

    NeuroGate& bind_pipeline(const std::string& primary, const std::string& secondary, PipelineAction handler);
    NeuroGate& BindPipeline(const std::string& primary, const std::string& secondary, PipelineAction handler) {
        return bind_pipeline(primary, secondary, std::move(handler));
    }

    NeuroGate& ambiguous(AmbiguousAction handler);
    NeuroGate& Ambiguous(AmbiguousAction handler) {
        return ambiguous(std::move(handler));
    }

    NeuroGate& fallback(RouteAction handler);
    NeuroGate& Fallback(RouteAction handler) {
        return fallback(std::move(handler));
    }

    void swap_model(std::shared_ptr<InferenceModel> new_model);
    void SwapModel(std::shared_ptr<InferenceModel> new_model) {
        swap_model(std::move(new_model));
    }

    std::shared_ptr<InferenceModel> model() const;
    std::shared_ptr<InferenceModel> Model() const { return model(); }

    GateTrace inspect(const std::string& text) const;
    GateTrace Inspect(const std::string& text) const { return inspect(text); }

    void filter(const std::string& text, void* payload = nullptr);
    void Filter(const std::string& text, void* payload = nullptr) { filter(text, payload); }

    void filter_pipeline(const std::string& text, void* payload = nullptr);
    void FilterPipeline(const std::string& text, void* payload = nullptr) { filter_pipeline(text, payload); }

    void filter_tokens(const std::vector<uint32_t>& tokens, void* payload = nullptr);
    void FilterTokens(const std::vector<uint32_t>& tokens, void* payload = nullptr) { filter_tokens(tokens, payload); }

private:
    friend class GateRouteBuilder;

    struct GateEvaluation {
        int primary_idx = -1;
        int secondary_idx = -1;
        bool is_fallback = false;
        bool is_pipeline = false;
        bool is_ambiguous = false;
    };

    void compute_baseline_centroid(const InferenceModel& model);
    GateEvaluation evaluate_fast(const std::string& text) const;
    GateEvaluation evaluate_fast_tokens(const std::vector<uint32_t>& tokens) const;

    mutable std::shared_mutex mu_;
    std::shared_ptr<InferenceModel> model_;

    std::array<std::string, MAX_GATE_CLASSES> labels_;
    std::array<RouteAction, MAX_GATE_CLASSES> routes_;
    size_t class_count_ = 0;
    std::unordered_map<std::string, int> label_to_index_;

    std::unordered_map<std::string, uint64_t> anchor_dict_;
    std::unordered_map<uint32_t, uint64_t> anchor_token_map_;
    std::vector<AnchorRule> anchor_rules_;

    std::array<float, MAX_GATE_EMB_DIM> domain_centroid_{};
    bool has_centroid_ = false;
    float min_cosine_sim_ = 0.25f;

    DispatchPolicy policy_;
    std::unordered_map<std::string, PipelineAction> pipelines_;
    AmbiguousAction ambiguous_;
    RouteAction fallback_;
};

} // namespace intellibranch
