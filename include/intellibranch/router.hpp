#pragma once

#include "intellibranch/common.hpp"
#include "intellibranch/runtime.hpp"
#include "intellibranch/telemetry.hpp"
#include <string>
#include <unordered_map>
#include <memory>
#include <shared_mutex>
#include <stop_token>

namespace intellibranch {

class Router {
public:
    static std::shared_ptr<Router> create(const std::string& model_path, double default_threshold = 0.60);

    explicit Router(std::shared_ptr<InferenceModel> model, double default_threshold = 0.60);

    void reload(const std::string& model_path);
    void swap_model(std::shared_ptr<InferenceModel> new_model);
    std::shared_ptr<InferenceModel> model() const;

    Router& enable_telemetry(size_t capacity = 1024);
    std::vector<TelemetryEvent> drain_telemetry();

    Router& set_policy(DispatchPolicy policy);

    Router& bind(const std::string& label, RouteAction action);
    Router& bind_pipeline(const std::string& primary, const std::string& secondary, PipelineAction action);
    Router& default_pipeline(PipelineAction action);
    Router& ambiguous(AmbiguousAction action);
    Router& fallback(RouteAction action);

    void dispatch(const std::string& text, void* payload = nullptr, std::stop_token stop_token = {});
    void dispatch_pipeline(const std::string& text, void* payload = nullptr, std::stop_token stop_token = {});
    RouteTrace inspect(const std::string& text) const;
    std::pair<RouteTrace, bool> dispatch_with_trace(const std::string& text, void* payload = nullptr);

private:
    void record_telemetry(
        const std::string& text,
        const std::string& primary,
        const std::string& secondary,
        double conf,
        double entropy,
        bool is_ambiguous,
        bool is_pipeline,
        bool is_fallback
    );

    static std::string pipeline_key(const std::string& primary, const std::string& secondary);

    mutable std::shared_mutex mu_;
    std::shared_ptr<InferenceModel> model_;
    double threshold_;
    DispatchPolicy policy_;
    std::unordered_map<std::string, RouteAction> routes_;
    std::unordered_map<std::string, PipelineAction> pipelines_;
    PipelineAction default_pipeline_;
    AmbiguousAction ambiguous_;
    RouteAction fallback_;
    std::unique_ptr<TelemetryRingBuffer> telemetry_;
};

} // namespace intellibranch
