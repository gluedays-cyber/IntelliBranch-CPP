#include "test_helpers.hpp"
#include "intellibranch/router.hpp"
#include "intellibranch/binary.hpp"
#include <thread>
#include <atomic>
#include <filesystem>

namespace fs = std::filesystem;
using namespace intellibranch;

std::shared_ptr<InferenceModel> get_sample_model() {
    Header header{
        .magic = MAGIC_BYTES,
        .version = 1,
        .vocab_size = 4,
        .embedding_dim = 4,
        .hidden_dim = 6,
        .num_classes = 2,
    };

    std::vector<std::string> labels = {"Refund", "Delivery"};
    std::vector<std::string> vocab = {"[PAD]", "refund", "cancel", "delivery"};
    std::vector<MergeRule> merge_rules = {
        {1, 2, 3}
    };

    Weights weights{
        .embedding = std::vector<float>(header.vocab_size * header.embedding_dim),
        .positional = std::vector<float>(MAX_SEQUENCE_TOKENS * header.embedding_dim, 0.0f),
        .w1 = std::vector<float>(header.embedding_dim * header.hidden_dim),
        .b1 = std::vector<float>(header.hidden_dim, 0.1f),
        .w2 = std::vector<float>(header.hidden_dim * header.num_classes),
        .b2 = std::vector<float>(header.num_classes, 0.2f),
    };

    for (size_t i = 0; i < weights.embedding.size(); ++i) weights.embedding[i] = static_cast<float>(i) * 0.05f;
    for (size_t i = 0; i < weights.w1.size(); ++i) weights.w1[i] = static_cast<float>(i) * 0.02f;
    for (size_t i = 0; i < weights.w2.size(); ++i) weights.w2[i] = static_cast<float>(i) * 0.03f;

    return std::make_shared<InferenceModel>(header, labels, vocab, merge_rules, weights);
}

void test_router_dispatch_and_fallback() {
    std::string path = "test_router_tmp.bin";
    auto model = get_sample_model();
    save_binary_model(path, *model);

    auto router = Router::create(path, 0.1);

    bool refund_called = false;
    bool fallback_called = false;

    router->bind("Refund", [&](void*) { refund_called = true; })
           .fallback([&](void*) { fallback_called = true; });

    router->dispatch("refund", nullptr);
    ASSERT_TRUE(refund_called || fallback_called);

    // Strict router
    auto strict_router = Router::create(path, 0.999999);
    bool strict_fallback = false;
    strict_router->bind("Refund", [&](void*) {})
                 .fallback([&](void*) { strict_fallback = true; });

    strict_router->dispatch("refund", nullptr);
    ASSERT_TRUE(strict_fallback);

    fs::remove(path);
}

void test_router_inspect_and_trace() {
    std::string path = "test_router_trace_tmp.bin";
    auto model = get_sample_model();
    save_binary_model(path, *model);

    auto router = Router::create(path, 0.5);
    router->bind("Refund", [](void*) {});

    auto trace = router->inspect("refund");
    ASSERT_FALSE(trace.token_ids.empty());
    ASSERT_FALSE(trace.class_probabilities.empty());
    ASSERT_TRUE(trace.confidence > 0.0);

    fs::remove(path);
}

void test_router_context_cancellation() {
    std::string path = "test_router_ctx_tmp.bin";
    auto model = get_sample_model();
    save_binary_model(path, *model);

    auto router = Router::create(path, 0.1);
    router->bind("Refund", [](void*) {});

    std::stop_source source;
    source.request_stop();

    bool caught = false;
    try {
        router->dispatch("refund", nullptr, source.get_token());
    } catch (const IntelliBranchException&) {
        caught = true;
    }
    ASSERT_TRUE(caught);

    fs::remove(path);
}

void test_router_3tier_dispatch() {
    std::string path = "test_router_3tier_tmp.bin";
    auto model = get_sample_model();
    save_binary_model(path, *model);

    auto router = Router::create(path, 0.5);
    router->set_policy(DispatchPolicy{
        .high_threshold = 0.80,
        .low_threshold = 0.30,
        .margin_cutoff = 0.20,
        .max_entropy = 2.0,
    });

    bool definite_called = false;
    bool ambiguous_called = false;
    bool fallback_called = false;

    router->bind("Refund", [&](void*) { definite_called = true; })
           .ambiguous([&](const std::string&, const std::string&, void*) { ambiguous_called = true; })
           .fallback([&](void*) { fallback_called = true; });

    auto trace = router->inspect("refund");
    router->dispatch("refund", nullptr);

    if (trace.is_ambiguous) {
        ASSERT_TRUE(ambiguous_called);
    } else if (trace.is_fallback) {
        ASSERT_TRUE(fallback_called);
    } else {
        ASSERT_TRUE(definite_called);
    }

    fs::remove(path);
}

void test_router_ood_entropy_isolation() {
    std::string path = "test_router_ood_tmp.bin";
    auto model = get_sample_model();
    save_binary_model(path, *model);

    auto router = Router::create(path, 0.5);
    router->set_policy(DispatchPolicy{
        .high_threshold = 0.70,
        .low_threshold = 0.20,
        .margin_cutoff = 0.10,
        .max_entropy = 0.001, // Force OOD
    });

    bool fallback_triggered = false;
    router->bind("Refund", [](void*) {})
           .fallback([&](void*) { fallback_triggered = true; });

    router->dispatch("refund", nullptr);
    ASSERT_TRUE(fallback_triggered);

    fs::remove(path);
}

void test_router_multi_intent_pipeline() {
    std::string path = "test_router_pipe_tmp.bin";
    auto model = get_sample_model();
    save_binary_model(path, *model);

    auto router = Router::create(path, 0.5);
    router->set_policy(DispatchPolicy{
        .high_threshold = 0.80,
        .low_threshold = 0.20,
        .margin_cutoff = 0.15,
        .max_entropy = 3.0,
        .pipeline_threshold = 0.20,
    });

    bool specific_pipeline_called = false;
    bool default_pipeline_called = false;

    router->bind_pipeline("Refund", "Delivery", [&](const std::string& p, const std::string& s, void*) {
        specific_pipeline_called = true;
    }).default_pipeline([&](const std::string&, const std::string&, void*) {
        default_pipeline_called = true;
    });

    router->dispatch_pipeline("refund delivery", nullptr);
    ASSERT_TRUE(specific_pipeline_called || default_pipeline_called);

    fs::remove(path);
}

void test_router_telemetry_drain() {
    std::string path = "test_router_telem_tmp.bin";
    auto model = get_sample_model();
    save_binary_model(path, *model);

    auto router = Router::create(path, 0.5);
    router->enable_telemetry(4);

    router->dispatch("unknown gibberish query 12345", nullptr);
    router->dispatch("another isolated query 67890", nullptr);

    auto events = router->drain_telemetry();
    ASSERT_TRUE(events.size() >= 2);
    for (const auto& ev : events) {
        ASSERT_TRUE(ev.is_fallback || ev.is_ambiguous || ev.is_pipeline);
    }

    auto drained_again = router->drain_telemetry();
    ASSERT_TRUE(drained_again.empty());

    fs::remove(path);
}

int main() {
    test_router_dispatch_and_fallback();
    test_router_inspect_and_trace();
    test_router_context_cancellation();
    test_router_3tier_dispatch();
    test_router_ood_entropy_isolation();
    test_router_multi_intent_pipeline();
    test_router_telemetry_drain();

    std::cout << "[PASS] test_router passed all test cases successfully.\n";
    return 0;
}
