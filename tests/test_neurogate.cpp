#include "test_helpers.hpp"
#include "intellibranch/neurogate.hpp"
#include "intellibranch/trainer.hpp"
#include "intellibranch/binary.hpp"
#include <filesystem>

namespace fs = std::filesystem;
using namespace intellibranch;

void test_neurogate_basic_and_anchors() {
    std::string path = "test_neurogate_tmp.bin";

    std::vector<DataSample> samples = {
        {"turn on living room lamps", "LightControl"},
        {"it is too dark here please switch on light", "LightControl"},
        {"cooling mode maximum fan temperature bedroom", "ClimateControl"},
        {"set ac thermostat cool air heat", "ClimateControl"},
    };

    TrainConfig cfg = default_train_config();
    cfg.epochs = 50;
    cfg.embedding_dim = 32;
    cfg.hidden_dim = 32;
    cfg.target_vocab_size = 64;

    auto model = train_model(samples, cfg);
    save_binary_model(path, *model);

    auto gate = NeuroGate::create(path);

    bool light_triggered = false;
    bool fallback_triggered = false;

    gate->bind("LightControl", [&](void*) {
        light_triggered = true;
    }).with_anchor(1.5f, "lamp", "lamps", "dark", "light");

    gate->bind("ClimateControl", [](void*) {}).with_anchor(1.5f, "cooling", "ac", "fan", "temp");

    gate->fallback([&](void*) {
        fallback_triggered = true;
    });

    // 1. Verify anchor boost flips or reinforces LightControl
    light_triggered = false;
    std::string test_query = "it is too dark in here please switch on lamps";
    gate->filter(test_query, nullptr);
    ASSERT_TRUE(light_triggered);

    // 2. Verify inspection trace
    trace = gate->inspect(test_query);
    ASSERT_EQ(trace.predicted_label, "LightControl");
    ASSERT_FALSE(trace.triggered_anchors.empty());

    // 3. Verify OOD boundary rejection
    gate->set_min_cosine_sim(0.99f);
    fallback_triggered = false;
    gate->filter("what is the meaning of quantum black holes", nullptr);
    ASSERT_TRUE(fallback_triggered);

    fs::remove(path);
}

int main() {
    test_neurogate_basic_and_anchors();

    std::cout << "[PASS] test_neurogate passed all test cases successfully.\n";
    return 0;
}
