#pragma once

#include "intellibranch/common.hpp"
#include "intellibranch/runtime.hpp"
#include <string>
#include <vector>
#include <memory>

namespace intellibranch {

struct TrainConfig {
    size_t embedding_dim = 64;
    size_t hidden_dim = 128;
    size_t target_vocab_size = 200;
    float learning_rate = 0.001f;
    float weight_decay = 0.01f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-8f;
    size_t epochs = 100;
    size_t batch_size = 32;
    size_t patience = 5;
    uint64_t seed = 42;
};

inline TrainConfig default_train_config() {
    return TrainConfig{};
}

class AdamWState {
public:
    explicit AdamWState(size_t size);

    void step(
        float* param,
        const float* grad,
        size_t n,
        float lr,
        float weight_decay,
        float beta1,
        float beta2,
        float eps
    );

private:
    std::vector<float> m_;
    std::vector<float> v_;
    uint64_t t_ = 0;
};

std::vector<DataSample> load_csv_dataset(const std::string& file_path);

std::shared_ptr<InferenceModel> train_model(
    const std::vector<DataSample>& samples,
    const TrainConfig& cfg = default_train_config()
);

} // namespace intellibranch
