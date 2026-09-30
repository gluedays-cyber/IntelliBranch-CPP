#include "test_helpers.hpp"
#include "intellibranch/ops.hpp"
#include <vector>
#include <limits>
#include <cmath>

using namespace intellibranch;

void test_gelu() {
    struct TestCase {
        float input;
        float expected;
        float epsilon;
    };
    std::vector<TestCase> testCases = {
        {0.0f, 0.0f, 1e-5f},
        {1.0f, 0.841192f, 1e-4f},
        {-1.0f, -0.158808f, 1e-4f},
        {2.0f, 1.9545f, 1e-3f},
        {-2.0f, -0.0454f, 1e-3f},
    };

    for (const auto& tc : testCases) {
        float res = gelu(tc.input);
        ASSERT_NEAR(res, tc.expected, tc.epsilon);
    }
}

void test_gelu_in_place() {
    std::vector<float> inputs = {0.0f, 1.0f, -1.0f};
    std::vector<float> expected = {0.0f, 0.841192f, -0.158808f};

    gelu_in_place(inputs);

    for (size_t i = 0; i < inputs.size(); ++i) {
        ASSERT_NEAR(inputs[i], expected[i], 1e-4f);
    }
}

void test_softmax() {
    std::vector<float> logits = {2.0f, 1.0f, 0.1f};
    std::vector<float> probs(logits.size());

    softmax(logits, 1.0f, probs);

    float sum = 0.0f;
    for (float p : probs) {
        ASSERT_TRUE(p >= 0.0f && p <= 1.0f);
        sum += p;
    }
    ASSERT_NEAR(sum, 1.0f, 1e-5f);
    ASSERT_TRUE(probs[0] > probs[1] && probs[1] > probs[2]);
}

void test_mean_pooling() {
    size_t emb_dim = 4;
    std::vector<float> embedding_table = {
        1.0f, 2.0f, 3.0f, 4.0f, // token 0
        5.0f, 6.0f, 7.0f, 8.0f, // token 1
    };

    std::vector<uint32_t> token_ids = {0, 1};
    std::vector<float> out(emb_dim);

    mean_pooling(token_ids, embedding_table, emb_dim, out);

    std::vector<float> expected = {3.0f, 4.0f, 5.0f, 6.0f};
    for (size_t i = 0; i < emb_dim; ++i) {
        ASSERT_NEAR(out[i], expected[i], 1e-5f);
    }
}

void test_matmul_vec_add() {
    size_t in_dim = 2;
    size_t out_dim = 3;

    std::vector<float> vec = {1.0f, 2.0f};
    std::vector<float> weights = {
        1.0f, 0.5f, 0.0f,
        0.0f, 1.0f, 2.0f,
    };
    std::vector<float> bias = {0.1f, 0.2f, 0.3f};
    std::vector<float> out(out_dim);

    matmul_vec_add(vec, weights, bias, in_dim, out_dim, out);

    // out[0] = 1*1.0 + 2*0.0 + 0.1 = 1.1
    // out[1] = 1*0.5 + 2*1.0 + 0.2 = 2.7
    // out[2] = 1*0.0 + 2*2.0 + 0.3 = 4.3
    std::vector<float> expected = {1.1f, 2.7f, 4.3f};
    for (size_t i = 0; i < out_dim; ++i) {
        ASSERT_NEAR(out[i], expected[i], 1e-5f);
    }
}

void test_numerical_hardening() {
    float nan_val = std::numeric_limits<float>::quiet_NaN();
    float inf_val = std::numeric_limits<float>::infinity();
    float neg_inf_val = -std::numeric_limits<float>::infinity();

    ASSERT_EQ(safe_clamp(nan_val, 100.0f), 0.0f);
    ASSERT_EQ(safe_clamp(inf_val, 100.0f), 0.0f);
    ASSERT_EQ(safe_clamp(neg_inf_val, 100.0f), 0.0f);
    ASSERT_EQ(safe_clamp(250.0f, 100.0f), 100.0f);
    ASSERT_EQ(safe_clamp(-250.0f, 100.0f), -100.0f);

    float extreme_gelu = gelu(1e20f);
    ASSERT_FALSE(std::isnan(extreme_gelu));
    ASSERT_FALSE(std::isinf(extreme_gelu));
    ASSERT_TRUE(extreme_gelu <= 100.0f);

    ASSERT_EQ(gelu(nan_val), 0.0f);

    std::vector<float> corrupted_vec = {nan_val, inf_val};
    std::vector<float> corrupted_weights = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> corrupted_bias = {nan_val, 10.0f};
    std::vector<float> corrupted_out(2);

    matmul_vec_add(corrupted_vec, corrupted_weights, corrupted_bias, 2, 2, corrupted_out);
    for (float v : corrupted_out) {
        ASSERT_FALSE(std::isnan(v));
        ASSERT_FALSE(std::isinf(v));
    }

    std::vector<float> deg_logits = {nan_val, inf_val, neg_inf_val};
    std::vector<float> deg_probs(3);
    softmax(deg_logits, 1.0f, deg_probs);
    for (float p : deg_probs) {
        ASSERT_FALSE(std::isnan(p));
        ASSERT_FALSE(std::isinf(p));
    }
}

int main() {
    test_gelu();
    test_gelu_in_place();
    test_softmax();
    test_mean_pooling();
    test_matmul_vec_add();
    test_numerical_hardening();

    std::cout << "[PASS] test_ops passed all test cases successfully.\n";
    return 0;
}
