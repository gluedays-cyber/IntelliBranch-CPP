#include "test_helpers.hpp"
#include "intellibranch/runtime.hpp"
#include "intellibranch/ops.hpp"
#include <cmath>

using namespace intellibranch;

std::shared_ptr<InferenceModel> create_standard_spec_model() {
    Header header{
        .magic = MAGIC_BYTES,
        .version = 1,
        .vocab_size = 500,
        .embedding_dim = 64,
        .hidden_dim = 128,
        .num_classes = 5,
    };

    std::vector<std::string> labels = {"Refund", "Delivery", "Account", "Payment", "General"};
    std::vector<std::string> vocab(header.vocab_size, "token");
    vocab[0] = "[PAD]";
    vocab[1] = "[UNK]";
    vocab[2] = "r";
    vocab[3] = "e";
    vocab[4] = "f";
    vocab[5] = "u";
    vocab[6] = "n";
    vocab[7] = "d";
    vocab[8] = " ";

    Weights weights{
        .embedding = std::vector<float>(header.vocab_size * header.embedding_dim),
        .positional = std::vector<float>(MAX_SEQUENCE_TOKENS * header.embedding_dim, 0.0f),
        .w1 = std::vector<float>(header.embedding_dim * header.hidden_dim),
        .b1 = std::vector<float>(header.hidden_dim, 0.0f),
        .w2 = std::vector<float>(header.hidden_dim * header.num_classes),
        .b2 = std::vector<float>(header.num_classes, 0.0f),
    };

    for (size_t i = 0; i < weights.embedding.size(); ++i) {
        weights.embedding[i] = 0.01f * static_cast<float>(i % 10);
    }
    for (size_t i = 0; i < weights.w1.size(); ++i) {
        weights.w1[i] = 0.005f * static_cast<float>(static_cast<int>(i % 20) - 10);
    }
    for (size_t i = 0; i < weights.w2.size(); ++i) {
        weights.w2[i] = 0.005f * static_cast<float>(static_cast<int>(i % 15) - 7);
    }

    return std::make_shared<InferenceModel>(header, labels, vocab, std::vector<MergeRule>{}, weights);
}

void test_model_forward() {
    auto model = create_standard_spec_model();
    std::vector<uint32_t> tokens = {1, 2, 5, 10};

    auto probs = model->forward(tokens, 1.0f);
    ASSERT_EQ(probs.size(), static_cast<size_t>(model->header().num_classes));

    float sum = 0.0f;
    for (float p : probs) {
        ASSERT_TRUE(p >= 0.0f && p <= 1.0f);
        sum += p;
    }
    ASSERT_NEAR(sum, 1.0f, 1e-4f);
}

void test_model_predict_tokens() {
    auto model = create_standard_spec_model();
    std::vector<uint32_t> tokens = {1, 2};

    auto [label, score] = model->predict_tokens(tokens);
    ASSERT_FALSE(label.empty());
    ASSERT_TRUE(score > 0.0 && score <= 1.0);
}

void test_input_truncation_guard() {
    auto model = create_standard_spec_model();
    std::string long_text;
    for (int i = 0; i < 200; ++i) {
        long_text += "refund ";
    }

    auto [label, score] = model->predict(long_text);
    ASSERT_FALSE(label.empty());
    ASSERT_TRUE(score > 0.0);
}

void test_oov_confidence_discounting() {
    auto model = create_standard_spec_model();
    std::string noise_text = "!@#%^&*()_+~`|}{[]:;?><";

    auto [label, score] = model->predict(noise_text);
    ASSERT_TRUE(score < 0.35);
}

void test_predict_slots_zero_alloc() {
    auto model = create_standard_spec_model();
    std::vector<uint32_t> tokens = {1, 2, 10};

    auto res = model->predict_slots(tokens, 1.0f);
    ASSERT_TRUE(res.total >= 1);
    ASSERT_TRUE(res.primary.index >= 0 && static_cast<size_t>(res.primary.index) < model->labels().size());
    ASSERT_TRUE(res.primary.confidence > 0.0f && res.primary.confidence <= 1.0f);
}

void test_truncate_to_rune_boundary() {
    std::string multi_byte_text = "Euro € Sign";
    std::string truncated6 = truncate_to_rune_boundary(multi_byte_text, 6);
    ASSERT_EQ(truncated6, "Euro ");

    std::string truncated8 = truncate_to_rune_boundary(multi_byte_text, 8);
    ASSERT_EQ(truncated8, "Euro €");

    std::string truncated0 = truncate_to_rune_boundary(multi_byte_text, 0);
    ASSERT_EQ(truncated0, "");

    std::string ascii = "hello world";
    std::string truncated_ascii = truncate_to_rune_boundary(ascii, 5);
    ASSERT_EQ(truncated_ascii, "hello");
}

void test_semantic_xor_positional_disambiguation() {
    size_t emb_dim = 4;
    std::vector<float> embedding_table = {
        1.0f, 0.0f, 0.0f, 0.0f, // token 0: "A"
        0.0f, 1.0f, 0.0f, 0.0f, // token 1: "B"
    };
    std::vector<float> pos_table = {
        0.1f, 0.2f, 0.0f, 0.0f, // pos 0
        0.0f, 0.0f, 0.3f, 0.4f, // pos 1
    };

    std::vector<uint32_t> seq_ab = {0, 1};
    std::vector<uint32_t> seq_ba = {1, 0};

    // 1. Without positional encoding:
    std::vector<float> out_legacy_ab(emb_dim);
    std::vector<float> out_legacy_ba(emb_dim);
    mean_pooling(seq_ab, embedding_table, emb_dim, out_legacy_ab);
    mean_pooling(seq_ba, embedding_table, emb_dim, out_legacy_ba);

    float diff_legacy = 0.0f;
    for (size_t d = 0; d < emb_dim; ++d) {
        diff_legacy += std::abs(out_legacy_ab[d] - out_legacy_ba[d]);
    }
    ASSERT_EQ(diff_legacy, 0.0f);

    // 2. With positional encoding:
    std::vector<float> out_pos_ab(emb_dim);
    std::vector<float> out_pos_ba(emb_dim);
    mean_pooling_with_pos(seq_ab, embedding_table, pos_table, emb_dim, out_pos_ab);
    mean_pooling_with_pos(seq_ba, embedding_table, pos_table, emb_dim, out_pos_ba);

    float diff_pos = 0.0f;
    for (size_t d = 0; d < emb_dim; ++d) {
        diff_pos += std::abs(out_pos_ab[d] - out_pos_ba[d]);
    }
    ASSERT_TRUE(diff_pos > 0.05f);
}

void test_empty_whitespace_input() {
    auto model = create_standard_spec_model();

    bool caught1 = false;
    try {
        model->predict("");
    } catch (const EmptyInputException&) {
        caught1 = true;
    }
    ASSERT_TRUE(caught1);

    bool caught2 = false;
    try {
        model->predict("   \t\n   ");
    } catch (const EmptyInputException&) {
        caught2 = true;
    }
    ASSERT_TRUE(caught2);
}

int main() {
    test_model_forward();
    test_model_predict_tokens();
    test_input_truncation_guard();
    test_oov_confidence_discounting();
    test_predict_slots_zero_alloc();
    test_truncate_to_rune_boundary();
    test_semantic_xor_positional_disambiguation();
    test_empty_whitespace_input();

    std::cout << "[PASS] test_runtime passed all test cases successfully.\n";
    return 0;
}
