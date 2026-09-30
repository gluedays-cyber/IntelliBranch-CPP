#include "test_helpers.hpp"
#include "intellibranch/binary.hpp"
#include <sstream>
#include <filesystem>

namespace fs = std::filesystem;
using namespace intellibranch;

std::shared_ptr<InferenceModel> create_sample_model(uint32_t version = 1) {
    Header header{
        .magic = MAGIC_BYTES,
        .version = version,
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

    for (size_t i = 0; i < weights.embedding.size(); ++i) {
        weights.embedding[i] = static_cast<float>(i) * 0.05f;
    }
    if (version >= 2) {
        for (size_t i = 0; i < weights.positional.size(); ++i) {
            weights.positional[i] = static_cast<float>(i + 1) * 0.01f;
        }
    }
    for (size_t i = 0; i < weights.w1.size(); ++i) {
        weights.w1[i] = static_cast<float>(i) * 0.02f;
    }
    for (size_t i = 0; i < weights.w2.size(); ++i) {
        weights.w2[i] = static_cast<float>(i) * 0.03f;
    }

    return std::make_shared<InferenceModel>(header, labels, vocab, merge_rules, weights);
}

void test_serialization_round_trip() {
    auto orig = create_sample_model(1);
    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);

    serialize_model(ss, *orig);
    ASSERT_TRUE(!ss.str().empty());

    auto loaded = deserialize_model(ss);
    ASSERT_TRUE(loaded->header().magic == orig->header().magic);
    ASSERT_EQ(loaded->header().version, orig->header().version);
    ASSERT_EQ(loaded->header().vocab_size, orig->header().vocab_size);
    ASSERT_EQ(loaded->header().embedding_dim, orig->header().embedding_dim);
    ASSERT_EQ(loaded->header().hidden_dim, orig->header().hidden_dim);
    ASSERT_EQ(loaded->header().num_classes, orig->header().num_classes);

    ASSERT_EQ(loaded->labels().size(), orig->labels().size());
    for (size_t i = 0; i < orig->labels().size(); ++i) {
        ASSERT_EQ(loaded->labels()[i], orig->labels()[i]);
    }

    ASSERT_EQ(loaded->vocab().size(), orig->vocab().size());
    for (size_t i = 0; i < orig->vocab().size(); ++i) {
        ASSERT_EQ(loaded->vocab()[i], orig->vocab()[i]);
    }

    ASSERT_EQ(loaded->merge_rules().size(), orig->merge_rules().size());
    for (size_t i = 0; i < orig->merge_rules().size(); ++i) {
        ASSERT_EQ(loaded->merge_rules()[i].token1, orig->merge_rules()[i].token1);
        ASSERT_EQ(loaded->merge_rules()[i].token2, orig->merge_rules()[i].token2);
        ASSERT_EQ(loaded->merge_rules()[i].target, orig->merge_rules()[i].target);
    }

    for (size_t i = 0; i < orig->weights().embedding.size(); ++i) {
        ASSERT_EQ(loaded->weights().embedding[i], orig->weights().embedding[i]);
    }
}

void test_invalid_magic_rejection() {
    auto orig = create_sample_model(1);
    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    serialize_model(ss, *orig);

    std::string data = ss.str();
    data[0] = 'X'; // corrupt magic

    std::stringstream corrupted_ss(data, std::ios::in | std::ios::out | std::ios::binary);
    bool caught = false;
    try {
        deserialize_model(corrupted_ss);
    } catch (const IntelliBranchException&) {
        caught = true;
    }
    ASSERT_TRUE(caught);
}

void test_checksum_verification_failure() {
    auto orig = create_sample_model(1);
    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    serialize_model(ss, *orig);

    std::string data = ss.str();
    data[data.size() - 40] ^= 0xFF; // corrupt tensor section before checksum

    std::stringstream corrupted_ss(data, std::ios::in | std::ios::out | std::ios::binary);
    bool caught = false;
    try {
        deserialize_model(corrupted_ss);
    } catch (const ChecksumFailedException&) {
        caught = true;
    }
    ASSERT_TRUE(caught);
}

void test_file_io() {
    std::string path = "test_model_temp.bin";
    auto orig = create_sample_model(1);

    save_binary_model(path, *orig);
    ASSERT_TRUE(fs::exists(path));

    auto loaded = load_binary_model(path);
    ASSERT_EQ(loaded->header().vocab_size, orig->header().vocab_size);

    fs::remove(path);
}

void test_format_v2_round_trip() {
    auto v2_model = create_sample_model(2);
    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);

    serialize_model(ss, *v2_model);
    auto loaded = deserialize_model(ss);

    ASSERT_EQ(loaded->header().version, 2u);
    size_t pos_len = MAX_SEQUENCE_TOKENS * loaded->header().embedding_dim;
    for (size_t i = 0; i < pos_len; ++i) {
        ASSERT_EQ(loaded->weights().positional[i], v2_model->weights().positional[i]);
    }
}

int main() {
    test_serialization_round_trip();
    test_invalid_magic_rejection();
    test_checksum_verification_failure();
    test_file_io();
    test_format_v2_round_trip();

    std::cout << "[PASS] test_binary passed all test cases successfully.\n";
    return 0;
}
