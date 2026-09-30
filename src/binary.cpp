#include "intellibranch/binary.hpp"
#include "intellibranch/sha256.hpp"
#include <fstream>
#include <cstring>
#include <vector>

namespace intellibranch {

namespace {

inline void write_u32_le(std::vector<uint8_t>& buf, uint32_t val) {
    buf.push_back(static_cast<uint8_t>(val & 0xFF));
    buf.push_back(static_cast<uint8_t>((val >> 8) & 0xFF));
    buf.push_back(static_cast<uint8_t>((val >> 16) & 0xFF));
    buf.push_back(static_cast<uint8_t>((val >> 24) & 0xFF));
}

inline void write_float32_le(std::vector<uint8_t>& buf, float val) {
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(float));
    write_u32_le(buf, bits);
}

inline uint32_t read_u32_le(const uint8_t* ptr) {
    return static_cast<uint32_t>(ptr[0]) |
          (static_cast<uint32_t>(ptr[1]) << 8) |
          (static_cast<uint32_t>(ptr[2]) << 16) |
          (static_cast<uint32_t>(ptr[3]) << 24);
}

inline float read_float32_le(const uint8_t* ptr) {
    uint32_t bits = read_u32_le(ptr);
    float val;
    std::memcpy(&val, &bits, sizeof(float));
    return val;
}

} // namespace

void serialize_model(std::ostream& os, const InferenceModel& model) {
    std::vector<uint8_t> payload;
    const auto& hdr = model.header();

    // 1. Header
    for (uint8_t b : hdr.magic) {
        payload.push_back(b);
    }
    write_u32_le(payload, hdr.version);
    write_u32_le(payload, hdr.vocab_size);
    write_u32_le(payload, hdr.embedding_dim);
    write_u32_le(payload, hdr.hidden_dim);
    write_u32_le(payload, hdr.num_classes);

    // 2. Labels Block
    const auto& labels = model.labels();
    write_u32_le(payload, static_cast<uint32_t>(labels.size()));
    for (const auto& lbl : labels) {
        write_u32_le(payload, static_cast<uint32_t>(lbl.size()));
        payload.insert(payload.end(), lbl.begin(), lbl.end());
    }

    // 3. Vocabulary Block
    const auto& vocab = model.vocab();
    write_u32_le(payload, static_cast<uint32_t>(vocab.size()));
    for (const auto& tok : vocab) {
        write_u32_le(payload, static_cast<uint32_t>(tok.size()));
        payload.insert(payload.end(), tok.begin(), tok.end());
    }

    // 4. Merge Rules Block
    const auto& rules = model.merge_rules();
    write_u32_le(payload, static_cast<uint32_t>(rules.size()));
    for (const auto& r : rules) {
        write_u32_le(payload, r.token1);
        write_u32_le(payload, r.token2);
        write_u32_le(payload, r.target);
    }

    // 5. Tensor Blocks
    const auto& w = model.weights();
    for (float f : w.embedding) {
        write_float32_le(payload, f);
    }

    if (hdr.version >= 2) {
        size_t pos_len = MAX_SEQUENCE_TOKENS * hdr.embedding_dim;
        for (size_t i = 0; i < pos_len; ++i) {
            float f = (i < w.positional.size()) ? w.positional[i] : 0.0f;
            write_float32_le(payload, f);
        }
    }

    for (float f : w.w1) {
        write_float32_le(payload, f);
    }
    for (float f : w.b1) {
        write_float32_le(payload, f);
    }
    for (float f : w.w2) {
        write_float32_le(payload, f);
    }
    for (float f : w.b2) {
        write_float32_le(payload, f);
    }

    // 6. Checksum Block
    auto checksum = SHA256::hash(payload.data(), payload.size());

    // Write all to os
    os.write(reinterpret_cast<const char*>(payload.data()), payload.size());
    os.write(reinterpret_cast<const char*>(checksum.data()), checksum.size());
}

std::shared_ptr<InferenceModel> deserialize_model(std::istream& is) {
    // Read entire stream into memory buffer
    std::vector<uint8_t> buffer((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
    if (buffer.size() < 32 + 24) { // At least header + checksum
        throw IntelliBranchException("model file too short");
    }

    size_t payload_len = buffer.size() - 32;
    const uint8_t* payload_ptr = buffer.data();
    const uint8_t* checksum_ptr = buffer.data() + payload_len;

    auto computed_checksum = SHA256::hash(payload_ptr, payload_len);
    if (std::memcmp(computed_checksum.data(), checksum_ptr, 32) != 0) {
        throw ChecksumFailedException();
    }

    size_t offset = 0;

    // 1. Read Header Block
    Header header;
    std::memcpy(header.magic.data(), payload_ptr + offset, 4);
    offset += 4;

    if (header.magic != MAGIC_BYTES) {
        throw InvalidMagicException();
    }

    header.version = read_u32_le(payload_ptr + offset); offset += 4;
    if (header.version != 1 && header.version != 2) {
        throw UnsupportedVersionException(header.version);
    }

    header.vocab_size = read_u32_le(payload_ptr + offset); offset += 4;
    header.embedding_dim = read_u32_le(payload_ptr + offset); offset += 4;
    header.hidden_dim = read_u32_le(payload_ptr + offset); offset += 4;
    header.num_classes = read_u32_le(payload_ptr + offset); offset += 4;

    // 2. Read Labels Block
    uint32_t num_labels = read_u32_le(payload_ptr + offset); offset += 4;
    if (num_labels != header.num_classes) {
        throw IntelliBranchException("num labels does not match num_classes");
    }

    std::vector<std::string> labels(num_labels);
    for (uint32_t i = 0; i < num_labels; ++i) {
        uint32_t str_len = read_u32_le(payload_ptr + offset); offset += 4;
        labels[i] = std::string(reinterpret_cast<const char*>(payload_ptr + offset), str_len);
        offset += str_len;
    }

    // 3. Read Vocabulary Block
    uint32_t vocab_count = read_u32_le(payload_ptr + offset); offset += 4;
    if (vocab_count != header.vocab_size) {
        throw IntelliBranchException("vocab count does not match vocab_size");
    }

    std::vector<std::string> vocab(vocab_count);
    for (uint32_t i = 0; i < vocab_count; ++i) {
        uint32_t str_len = read_u32_le(payload_ptr + offset); offset += 4;
        vocab[i] = std::string(reinterpret_cast<const char*>(payload_ptr + offset), str_len);
        offset += str_len;
    }

    // 4. Read Merge Rules Block
    uint32_t num_rules = read_u32_le(payload_ptr + offset); offset += 4;
    std::vector<MergeRule> merge_rules(num_rules);
    for (uint32_t i = 0; i < num_rules; ++i) {
        merge_rules[i].token1 = read_u32_le(payload_ptr + offset); offset += 4;
        merge_rules[i].token2 = read_u32_le(payload_ptr + offset); offset += 4;
        merge_rules[i].target = read_u32_le(payload_ptr + offset); offset += 4;
    }

    // 5. Read Tensor Blocks
    Weights weights;
    size_t emb_len = header.vocab_size * header.embedding_dim;
    weights.embedding.resize(emb_len);
    for (size_t i = 0; i < emb_len; ++i) {
        weights.embedding[i] = read_float32_le(payload_ptr + offset); offset += 4;
    }

    size_t pos_len = MAX_SEQUENCE_TOKENS * header.embedding_dim;
    weights.positional.resize(pos_len, 0.0f);
    if (header.version >= 2) {
        for (size_t i = 0; i < pos_len; ++i) {
            weights.positional[i] = read_float32_le(payload_ptr + offset); offset += 4;
        }
    }

    size_t w1_len = header.embedding_dim * header.hidden_dim;
    weights.w1.resize(w1_len);
    for (size_t i = 0; i < w1_len; ++i) {
        weights.w1[i] = read_float32_le(payload_ptr + offset); offset += 4;
    }

    size_t b1_len = header.hidden_dim;
    weights.b1.resize(b1_len);
    for (size_t i = 0; i < b1_len; ++i) {
        weights.b1[i] = read_float32_le(payload_ptr + offset); offset += 4;
    }

    size_t w2_len = header.hidden_dim * header.num_classes;
    weights.w2.resize(w2_len);
    for (size_t i = 0; i < w2_len; ++i) {
        weights.w2[i] = read_float32_le(payload_ptr + offset); offset += 4;
    }

    size_t b2_len = header.num_classes;
    weights.b2.resize(b2_len);
    for (size_t i = 0; i < b2_len; ++i) {
        weights.b2[i] = read_float32_le(payload_ptr + offset); offset += 4;
    }

    return std::make_shared<InferenceModel>(
        header, std::move(labels), std::move(vocab), std::move(merge_rules), std::move(weights)
    );
}

void save_binary_model(const std::string& file_path, const InferenceModel& model) {
    std::ofstream ofs(file_path, std::ios::binary);
    if (!ofs) {
        throw IntelliBranchException("failed to open file for writing: " + file_path);
    }
    serialize_model(ofs, model);
}

std::shared_ptr<InferenceModel> load_binary_model(const std::string& file_path) {
    std::ifstream ifs(file_path, std::ios::binary);
    if (!ifs) {
        throw IntelliBranchException("failed to open model file: " + file_path);
    }
    return deserialize_model(ifs);
}

} // namespace intellibranch
