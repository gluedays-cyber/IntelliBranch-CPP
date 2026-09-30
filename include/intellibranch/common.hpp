#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>

namespace intellibranch {

// Magic bytes for IBRN format ("IBRN")
inline constexpr std::array<uint8_t, 4> MAGIC_BYTES = {'I', 'B', 'R', 'N'};

// Format version: 2 supports positional embeddings
inline constexpr uint32_t CURRENT_FORMAT_VERSION = 2;

inline constexpr size_t MAX_INPUT_BYTES = 512;
inline constexpr size_t MAX_SEQUENCE_TOKENS = 128;
inline constexpr size_t MAX_GATE_CLASSES = 16;
inline constexpr size_t MAX_GATE_EMB_DIM = 64;

inline constexpr float SQRT_2_OVER_PI = 0.7978845608f;
inline constexpr float GELU_COEFF = 0.044715f;
inline constexpr float NUMERICAL_CLAMP_LIMIT = 100.0f;

struct Header {
    std::array<uint8_t, 4> magic = MAGIC_BYTES;
    uint32_t version = CURRENT_FORMAT_VERSION;
    uint32_t vocab_size = 0;
    uint32_t embedding_dim = 0;
    uint32_t hidden_dim = 0;
    uint32_t num_classes = 0;

    bool operator==(const Header& o) const = default;
};

struct MergeRule {
    uint32_t token1 = 0;
    uint32_t token2 = 0;
    uint32_t target = 0;

    bool operator==(const MergeRule& o) const = default;
};

struct Weights {
    std::vector<float> embedding;   // [vocab_size * embedding_dim]
    std::vector<float> positional;  // [max_seq_tokens * embedding_dim]
    std::vector<float> w1;          // [embedding_dim * hidden_dim]
    std::vector<float> b1;          // [hidden_dim]
    std::vector<float> w2;          // [hidden_dim * num_classes]
    std::vector<float> b2;          // [num_classes]
};

struct DataSample {
    std::string text;
    std::string label;
};

struct MatchSlot {
    int16_t index = -1;
    float confidence = 0.0f;
};

struct StaticInferenceResult {
    MatchSlot primary;
    MatchSlot secondary;
    float entropy = 0.0f;
    uint8_t total = 0;
};

struct DispatchPolicy {
    double high_threshold = 0.75;
    double low_threshold = 0.40;
    double margin_cutoff = 0.15;
    double max_entropy = 2.0;
    double pipeline_threshold = 0.30;
    double min_log_sum_exp = 0.0;
};

inline DispatchPolicy default_dispatch_policy() {
    return DispatchPolicy{};
}

struct RouteTrace {
    std::string input_text;
    std::vector<uint32_t> token_ids;
    std::vector<std::string> subwords;
    double unknown_token_ratio = 0.0;
    std::map<std::string, float> class_probabilities;
    std::string predicted_label;
    std::string secondary_label;
    double confidence = 0.0;
    double margin = 0.0;
    double entropy = 0.0;
    double threshold = 0.0;
    bool is_ambiguous = false;
    bool is_pipeline = false;
    bool is_fallback = false;
    std::string fallback_reason;
    int64_t latency_micros = 0;
};

struct GateTrace {
    std::string input_text;
    std::vector<uint32_t> token_ids;
    std::vector<std::string> subwords;
    double unknown_token_ratio = 0.0;
    float cosine_similarity = 1.0f;
    bool is_ood = false;
    uint64_t anchor_bitmask = 0;
    std::vector<std::string> triggered_anchors;
    std::map<std::string, float> class_probabilities;
    std::string predicted_label;
    std::string secondary_label;
    double confidence = 0.0;
    double margin = 0.0;
    double entropy = 0.0;
    double log_sum_exp = 0.0;
    double free_energy = 0.0;
    double threshold = 0.0;
    bool is_ambiguous = false;
    bool is_pipeline = false;
    bool is_fallback = false;
    std::string fallback_reason;
    int64_t latency_micros = 0;
};

struct TelemetryEvent {
    std::string input_text;
    std::string predicted_label;
    std::string secondary_label;
    double confidence = 0.0;
    double entropy = 0.0;
    bool is_ambiguous = false;
    bool is_pipeline = false;
    bool is_fallback = false;
    int64_t timestamp_nano = 0;
};

struct AnchorRule {
    int class_index = 0;
    uint64_t mask = 0;
    float weight = 0.0f;
    std::vector<std::string> keywords;
};

using RouteAction = std::function<void(void* payload)>;
using AmbiguousAction = std::function<void(const std::string& primary, const std::string& secondary, void* payload)>;
using PipelineAction = std::function<void(const std::string& primary, const std::string& secondary, void* payload)>;

// Exceptions
class IntelliBranchException : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class ChecksumFailedException : public IntelliBranchException {
public:
    ChecksumFailedException() : IntelliBranchException("checksum verification failed: model file corrupted") {}
};

class InvalidMagicException : public IntelliBranchException {
public:
    InvalidMagicException() : IntelliBranchException("invalid binary format: missing IBRN magic header") {}
};

class UnsupportedVersionException : public IntelliBranchException {
public:
    explicit UnsupportedVersionException(uint32_t v) 
        : IntelliBranchException("unsupported model format version: " + std::to_string(v)) {}
};

class EmptyInputException : public IntelliBranchException {
public:
    EmptyInputException() : IntelliBranchException("input token slice or text cannot be empty") {}
};

} // namespace intellibranch
