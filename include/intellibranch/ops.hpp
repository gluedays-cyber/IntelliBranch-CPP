#pragma once

#include "intellibranch/common.hpp"
#include <cstdint>
#include <cstddef>
#include <span>
#include <vector>

namespace intellibranch {

// Clamps value within [-limit, limit] and replaces NaN/Inf with 0.0f
float safe_clamp(float val, float limit = NUMERICAL_CLAMP_LIMIT);

// GELU activation via tanh approximation:
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
float gelu(float x);

// Applies GELU in-place across a float slice
void gelu_in_place(float* vec, size_t n);
inline void gelu_in_place(std::span<float> vec) {
    gelu_in_place(vec.data(), vec.size());
}
inline void gelu_in_place(std::vector<float>& vec) {
    gelu_in_place(vec.data(), vec.size());
}

// Derivative of GELU with respect to x for backpropagation
float gelu_derivative(float x);

// Computes average embedding vector across token IDs with learned positional embeddings.
// out must have size at least emb_dim.
void mean_pooling_with_pos(
    const uint32_t* token_ids,
    size_t seq_len,
    const float* embedding_table,
    size_t vocab_size,
    const float* pos_table,
    size_t max_pos_tokens,
    size_t emb_dim,
    float* out
);

inline void mean_pooling_with_pos(
    std::span<const uint32_t> token_ids,
    std::span<const float> embedding_table,
    std::span<const float> pos_table,
    size_t emb_dim,
    std::span<float> out
) {
    size_t vocab_size = emb_dim > 0 ? embedding_table.size() / emb_dim : 0;
    size_t max_pos = (emb_dim > 0 && !pos_table.empty()) ? pos_table.size() / emb_dim : 0;
    mean_pooling_with_pos(
        token_ids.data(), token_ids.size(),
        embedding_table.data(), vocab_size,
        pos_table.empty() ? nullptr : pos_table.data(), max_pos,
        emb_dim, out.data()
    );
}

// Computes average embedding vector without positional encoding
inline void mean_pooling(
    const uint32_t* token_ids,
    size_t seq_len,
    const float* embedding_table,
    size_t vocab_size,
    size_t emb_dim,
    float* out
) {
    mean_pooling_with_pos(token_ids, seq_len, embedding_table, vocab_size, nullptr, 0, emb_dim, out);
}

inline void mean_pooling(
    std::span<const uint32_t> token_ids,
    std::span<const float> embedding_table,
    size_t emb_dim,
    std::span<float> out
) {
    size_t vocab_size = emb_dim > 0 ? embedding_table.size() / emb_dim : 0;
    mean_pooling(token_ids.data(), token_ids.size(), embedding_table.data(), vocab_size, emb_dim, out.data());
}

// Computes out = vec * weights + bias
// vec: [1 x in_dim], weights: [in_dim x out_dim] (row-major), bias: [out_dim], out: [out_dim]
void matmul_vec_add(
    const float* vec,
    const float* weights,
    const float* bias,
    size_t in_dim,
    size_t out_dim,
    float* out
);

inline void matmul_vec_add(
    std::span<const float> vec,
    std::span<const float> weights,
    std::span<const float> bias,
    size_t in_dim,
    size_t out_dim,
    std::span<float> out
) {
    matmul_vec_add(vec.data(), weights.data(), bias.data(), in_dim, out_dim, out.data());
}

// Numerically stable softmax with temperature scaling
void softmax(const float* logits, size_t n, float temperature, float* out);

inline void softmax(std::span<const float> logits, float temperature, std::span<float> out) {
    softmax(logits.data(), logits.size(), temperature, out.data());
}

// Computes Euclidean L2 normalization: out = vec / ||vec||2
// Returns the original Euclidean norm.
float l2_normalize(const float* vec, size_t n, float* out);

inline float l2_normalize(std::span<const float> vec, std::span<float> out) {
    return l2_normalize(vec.data(), vec.size(), out.data());
}

// Dot product between two vectors
float dot_product(const float* a, const float* b, size_t n);

inline float dot_product(std::span<const float> a, std::span<const float> b) {
    size_t n = std::min(a.size(), b.size());
    return dot_product(a.data(), b.data(), n);
}

// Shannon entropy in bits with epsilon guard
float compute_entropy(const float* probs, size_t n);

inline float compute_entropy(std::span<const float> probs) {
    return compute_entropy(probs.data(), probs.size());
}

} // namespace intellibranch
