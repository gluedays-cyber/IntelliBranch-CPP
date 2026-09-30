#include "intellibranch/ops.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace intellibranch {

float safe_clamp(float val, float limit) {
    if (std::isnan(val) || std::isinf(val)) {
        return 0.0f;
    }
    if (val < -limit) {
        return -limit;
    }
    if (val > limit) {
        return limit;
    }
    return val;
}

float gelu(float x) {
    x = safe_clamp(x, NUMERICAL_CLAMP_LIMIT);
    float cube = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * cube);
    inner = safe_clamp(inner, NUMERICAL_CLAMP_LIMIT);
    float tanh_val = std::tanh(inner);
    return safe_clamp(0.5f * x * (1.0f + tanh_val), NUMERICAL_CLAMP_LIMIT);
}

void gelu_in_place(float* vec, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        vec[i] = gelu(vec[i]);
    }
}

float gelu_derivative(float x) {
    float cube = x * x * x;
    float u = SQRT_2_OVER_PI * (x + GELU_COEFF * cube);
    float tanh_u = std::tanh(u);
    float du = SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_COEFF * x * x);
    float sech2 = 1.0f - tanh_u * tanh_u;
    return 0.5f * (1.0f + tanh_u) + 0.5f * x * sech2 * du;
}

void mean_pooling_with_pos(
    const uint32_t* token_ids,
    size_t seq_len,
    const float* embedding_table,
    size_t vocab_size,
    const float* pos_table,
    size_t max_pos_tokens,
    size_t emb_dim,
    float* out
) {
    if (seq_len == 0) {
        throw IntelliBranchException("cannot pool over zero tokens");
    }

    for (size_t d = 0; d < emb_dim; ++d) {
        out[d] = 0.0f;
    }

    for (size_t pos = 0; pos < seq_len; ++pos) {
        uint32_t id = token_ids[pos];
        if (id >= vocab_size) {
            throw IntelliBranchException("token ID exceeds embedding table bounds");
        }

        size_t tok_offset = static_cast<size_t>(id) * emb_dim;
        bool has_pos = (pos_table != nullptr && pos < max_pos_tokens);
        size_t pos_offset = pos * emb_dim;

        for (size_t d = 0; d < emb_dim; ++d) {
            float val = safe_clamp(embedding_table[tok_offset + d], NUMERICAL_CLAMP_LIMIT);
            if (has_pos) {
                val = gelu(val + pos_table[pos_offset + d]);
            }
            out[d] = safe_clamp(out[d] + val, NUMERICAL_CLAMP_LIMIT);
        }
    }

    float inv_len = 1.0f / static_cast<float>(seq_len);
    for (size_t d = 0; d < emb_dim; ++d) {
        out[d] = safe_clamp(out[d] * inv_len, NUMERICAL_CLAMP_LIMIT);
    }
}

void matmul_vec_add(
    const float* vec,
    const float* weights,
    const float* bias,
    size_t in_dim,
    size_t out_dim,
    float* out
) {
    for (size_t j = 0; j < out_dim; ++j) {
        out[j] = safe_clamp(bias[j], NUMERICAL_CLAMP_LIMIT);
    }

    for (size_t i = 0; i < in_dim; ++i) {
        float v = safe_clamp(vec[i], NUMERICAL_CLAMP_LIMIT);
        if (v == 0.0f) {
            continue;
        }
        size_t row_offset = i * out_dim;
        for (size_t j = 0; j < out_dim; ++j) {
            float w = safe_clamp(weights[row_offset + j], NUMERICAL_CLAMP_LIMIT);
            out[j] = safe_clamp(out[j] + v * w, NUMERICAL_CLAMP_LIMIT);
        }
    }
}

void softmax(const float* logits, size_t n, float temperature, float* out) {
    if (n == 0) {
        throw IntelliBranchException("empty logits");
    }
    if (temperature <= 0.0f || std::isnan(temperature) || std::isinf(temperature)) {
        temperature = 1.0f;
    }

    float inv_temp = 1.0f / temperature;

    float max_logit = safe_clamp(logits[0], NUMERICAL_CLAMP_LIMIT) * inv_temp;
    for (size_t i = 1; i < n; ++i) {
        float scaled = safe_clamp(logits[i], NUMERICAL_CLAMP_LIMIT) * inv_temp;
        if (scaled > max_logit) {
            max_logit = scaled;
        }
    }

    float sum_exp = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float val = safe_clamp(logits[i], NUMERICAL_CLAMP_LIMIT) * inv_temp - max_logit;
        float e = std::exp(val);
        if (std::isnan(e) || std::isinf(e)) {
            e = 0.0f;
        }
        out[i] = e;
        sum_exp += e;
    }

    if (sum_exp <= 0.0f || std::isnan(sum_exp) || std::isinf(sum_exp)) {
        float uniform = 1.0f / static_cast<float>(n);
        for (size_t i = 0; i < n; ++i) {
            out[i] = uniform;
        }
        return;
    }

    float inv_sum = 1.0f / sum_exp;
    for (size_t i = 0; i < n; ++i) {
        out[i] *= inv_sum;
    }
}

float l2_normalize(const float* vec, size_t n, float* out) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum_sq += static_cast<double>(vec[i]) * static_cast<double>(vec[i]);
    }
    float norm = static_cast<float>(std::sqrt(sum_sq));
    if (norm < 1e-7f || std::isnan(norm) || std::isinf(norm)) {
        for (size_t i = 0; i < n; ++i) {
            out[i] = 0.0f;
        }
        return 0.0f;
    }
    float inv_norm = 1.0f / norm;
    for (size_t i = 0; i < n; ++i) {
        out[i] = vec[i] * inv_norm;
    }
    return norm;
}

float dot_product(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

float compute_entropy(const float* probs, size_t n) {
    double entropy = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float p = probs[i];
        if (p > 1e-7f) {
            entropy -= static_cast<double>(p) * std::log2(static_cast<double>(p));
        }
    }
    return static_cast<float>(entropy);
}

} // namespace intellibranch
