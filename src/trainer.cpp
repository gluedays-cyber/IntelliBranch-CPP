#include "intellibranch/trainer.hpp"
#include "intellibranch/ops.hpp"
#include <fstream>
#include <sstream>
#include <random>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <algorithm>

namespace intellibranch {

AdamWState::AdamWState(size_t size) : m_(size, 0.0f), v_(size, 0.0f), t_(0) {}

void AdamWState::step(
    float* param,
    const float* grad,
    size_t n,
    float lr,
    float weight_decay,
    float beta1,
    float beta2,
    float eps
) {
    t_++;
    double bc1 = 1.0 - std::pow(static_cast<double>(beta1), static_cast<double>(t_));
    double bc2 = 1.0 - std::pow(static_cast<double>(beta2), static_cast<double>(t_));

    for (size_t i = 0; i < n; ++i) {
        float g = grad[i];
        m_[i] = beta1 * m_[i] + (1.0f - beta1) * g;
        v_[i] = beta2 * v_[i] + (1.0f - beta2) * (g * g);

        float m_hat = m_[i] / static_cast<float>(bc1);
        float v_hat = v_[i] / static_cast<float>(bc2);

        param[i] -= lr * (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * param[i]);
    }
}

namespace {

std::string trim(const std::string& str) {
    size_t start = 0;
    while (start < str.size() && (std::isspace(static_cast<unsigned char>(str[start])) != 0)) start++;
    if (start == str.size()) return "";
    size_t end = str.size() - 1;
    while (end > start && (std::isspace(static_cast<unsigned char>(str[end])) != 0)) end--;
    return str.substr(start, end - start + 1);
}

std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    bool in_quotes = false;

    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            if (in_quotes && i + 1 < line.size() && line[i + 1] == '"') {
                current += '"';
                i++;
            } else {
                in_quotes = !in_quotes;
            }
        } else if (c == ',' && !in_quotes) {
            fields.push_back(trim(current));
            current.clear();
        } else {
            current += c;
        }
    }
    fields.push_back(trim(current));
    return fields;
}

} // namespace

std::vector<DataSample> load_csv_dataset(const std::string& file_path) {
    std::ifstream ifs(file_path);
    if (!ifs) {
        throw IntelliBranchException("failed to open dataset file: " + file_path);
    }

    std::string line;
    // Read header line
    if (!std::getline(ifs, line)) {
        throw IntelliBranchException("dataset is empty or failed to read header: " + file_path);
    }

    auto header = parse_csv_line(line);
    if (header.size() < 2) {
        throw IntelliBranchException("CSV must contain at least 2 columns (text, label)");
    }

    std::vector<DataSample> samples;
    while (std::getline(ifs, line)) {
        if (trim(line).empty()) continue;
        auto row = parse_csv_line(line);
        if (row.size() < 2) continue;
        if (!row[0].empty() && !row[1].empty()) {
            samples.push_back(DataSample{row[0], row[1]});
        }
    }

    if (samples.empty()) {
        throw IntelliBranchException("dataset contains no valid records");
    }

    return samples;
}

std::shared_ptr<InferenceModel> train_model(const std::vector<DataSample>& samples, const TrainConfig& cfg) {
    std::mt19937_64 rng(cfg.seed);

    // 1. Collect unique labels
    std::unordered_map<std::string, uint32_t> label_map;
    std::vector<std::string> labels;
    for (const auto& s : samples) {
        if (!label_map.contains(s.label)) {
            label_map[s.label] = static_cast<uint32_t>(labels.size());
            labels.push_back(s.label);
        }
    }

    size_t num_classes = labels.size();
    if (num_classes < 2) {
        throw IntelliBranchException("dataset must contain at least 2 distinct classes");
    }

    // 2. Train Pure C++ BPE Tokenizer
    std::vector<std::string> corpus;
    corpus.reserve(samples.size());
    for (const auto& s : samples) {
        corpus.push_back(s.text);
    }
    BPETokenizer tokenizer = BPETokenizer::train_bpe(corpus, cfg.target_vocab_size);
    size_t vocab_size = tokenizer.vocab_size();

    // 3. Prepare Encoded Dataset
    struct EncodedSample {
        std::vector<uint32_t> tokens;
        uint32_t class_id;
    };

    std::unordered_map<uint32_t, std::vector<EncodedSample>> class_buckets;
    for (const auto& s : samples) {
        auto tokens = tokenizer.encode(s.text);
        if (tokens.empty()) {
            tokens = {0};
        }
        uint32_t cid = label_map[s.label];
        class_buckets[cid].push_back(EncodedSample{tokens, cid});
    }

    std::vector<EncodedSample> train_set;
    std::vector<EncodedSample> val_set;

    // Stratified split
    for (auto& [cid, bucket] : class_buckets) {
        std::shuffle(bucket.begin(), bucket.end(), rng);
        if (bucket.size() <= 2) {
            train_set.insert(train_set.end(), bucket.begin(), bucket.end());
            val_set.insert(val_set.end(), bucket.begin(), bucket.end());
        } else {
            size_t val_cnt = static_cast<size_t>(static_cast<double>(bucket.size()) * 0.15);
            if (val_cnt < 1) val_cnt = 1;
            val_set.insert(val_set.end(), bucket.begin(), bucket.begin() + val_cnt);
            train_set.insert(train_set.end(), bucket.begin() + val_cnt, bucket.end());
        }
    }

    // 4. Initialize Network Weights
    Header header{
        .magic = MAGIC_BYTES,
        .version = CURRENT_FORMAT_VERSION,
        .vocab_size = static_cast<uint32_t>(vocab_size),
        .embedding_dim = static_cast<uint32_t>(cfg.embedding_dim),
        .hidden_dim = static_cast<uint32_t>(cfg.hidden_dim),
        .num_classes = static_cast<uint32_t>(num_classes),
    };

    Weights weights{
        .embedding = std::vector<float>(vocab_size * cfg.embedding_dim),
        .positional = std::vector<float>(MAX_SEQUENCE_TOKENS * cfg.embedding_dim),
        .w1 = std::vector<float>(cfg.embedding_dim * cfg.hidden_dim),
        .b1 = std::vector<float>(cfg.hidden_dim, 0.0f),
        .w2 = std::vector<float>(cfg.hidden_dim * num_classes),
        .b2 = std::vector<float>(num_classes, 0.0f),
    };

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    float emb_scale = std::sqrt(1.0f / static_cast<float>(cfg.embedding_dim));
    for (auto& f : weights.embedding) f = dist(rng) * emb_scale;
    for (auto& f : weights.positional) f = dist(rng) * emb_scale;

    float w1_scale = std::sqrt(2.0f / static_cast<float>(cfg.embedding_dim));
    for (auto& f : weights.w1) f = dist(rng) * w1_scale;

    float w2_scale = std::sqrt(2.0f / static_cast<float>(cfg.hidden_dim));
    for (auto& f : weights.w2) f = dist(rng) * w2_scale;

    // AdamW Optimizers
    AdamWState opt_emb(weights.embedding.size());
    AdamWState opt_pos(weights.positional.size());
    AdamWState opt_w1(weights.w1.size());
    AdamWState opt_b1(weights.b1.size());
    AdamWState opt_w2(weights.w2.size());
    AdamWState opt_b2(weights.b2.size());

    // Gradient Accumulator Buffers
    std::vector<float> grad_emb(weights.embedding.size(), 0.0f);
    std::vector<float> grad_pos(weights.positional.size(), 0.0f);
    std::vector<float> grad_w1(weights.w1.size(), 0.0f);
    std::vector<float> grad_b1(weights.b1.size(), 0.0f);
    std::vector<float> grad_w2(weights.w2.size(), 0.0f);
    std::vector<float> grad_b2(weights.b2.size(), 0.0f);

    // Scratch buffers
    std::vector<float> pooled(cfg.embedding_dim);
    std::vector<float> z1(cfg.hidden_dim);
    std::vector<float> a1(cfg.hidden_dim);
    std::vector<float> z2(num_classes);
    std::vector<float> probs(num_classes);

    std::vector<float> d_z2(num_classes);
    std::vector<float> d_a1(cfg.hidden_dim);
    std::vector<float> d_z1(cfg.hidden_dim);
    std::vector<float> d_mean(cfg.embedding_dim);

    float best_val_loss = std::numeric_limits<float>::max();
    float best_train_loss = std::numeric_limits<float>::max();
    size_t patience_counter = 0;
    Weights best_weights = weights;

    size_t batch_size = cfg.batch_size;
    if (train_set.size() <= 32) {
        batch_size = 4;
    } else if (batch_size > train_set.size()) {
        batch_size = train_set.size();
    }
    if (batch_size == 0) batch_size = 1;

    // 5. Training Loop
    for (size_t epoch = 1; epoch <= cfg.epochs; ++epoch) {
        std::shuffle(train_set.begin(), train_set.end(), rng);

        std::fill(grad_emb.begin(), grad_emb.end(), 0.0f);
        std::fill(grad_pos.begin(), grad_pos.end(), 0.0f);
        std::fill(grad_w1.begin(), grad_w1.end(), 0.0f);
        std::fill(grad_b1.begin(), grad_b1.end(), 0.0f);
        std::fill(grad_w2.begin(), grad_w2.end(), 0.0f);
        std::fill(grad_b2.begin(), grad_b2.end(), 0.0f);

        size_t accum_count = 0;

        for (size_t idx = 0; idx < train_set.size(); ++idx) {
            const auto& sample = train_set[idx];

            // Forward
            mean_pooling_with_pos(
                sample.tokens.data(), sample.tokens.size(),
                weights.embedding.data(), vocab_size,
                weights.positional.data(), MAX_SEQUENCE_TOKENS,
                cfg.embedding_dim, pooled.data()
            );

            matmul_vec_add(pooled.data(), weights.w1.data(), weights.b1.data(), cfg.embedding_dim, cfg.hidden_dim, z1.data());
            for (size_t i = 0; i < cfg.hidden_dim; ++i) {
                a1[i] = gelu(z1[i]);
            }
            matmul_vec_add(a1.data(), weights.w2.data(), weights.b2.data(), cfg.hidden_dim, num_classes, z2.data());
            softmax(z2.data(), num_classes, 1.0f, probs.data());

            // Backward
            for (size_t c = 0; c < num_classes; ++c) {
                d_z2[c] = probs[c];
            }
            d_z2[sample.class_id] -= 1.0f;

            for (size_t h = 0; h < cfg.hidden_dim; ++h) {
                float ah = a1[h];
                size_t row = h * num_classes;
                for (size_t c = 0; c < num_classes; ++c) {
                    grad_w2[row + c] += ah * d_z2[c];
                }
            }
            for (size_t c = 0; c < num_classes; ++c) {
                grad_b2[c] += d_z2[c];
            }

            for (size_t h = 0; h < cfg.hidden_dim; ++h) {
                float sum = 0.0f;
                size_t row = h * num_classes;
                for (size_t c = 0; c < num_classes; ++c) {
                    sum += d_z2[c] * weights.w2[row + c];
                }
                d_a1[h] = sum;
            }

            for (size_t h = 0; h < cfg.hidden_dim; ++h) {
                d_z1[h] = d_a1[h] * gelu_derivative(z1[h]);
            }

            for (size_t e = 0; e < cfg.embedding_dim; ++e) {
                float pe = pooled[e];
                size_t row = e * cfg.hidden_dim;
                for (size_t h = 0; h < cfg.hidden_dim; ++h) {
                    grad_w1[row + h] += pe * d_z1[h];
                }
            }
            for (size_t h = 0; h < cfg.hidden_dim; ++h) {
                grad_b1[h] += d_z1[h];
            }

            for (size_t e = 0; e < cfg.embedding_dim; ++e) {
                float sum = 0.0f;
                size_t row = e * cfg.hidden_dim;
                for (size_t h = 0; h < cfg.hidden_dim; ++h) {
                    sum += d_z1[h] * weights.w1[row + h];
                }
                d_mean[e] = sum;
            }

            float inv_len = 1.0f / static_cast<float>(sample.tokens.size());
            for (size_t pos = 0; pos < sample.tokens.size(); ++pos) {
                uint32_t tok = sample.tokens[pos];
                size_t tok_offset = static_cast<size_t>(tok) * cfg.embedding_dim;
                size_t pos_offset = pos * cfg.embedding_dim;
                for (size_t e = 0; e < cfg.embedding_dim; ++e) {
                    float sum_val = weights.embedding[tok_offset + e] + weights.positional[pos_offset + e];
                    float g = d_mean[e] * inv_len * gelu_derivative(sum_val);
                    grad_emb[tok_offset + e] += g;
                    if (pos_offset + cfg.embedding_dim <= weights.positional.size()) {
                        grad_pos[pos_offset + e] += g;
                    }
                }
            }

            accum_count++;

            if (accum_count % batch_size == 0 || idx == train_set.size() - 1) {
                float scale = 1.0f / static_cast<float>(accum_count);
                for (auto& g : grad_emb) g *= scale;
                for (auto& g : grad_pos) g *= scale;
                for (auto& g : grad_w1) g *= scale;
                for (auto& g : grad_b1) g *= scale;
                for (auto& g : grad_w2) g *= scale;
                for (auto& g : grad_b2) g *= scale;

                opt_emb.step(weights.embedding.data(), grad_emb.data(), weights.embedding.size(), cfg.learning_rate, 0.0f, cfg.beta1, cfg.beta2, cfg.epsilon);
                opt_pos.step(weights.positional.data(), grad_pos.data(), weights.positional.size(), cfg.learning_rate, 0.0f, cfg.beta1, cfg.beta2, cfg.epsilon);
                opt_w1.step(weights.w1.data(), grad_w1.data(), weights.w1.size(), cfg.learning_rate, cfg.weight_decay, cfg.beta1, cfg.beta2, cfg.epsilon);
                opt_b1.step(weights.b1.data(), grad_b1.data(), weights.b1.size(), cfg.learning_rate, 0.0f, cfg.beta1, cfg.beta2, cfg.epsilon);
                opt_w2.step(weights.w2.data(), grad_w2.data(), weights.w2.size(), cfg.learning_rate, cfg.weight_decay, cfg.beta1, cfg.beta2, cfg.epsilon);
                opt_b2.step(weights.b2.data(), grad_b2.data(), weights.b2.size(), cfg.learning_rate, 0.0f, cfg.beta1, cfg.beta2, cfg.epsilon);

                std::fill(grad_emb.begin(), grad_emb.end(), 0.0f);
                std::fill(grad_pos.begin(), grad_pos.end(), 0.0f);
                std::fill(grad_w1.begin(), grad_w1.end(), 0.0f);
                std::fill(grad_b1.begin(), grad_b1.end(), 0.0f);
                std::fill(grad_w2.begin(), grad_w2.end(), 0.0f);
                std::fill(grad_b2.begin(), grad_b2.end(), 0.0f);
                accum_count = 0;
            }
        }

        // Validation Evaluation
        float train_loss = 0.0f;
        size_t train_correct = 0;
        for (const auto& sample : train_set) {
            mean_pooling_with_pos(
                sample.tokens.data(), sample.tokens.size(),
                weights.embedding.data(), vocab_size,
                weights.positional.data(), MAX_SEQUENCE_TOKENS,
                cfg.embedding_dim, pooled.data()
            );
            matmul_vec_add(pooled.data(), weights.w1.data(), weights.b1.data(), cfg.embedding_dim, cfg.hidden_dim, z1.data());
            for (size_t i = 0; i < cfg.hidden_dim; ++i) a1[i] = gelu(z1[i]);
            matmul_vec_add(a1.data(), weights.w2.data(), weights.b2.data(), cfg.hidden_dim, num_classes, z2.data());
            softmax(z2.data(), num_classes, 1.0f, probs.data());

            float p = probs[sample.class_id];
            if (p < 1e-7f) p = 1e-7f;
            train_loss -= std::log(p);

            uint32_t pred = 0;
            float max_p = -1.0f;
            for (size_t c = 0; c < num_classes; ++c) {
                if (probs[c] > max_p) {
                    max_p = probs[c];
                    pred = static_cast<uint32_t>(c);
                }
            }
            if (pred == sample.class_id) train_correct++;
        }
        train_loss /= static_cast<float>(train_set.size());

        float val_loss = 0.0f;
        size_t val_correct = 0;
        for (const auto& sample : val_set) {
            mean_pooling_with_pos(
                sample.tokens.data(), sample.tokens.size(),
                weights.embedding.data(), vocab_size,
                weights.positional.data(), MAX_SEQUENCE_TOKENS,
                cfg.embedding_dim, pooled.data()
            );
            matmul_vec_add(pooled.data(), weights.w1.data(), weights.b1.data(), cfg.embedding_dim, cfg.hidden_dim, z1.data());
            for (size_t i = 0; i < cfg.hidden_dim; ++i) a1[i] = gelu(z1[i]);
            matmul_vec_add(a1.data(), weights.w2.data(), weights.b2.data(), cfg.hidden_dim, num_classes, z2.data());
            softmax(z2.data(), num_classes, 1.0f, probs.data());

            float p = probs[sample.class_id];
            if (p < 1e-7f) p = 1e-7f;
            val_loss -= std::log(p);

            uint32_t pred = 0;
            float max_p = -1.0f;
            for (size_t c = 0; c < num_classes; ++c) {
                if (probs[c] > max_p) {
                    max_p = probs[c];
                    pred = static_cast<uint32_t>(c);
                }
            }
            if (pred == sample.class_id) val_correct++;
        }
        val_loss /= static_cast<float>(val_set.size());

        bool is_better = false;
        if (samples.size() < 50) {
            if (train_loss < best_train_loss) {
                best_train_loss = train_loss;
                is_better = true;
            }
        } else {
            if (val_loss < best_val_loss) {
                best_val_loss = val_loss;
                is_better = true;
            }
        }

        if (is_better) {
            best_val_loss = val_loss;
            patience_counter = 0;
            best_weights = weights;
        } else {
            patience_counter++;
            if (patience_counter >= cfg.patience && epoch >= 30) {
                std::cout << "[Early Stopping] Triggered at epoch " << epoch
                          << " (Train Loss: " << train_loss << ", Val Loss: " << val_loss << ")" << std::endl;
                break;
            }
        }

        if (epoch % 10 == 0 || epoch == cfg.epochs) {
            float train_acc = static_cast<float>(train_correct) / static_cast<float>(train_set.size()) * 100.0f;
            float val_acc = static_cast<float>(val_correct) / static_cast<float>(val_set.size()) * 100.0f;
            std::cout << "Epoch " << epoch << "/" << cfg.epochs
                      << " - Train Loss: " << train_loss << " (Acc: " << train_acc << "%)"
                      << " | Val Loss: " << val_loss << " (Acc: " << val_acc << "%)" << std::endl;
        }
    }

    return std::make_shared<InferenceModel>(
        header, labels, tokenizer.vocab(), tokenizer.merge_rules(), best_weights
    );
}

} // namespace intellibranch
