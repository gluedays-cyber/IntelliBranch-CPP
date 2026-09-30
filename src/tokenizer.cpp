#include "intellibranch/tokenizer.hpp"
#include <algorithm>
#include <cctype>
#include <sstream>

namespace intellibranch {

namespace {

std::string trim_and_lower(const std::string& str) {
    if (str.empty()) return "";
    size_t start = 0;
    while (start < str.size() && (std::isspace(static_cast<unsigned char>(str[start])) != 0)) {
        start++;
    }
    if (start == str.size()) return "";
    size_t end = str.size() - 1;
    while (end > start && (std::isspace(static_cast<unsigned char>(str[end])) != 0)) {
        end--;
    }
    std::string result;
    result.reserve(end - start + 1);
    for (size_t i = start; i <= end; ++i) {
        result.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(str[i]))));
    }
    return result;
}

std::vector<std::string> split_utf8_chars(const std::string& str) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < str.size()) {
        unsigned char c = static_cast<unsigned char>(str[i]);
        size_t len = 1;
        if ((c & 0x80) == 0) {
            len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            len = 4;
        } else {
            len = 1;
        }
        if (i + len > str.size()) {
            len = str.size() - i;
        }
        chars.push_back(str.substr(i, len));
        i += len;
    }
    return chars;
}

} // namespace

std::string truncate_to_rune_boundary(const std::string& text, size_t max_bytes) {
    if (text.size() <= max_bytes) {
        return text;
    }
    size_t idx = max_bytes;
    while (idx > 0 && (static_cast<unsigned char>(text[idx]) & 0xC0) == 0x80) {
        idx--;
    }
    return text.substr(0, idx);
}

bool is_valid_utf8(const std::string& text) {
    size_t i = 0;
    size_t n = text.size();
    while (i < n) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        if (c <= 0x7F) {
            i += 1;
        } else if ((c & 0xE0) == 0xC0) {
            if (i + 1 >= n || (static_cast<unsigned char>(text[i + 1]) & 0xC0) != 0x80) return false;
            i += 2;
        } else if ((c & 0xF0) == 0xE0) {
            if (i + 2 >= n || (static_cast<unsigned char>(text[i + 1]) & 0xC0) != 0x80 ||
                (static_cast<unsigned char>(text[i + 2]) & 0xC0) != 0x80) return false;
            i += 3;
        } else if ((c & 0xF8) == 0xF0) {
            if (i + 3 >= n || (static_cast<unsigned char>(text[i + 1]) & 0xC0) != 0x80 ||
                (static_cast<unsigned char>(text[i + 2]) & 0xC0) != 0x80 ||
                (static_cast<unsigned char>(text[i + 3]) & 0xC0) != 0x80) return false;
            i += 4;
        } else {
            return false;
        }
    }
    return true;
}

BPETokenizer::BPETokenizer(std::vector<std::string> vocab, std::vector<MergeRule> rules)
    : vocab_(std::move(vocab)), merge_rules_(std::move(rules)) {
    for (size_t idx = 0; idx < vocab_.size(); ++idx) {
        vocab_map_[vocab_[idx]] = static_cast<uint32_t>(idx);
        if (vocab_[idx] == "[UNK]") {
            has_unk_ = true;
            unk_id_ = static_cast<uint32_t>(idx);
        }
    }
    for (const auto& r : merge_rules_) {
        uint64_t key = (static_cast<uint64_t>(r.token1) << 32) | static_cast<uint64_t>(r.token2);
        rule_lookup_[key] = r.target;
    }
}

BPETokenizer BPETokenizer::train_bpe(const std::vector<std::string>& corpus, size_t target_vocab_size) {
    if (target_vocab_size < 10) {
        target_vocab_size = 10;
    }

    std::unordered_map<std::string, uint32_t> vocab_map;
    std::vector<std::string> vocab;

    std::vector<std::string> special_tokens = {"[PAD]", "[UNK]"};
    for (const auto& st : special_tokens) {
        vocab_map[st] = static_cast<uint32_t>(vocab.size());
        vocab.push_back(st);
    }

    // 1. Collect unique characters
    for (const auto& raw_text : corpus) {
        std::string text = trim_and_lower(raw_text);
        auto chars = split_utf8_chars(text);
        for (const auto& ch : chars) {
            if (!vocab_map.contains(ch)) {
                vocab_map[ch] = static_cast<uint32_t>(vocab.size());
                vocab.push_back(ch);
            }
        }
    }

    // 2. Tokenize corpus into character token ID sequences
    std::vector<std::vector<uint32_t>> tokenized_corpus;
    for (const auto& raw_text : corpus) {
        std::string text = trim_and_lower(raw_text);
        if (text.empty()) continue;
        auto chars = split_utf8_chars(text);
        std::vector<uint32_t> seq;
        seq.reserve(chars.size());
        for (const auto& ch : chars) {
            seq.push_back(vocab_map[ch]);
        }
        tokenized_corpus.push_back(std::move(seq));
    }

    std::vector<MergeRule> merge_rules;
    std::unordered_map<uint64_t, uint32_t> rule_lookup;

    // 3. Iteratively merge most frequent adjacent pairs
    while (vocab.size() < target_vocab_size) {
        std::unordered_map<uint64_t, int> pair_counts;
        for (const auto& seq : tokenized_corpus) {
            if (seq.size() < 2) continue;
            for (size_t i = 0; i < seq.size() - 1; ++i) {
                uint64_t key = (static_cast<uint64_t>(seq[i]) << 32) | static_cast<uint64_t>(seq[i + 1]);
                pair_counts[key]++;
            }
        }

        if (pair_counts.empty()) {
            break;
        }

        uint64_t best_key = 0;
        int max_freq = 0;
        for (const auto& [key, freq] : pair_counts) {
            if (freq > max_freq) {
                max_freq = freq;
                best_key = key;
            }
        }

        if (max_freq < 2 && vocab.size() >= target_vocab_size / 2) {
            break;
        }

        uint32_t t1 = static_cast<uint32_t>(best_key >> 32);
        uint32_t t2 = static_cast<uint32_t>(best_key & 0xFFFFFFFF);

        std::string str1 = vocab[t1];
        std::string str2 = vocab[t2];
        std::string merged_str = str1 + str2;

        uint32_t new_id = static_cast<uint32_t>(vocab.size());
        vocab_map[merged_str] = new_id;
        vocab.push_back(merged_str);

        MergeRule rule{t1, t2, new_id};
        merge_rules.push_back(rule);
        rule_lookup[best_key] = new_id;

        // Apply merge in-place across tokenized corpus
        for (auto& seq : tokenized_corpus) {
            if (seq.size() < 2) continue;
            std::vector<uint32_t> new_seq;
            new_seq.reserve(seq.size());
            size_t i = 0;
            while (i < seq.size()) {
                if (i + 1 < seq.size() && seq[i] == t1 && seq[i + 1] == t2) {
                    new_seq.push_back(new_id);
                    i += 2;
                } else {
                    new_seq.push_back(seq[i]);
                    i++;
                }
            }
            seq = std::move(new_seq);
        }
    }

    return BPETokenizer(std::move(vocab), std::move(merge_rules));
}

std::vector<uint32_t> BPETokenizer::encode(const std::string& text) const {
    std::string clean = trim_and_lower(text);
    if (clean.empty()) {
        return {};
    }

    auto chars = split_utf8_chars(clean);
    std::vector<uint32_t> tokens;
    tokens.reserve(chars.size());

    for (const auto& ch : chars) {
        auto it = vocab_map_.find(ch);
        if (it != vocab_map_.end()) {
            tokens.push_back(it->second);
        } else if (has_unk_) {
            tokens.push_back(unk_id_);
        } else if (!vocab_.empty()) {
            tokens.push_back(0);
        }
    }

    if (tokens.size() <= 1) {
        return tokens;
    }

    while (true) {
        bool merged = false;
        std::vector<uint32_t> next_tokens;
        next_tokens.reserve(tokens.size());
        size_t i = 0;
        while (i < tokens.size()) {
            if (i + 1 < tokens.size()) {
                uint64_t key = (static_cast<uint64_t>(tokens[i]) << 32) | static_cast<uint64_t>(tokens[i + 1]);
                auto it = rule_lookup_.find(key);
                if (it != rule_lookup_.end()) {
                    next_tokens.push_back(it->second);
                    i += 2;
                    merged = true;
                    continue;
                }
            }
            next_tokens.push_back(tokens[i]);
            i++;
        }
        tokens = std::move(next_tokens);
        if (!merged) {
            break;
        }
    }

    return tokens;
}

std::string BPETokenizer::decode(const std::vector<uint32_t>& tokens) const {
    std::string res;
    for (uint32_t tok : tokens) {
        if (tok < vocab_.size()) {
            const std::string& str = vocab_[tok];
            if (str != "[PAD]" && str != "[UNK]") {
                res += str;
            }
        }
    }
    return res;
}

} // namespace intellibranch
