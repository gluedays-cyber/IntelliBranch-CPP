#pragma once

#include "intellibranch/common.hpp"
#include <string>
#include <vector>
#include <unordered_map>

namespace intellibranch {

std::string truncate_to_rune_boundary(const std::string& text, size_t max_bytes);
bool is_valid_utf8(const std::string& text);

class BPETokenizer {
public:
    BPETokenizer() = default;
    BPETokenizer(std::vector<std::string> vocab, std::vector<MergeRule> rules);

    static BPETokenizer train_bpe(const std::vector<std::string>& corpus, size_t target_vocab_size);

    std::vector<uint32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<uint32_t>& tokens) const;

    const std::vector<std::string>& vocab() const { return vocab_; }
    const std::unordered_map<std::string, uint32_t>& vocab_map() const { return vocab_map_; }
    const std::vector<MergeRule>& merge_rules() const { return merge_rules_; }
    size_t vocab_size() const { return vocab_.size(); }

    bool has_unk() const { return has_unk_; }
    uint32_t unk_id() const { return unk_id_; }

private:
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, uint32_t> vocab_map_;
    std::vector<MergeRule> merge_rules_;
    std::unordered_map<uint64_t, uint32_t> rule_lookup_;
    bool has_unk_ = false;
    uint32_t unk_id_ = 0;
};

} // namespace intellibranch
