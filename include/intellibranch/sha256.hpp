#pragma once

#include <cstdint>
#include <cstddef>
#include <array>
#include <span>

namespace intellibranch {

class SHA256 {
public:
    SHA256();
    void reset();
    void update(const uint8_t* data, size_t length);
    void update(std::span<const uint8_t> data);
    std::array<uint8_t, 32> finalize();

    static std::array<uint8_t, 32> hash(const uint8_t* data, size_t length);
    static std::array<uint8_t, 32> hash(std::span<const uint8_t> data);

private:
    void transform(const uint8_t* chunk);

    uint32_t state_[8];
    uint64_t bit_len_;
    uint8_t buffer_[64];
    size_t buffer_len_;
};

} // namespace intellibranch
