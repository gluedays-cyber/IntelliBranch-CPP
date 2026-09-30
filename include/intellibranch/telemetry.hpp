#pragma once

#include "intellibranch/common.hpp"
#include <vector>
#include <mutex>

namespace intellibranch {

class TelemetryRingBuffer {
public:
    explicit TelemetryRingBuffer(size_t capacity = 1024);

    void push(TelemetryEvent event);
    std::vector<TelemetryEvent> drain();
    size_t count() const;

private:
    mutable std::mutex mu_;
    size_t capacity_;
    std::vector<TelemetryEvent> events_;
    size_t head_ = 0;
    size_t tail_ = 0;
    size_t count_ = 0;
};

} // namespace intellibranch
