#include "intellibranch/telemetry.hpp"
#include <chrono>

namespace intellibranch {

TelemetryRingBuffer::TelemetryRingBuffer(size_t capacity)
    : capacity_(capacity > 0 ? capacity : 1024), events_(capacity_ > 0 ? capacity_ : 1024) {}

void TelemetryRingBuffer::push(TelemetryEvent event) {
    std::lock_guard<std::mutex> lock(mu_);

    auto now = std::chrono::system_clock::now().time_since_epoch();
    event.timestamp_nano = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();

    events_[head_] = std::move(event);
    head_ = (head_ + 1) % capacity_;

    if (count_ < capacity_) {
        count_++;
    } else {
        tail_ = (tail_ + 1) % capacity_;
    }
}

std::vector<TelemetryEvent> TelemetryRingBuffer::drain() {
    std::lock_guard<std::mutex> lock(mu_);

    if (count_ == 0) {
        return {};
    }

    std::vector<TelemetryEvent> result(count_);
    for (size_t i = 0; i < count_; ++i) {
        size_t idx = (tail_ + i) % capacity_;
        result[i] = std::move(events_[idx]);
    }

    head_ = 0;
    tail_ = 0;
    count_ = 0;

    return result;
}

size_t TelemetryRingBuffer::count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return count_;
}

} // namespace intellibranch
