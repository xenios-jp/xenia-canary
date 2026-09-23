/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */
#ifndef XENIA_UI_METAL_GPU_TIMING_LEDGER_H_
#define XENIA_UI_METAL_GPU_TIMING_LEDGER_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xe {
namespace ui {
namespace metal {

// What submitted a timed command buffer.
enum class GpuTimingSource : uint32_t { kBackend, kPresenterCopy, kCount };

// Completion ledger of command buffer GPU intervals. Intervals may overlap:
// their sum and their union are different measurements, not shader execution
// time.
class GpuTimingLedger {
 public:
  struct Result {
    uint64_t completed = 0, pending = 0, invalid = 0, failed = 0;
    uint64_t duration_sum_ns = 0, interval_union_ns = 0;
    // Completed buffers with a valid interval, by source.
    uint64_t buffers[size_t(GpuTimingSource::kCount)] = {};
    bool valid() const { return completed && !pending && !invalid && !failed; }
  };

  uint64_t Register(uintptr_t identity, GpuTimingSource source) {
    std::lock_guard lock(mutex_);
    if (!identity || live_.count(identity)) {
      ++invalid_;
      return 0;
    }
    const uint64_t token = entries_.size() + 1;
    entries_.push_back({identity, source});
    live_.emplace(identity, token);
    ++pending_;
    return token;
  }

  void Cancel(uintptr_t identity) {
    std::lock_guard lock(mutex_);
    auto found = live_.find(identity);
    if (found == live_.end()) {
      ++invalid_;
      return;
    }
    entries_[found->second - 1].state = State::kCancelled;
    live_.erase(found);
    --pending_;
    completed_cv_.notify_all();
  }

  void Complete(uint64_t token, double start_seconds, double end_seconds,
                bool succeeded) {
    std::lock_guard lock(mutex_);
    if (!token || token > entries_.size() ||
        entries_[token - 1].state != State::kPending) {
      ++invalid_;
      return;
    }
    auto& entry = entries_[token - 1];
    entry.state = State::kCompleted;
    entry.succeeded = succeeded;
    // Leave headroom for integer conversion; this is over 285 years of uptime.
    entry.valid = std::isfinite(start_seconds) && std::isfinite(end_seconds) &&
                  start_seconds > 0 && end_seconds >= start_seconds &&
                  end_seconds < 9e9;
    if (entry.valid) {
      entry.start_ns = uint64_t(start_seconds * 1e9);
      entry.end_ns = uint64_t(end_seconds * 1e9);
    }
    live_.erase(entry.identity);
    --pending_;
    completed_cv_.notify_all();
  }

  bool Wait(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    return completed_cv_.wait_for(lock, timeout, [&] { return !pending_; });
  }

  Result Snapshot() const {
    std::lock_guard lock(mutex_);
    Result result;
    result.pending = pending_;
    result.invalid = invalid_;
    std::vector<std::pair<uint64_t, uint64_t>> intervals;
    for (const auto& entry : entries_) {
      if (entry.state != State::kCompleted) {
        continue;
      }
      ++result.completed;
      result.failed += !entry.succeeded;
      if (!entry.valid) {
        ++result.invalid;
        continue;
      }
      const uint64_t duration = entry.end_ns - entry.start_ns;
      if (duration >
          std::numeric_limits<uint64_t>::max() - result.duration_sum_ns) {
        ++result.invalid;
        continue;
      }
      result.duration_sum_ns += duration;
      ++result.buffers[size_t(entry.source)];
      intervals.emplace_back(entry.start_ns, entry.end_ns);
    }
    std::sort(intervals.begin(), intervals.end());
    if (!intervals.empty()) {
      uint64_t begin = intervals.front().first, end = intervals.front().second;
      for (const auto& interval : intervals) {
        if (interval.first > end) {
          result.interval_union_ns += end - begin;
          begin = interval.first;
        }
        end = std::max(end, interval.second);
      }
      result.interval_union_ns += end - begin;
    }
    return result;
  }

 private:
  enum class State { kPending, kCompleted, kCancelled };
  struct Entry {
    uintptr_t identity;
    GpuTimingSource source;
    State state = State::kPending;
    bool valid = false, succeeded = false;
    uint64_t start_ns = 0, end_ns = 0;
  };
  mutable std::mutex mutex_;
  std::condition_variable completed_cv_;
  std::vector<Entry> entries_;
  std::unordered_map<uintptr_t, uint64_t> live_;
  uint64_t pending_ = 0, invalid_ = 0;
};

}  // namespace metal
}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_METAL_GPU_TIMING_LEDGER_H_
