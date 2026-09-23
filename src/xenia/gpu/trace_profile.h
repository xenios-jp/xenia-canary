/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_TRACE_PROFILE_H_
#define XENIA_GPU_TRACE_PROFILE_H_

#include <array>
#include <atomic>
#include <cstdint>
#include <iterator>
#include <mutex>
#include <utility>
#include <vector>

namespace xe {
namespace gpu {

enum class TraceCount : size_t {
  kDrawRequests,
  kDxilDraws,
  kMemexportDraws,
  kDrawFailures,
  kShaderTranslations,
  kHostShaderCompiles,
  kPipelineCreations,
  kResolveRequests,
  kCount
};
inline constexpr const char* kTraceCountNames[] = {
    "draw_requests",      "dxil_draws",          "memexport_draws",
    "draw_failures",      "shader_translations", "host_shader_compiles",
    "pipeline_creations", "resolve_requests"};
static_assert(std::size(kTraceCountNames) == size_t(TraceCount::kCount));

struct TraceStatsSnapshot {
  std::array<uint64_t, size_t(TraceCount::kCount)> counts{};
  // Guest memory ranges written by the GPU, for verifying selected commands.
  std::vector<std::pair<uint32_t, uint32_t>> memory_writes;
};

// Counters for one measured trace replay pass. Written from the command
// processor thread and from command buffer completion handlers.
class TraceProfileStats {
 public:
  explicit TraceProfileStats(bool capture_memory)
      : capture_memory_(capture_memory) {}
  void Add(TraceCount count, uint64_t amount = 1) {
    counts_[size_t(count)].fetch_add(amount, std::memory_order_relaxed);
  }
  void MemoryWrite(uint32_t address, uint32_t size) {
    if (!capture_memory_ || !size) {
      return;
    }
    std::lock_guard lock(mutex_);
    memory_writes_.emplace_back(address, size);
  }
  TraceStatsSnapshot Snapshot() const {
    TraceStatsSnapshot result;
    for (size_t i = 0; i < counts_.size(); ++i) {
      result.counts[i] = counts_[i].load(std::memory_order_relaxed);
    }
    std::lock_guard lock(mutex_);
    result.memory_writes = memory_writes_;
    return result;
  }

 private:
  const bool capture_memory_;
  std::array<std::atomic<uint64_t>, size_t(TraceCount::kCount)> counts_{};
  mutable std::mutex mutex_;
  std::vector<std::pair<uint32_t, uint32_t>> memory_writes_;
};

// Platform CPU clocks. Zero denotes unsupported or failure.
uint64_t TraceThreadCpuNs();
uint64_t TraceProcessCpuNs();
uint64_t TraceWallNs();

struct TraceProfileSample {
  // The clocks worked, and every command buffer the backend and the presenter
  // submitted during the pass completed with a GPU interval.
  bool accounting_valid = false;
  uint64_t command_thread_cpu_ns = 0, process_cpu_ns = 0, replay_wall_ns = 0;
  uint64_t gpu_drain_wall_ns = 0;
  // Sum and union of the command buffer GPU intervals.
  uint64_t gpu_buffer_duration_sum_ns = 0, gpu_buffer_interval_union_ns = 0;
  TraceStatsSnapshot stats;
  bool valid() const {
    return accounting_valid && command_thread_cpu_ns &&
           !stats.counts[size_t(TraceCount::kDrawFailures)];
  }
  // No shader or pipeline was created during the pass.
  bool warm() const {
    return !stats.counts[size_t(TraceCount::kShaderTranslations)] &&
           !stats.counts[size_t(TraceCount::kHostShaderCompiles)] &&
           !stats.counts[size_t(TraceCount::kPipelineCreations)];
  }
};

}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_TRACE_PROFILE_H_
