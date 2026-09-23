/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/ui/metal/gpu_timing_ledger.h"

#include <functional>
#include <limits>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#include "third_party/catch/include/catch.hpp"

namespace xe::ui::metal::test {

constexpr auto kBackend = GpuTimingSource::kBackend;
constexpr auto kPresenterCopy = GpuTimingSource::kPresenterCopy;

TEST_CASE("GpuTimingLedger interval accounting", "[gpu_timing_ledger]") {
  GpuTimingLedger ledger;
  auto a = ledger.Register(1, kBackend);
  auto b = ledger.Register(2, kPresenterCopy);
  auto c = ledger.Register(3, kBackend);
  ledger.Register(4, kBackend);
  ledger.Cancel(4);
  ledger.Complete(b, 2, 5, true);
  ledger.Complete(c, 7, 8, true);
  ledger.Complete(a, 1, 3, true);
  REQUIRE(ledger.Wait(std::chrono::milliseconds(0)));
  auto result = ledger.Snapshot();
  REQUIRE(result.valid());
  REQUIRE(result.completed == 3);
  REQUIRE(result.duration_sum_ns == 6000000000);
  REQUIRE(result.interval_union_ns == 5000000000);
  REQUIRE(result.buffers[size_t(kBackend)] == 2);
  REQUIRE(result.buffers[size_t(kPresenterCopy)] == 1);
  // A completed identity may be registered again.
  auto reused = ledger.Register(1, kBackend);
  ledger.Complete(reused, 9, 10, true);
  REQUIRE(ledger.Snapshot().valid());
}

TEST_CASE("GpuTimingLedger rejects invalid times", "[gpu_timing_ledger]") {
  for (auto times : std::vector<std::pair<double, double>>{
           {0, 1},
           {-1, 1},
           {2, 1},
           {1, std::numeric_limits<double>::infinity()},
           {std::numeric_limits<double>::quiet_NaN(), 2},
           {1, 9e9}}) {
    GpuTimingLedger ledger;
    auto token = ledger.Register(1, kBackend);
    ledger.Complete(token, times.first, times.second, true);
    REQUIRE_FALSE(ledger.Snapshot().valid());
    REQUIRE(ledger.Snapshot().invalid == 1);
  }
}

TEST_CASE("GpuTimingLedger failure and cancellation", "[gpu_timing_ledger]") {
  {
    GpuTimingLedger ledger;
    auto token = ledger.Register(1, kBackend);
    REQUIRE_FALSE(ledger.Wait(std::chrono::milliseconds(0)));
    REQUIRE_FALSE(ledger.Register(1, kBackend));
    ledger.Complete(token, 1, 2, false);
    REQUIRE(ledger.Snapshot().failed == 1);
    REQUIRE_FALSE(ledger.Snapshot().valid());
    ledger.Complete(token, 1, 2, true);
    REQUIRE(ledger.Snapshot().invalid == 2);
  }
  {
    GpuTimingLedger ledger;
    auto token = ledger.Register(1, kBackend);
    ledger.Cancel(1);
    ledger.Complete(token, 1, 2, true);
    REQUIRE_FALSE(ledger.Snapshot().valid());
    REQUIRE(ledger.Snapshot().pending == 0);
  }
}

TEST_CASE("GpuTimingLedger outlives its session through callbacks",
          "[gpu_timing_ledger]") {
  auto session = std::make_shared<GpuTimingLedger>();
  std::weak_ptr<GpuTimingLedger> weak = session;
  auto token = session->Register(1, kBackend);
  std::function<void()> callback = [session, token] {
    session->Complete(token, 1, 2, true);
  };
  session.reset();
  REQUIRE_FALSE(weak.expired());
  callback();
  callback = {};
  REQUIRE(weak.expired());
}

TEST_CASE("GpuTimingLedger concurrent completion", "[gpu_timing_ledger]") {
  auto session = std::make_shared<GpuTimingLedger>();
  std::vector<std::thread> threads;
  for (unsigned i = 0; i < 64; ++i) {
    auto token = session->Register(i + 1, kBackend);
    threads.emplace_back(
        [session, token] { session->Complete(token, 1, 2, true); });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  auto result = session->Snapshot();
  REQUIRE(result.valid());
  REQUIRE(result.completed == 64);
  REQUIRE(result.duration_sum_ns == 64000000000);
  REQUIRE(result.interval_union_ns == 1000000000);
}

}  // namespace xe::ui::metal::test
