/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <chrono>
#include <functional>
#include <future>
#include <initializer_list>
#include <memory>

#include "third_party/catch/include/catch.hpp"
#include "xenia/base/byte_order.h"
#include "xenia/base/memory.h"
#include "xenia/gpu/command_processor.h"
#include "xenia/gpu/graphics_system.h"
#include "xenia/gpu/registers.h"
#include "xenia/gpu/xenos.h"
#include "xenia/memory.h"

namespace xe::gpu::test {
namespace {

constexpr uint32_t kGuestBase = 0x01000000;
constexpr uint32_t kPacketBase = kGuestBase + 0x3000;

class TestGraphicsSystem final : public GraphicsSystem {
 public:
  explicit TestGraphicsSystem(Memory& memory) { memory_ = &memory; }
  ~TestGraphicsSystem() override {
    xe::memory::DeallocFixed(register_file_, 0,
                             xe::memory::DeallocationType::kRelease);
  }
  std::string name() const override { return "PM4 memory test"; }

 protected:
  std::unique_ptr<CommandProcessor> CreateCommandProcessor() override {
    return nullptr;
  }
};

class TestCommandProcessor final : public CommandProcessor {
 public:
  explicit TestCommandProcessor(GraphicsSystem& graphics)
      : CommandProcessor(&graphics, nullptr) {}
  bool ExecuteTestPacket(uint32_t address, uint32_t count) {
    reader_ = RingBuffer(memory_->TranslatePhysical(address),
                         count * sizeof(uint32_t));
    reader_.set_write_offset(count * sizeof(uint32_t));
    return ExecutePacket();
  }
  void Stop() { worker_running_.store(false); }
  std::function<void()> on_submit;
  void RestoreEdramSnapshot(const void*) override {}
  void TracePlaybackWroteMemory(uint32_t, uint32_t) override {}

 protected:
  bool SetupContext() override { return true; }
  void ShutdownContext() override {}
  void SubmitBeforeShortMemoryPoll() override {
    if (on_submit) {
      on_submit();
    }
  }
};

struct Fixture {
  Memory memory;
  std::unique_ptr<TestGraphicsSystem> graphics;
  std::unique_ptr<TestCommandProcessor> processor;

  Fixture() {
    REQUIRE(memory.Initialize());
    REQUIRE(memory.LookupHeap(0xA0000000 + kGuestBase)
                ->AllocFixed(0xA0000000 + kGuestBase, 0x10000, 0x10000,
                             kMemoryAllocationReserve | kMemoryAllocationCommit,
                             kMemoryProtectRead | kMemoryProtectWrite));
    graphics = std::make_unique<TestGraphicsSystem>(memory);
    processor = std::make_unique<TestCommandProcessor>(*graphics);
  }

  bool Packet(uint32_t opcode, std::initializer_list<uint32_t> words) {
    auto* destination = memory.TranslatePhysical<uint32_t*>(kPacketBase);
    *destination++ = xe::byte_swap(
        uint32_t(0xC0000000 | (opcode << 8) | ((words.size() - 1) << 16)));
    for (uint32_t word : words) {
      *destination++ = xe::byte_swap(word);
    }
    return processor->ExecuteTestPacket(kPacketBase,
                                        uint32_t(words.size() + 1));
  }
};

}  // namespace

TEST_CASE("PM4 zero-delay memory poll submits its pending GPU producer",
          "[pm4][memory-order]") {
  Fixture fixture;
  auto* output = fixture.memory.TranslatePhysical<uint32_t*>(kGuestBase);
  *output = 0;
  uint32_t submissions = 0;
  fixture.processor->on_submit = [&]() {
    ++submissions;
    *output = 7;
  };
  auto execute = std::async(std::launch::async, [&]() {
    return fixture.Packet(xenos::PM4_WAIT_REG_MEM,
                          {0x13, kGuestBase, 7, UINT32_MAX, 0});
  });
  const auto status = execute.wait_for(std::chrono::seconds(5));
  // Always let a broken polling path exit before making assertions.
  fixture.processor->Stop();
  const bool succeeded = execute.get();
  REQUIRE(status == std::future_status::ready);
  REQUIRE(succeeded);
  REQUIRE(submissions == 1);
  REQUIRE(*output == 7);
}

TEST_CASE("PM4 satisfied memory poll preserves unrelated queued GPU work",
          "[pm4][memory-order]") {
  Fixture fixture;
  *fixture.memory.TranslatePhysical<uint32_t*>(kGuestBase) = 7;
  uint32_t submissions = 0;
  fixture.processor->on_submit = [&]() { ++submissions; };
  REQUIRE(fixture.Packet(xenos::PM4_WAIT_REG_MEM,
                         {0x13, kGuestBase, 7, UINT32_MAX, 0}));
  REQUIRE(submissions == 0);
}

TEST_CASE("PM4 redirected shader fence stays inside the writeback window",
          "[pm4][memory-order]") {
  Fixture fixture;
  constexpr uint32_t kWritebackOffset = 0x00800000;
  constexpr uint32_t kWritebackBase = 0x7E000000;
  REQUIRE(fixture.memory.LookupHeap(0xA0000000 + kWritebackOffset)
              ->AllocFixed(0xA0000000 + kWritebackOffset, 0x10000, 0x10000,
                           kMemoryAllocationReserve | kMemoryAllocationCommit,
                           kMemoryProtectRead | kMemoryProtectWrite));
  auto* fence = fixture.memory.TranslatePhysical<uint32_t*>(kWritebackOffset);
  *fence = 0;
  auto& regs = fixture.graphics->register_file()->values;
  regs[XE_GPU_REG_WRITEBACK_START] = kWritebackBase;
  regs[XE_GPU_REG_WRITEBACK_SIZE] = 0x02000000;
  // Inside the 16 MiB window: lands at the physical alias of the offset.
  REQUIRE(fixture.Packet(xenos::PM4_EVENT_WRITE_SHD,
                         {0, kWritebackBase + kWritebackOffset, 31}));
  REQUIRE(*fence == 31);
  // Past the window, even though WRITEBACK_SIZE covers it: not redirected, so
  // the store goes to the packet's own physical address instead.
  constexpr uint32_t kOutsideOffset = 0x01000000 + kWritebackOffset;
  constexpr uint32_t kOutsidePhysical =
      (kWritebackBase + kOutsideOffset) & 0x1FFFFFFF;
  REQUIRE(fixture.memory.LookupHeap(0xA0000000 + kOutsidePhysical)
              ->AllocFixed(0xA0000000 + kOutsidePhysical, 0x10000, 0x10000,
                           kMemoryAllocationReserve | kMemoryAllocationCommit,
                           kMemoryProtectRead | kMemoryProtectWrite));
  auto* outside = fixture.memory.TranslatePhysical<uint32_t*>(kOutsidePhysical);
  *outside = 0;
  REQUIRE(fixture.Packet(xenos::PM4_EVENT_WRITE_SHD,
                         {0, kWritebackBase + kOutsideOffset, 47}));
  REQUIRE(*outside == 47);
  REQUIRE(*fence == 31);
}

}  // namespace xe::gpu::test
