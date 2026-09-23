/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "xenia/base/console_app_main.h"
#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/gpu/metal/metal_command_processor.h"
#include "xenia/gpu/metal/metal_graphics_system.h"
#include "xenia/gpu/metal/metal_shared_memory.h"
#include "xenia/gpu/trace_dump.h"
#include "xenia/ui/metal/metal_api.h"
#include "xenia/ui/metal/metal_provider.h"

DECLARE_bool(shared_memory_zero_copy);
DECLARE_bool(async_shader_compilation);
DECLARE_bool(async_shader_skip_draws);

DEFINE_string(
    metal_trace_dump_capture, "",
    "Path of a .gputrace to write around the replayed frame. Empty takes no "
    "capture. Needs METAL_CAPTURE_ENABLED=1 in the environment.",
    "Metal");

namespace xe {
namespace gpu {
namespace metal {

class MetalTraceDump : public TraceDump {
 public:
  std::unique_ptr<gpu::GraphicsSystem> CreateGraphicsSystem() override {
    // Playback restores recorded RAM on the CPU while preceding draws may
    // still be queued. A bytes-no-copy buffer would expose those later bytes
    // to earlier draws. This applies only to this tool, before the shared
    // buffer is created.
    if (cvars::shared_memory_zero_copy) {
      XELOGI(
          "Metal trace replay: disabling zero-copy guest RAM for ordered "
          "snapshot restoration");
      cvars::shared_memory_zero_copy = false;
    }
    return std::unique_ptr<gpu::GraphicsSystem>(new MetalGraphicsSystem());
  }

  void BeginHostCapture() override {
    if (cvars::metal_trace_dump_capture.empty()) {
      return;
    }
    auto* provider = static_cast<const ui::metal::MetalProvider*>(
        graphics_system_->provider());
    MTL::Device* device = provider ? provider->GetDevice() : nullptr;
    if (!device) {
      XELOGE("Metal trace dump: no device to capture");
      return;
    }

    MTL::CaptureManager* manager = MTL::CaptureManager::sharedCaptureManager();
    if (!manager || !manager->supportsDestination(
                        MTL::CaptureDestinationGPUTraceDocument)) {
      XELOGE(
          "Metal trace dump: GPU trace documents unavailable - run with "
          "METAL_CAPTURE_ENABLED=1");
      return;
    }

    // Metal refuses to overwrite, and a .gputrace is a directory.
    std::filesystem::path capture_path(cvars::metal_trace_dump_capture);
    std::error_code remove_error;
    std::filesystem::remove_all(capture_path, remove_error);

    NS::AutoreleasePool* pool = NS::AutoreleasePool::alloc()->init();
    NS::String* path_string = NS::String::string(capture_path.string().c_str(),
                                                 NS::UTF8StringEncoding);
    MTL::CaptureDescriptor* descriptor =
        MTL::CaptureDescriptor::alloc()->init();
    descriptor->setCaptureObject(device);
    descriptor->setDestination(MTL::CaptureDestinationGPUTraceDocument);
    descriptor->setOutputURL(NS::URL::fileURLWithPath(path_string));

    NS::Error* error = nullptr;
    if (manager->startCapture(descriptor, &error)) {
      capturing_ = true;
      XELOGI("Metal trace dump: capturing to {}", capture_path.string());
    } else {
      const char* message = "unknown error";
      if (error && error->localizedDescription()) {
        message = error->localizedDescription()->utf8String();
      }
      XELOGE("Metal trace dump: failed to start capture - {}", message);
    }
    descriptor->release();
    pool->release();
  }

  void EndHostCapture() override {
    // A trace whose frame holds no swap never reaches the presenter, so there
    // would be nothing for the PNG.
    auto* command_processor = static_cast<MetalCommandProcessor*>(
        graphics_system_->command_processor());
    if (command_processor && !command_processor->HasSeenSwap()) {
      command_processor->ForceIssueSwap();
    }

    if (!capturing_) {
      return;
    }
    capturing_ = false;
    MTL::CaptureManager* manager = MTL::CaptureManager::sharedCaptureManager();
    if (manager && manager->isCapturing()) {
      manager->stopCapture();
    }
    XELOGI("Metal trace dump: capture written to {}",
           cvars::metal_trace_dump_capture);
  }

  bool HasTraceProfiling() const override { return true; }

  void PrepareTraceProfileReplay() override {
    RunOnCommandThread(
        [&] { command_processor()->PrepareTraceProfileReplay(); });
  }

  bool BeginTraceProfile(bool reset_state) override {
    if (!cvars::metal_trace_dump_capture.empty() ||
        cvars::async_shader_compilation || cvars::async_shader_skip_draws) {
      XELOGE(
          "Trace profiling requires no GPU capture and synchronous shader "
          "compilation");
      return false;
    }
    bool started = false;
    RunOnCommandThread(
        [&] { started = command_processor()->BeginTraceProfile(reset_state); });
    return started;
  }

  TraceProfileSample EndTraceProfile() override {
    TraceProfileSample sample;
    RunOnCommandThread(
        [&] { sample = command_processor()->EndTraceProfile(); });
    return sample;
  }

  // Each range is written as recorded: its address, its length and its bytes,
  // after the range count.
  bool ReadTraceProfileMemory(
      const std::vector<std::pair<uint32_t, uint32_t>>& ranges,
      std::vector<uint8_t>& bytes) override {
    MTL::Buffer* buffer = command_processor()->shared_memory()->GetBuffer();
    if (!buffer || !buffer->contents()) {
      return ranges.empty();
    }
    uint64_t size = sizeof(uint64_t);
    for (const auto& range : ranges) {
      if (uint64_t(range.first) + range.second > buffer->length()) {
        return false;
      }
      size += sizeof(uint32_t) * 2 + range.second;
    }
    bytes.resize(size);
    uint64_t count = ranges.size();
    std::memcpy(bytes.data(), &count, sizeof(count));
    size_t offset = sizeof(count);
    const auto* contents = static_cast<const uint8_t*>(buffer->contents());
    for (const auto& range : ranges) {
      std::memcpy(bytes.data() + offset, &range.first, sizeof(uint32_t));
      std::memcpy(bytes.data() + offset + sizeof(uint32_t), &range.second,
                  sizeof(uint32_t));
      offset += sizeof(uint32_t) * 2;
      std::memcpy(bytes.data() + offset, contents + range.first, range.second);
      offset += range.second;
    }
    return true;
  }

  std::string TraceProfileDevice() override {
    auto* provider =
        static_cast<ui::metal::MetalProvider*>(graphics_system_->provider());
    return provider->GetDevice()->name()->utf8String();
  }

 private:
  MetalCommandProcessor* command_processor() const {
    return static_cast<MetalCommandProcessor*>(
        graphics_system_->command_processor());
  }

  bool capturing_ = false;
};

int trace_dump_main(const std::vector<std::string>& args) {
  MetalTraceDump trace_dump;
  return trace_dump.Main(args);
}

}  // namespace metal
}  // namespace gpu
}  // namespace xe

XE_DEFINE_CONSOLE_APP("xenia-gpu-metal-trace-dump",
                      xe::gpu::metal::trace_dump_main, "some.trace",
                      "target_trace_file");
