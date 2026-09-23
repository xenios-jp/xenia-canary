/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2021 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_TRACE_DUMP_H_
#define XENIA_GPU_TRACE_DUMP_H_

#include <functional>
#include <string>

#include "xenia/emulator.h"
#include "xenia/gpu/shader.h"
#include "xenia/gpu/trace_player.h"
#include "xenia/gpu/trace_profile.h"
#include "xenia/gpu/trace_protocol.h"
#include "xenia/gpu/xenos.h"
#include "xenia/memory.h"

namespace xe {
namespace gpu {

struct SamplerInfo;
struct TextureInfo;

class TraceDump {
 public:
  virtual ~TraceDump();

  int Main(const std::vector<std::string>& args);

 protected:
  TraceDump();

  virtual std::unique_ptr<gpu::GraphicsSystem> CreateGraphicsSystem() = 0;

  virtual void BeginHostCapture() = 0;
  virtual void EndHostCapture() = 0;
  // Trace profiling (trace_profile_path), implemented by the backend. Without
  // it, profiling fails before replaying. All of these are called from the
  // main thread while playback is idle.
  virtual bool HasTraceProfiling() const { return false; }
  // Resets the texture and memory state so that a selected command range can
  // be replayed again from its prerequisites.
  virtual void PrepareTraceProfileReplay() {}
  virtual bool BeginTraceProfile(bool reset_state) { return false; }
  virtual TraceProfileSample EndTraceProfile() { return {}; }
  // Appends the current contents of the given guest memory ranges.
  virtual bool ReadTraceProfileMemory(
      const std::vector<std::pair<uint32_t, uint32_t>>& ranges,
      std::vector<uint8_t>& bytes) {
    return ranges.empty();
  }
  virtual std::string TraceProfileDevice() { return {}; }

  // Runs the function on the command processor thread and waits for it.
  void RunOnCommandThread(const std::function<void()>& function);

  std::unique_ptr<Emulator> emulator_;
  GraphicsSystem* graphics_system_ = nullptr;
  std::unique_ptr<TracePlayer> player_;

 private:
  bool Setup();
  bool CaptureToPng(const std::filesystem::path& png_path);
  bool Load(const std::filesystem::path& trace_file_path);
  int Run();
  void ReplayFrames();
  int RunProfile();

  std::filesystem::path trace_file_path_;
  std::filesystem::path base_output_path_;
};

}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_TRACE_DUMP_H_
