/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_UI_METAL_METAL_PROVIDER_H_
#define XENIA_UI_METAL_METAL_PROVIDER_H_

#include <atomic>
#include <cstdint>
#include <memory>

#include "xenia/ui/graphics_provider.h"
#include "xenia/ui/metal/metal_api.h"

namespace xe {
namespace ui {
namespace metal {

class GpuTimingLedger;
enum class GpuTimingSource : uint32_t;

class MetalProvider : public GraphicsProvider {
 public:
  static std::unique_ptr<MetalProvider> Create();

  ~MetalProvider() override;

  std::unique_ptr<Presenter> CreatePresenter(
      Presenter::HostGpuLossCallback host_gpu_loss_callback =
          Presenter::FatalErrorHostGpuLossCallback) override;

  std::unique_ptr<ImmediateDrawer> CreateImmediateDrawer() override;

  MTL::Device* GetDevice() const { return device_; }

  MTL::CommandQueue* GetCommandQueue() const { return command_queue_; }

  static bool IsMetalAPIAvailable();

  // GPU interval timing of submitted command buffers, for trace profiling.
  // Tracking costs one atomic load while no session is active.
  bool BeginGpuTiming();
  std::shared_ptr<GpuTimingLedger> EndGpuTiming();
  void TrackGpuTiming(MTL::CommandBuffer* buffer, GpuTimingSource source) const;
  void CancelGpuTiming(MTL::CommandBuffer* buffer) const;

 private:
  MetalProvider();

  bool Initialize();

  MTL::Device* device_ = nullptr;
  MTL::CommandQueue* command_queue_ = nullptr;
  // Atomically published; completion handlers retain the session, not provider.
  std::shared_ptr<GpuTimingLedger> gpu_timing_;
  std::atomic<bool> gpu_timing_active_{false};
};

}  // namespace metal
}  // namespace ui
}  // namespace xe

#endif  // XENIA_UI_METAL_METAL_PROVIDER_H_
