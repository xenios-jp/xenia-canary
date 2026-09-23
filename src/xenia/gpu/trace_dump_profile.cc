/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "third_party/rapidjson/include/rapidjson/stringbuffer.h"
#include "third_party/rapidjson/include/rapidjson/writer.h"
#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/gpu/command_processor.h"
#include "xenia/gpu/graphics_system.h"
#include "xenia/gpu/trace_dump.h"
#include "xenia/ui/presenter.h"

DEFINE_path(trace_profile_path, "",
            "Replay the trace repeatedly, verify that every pass reproduces "
            "the output, and write timing JSON to this path.",
            "GPU.Debug");
DEFINE_uint32(trace_profile_warmup, 2,
              "Replay passes before the measured samples, at least 2.",
              "GPU.Debug");
DEFINE_uint32(trace_profile_samples, 8, "Measured replay passes.", "GPU.Debug");
DEFINE_int32(trace_profile_frame, -1,
             "Frame containing the selected command range, or -1 to replay "
             "the whole trace.",
             "GPU.Debug");
DEFINE_int32(trace_profile_first_command, -1,
             "First selected command. Earlier commands are replayed outside "
             "the measurement.",
             "GPU.Debug");
DEFINE_int32(trace_profile_last_command, -1,
             "Last selected command, inclusive.", "GPU.Debug");
DECLARE_int32(trace_dump_stop_command);
DECLARE_int32(trace_dump_edram_frame);
DECLARE_int32(trace_dump_trace_frame);
DECLARE_int32(trace_dump_memory_frame);
DECLARE_uint32(trace_dump_interval);

namespace xe {
namespace gpu {

namespace {

using Writer = rapidjson::Writer<rapidjson::StringBuffer>;

void Number(Writer& w, const char* key, uint64_t value) {
  w.Key(key);
  w.Uint64(value);
}

void WriteSample(Writer& w, const TraceProfileSample& s) {
  w.StartObject();
  Number(w, "command_thread_cpu_ns", s.command_thread_cpu_ns);
  Number(w, "process_cpu_ns", s.process_cpu_ns);
  Number(w, "replay_wall_ns", s.replay_wall_ns);
  Number(w, "gpu_drain_wall_ns", s.gpu_drain_wall_ns);
  Number(w, "gpu_buffer_duration_sum_ns", s.gpu_buffer_duration_sum_ns);
  Number(w, "gpu_buffer_interval_union_ns", s.gpu_buffer_interval_union_ns);
  w.Key("counts");
  w.StartObject();
  for (size_t i = 0; i < s.stats.counts.size(); ++i) {
    Number(w, kTraceCountNames[i], s.stats.counts[i]);
  }
  w.EndObject();
  w.EndObject();
}

}  // namespace

int TraceDump::RunProfile() {
  if (!HasTraceProfiling()) {
    XELOGE("Trace profiling: this backend has no trace profiling");
    return 6;
  }
  const bool selected = cvars::trace_profile_frame >= 0 ||
                        cvars::trace_profile_first_command >= 0 ||
                        cvars::trace_profile_last_command >= 0;
  if (selected &&
      (cvars::trace_profile_frame < 0 ||
       cvars::trace_profile_frame >= player_->frame_count() ||
       cvars::trace_profile_first_command < 0 ||
       cvars::trace_profile_last_command < cvars::trace_profile_first_command ||
       size_t(cvars::trace_profile_last_command) >=
           player_->frame(cvars::trace_profile_frame)->commands.size())) {
    XELOGE("Invalid selected trace command range");
    return 6;
  }
  if (cvars::trace_profile_warmup < 2 || cvars::trace_profile_warmup > 64 ||
      !cvars::trace_profile_samples || cvars::trace_profile_samples > 1024 ||
      cvars::trace_dump_stop_command >= 0 ||
      cvars::trace_dump_edram_frame >= 0 ||
      cvars::trace_dump_trace_frame >= 0 ||
      cvars::trace_dump_memory_frame >= 0 || cvars::trace_dump_interval) {
    XELOGE(
        "Trace profiling needs 2..64 warmups, 1..1024 samples, and no "
        "diagnostic dumps");
    return 6;
  }
  const auto verification_path =
      std::filesystem::path(cvars::trace_profile_path.string() + ".verify.bin");
  const auto temporary_edram =
      std::filesystem::path(cvars::trace_profile_path.string() + ".edram.tmp");
  if (std::filesystem::exists(verification_path) ||
      std::filesystem::exists(temporary_edram)) {
    XELOGE("Profile verification output already exists");
    return 6;
  }
  std::filesystem::create_directories(
      cvars::trace_profile_path.parent_path().empty()
          ? std::filesystem::path(".")
          : cvars::trace_profile_path.parent_path());

  // The verification record is the guest output image (full replays only),
  // the EDRAM contents and, for a selected range, the guest memory the GPU
  // wrote.
  std::vector<uint8_t> reference;
  auto verify = [&](const TraceProfileSample& sample) {
    ui::RawImage image{};
    if (!selected &&
        !graphics_system_->presenter()->CaptureGuestOutput(image)) {
      return false;
    }
    bool dumped = false;
    RunOnCommandThread([&] {
      dumped = graphics_system_->command_processor()->DumpEdramSnapshotToFile(
          temporary_edram);
    });
    if (!dumped) {
      return false;
    }
    std::ifstream input(temporary_edram, std::ios::binary);
    std::vector<uint8_t> edram((std::istreambuf_iterator<char>(input)), {});
    input.close();
    std::filesystem::remove(temporary_edram);
    if (edram.size() != xenos::kEdramSizeBytes) {
      return false;
    }
    const uint64_t dimensions[] = {image.width, image.height, image.stride};
    const auto* dimensions_bytes = reinterpret_cast<const uint8_t*>(dimensions);
    std::vector<uint8_t> bytes(dimensions_bytes,
                               dimensions_bytes + sizeof(dimensions));
    bytes.insert(bytes.end(), image.data.begin(), image.data.end());
    bytes.insert(bytes.end(), edram.begin(), edram.end());
    if (selected) {
      std::vector<uint8_t> memory;
      if (!ReadTraceProfileMemory(sample.stats.memory_writes, memory)) {
        return false;
      }
      bytes.insert(bytes.end(), memory.begin(), memory.end());
    }
    if (reference.empty()) {
      reference = std::move(bytes);
      std::ofstream file(verification_path, std::ios::binary);
      file.write(reinterpret_cast<const char*>(reference.data()),
                 reference.size());
      return bool(file);
    }
    if (bytes != reference) {
      std::ofstream file(cvars::trace_profile_path.string() + ".mismatch.bin",
                         std::ios::binary);
      file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
      XELOGE("Repeated replay changed the image, EDRAM or memory output");
      return false;
    }
    return true;
  };

  // A pass is valid if its accounting is complete and it reproduced the
  // output of the first warmup pass.
  auto measure = [&]() {
    player_->RewindPlayback();
    if (selected) {
      PrepareTraceProfileReplay();
      for (int frame = 0; frame < cvars::trace_profile_frame; ++frame) {
        player_->SeekFrame(frame);
        player_->WaitOnPlayback();
      }
      // Commands are contiguous, so the prefix up to the previous command ends
      // where the selected range starts.
      if (cvars::trace_profile_first_command > 0) {
        player_->PlayFramePrefix(cvars::trace_profile_frame,
                                 cvars::trace_profile_first_command - 1);
        player_->WaitOnPlayback();
      }
    }
    TraceProfileSample sample;
    if (!BeginTraceProfile(!selected)) {
      return std::make_pair(sample, false);
    }
    if (selected) {
      player_->PlayCommandRange(cvars::trace_profile_frame,
                                cvars::trace_profile_first_command,
                                cvars::trace_profile_last_command);
      player_->WaitOnPlayback();
    } else {
      ReplayFrames();
    }
    sample = EndTraceProfile();
    return std::make_pair(sample, sample.valid() && verify(sample));
  };

  bool warm = false;
  for (uint32_t i = 0; i < cvars::trace_profile_warmup; ++i) {
    auto [sample, valid] = measure();
    if (!valid) {
      XELOGE("Trace profile warmup failed accounting or verification");
      return 7;
    }
    warm = sample.warm();
  }
  if (!warm) {
    XELOGE("Shader and pipeline creation did not settle during warmup");
    return 7;
  }
  bool valid = true;
  std::vector<TraceProfileSample> samples;
  for (uint32_t i = 0; valid && i < cvars::trace_profile_samples; ++i) {
    auto [sample, sample_valid] = measure();
    valid = sample_valid && sample.warm();
    samples.push_back(std::move(sample));
  }

  rapidjson::StringBuffer text;
  Writer w(text);
  w.StartObject();
  w.Key("valid");
  w.Bool(valid);
  w.Key("verification_file");
  w.String(xe::path_to_utf8(verification_path).c_str());
  w.Key("device");
  w.String(TraceProfileDevice().c_str());
  w.Key("samples");
  w.StartArray();
  for (const TraceProfileSample& sample : samples) {
    WriteSample(w, sample);
  }
  w.EndArray();
  w.EndObject();
  std::ofstream file(cvars::trace_profile_path);
  file << text.GetString() << '\n';
  file.close();
  valid &= bool(file);
  if (!selected && !CaptureToPng(base_output_path_.replace_extension(".png"))) {
    valid = false;
  }
  player_.reset();
  emulator_.reset();
  return valid ? 0 : 7;
}

}  // namespace gpu
}  // namespace xe
