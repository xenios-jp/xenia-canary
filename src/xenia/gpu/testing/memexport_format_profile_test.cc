/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/memexport_format_profile.h"

#include <cstring>

#include "third_party/catch/include/catch.hpp"

namespace xe {
namespace gpu {
namespace test {

namespace profile = memexport_format_profile;

namespace {

void WriteDescriptor(RegisterFile& regs, uint32_t index) {
  xenos::xe_gpu_memexport_stream_t descriptor{};
  descriptor.const_0x1 = 1;
  descriptor.const_0x4b0 = 0x4B0;
  descriptor.const_0x96 = 0x96;
  descriptor.index_count = 1;
  std::memcpy(&regs.values[XE_GPU_REG_SHADER_CONSTANT_000_X + index * 4],
              &descriptor, sizeof(descriptor));
}

Shader::MemExportStream MakeStream(uint32_t address, uint32_t constant,
                                   Shader::MemExportStream::Scale scale) {
  Shader::MemExportStream stream;
  stream.instruction_address = address;
  stream.stream_constant = constant;
  stream.scale = scale;
  return stream;
}

}  // namespace

TEST_CASE("MemExport format profile selects provable sites",
          "[memexport_format_profile]") {
  RegisterFile regs{};
  WriteDescriptor(regs, 8);
  const auto good =
      MakeStream(100, 8, Shader::MemExportStream::Scale::kLiteralZero);
  const auto unknown =
      MakeStream(101, 8, Shader::MemExportStream::Scale::kUnknown);
  std::shared_ptr<const Shader::Specialization> specialization;
  REQUIRE(profile::SelectStreams({good, unknown}, regs, 0, specialization));
  uint32_t word;
  REQUIRE(specialization->GetMemExportFormat(100, word));
  REQUIRE_FALSE(specialization->GetMemExportFormat(101, word));
  // Out-of-range constants and invalid descriptors are unprovable.
  REQUIRE_FALSE(profile::SelectStreams({good}, regs, 511, specialization));
  REQUIRE_FALSE(specialization);
  regs.values[XE_GPU_REG_SHADER_CONSTANT_000_X + 8 * 4 + 3] = 0;
  REQUIRE_FALSE(profile::SelectStreams({good}, regs, 0, specialization));
}

TEST_CASE("MemExport format profile requires a +0 scale",
          "[memexport_format_profile]") {
  RegisterFile regs{};
  WriteDescriptor(regs, 8);
  auto scale =
      MakeStream(110, 8, Shader::MemExportStream::Scale::kConstantRegister);
  scale.scale_constant = 5;
  scale.scale_component = 2;
  std::shared_ptr<const Shader::Specialization> specialization;
  REQUIRE(profile::SelectStreams({scale}, regs, 0, specialization));
  for (uint32_t bits : {0x80000000u, 1u, 0x7FC00000u}) {
    regs.values[XE_GPU_REG_SHADER_CONSTANT_000_X + 5 * 4 + 2] = bits;
    REQUIRE_FALSE(profile::SelectStreams({scale}, regs, 0, specialization));
  }
  // Either multiplicand being +0 proves the product.
  scale.has_alternate_scale = true;
  scale.alternate_scale_constant = 6;
  scale.alternate_scale_component = 2;
  REQUIRE(profile::SelectStreams({scale}, regs, 0, specialization));
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
