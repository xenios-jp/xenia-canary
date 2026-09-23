/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <cstdint>
#include <vector>

#include "third_party/SPIRV-Headers/include/spirv/unified1/spirv.hpp"
#include "third_party/catch/include/catch.hpp"
#include "xenia/gpu/edram_dump_shader.h"

namespace xe {
namespace gpu {
namespace test {

TEST_CASE("Resolve texture output is binding 1 of the destination set",
          "[gpu]") {
  for (auto format : {xenos::ColorRenderTargetFormat::k_8_8_8_8,
                      xenos::ColorRenderTargetFormat::k_16_16_16_16_FLOAT}) {
    EdramDumpShaderKey key;
    key.resource_format = uint32_t(format);
    key.direct_resolve = 1;
    key.direct_resolve_texture = 1;
    EdramDumpShaderOptions options;
    options.descriptor_set_dest = 3;
    std::vector<uint32_t> spirv = BuildEdramDumpShaderSpirv(key, options);
    REQUIRE(spirv.size() > 5);
    // The only binding-1 variable is the resolve texture, in the destination
    // set, written once per resolved pixel of a thread.
    uint32_t binding_1_id = 0, image_writes = 0;
    for (size_t offset = 5; offset < spirv.size();) {
      const uint32_t word_count = spirv[offset] >> 16;
      const spv::Op opcode = spv::Op(spirv[offset] & 0xFFFF);
      REQUIRE(word_count != 0);
      if (opcode == spv::OpDecorate &&
          spv::Decoration(spirv[offset + 2]) == spv::DecorationBinding &&
          spirv[offset + 3] == 1) {
        REQUIRE(binding_1_id == 0);
        binding_1_id = spirv[offset + 1];
      } else if (opcode == spv::OpImageWrite) {
        ++image_writes;
      }
      offset += word_count;
    }
    REQUIRE(binding_1_id != 0);
    bool in_dest_set = false;
    for (size_t offset = 5; offset < spirv.size();
         offset += spirv[offset] >> 16) {
      in_dest_set |=
          spv::Op(spirv[offset] & 0xFFFF) == spv::OpDecorate &&
          spirv[offset + 1] == binding_1_id &&
          spv::Decoration(spirv[offset + 2]) == spv::DecorationDescriptorSet &&
          spirv[offset + 3] == options.descriptor_set_dest;
    }
    REQUIRE(in_dest_set);
    REQUIRE(image_writes == GetEdramDumpShaderResolvePixelsPerThread(
                                xenos::IsColorRenderTargetFormat64bpp(format)));
  }
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
