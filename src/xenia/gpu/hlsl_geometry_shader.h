/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_HLSL_GEOMETRY_SHADER_H_
#define XENIA_GPU_HLSL_GEOMETRY_SHADER_H_

#include <string>

#include "xenia/gpu/dxbc_geometry_shader.h"

namespace xe {
namespace gpu {

// Generates the HLSL geometry shader source for the given shared geometry key.
// This is used by DXIL-based backends that need the Xenos point / rectangle /
// quad primitive expansion without going through the legacy DXBC generator.
std::string CreateHlslGeometryShader(GeometryShaderKey key);

}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_HLSL_GEOMETRY_SHADER_H_
