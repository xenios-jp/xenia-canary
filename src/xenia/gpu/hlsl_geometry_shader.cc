/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/hlsl_geometry_shader.h"

#include <cstdint>
#include <sstream>
#include <string>

#include "xenia/base/assert.h"

namespace xe {
namespace gpu {
namespace {

const char* HlslFloatVectorType(uint32_t component_count) {
  switch (component_count) {
    case 1:
      return "float";
    case 2:
      return "float2";
    case 3:
      return "float3";
    case 4:
      return "float4";
    default:
      assert_unhandled_case(component_count);
      return "float4";
  }
}

const char* HlslGeometryInputPrimitive(PipelineGeometryShader type) {
  switch (type) {
    case PipelineGeometryShader::kPointList:
      return "point";
    case PipelineGeometryShader::kRectangleList:
      return "triangle";
    case PipelineGeometryShader::kQuadList:
      return "lineadj";
    default:
      assert_unhandled_case(type);
      return "point";
  }
}

uint32_t GeometryInputVertexCount(PipelineGeometryShader type) {
  switch (type) {
    case PipelineGeometryShader::kPointList:
      return 1;
    case PipelineGeometryShader::kRectangleList:
      return 3;
    case PipelineGeometryShader::kQuadList:
      return 4;
    default:
      assert_unhandled_case(type);
      return 1;
  }
}

void AppendHlslClipDeclarations(std::ostringstream& hlsl,
                                uint32_t clip_distance_count) {
  if (!clip_distance_count) {
    return;
  }
  if (clip_distance_count <= 4) {
    hlsl << "  " << HlslFloatVectorType(clip_distance_count)
         << " xe_clip_distance : SV_ClipDistance0;\n";
  } else {
    hlsl << "  float4 xe_clip_distance_0123 : SV_ClipDistance0;\n";
    hlsl << "  " << HlslFloatVectorType(clip_distance_count - 4)
         << " xe_clip_distance_45 : SV_ClipDistance1;\n";
  }
}

void AppendHlslCullDeclarations(std::ostringstream& hlsl,
                                uint32_t cull_distance_count) {
  if (!cull_distance_count) {
    return;
  }
  if (cull_distance_count <= 4) {
    hlsl << "  " << HlslFloatVectorType(cull_distance_count)
         << " xe_cull_distance : SV_CullDistance0;\n";
  } else {
    hlsl << "  float4 xe_cull_distance_0123 : SV_CullDistance0;\n";
    hlsl << "  " << HlslFloatVectorType(cull_distance_count - 4)
         << " xe_cull_distance_45 : SV_CullDistance1;\n";
  }
}

std::string HlslDistanceComponent(const char* name, uint32_t distance_count,
                                  uint32_t index,
                                  const std::string& source) {
  const char* components = "xyzw";
  std::string field = source + "." + name;
  if (distance_count <= 4) {
    if (distance_count == 1) {
      return field;
    }
    return field + "." + components[index];
  }
  if (index < 4) {
    return field + "_0123." + components[index];
  }
  uint32_t tail_index = index - 4;
  if (distance_count - 4 == 1) {
    return field + "_45";
  }
  return field + "_45." + components[tail_index];
}

void AppendHlslClipCopy(std::ostringstream& hlsl,
                        uint32_t clip_distance_count,
                        const char* destination, const char* source) {
  if (!clip_distance_count) {
    return;
  }
  if (clip_distance_count <= 4) {
    hlsl << "  " << destination << ".xe_clip_distance = " << source
         << ".xe_clip_distance;\n";
  } else {
    hlsl << "  " << destination << ".xe_clip_distance_0123 = " << source
         << ".xe_clip_distance_0123;\n";
    hlsl << "  " << destination << ".xe_clip_distance_45 = " << source
         << ".xe_clip_distance_45;\n";
  }
}

void AppendHlslClipMirror(std::ostringstream& hlsl,
                          uint32_t clip_distance_count) {
  if (!clip_distance_count) {
    return;
  }
  if (clip_distance_count <= 4) {
    hlsl << "  output_vertex.xe_clip_distance = -a.xe_clip_distance + "
            "b.xe_clip_distance + c.xe_clip_distance;\n";
  } else {
    hlsl << "  output_vertex.xe_clip_distance_0123 = "
            "-a.xe_clip_distance_0123 + b.xe_clip_distance_0123 + "
            "c.xe_clip_distance_0123;\n";
    hlsl << "  output_vertex.xe_clip_distance_45 = "
            "-a.xe_clip_distance_45 + b.xe_clip_distance_45 + "
            "c.xe_clip_distance_45;\n";
  }
}

void AppendHlslPointVertex(std::ostringstream& hlsl, const char* uv,
                           const char* offset) {
  hlsl << "  XeCopyVertex(input[0], output_vertex);\n";
  hlsl << "  output_vertex.xe_position.xy = input[0].xe_position.xy + "
       << offset << ";\n";
  hlsl << "  output_vertex.xe_point_parameters = float3(" << uv
       << ", 0.0);\n";
  hlsl << "  stream.Append(output_vertex);\n";
}

}  // namespace

std::string CreateHlslGeometryShader(GeometryShaderKey key) {
  std::ostringstream hlsl;
  const uint32_t clip_distance_count =
      key.user_clip_plane_cull ? 0 : key.user_clip_plane_count;
  const uint32_t cull_distance_count =
      (key.user_clip_plane_cull ? key.user_clip_plane_count : 0) +
      key.has_vertex_kill_and;
  const uint32_t input_vertex_count = GeometryInputVertexCount(key.type);

  hlsl << "// Generated HLSL geometry shader - Xenia Xbox 360 Emulator\n";
  hlsl << "// Shader Model 6.0\n\n";

  if (key.type == PipelineGeometryShader::kPointList) {
    hlsl << "cbuffer xe_system_cbuffer : register(b0) {\n";
    hlsl << "  uint4 xe_system_constants_padding[10];\n";
    hlsl << "  float2 xe_point_constant_diameter;\n";
    hlsl << "  float2 xe_point_screen_diameter_to_ndc_radius;\n";
    hlsl << "};\n\n";
  }

  hlsl << "struct GSIn {\n";
  for (uint32_t i = 0; i < key.interpolator_count; ++i) {
    hlsl << "  float4 xe_interpolator_" << i << " : TEXCOORD" << i << ";\n";
  }
  if (key.has_point_size) {
    hlsl << "  float3 xe_point_parameters : TEXCOORD"
         << key.interpolator_count << ";\n";
  }
  hlsl << "  precise float4 xe_position : SV_Position;\n";
  AppendHlslClipDeclarations(hlsl, clip_distance_count);
  AppendHlslCullDeclarations(hlsl, cull_distance_count);
  hlsl << "};\n\n";

  hlsl << "struct GSOut {\n";
  for (uint32_t i = 0; i < key.interpolator_count; ++i) {
    hlsl << "  float4 xe_interpolator_" << i << " : TEXCOORD" << i << ";\n";
  }
  if (key.has_point_coordinates) {
    hlsl << "  float3 xe_point_parameters : TEXCOORD"
         << key.interpolator_count << ";\n";
  }
  hlsl << "  precise float4 xe_position : SV_Position;\n";
  AppendHlslClipDeclarations(hlsl, clip_distance_count);
  hlsl << "};\n\n";

  hlsl << "bool XePositionIsNaN(float4 position) {\n";
  hlsl << "  return any(position != position);\n";
  hlsl << "}\n\n";

  hlsl << "void XeCopyVertex(GSIn input_vertex, out GSOut output_vertex) {\n";
  hlsl << "  output_vertex = (GSOut)0;\n";
  for (uint32_t i = 0; i < key.interpolator_count; ++i) {
    hlsl << "  output_vertex.xe_interpolator_" << i
         << " = input_vertex.xe_interpolator_" << i << ";\n";
  }
  if (key.has_point_coordinates) {
    hlsl << "  output_vertex.xe_point_parameters = float3(0.0, 0.0, 0.0);\n";
  }
  hlsl << "  output_vertex.xe_position = input_vertex.xe_position;\n";
  AppendHlslClipCopy(hlsl, clip_distance_count, "output_vertex",
                     "input_vertex");
  hlsl << "}\n\n";

  if (key.type == PipelineGeometryShader::kRectangleList) {
    hlsl << "GSOut XeMakeMirroredVertex(GSIn a, GSIn b, GSIn c) {\n";
    hlsl << "  GSOut output_vertex = (GSOut)0;\n";
    for (uint32_t i = 0; i < key.interpolator_count; ++i) {
      hlsl << "  output_vertex.xe_interpolator_" << i
           << " = -a.xe_interpolator_" << i << " + b.xe_interpolator_" << i
           << " + c.xe_interpolator_" << i << ";\n";
    }
    if (key.has_point_coordinates) {
      hlsl << "  output_vertex.xe_point_parameters = float3(0.0, 0.0, 0.0);\n";
    }
    hlsl << "  output_vertex.xe_position = -a.xe_position + b.xe_position + "
            "c.xe_position;\n";
    AppendHlslClipMirror(hlsl, clip_distance_count);
    hlsl << "  return output_vertex;\n";
    hlsl << "}\n\n";
  }

  hlsl << "[maxvertexcount(4)]\n";
  hlsl << "void main(" << HlslGeometryInputPrimitive(key.type)
       << " GSIn input[" << input_vertex_count
       << "], inout TriangleStream<GSOut> stream) {\n";

  for (uint32_t i = 0; i < input_vertex_count; ++i) {
    hlsl << "  if (XePositionIsNaN(input[" << i << "].xe_position)) return;\n";
  }
  for (uint32_t i = 0; i < cull_distance_count; ++i) {
    hlsl << "  if (";
    for (uint32_t j = 0; j < input_vertex_count; ++j) {
      if (j) {
        hlsl << " && ";
      }
      hlsl << HlslDistanceComponent(
                  "xe_cull_distance", cull_distance_count, i,
                  "input[" + std::to_string(j) + "]")
           << " < 0.0";
    }
    hlsl << ") return;\n";
  }

  switch (key.type) {
    case PipelineGeometryShader::kPointList:
      hlsl << "  float2 point_size = xe_point_constant_diameter;\n";
      if (key.has_point_size) {
        hlsl << "  if (input[0].xe_point_parameters.x >= 0.0) {\n";
        hlsl << "    point_size = input[0].xe_point_parameters.xx;\n";
        hlsl << "  }\n";
      }
      hlsl << "  if (point_size.x <= 0.0 || point_size.y <= 0.0) return;\n";
      hlsl << "  float2 point_radius = point_size * "
              "xe_point_screen_diameter_to_ndc_radius * "
              "input[0].xe_position.w;\n";
      hlsl << "  GSOut output_vertex;\n";
      if (key.has_point_coordinates) {
        AppendHlslPointVertex(hlsl, "0.0, 0.0",
                              "float2(-point_radius.x, point_radius.y)");
        AppendHlslPointVertex(hlsl, "1.0, 0.0",
                              "float2(point_radius.x, point_radius.y)");
        AppendHlslPointVertex(hlsl, "0.0, 1.0",
                              "float2(-point_radius.x, -point_radius.y)");
        AppendHlslPointVertex(hlsl, "1.0, 1.0",
                              "float2(point_radius.x, -point_radius.y)");
      } else {
        hlsl << "  XeCopyVertex(input[0], output_vertex);\n";
        hlsl << "  output_vertex.xe_position.xy = input[0].xe_position.xy + "
                "float2(-point_radius.x, point_radius.y);\n";
        hlsl << "  stream.Append(output_vertex);\n";
        hlsl << "  XeCopyVertex(input[0], output_vertex);\n";
        hlsl << "  output_vertex.xe_position.xy = input[0].xe_position.xy + "
                "float2(point_radius.x, point_radius.y);\n";
        hlsl << "  stream.Append(output_vertex);\n";
        hlsl << "  XeCopyVertex(input[0], output_vertex);\n";
        hlsl << "  output_vertex.xe_position.xy = input[0].xe_position.xy + "
                "float2(-point_radius.x, -point_radius.y);\n";
        hlsl << "  stream.Append(output_vertex);\n";
        hlsl << "  XeCopyVertex(input[0], output_vertex);\n";
        hlsl << "  output_vertex.xe_position.xy = input[0].xe_position.xy + "
                "float2(point_radius.x, -point_radius.y);\n";
        hlsl << "  stream.Append(output_vertex);\n";
      }
      hlsl << "  stream.RestartStrip();\n";
      break;
    case PipelineGeometryShader::kRectangleList:
      hlsl << "  float2 edge12 = input[2].xe_position.xy - "
              "input[1].xe_position.xy;\n";
      hlsl << "  float2 edge20 = input[0].xe_position.xy - "
              "input[2].xe_position.xy;\n";
      hlsl << "  float2 edge01 = input[1].xe_position.xy - "
              "input[0].xe_position.xy;\n";
      hlsl << "  float length12 = dot(edge12, edge12);\n";
      hlsl << "  float length20 = dot(edge20, edge20);\n";
      hlsl << "  float length01 = dot(edge01, edge01);\n";
      hlsl << "  uint i0 = 2, i1 = 0, i2 = 1;\n";
      hlsl << "  if (length12 > length20 && length12 > length01) {\n";
      hlsl << "    i0 = 0; i1 = 1; i2 = 2;\n";
      hlsl << "  } else if (length20 > length01) {\n";
      hlsl << "    i0 = 1; i1 = 2; i2 = 0;\n";
      hlsl << "  }\n";
      hlsl << "  GSOut output_vertex;\n";
      hlsl << "  XeCopyVertex(input[i0], output_vertex);\n";
      hlsl << "  stream.Append(output_vertex);\n";
      hlsl << "  XeCopyVertex(input[i1], output_vertex);\n";
      hlsl << "  stream.Append(output_vertex);\n";
      hlsl << "  XeCopyVertex(input[i2], output_vertex);\n";
      hlsl << "  stream.Append(output_vertex);\n";
      hlsl << "  output_vertex = XeMakeMirroredVertex(input[i0], input[i1], "
              "input[i2]);\n";
      hlsl << "  stream.Append(output_vertex);\n";
      hlsl << "  stream.RestartStrip();\n";
      break;
    case PipelineGeometryShader::kQuadList:
      hlsl << "  GSOut output_vertex;\n";
      hlsl << "  XeCopyVertex(input[0], output_vertex);\n";
      hlsl << "  stream.Append(output_vertex);\n";
      hlsl << "  XeCopyVertex(input[1], output_vertex);\n";
      hlsl << "  stream.Append(output_vertex);\n";
      hlsl << "  XeCopyVertex(input[3], output_vertex);\n";
      hlsl << "  stream.Append(output_vertex);\n";
      hlsl << "  XeCopyVertex(input[2], output_vertex);\n";
      hlsl << "  stream.Append(output_vertex);\n";
      hlsl << "  stream.RestartStrip();\n";
      break;
    default:
      assert_unhandled_case(key.type);
      break;
  }

  hlsl << "}\n";
  return hlsl.str();
}

}  // namespace gpu
}  // namespace xe
