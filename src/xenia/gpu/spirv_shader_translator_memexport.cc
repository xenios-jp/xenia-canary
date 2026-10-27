/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2024 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/spirv_shader_translator.h"

#include "third_party/glslang/SPIRV/GLSL.std.450.h"
#include "xenia/base/assert.h"
#include "xenia/base/math.h"
#include "xenia/gpu/spirv_compatibility.h"
#include "xenia/gpu/ucode.h"

namespace xe {
namespace gpu {

void SpirvShaderTranslator::ExportToMemory(uint8_t export_eM) {
  if (!export_eM) {
    return;
  }

  assert_zero(export_eM & ~current_shader().memexport_eM_written());

  if (!IsMemoryExportSupported()) {
    return;
  }

  // Check if memory export is allowed in this guest shader invocation.
  std::optional<SpirvBuilder::IfBuilder> if_memexport_allowed;
  spv::Id memexport_allowed = main_memexport_allowed_;

  // For pixel shaders with resolution scaling, only allow memory export from
  // the center host pixel to avoid duplicate exports.
  if (is_pixel_shader() && (GetCurrentDrawResolutionScaleX() > 1 ||
                            GetCurrentDrawResolutionScaleY() > 1)) {
    assert_true(input_fragment_coordinates_ != spv::NoResult);

    // Check if we're at the center pixel (scale/2 for both X and Y).
    spv::Id is_center_pixel = builder_->makeBoolConstant(true);

    // Check X coordinate.
    if (GetCurrentDrawResolutionScaleX() > 1) {
      id_vector_temp_.clear();
      id_vector_temp_.push_back(const_int_0_);
      spv::Id pixel_x = builder_->createUnaryOp(
          spv::OpConvertFToU, type_uint_,
          builder_->createLoad(
              builder_->createAccessChain(spv::StorageClassInput,
                                          input_fragment_coordinates_,
                                          id_vector_temp_),
              spv::NoPrecision));
      spv::Id pixel_x_remainder = builder_->createBinOp(
          spv::OpUMod, type_uint_, pixel_x,
          builder_->makeUintConstant(GetCurrentDrawResolutionScaleX()));
      is_center_pixel = builder_->createBinOp(
          spv::OpLogicalAnd, type_bool_, is_center_pixel,
          builder_->createBinOp(spv::OpIEqual, type_bool_, pixel_x_remainder,
                                builder_->makeUintConstant(
                                    GetCurrentDrawResolutionScaleX() >> 1)));
    }

    // Check Y coordinate.
    if (GetCurrentDrawResolutionScaleY() > 1) {
      id_vector_temp_.clear();
      id_vector_temp_.push_back(builder_->makeIntConstant(1));
      spv::Id pixel_y = builder_->createUnaryOp(
          spv::OpConvertFToU, type_uint_,
          builder_->createLoad(
              builder_->createAccessChain(spv::StorageClassInput,
                                          input_fragment_coordinates_,
                                          id_vector_temp_),
              spv::NoPrecision));
      spv::Id pixel_y_remainder = builder_->createBinOp(
          spv::OpUMod, type_uint_, pixel_y,
          builder_->makeUintConstant(GetCurrentDrawResolutionScaleY()));
      is_center_pixel = builder_->createBinOp(
          spv::OpLogicalAnd, type_bool_, is_center_pixel,
          builder_->createBinOp(spv::OpIEqual, type_bool_, pixel_y_remainder,
                                builder_->makeUintConstant(
                                    GetCurrentDrawResolutionScaleY() >> 1)));
    }

    // Combine with existing memexport_allowed condition.
    memexport_allowed =
        memexport_allowed != spv::NoResult
            ? builder_->createBinOp(spv::OpLogicalAnd, type_bool_,
                                    memexport_allowed, is_center_pixel)
            : is_center_pixel;
  }

  if (memexport_allowed != spv::NoResult) {
    if_memexport_allowed.emplace(
        memexport_allowed, spv::SelectionControlDontFlattenMask, *builder_);
  }

  // If the pixel was killed (but the actual killing on the SPIR-V side has not
  // been performed yet because the device doesn't support demotion to helper
  // invocation that doesn't interfere with control flow), the current
  // invocation is not considered active anymore.
  std::optional<SpirvBuilder::IfBuilder> if_pixel_not_killed;
  if (var_main_kill_pixel_ != spv::NoResult) {
    if_pixel_not_killed.emplace(
        builder_->createUnaryOp(
            spv::OpLogicalNot, type_bool_,
            builder_->createLoad(var_main_kill_pixel_, spv::NoPrecision)),
        spv::SelectionControlDontFlattenMask, *builder_);
  }

  // Check if the address with the correct sign and exponent was written, and
  // that the index doesn't overflow the mantissa bits. Z takes all 12 bits of
  // const_0x4b0 rather than the top 9, so the constants the shader accepts
  // match the ones draw_util::AddMemExportRanges derives ranges from.
  // all((eA_vector >> uvec4(30, 23, 20, 23)) == uvec4(0x1, 0x96, 0x4B0, 0x96))
  spv::Id eA_vector = builder_->createUnaryOp(
      spv::OpBitcast, type_uint4_,
      builder_->createLoad(var_main_memexport_address_, spv::NoPrecision));
  id_vector_temp_.clear();
  id_vector_temp_.push_back(builder_->makeUintConstant(30));
  id_vector_temp_.push_back(builder_->makeUintConstant(23));
  id_vector_temp_.push_back(builder_->makeUintConstant(20));
  id_vector_temp_.push_back(builder_->makeUintConstant(23));
  spv::Id address_validation_shift =
      builder_->makeCompositeConstant(type_uint4_, id_vector_temp_);
  id_vector_temp_.clear();
  id_vector_temp_.push_back(builder_->makeUintConstant(0x1));
  id_vector_temp_.push_back(builder_->makeUintConstant(0x96));
  id_vector_temp_.push_back(builder_->makeUintConstant(0x4B0));
  id_vector_temp_.push_back(builder_->makeUintConstant(0x96));
  spv::Id address_validation_value =
      builder_->makeCompositeConstant(type_uint4_, id_vector_temp_);
  SpirvBuilder::IfBuilder if_address_valid(
      builder_->createUnaryOp(
          spv::OpAll, type_bool_,
          builder_->createBinOp(
              spv::OpIEqual, type_bool4_,
              builder_->createBinOp(spv::OpShiftRightLogical, type_uint4_,
                                    eA_vector, address_validation_shift),
              address_validation_value)),
      spv::SelectionControlDontFlattenMask, *builder_, 2, 1);

  using EMIdArray = std::array<spv::Id, ucode::kMaxMemExportElementCount>;

  auto for_each_eM = [&](std::function<void(uint32_t eM_index)> fn) {
    uint8_t eM_remaining = export_eM;
    uint32_t eM_index;
    while (xe::bit_scan_forward(eM_remaining, &eM_index)) {
      eM_remaining &= ~(uint8_t(1) << eM_index);
      fn(eM_index);
    }
  };

  // Load the original eM.
  EMIdArray eM_original;
  for_each_eM([&](uint32_t eM_index) {
    eM_original[eM_index] = builder_->createLoad(
        var_main_memexport_data_[eM_index], spv::NoPrecision);
  });

  // Swap red and blue if needed.
  spv::Id format_info =
      builder_->createCompositeExtract(eA_vector, type_uint_, 2);
  spv::Id swap_red_blue = builder_->createBinOp(
      spv::OpINotEqual, type_bool_,
      builder_->createBinOp(spv::OpBitwiseAnd, type_uint_, format_info,
                            builder_->makeUintConstant(uint32_t(1) << 19)),
      const_uint_0_);
  EMIdArray eM_swapped;
  uint_vector_temp_.clear();
  uint_vector_temp_.push_back(2);
  uint_vector_temp_.push_back(1);
  uint_vector_temp_.push_back(0);
  uint_vector_temp_.push_back(3);
  spv::Id swap_red_blue_bool4 =
      builder_->smearScalar(spv::NoPrecision, swap_red_blue, type_bool4_);
  for_each_eM([&](uint32_t eM_index) {
    eM_swapped[eM_index] = builder_->createTriOp(
        spv::OpSelect, type_float4_, swap_red_blue_bool4,
        builder_->createRvalueSwizzle(spv::NoPrecision, type_float4_,
                                      eM_original[eM_index], uint_vector_temp_),
        eM_original[eM_index]);
  });

  // Extract the numeric format.
  spv::Id is_signed = builder_->createBinOp(
      spv::OpINotEqual, type_bool_,
      builder_->createBinOp(spv::OpBitwiseAnd, type_uint_, format_info,
                            builder_->makeUintConstant(uint32_t(1) << 16)),
      const_uint_0_);
  spv::Id is_norm = builder_->createBinOp(
      spv::OpIEqual, type_bool_,
      builder_->createBinOp(spv::OpBitwiseAnd, type_uint_, format_info,
                            builder_->makeUintConstant(uint32_t(1) << 17)),
      const_uint_0_);

  // Perform format packing.

  auto saturate_nan = [&](const EMIdArray& eM) -> EMIdArray {
    EMIdArray eM_flushed;
    for_each_eM([&](uint32_t eM_index) {
      spv::Id element_unflushed = eM[eM_index];
      unsigned int component_count =
          builder_->getNumComponents(element_unflushed);
      eM_flushed[eM_index] = builder_->createTriOp(
          spv::OpSelect, type_float_vectors_[component_count - 1],
          builder_->createUnaryOp(spv::OpIsNan,
                                  type_bool_vectors_[component_count - 1],
                                  element_unflushed),
          builder_->createUnaryOp(
              spv::OpBitcast, type_float_vectors_[component_count - 1],
              builder_->createBinOp(
                  spv::OpBitwiseOr, type_uint_vectors_[component_count - 1],
                  builder_->createBinOp(
                      spv::OpBitwiseAnd,
                      type_uint_vectors_[component_count - 1],
                      builder_->createUnaryOp(
                          spv::OpBitcast,
                          type_uint_vectors_[component_count - 1],
                          element_unflushed),
                      builder_->smearScalar(
                          spv::NoPrecision,
                          builder_->makeUintConstant(0x80000000),
                          type_uint_vectors_[component_count - 1])),
                  builder_->smearScalar(
                      spv::NoPrecision, builder_->makeUintConstant(0x7F800000),
                      type_uint_vectors_[component_count - 1]))),
          element_unflushed);
    });
    return eM_flushed;
  };

  auto make_float_constant_vectors =
      [&](float value) -> std::array<spv::Id, 4> {
    std::array<spv::Id, 4> const_vectors;
    const_vectors[0] = builder_->makeFloatConstant(value);
    id_vector_temp_.clear();
    id_vector_temp_.push_back(const_vectors[0]);
    for (unsigned int component_count_minus_1 = 1; component_count_minus_1 < 4;
         ++component_count_minus_1) {
      id_vector_temp_.push_back(const_vectors[0]);
      const_vectors[component_count_minus_1] = builder_->makeCompositeConstant(
          type_float_vectors_[component_count_minus_1], id_vector_temp_);
    }
    return const_vectors;
  };
  std::array<spv::Id, 4> const_float_vectors_minus_1 =
      make_float_constant_vectors(-1.0f);
  std::array<spv::Id, 4> const_float_vectors_minus_0_5 =
      make_float_constant_vectors(-0.5f);
  std::array<spv::Id, 4> const_float_vectors_0_5 =
      make_float_constant_vectors(0.5f);

  // Normalized export first truncates to 23 fractional bits, then rounds
  // the exact scaled fixed-point value with ties towards positive infinity.
  // Split q * (2^m - 1) into quotient and remainder to avoid 64-bit arithmetic.
  auto normalize_fixed = [&](spv::Id value,
                             const std::array<uint32_t, 4>& widths,
                             unsigned int count, bool signed_value) {
    spv::Id float_type = type_float_vectors_[count - 1];
    spv::Id int_type = type_int_vectors_[count - 1];
    spv::Id clamped = builder_->createTriBuiltinCall(
        float_type, ext_inst_glsl_std_450_, GLSLstd450FClamp, value,
        signed_value ? const_float_vectors_minus_1[count - 1]
                     : const_float_vectors_0_[count - 1],
        const_float_vectors_1_[count - 1]);
    spv::Id q = builder_->createUnaryOp(
        spv::OpConvertFToS, int_type,
        builder_->createNoContractionBinOp(
            spv::OpFMul, float_type, clamped,
            builder_->smearScalar(spv::NoPrecision,
                                  builder_->makeFloatConstant(0x1p23f),
                                  float_type)));
    auto int_components = [&](auto component) {
      id_vector_temp_.clear();
      for (unsigned int i = 0; i < count; ++i) {
        id_vector_temp_.push_back(builder_->makeIntConstant(component(i)));
      }
      return count > 1
                 ? builder_->makeCompositeConstant(int_type, id_vector_temp_)
                 : id_vector_temp_.front();
    };
    spv::Id magnitude_bits = int_components([&](unsigned int i) {
      return int(widths[i] - uint32_t(signed_value));
    });
    spv::Id quotient_shift = int_components([&](unsigned int i) {
      return int(23 - widths[i] + uint32_t(signed_value));
    });
    spv::Id remainder_mask = int_components([&](unsigned int i) {
      return int((uint32_t(1)
                  << (23 - widths[i] + uint32_t(signed_value))) - 1);
    });
    spv::Id quotient = builder_->createBinOp(
        spv::OpShiftRightArithmetic, int_type, q, quotient_shift);
    spv::Id remainder = builder_->createBinOp(
        spv::OpBitwiseAnd, int_type, q, remainder_mask);
    spv::Id correction = builder_->createBinOp(
        spv::OpISub, int_type,
        builder_->createBinOp(spv::OpShiftLeftLogical, int_type, remainder,
                              magnitude_bits),
        q);
    correction = builder_->createBinOp(
        spv::OpIAdd, int_type, correction,
        builder_->smearScalar(spv::NoPrecision,
                              builder_->makeIntConstant(1 << 22), int_type));
    correction = builder_->createBinOp(
        spv::OpShiftRightArithmetic, int_type, correction,
        builder_->smearScalar(spv::NoPrecision, builder_->makeIntConstant(23),
                              int_type));
    return builder_->createUnaryOp(
        spv::OpConvertSToF, float_type,
        builder_->createBinOp(spv::OpIAdd, int_type, quotient, correction));
  };

  // Integer conversion has an asymmetric negative half-step threshold.
  // Separating integral and fractional parts also avoids rounding a value just
  // below +0.5 up to 1.0 while adding the rounding bias in float32.
  auto round_integer = [&](spv::Id value, const std::array<uint32_t, 4>& widths,
                           unsigned int count, bool signed_value) {
    spv::Id type = type_float_vectors_[count - 1];
    spv::Id negative = spv::NoResult;
    spv::Id bias = const_float_vectors_0_5[count - 1];
    spv::Id threshold = bias;
    spv::Id direction = const_float_vectors_1_[count - 1];
    if (signed_value) {
      negative = builder_->createBinOp(
          spv::OpFOrdLessThan, type_bool_vectors_[count - 1], value,
          const_float_vectors_0_[count - 1]);
      bias = builder_->createTriOp(
          spv::OpSelect, type, negative,
          const_float_vectors_minus_0_5[count - 1], bias);
      direction = builder_->createTriOp(
          spv::OpSelect, type, negative,
          const_float_vectors_minus_1[count - 1], direction);
      id_vector_temp_.clear();
      for (unsigned int i = 0; i < count; ++i) {
        id_vector_temp_.push_back(builder_->makeFloatConstant(
            0.5f + float(uint32_t(1) << widths[i]) * 0x1p-24f));
      }
      spv::Id negative_threshold =
          count > 1 ? builder_->makeCompositeConstant(type, id_vector_temp_)
                    : id_vector_temp_.front();
      threshold = builder_->createTriOp(spv::OpSelect, type, negative,
                                        negative_threshold, threshold);
    }
    spv::Id integral = builder_->createUnaryBuiltinCall(
        type, ext_inst_glsl_std_450_, GLSLstd450Trunc, value);
    spv::Id fraction = builder_->createUnaryBuiltinCall(
        type, ext_inst_glsl_std_450_, GLSLstd450FAbs,
        builder_->createNoContractionBinOp(spv::OpFSub, type, value, integral));
    spv::Id increment = builder_->createTriOp(
        spv::OpSelect, type,
        builder_->createBinOp(spv::OpFOrdGreaterThanEqual,
                              type_bool_vectors_[count - 1], fraction,
                              threshold),
        direction, const_float_vectors_0_[count - 1]);
    spv::Id integer_rounded = builder_->createNoContractionBinOp(
        spv::OpFAdd, type, integral, increment);
    spv::Id normalized_rounded = builder_->createNoContractionBinOp(
        spv::OpFAdd, type, value, bias);
    return builder_->createTriOp(
        spv::OpSelect, type,
        count > 1 ? builder_->smearScalar(spv::NoPrecision, is_norm,
                                          type_bool_vectors_[count - 1])
                  : is_norm,
        normalized_rounded, integer_rounded);
  };

  // The widths must be without holes (R, RG, RGB, RGBA), and expecting the
  // widths to add up to the size of the stored texel (8, 16 or 32 bits), as the
  // unused upper bits will contain junk from the sign extension of X if the
  // number is signed.
  auto pack_8_16_32 = [&](std::array<uint32_t, 4> widths) -> EMIdArray {
    unsigned int component_count;
    std::array<uint32_t, 4> offsets{};
    for (component_count = 0; component_count < widths.size();
         ++component_count) {
      if (!widths[component_count]) {
        break;
      }
      // Only formats for which max + 0.5 can be represented exactly.
      assert(widths[component_count] <= 23);
      if (component_count) {
        offsets[component_count] =
            offsets[component_count - 1] + widths[component_count - 1];
      }
    }
    assert_not_zero(component_count);

    // Memory export saturates integer channels to the destination range before
    // packing. Masking an out-of-range converted value would wrap instead.
    auto component_limit = [&](bool is_signed, bool minimum) {
      id_vector_temp_.clear();
      for (unsigned int i = 0; i < component_count; ++i) {
        uint32_t magnitude = uint32_t(1) << (widths[i] - uint32_t(is_signed));
        float limit = minimum ? (is_signed ? -float(magnitude) : 0.0f)
                              : float(magnitude - 1);
        id_vector_temp_.push_back(builder_->makeFloatConstant(limit));
      }
      return component_count > 1
                 ? builder_->makeCompositeConstant(
                       type_float_vectors_[component_count - 1], id_vector_temp_)
                 : id_vector_temp_.front();
    };
    spv::Id signed_min = component_limit(true, true);
    spv::Id signed_max = component_limit(true, false);
    spv::Id unsigned_max = component_limit(false, false);

    // Extract the needed components.
    EMIdArray eM_unflushed = eM_swapped;
    if (component_count < 4) {
      if (component_count == 1) {
        for_each_eM([&](uint32_t eM_index) {
          eM_unflushed[eM_index] = builder_->createCompositeExtract(
              eM_unflushed[eM_index], type_float_, 0);
        });
      } else {
        uint_vector_temp_.clear();
        for (unsigned int component_index = 0;
             component_index < component_count; ++component_index) {
          uint_vector_temp_.push_back(component_index);
        }
        for_each_eM([&](uint32_t eM_index) {
          eM_unflushed[eM_index] = builder_->createRvalueSwizzle(
              spv::NoPrecision, type_float_vectors_[component_count - 1],
              eM_unflushed[eM_index], uint_vector_temp_);
        });
      }
    }

    // Saturate NaNs according to their sign, like infinities.
    EMIdArray eM_flushed = saturate_nan(eM_unflushed);

    // Convert to integers.
    SpirvBuilder::IfBuilder if_signed(
        is_signed, spv::SelectionControlDontFlattenMask, *builder_);
    EMIdArray eM_signed;
    {
      // Signed.
      SpirvBuilder::IfBuilder if_norm(
          is_norm, spv::SelectionControlDontFlattenMask, *builder_);
      EMIdArray eM_norm;
      {
        // Quantize and scale using the console's fixed-point conversion.
        for_each_eM([&](uint32_t eM_index) {
          eM_norm[eM_index] = normalize_fixed(
              eM_flushed[eM_index], widths, component_count, true);
        });
      }
      if_norm.makeEndIf();
      // All phi instructions must be in the beginning of the block.
      for_each_eM([&](uint32_t eM_index) {
        eM_signed[eM_index] =
            if_norm.createMergePhi(eM_norm[eM_index], eM_flushed[eM_index]);
      });
      // Convert to signed integer, adding plus/minus 0.5 before truncating
      // according to the Direct3D format conversion rules.
      for_each_eM([&](uint32_t eM_index) {
        eM_signed[eM_index] = builder_->createTriBuiltinCall(
            type_float_vectors_[component_count - 1], ext_inst_glsl_std_450_,
            GLSLstd450FClamp, eM_signed[eM_index], signed_min, signed_max);
        eM_signed[eM_index] = builder_->createUnaryOp(
            spv::OpBitcast, type_uint_vectors_[component_count - 1],
            builder_->createUnaryOp(spv::OpConvertFToS, type_int_vectors_[component_count - 1],
                                    round_integer(eM_signed[eM_index], widths, component_count, true)));
      });
    }
    if_signed.makeBeginElse();
    EMIdArray eM_unsigned;
    {
      SpirvBuilder::IfBuilder if_norm(
          is_norm, spv::SelectionControlDontFlattenMask, *builder_);
      EMIdArray eM_norm;
      {
        // Quantize and scale using the console's fixed-point conversion.
        for_each_eM([&](uint32_t eM_index) {
          eM_norm[eM_index] = normalize_fixed(
              eM_flushed[eM_index], widths, component_count, false);
        });
      }
      if_norm.makeEndIf();
      // All phi instructions must be in the beginning of the block.
      for_each_eM([&](uint32_t eM_index) {
        eM_unsigned[eM_index] =
            if_norm.createMergePhi(eM_norm[eM_index], eM_flushed[eM_index]);
      });
      // Convert to unsigned integer, adding 0.5 before truncating according to
      // the Direct3D format conversion rules.
      for_each_eM([&](uint32_t eM_index) {
        eM_unsigned[eM_index] = builder_->createTriBuiltinCall(
            type_float_vectors_[component_count - 1], ext_inst_glsl_std_450_,
            GLSLstd450FClamp, eM_unsigned[eM_index],
            const_float_vectors_0_[component_count - 1], unsigned_max);
        eM_unsigned[eM_index] = builder_->createUnaryOp(
            spv::OpConvertFToU, type_uint_vectors_[component_count - 1], round_integer(eM_unsigned[eM_index], widths, component_count, false));
      });
    }
    if_signed.makeEndIf();
    EMIdArray eM_unpacked;
    for_each_eM([&](uint32_t eM_index) {
      eM_unpacked[eM_index] =
          if_signed.createMergePhi(eM_signed[eM_index], eM_unsigned[eM_index]);
    });

    // Pack into a 32-bit value, and pad to a 4-component vector for the phi.
    EMIdArray eM_packed;
    for_each_eM([&](uint32_t eM_index) {
      spv::Id element_unpacked = eM_unpacked[eM_index];
      eM_packed[eM_index] = component_count > 1
                                ? builder_->createCompositeExtract(
                                      element_unpacked, type_uint_, 0)
                                : element_unpacked;
      for (unsigned int component_index = 1; component_index < component_count;
           ++component_index) {
        eM_packed[eM_index] = builder_->createQuadOp(
            spv::OpBitFieldInsert, type_uint_, eM_packed[eM_index],
            builder_->createCompositeExtract(element_unpacked, type_uint_,
                                             component_index),
            builder_->makeUintConstant(offsets[component_index]),
            builder_->makeUintConstant(widths[component_index]));
      }
      id_vector_temp_.clear();
      id_vector_temp_.resize(4, const_uint_0_);
      id_vector_temp_.front() = eM_packed[eM_index];
      eM_packed[eM_index] =
          builder_->createCompositeConstruct(type_uint4_, id_vector_temp_);
    });

    return eM_packed;
  };

  SpirvBuilder::SwitchBuilder format_switch(
      builder_->createTriOp(spv::OpBitFieldUExtract, type_uint_, format_info,
                            builder_->makeUintConstant(8),
                            builder_->makeUintConstant(6)),
      spv::SelectionControlDontFlattenMask, *builder_);

  struct FormatCase {
    EMIdArray eM_packed;
    uint32_t element_bytes_log2;
    spv::Id phi_parent;
  };
  std::vector<FormatCase> format_cases;
  // Must be called at the end of the switch case segment for the correct phi
  // parent.
  auto add_format_case = [&](const EMIdArray& eM_packed,
                             uint32_t element_bytes_log2) {
    FormatCase& format_case = format_cases.emplace_back();
    format_case.eM_packed = eM_packed;
    format_case.element_bytes_log2 = element_bytes_log2;
    format_case.phi_parent = builder_->getBuildPoint()->getId();
  };

  // k_8, k_8_A, k_8_B
  format_switch.makeBeginCase(
      static_cast<unsigned int>(xenos::ColorFormat::k_8));
  format_switch.addCurrentCaseLiteral(
      static_cast<unsigned int>(xenos::ColorFormat::k_8_A));
  format_switch.addCurrentCaseLiteral(
      static_cast<unsigned int>(xenos::ColorFormat::k_8_B));
  add_format_case(pack_8_16_32({8}), 0);

  // k_1_5_5_5
  format_switch.makeBeginCase(
      static_cast<unsigned int>(xenos::ColorFormat::k_1_5_5_5));
  add_format_case(pack_8_16_32({5, 5, 5, 1}), 1);

  // k_5_6_5
  format_switch.makeBeginCase(
      static_cast<unsigned int>(xenos::ColorFormat::k_5_6_5));
  add_format_case(pack_8_16_32({5, 6, 5}), 1);

  // k_6_5_5
  format_switch.makeBeginCase(
      static_cast<unsigned int>(xenos::ColorFormat::k_6_5_5));
  add_format_case(pack_8_16_32({5, 5, 6}), 1);

  // k_8_8_8_8, k_8_8_8_8_A, k_8_8_8_8_AS_16_16_16_16
  format_switch.makeBeginCase(
      static_cast<unsigned int>(xenos::ColorFormat::k_8_8_8_8));
  format_switch.addCurrentCaseLiteral(
      static_cast<unsigned int>(xenos::ColorFormat::k_8_8_8_8_A));
  format_switch.addCurrentCaseLiteral(
      static_cast<unsigned int>(xenos::ColorFormat::k_8_8_8_8_AS_16_16_16_16));
  add_format_case(pack_8_16_32({8, 8, 8, 8}), 2);

  // k_2_10_10_10, k_2_10_10_10_AS_16_16_16_16
  format_switch.makeBeginCase(
      static_cast<unsigned int>(xenos::ColorFormat::k_2_10_10_10));
  format_switch.addCurrentCaseLiteral(static_cast<unsigned int>(
      xenos::ColorFormat::k_2_10_10_10_AS_16_16_16_16));
  add_format_case(pack_8_16_32({10, 10, 10, 2}), 2);

  // k_8_8
  format_switch.makeBeginCase(
      static_cast<unsigned int>(xenos::ColorFormat::k_8_8));
  add_format_case(pack_8_16_32({8, 8}), 1);

  // k_4_4_4_4
  format_switch.makeBeginCase(
      static_cast<unsigned int>(xenos::ColorFormat::k_4_4_4_4));
  add_format_case(pack_8_16_32({4, 4, 4, 4}), 1);

  // k_10_11_11, k_10_11_11_AS_16_16_16_16
  format_switch.makeBeginCase(
      static_cast<unsigned int>(xenos::ColorFormat::k_10_11_11));
  format_switch.addCurrentCaseLiteral(
      static_cast<unsigned int>(xenos::ColorFormat::k_10_11_11_AS_16_16_16_16));
  add_format_case(pack_8_16_32({11, 11, 10}), 2);

  // k_11_11_10, k_11_11_10_AS_16_16_16_16
  format_switch.makeBeginCase(
      static_cast<unsigned int>(xenos::ColorFormat::k_11_11_10));
  format_switch.addCurrentCaseLiteral(
      static_cast<unsigned int>(xenos::ColorFormat::k_11_11_10_AS_16_16_16_16));
  add_format_case(pack_8_16_32({10, 11, 11}), 2);

  // k_16
  format_switch.makeBeginCase(
      static_cast<unsigned int>(xenos::ColorFormat::k_16));
  add_format_case(pack_8_16_32({16}), 1);

  // k_16_16
  format_switch.makeBeginCase(
      static_cast<unsigned int>(xenos::ColorFormat::k_16_16));
  add_format_case(pack_8_16_32({16, 16}), 2);

  // k_16_16_16_16
  format_switch.makeBeginCase(
      static_cast<unsigned int>(xenos::ColorFormat::k_16_16_16_16));
  {
    // Saturate NaNs according to their sign, like infinities.
    EMIdArray fixed16_flushed = saturate_nan(eM_swapped);
    spv::Id fixed16_signed_min = make_float_constant_vectors(-32768.0f)[3];
    spv::Id fixed16_signed_max = make_float_constant_vectors(32767.0f)[3];
    spv::Id fixed16_unsigned_max = make_float_constant_vectors(65535.0f)[3];

    // Convert to integers.
    SpirvBuilder::IfBuilder if_signed(
        is_signed, spv::SelectionControlDontFlattenMask, *builder_);
    EMIdArray fixed16_signed;
    {
      // Signed.
      SpirvBuilder::IfBuilder if_norm(
          is_norm, spv::SelectionControlDontFlattenMask, *builder_);
      EMIdArray fixed16_norm;
      {
        // Quantize and scale using the console's fixed-point conversion.
        for_each_eM([&](uint32_t eM_index) {
          fixed16_norm[eM_index] = normalize_fixed(
              fixed16_flushed[eM_index], std::array<uint32_t, 4>{16, 16, 16, 16}, 4, true);
        });
      }
      if_norm.makeEndIf();
      // All phi instructions must be in the beginning of the block.
      for_each_eM([&](uint32_t eM_index) {
        fixed16_signed[eM_index] = if_norm.createMergePhi(
            fixed16_norm[eM_index], fixed16_flushed[eM_index]);
      });
      // Convert to signed integer, adding plus/minus 0.5 before truncating
      // according to the Direct3D format conversion rules.
      for_each_eM([&](uint32_t eM_index) {
        fixed16_signed[eM_index] = builder_->createTriBuiltinCall(
            type_float4_, ext_inst_glsl_std_450_, GLSLstd450FClamp,
            fixed16_signed[eM_index], fixed16_signed_min, fixed16_signed_max);
        fixed16_signed[eM_index] = builder_->createUnaryOp(
            spv::OpBitcast, type_uint4_,
            builder_->createUnaryOp(spv::OpConvertFToS, type_int4_,
                                    round_integer(fixed16_signed[eM_index], std::array<uint32_t, 4>{16, 16, 16, 16}, 4, true)));
      });
    }
    if_signed.makeBeginElse();
    EMIdArray fixed16_unsigned;
    {
      // Unsigned.
      SpirvBuilder::IfBuilder if_norm(
          is_norm, spv::SelectionControlDontFlattenMask, *builder_);
      EMIdArray fixed16_norm;
      {
        // Quantize and scale using the console's fixed-point conversion.
        for_each_eM([&](uint32_t eM_index) {
          fixed16_norm[eM_index] = normalize_fixed(
              fixed16_flushed[eM_index], std::array<uint32_t, 4>{16, 16, 16, 16}, 4, false);
        });
      }
      if_norm.makeEndIf();
      // All phi instructions must be in the beginning of the block.
      for_each_eM([&](uint32_t eM_index) {
        fixed16_unsigned[eM_index] = if_norm.createMergePhi(
            fixed16_norm[eM_index], fixed16_flushed[eM_index]);
      });
      // Convert to unsigned integer, adding 0.5 before truncating according to
      // the Direct3D format conversion rules.
      for_each_eM([&](uint32_t eM_index) {
        fixed16_unsigned[eM_index] = builder_->createTriBuiltinCall(
            type_float4_, ext_inst_glsl_std_450_, GLSLstd450FClamp,
            fixed16_unsigned[eM_index], const_float4_0_, fixed16_unsigned_max);
        fixed16_unsigned[eM_index] = builder_->createUnaryOp(
            spv::OpConvertFToU, type_uint4_, round_integer(fixed16_unsigned[eM_index], std::array<uint32_t, 4>{16, 16, 16, 16}, 4, false));
      });
    }
    if_signed.makeEndIf();
    EMIdArray fixed16_unpacked;
    for_each_eM([&](uint32_t eM_index) {
      fixed16_unpacked[eM_index] = if_signed.createMergePhi(
          fixed16_signed[eM_index], fixed16_unsigned[eM_index]);
    });

    // Pack into two 32-bit values, and pad to a 4-component vector for the phi.
    EMIdArray fixed16_packed;
    spv::Id const_uint_16 = builder_->makeUintConstant(16);
    for_each_eM([&](uint32_t eM_index) {
      spv::Id fixed16_element_unpacked = fixed16_unpacked[eM_index];
      id_vector_temp_.clear();
      for (uint32_t component_index = 0; component_index < 2;
           ++component_index) {
        id_vector_temp_.push_back(builder_->createQuadOp(
            spv::OpBitFieldInsert, type_uint_,
            builder_->createCompositeExtract(fixed16_element_unpacked,
                                             type_uint_, 2 * component_index),
            builder_->createCompositeExtract(
                fixed16_element_unpacked, type_uint_, 2 * component_index + 1),
            const_uint_16, const_uint_16));
      }
      for (uint32_t component_index = 2; component_index < 4;
           ++component_index) {
        id_vector_temp_.push_back(const_uint_0_);
      }
      fixed16_packed[eM_index] =
          builder_->createCompositeConstruct(type_uint4_, id_vector_temp_);
    });

    add_format_case(fixed16_packed, 3);
  }

  // Xbox 360 float16 uses extended range: exponent 31 is a large finite value,
  // not Inf/NaN. See PackFloat16x2ExtendedRange.

  // k_16_FLOAT
  format_switch.makeBeginCase(
      static_cast<unsigned int>(xenos::ColorFormat::k_16_FLOAT));
  {
    EMIdArray format_packed_16_float;
    for_each_eM([&](uint32_t eM_index) {
      id_vector_temp_.clear();
      id_vector_temp_.push_back(builder_->createCompositeExtract(
          eM_swapped[eM_index], type_float_, 0));
      id_vector_temp_.push_back(const_float_0_);
      spv::Id format_packed_16_float_x = PackFloat16x2ExtendedRange(
          builder_->createCompositeConstruct(type_float2_, id_vector_temp_));
      id_vector_temp_.clear();
      id_vector_temp_.resize(4, const_uint_0_);
      id_vector_temp_.front() = format_packed_16_float_x;
      format_packed_16_float[eM_index] =
          builder_->createCompositeConstruct(type_uint4_, id_vector_temp_);
    });
    add_format_case(format_packed_16_float, 1);
  }

  // k_16_16_FLOAT
  format_switch.makeBeginCase(
      static_cast<unsigned int>(xenos::ColorFormat::k_16_16_FLOAT));
  {
    EMIdArray format_packed_16_16_float;
    for_each_eM([&](uint32_t eM_index) {
      uint_vector_temp_.clear();
      uint_vector_temp_.push_back(0);
      uint_vector_temp_.push_back(1);
      spv::Id format_packed_16_16_float_xy =
          PackFloat16x2ExtendedRange(builder_->createRvalueSwizzle(
              spv::NoPrecision, type_float2_, eM_swapped[eM_index],
              uint_vector_temp_));
      id_vector_temp_.clear();
      id_vector_temp_.resize(4, const_uint_0_);
      id_vector_temp_.front() = format_packed_16_16_float_xy;
      format_packed_16_16_float[eM_index] =
          builder_->createCompositeConstruct(type_uint4_, id_vector_temp_);
    });
    add_format_case(format_packed_16_16_float, 2);
  }

  // k_16_16_16_16_FLOAT
  format_switch.makeBeginCase(
      static_cast<unsigned int>(xenos::ColorFormat::k_16_16_16_16_FLOAT));
  {
    EMIdArray format_packed_16_16_16_16_float;
    for_each_eM([&](uint32_t eM_index) {
      spv::Id format_packed_16_16_16_16_float_xy_zw[2];
      for (uint32_t component_index = 0; component_index < 2;
           ++component_index) {
        uint_vector_temp_.clear();
        uint_vector_temp_.push_back(2 * component_index);
        uint_vector_temp_.push_back(2 * component_index + 1);
        format_packed_16_16_16_16_float_xy_zw[component_index] =
            PackFloat16x2ExtendedRange(builder_->createRvalueSwizzle(
                spv::NoPrecision, type_float2_, eM_swapped[eM_index],
                uint_vector_temp_));
      }
      id_vector_temp_.clear();
      id_vector_temp_.push_back(format_packed_16_16_16_16_float_xy_zw[0]);
      id_vector_temp_.push_back(format_packed_16_16_16_16_float_xy_zw[1]);
      id_vector_temp_.push_back(const_uint_0_);
      id_vector_temp_.push_back(const_uint_0_);
      format_packed_16_16_16_16_float[eM_index] =
          builder_->createCompositeConstruct(type_uint4_, id_vector_temp_);
    });
    add_format_case(format_packed_16_16_16_16_float, 3);
  }

  // k_32_FLOAT
  format_switch.makeBeginCase(
      static_cast<unsigned int>(xenos::ColorFormat::k_32_FLOAT));
  {
    EMIdArray format_packed_32_float;
    for_each_eM([&](uint32_t eM_index) {
      // For single-channel float export, swap_rb selects alpha rather than
      // applying the four-channel red/blue permutation.
      spv::Id value = builder_->createTriOp(
          spv::OpSelect, type_float_, swap_red_blue,
          builder_->createCompositeExtract(eM_original[eM_index], type_float_, 3),
          builder_->createCompositeExtract(eM_original[eM_index], type_float_, 0));
      format_packed_32_float[eM_index] = builder_->createUnaryOp(
          spv::OpBitcast, type_uint4_,
          builder_->createCompositeInsert(value, eM_original[eM_index],
                                          type_float4_, 0));
    });
    add_format_case(format_packed_32_float, 2);
  }

  // k_32_32_FLOAT
  format_switch.makeBeginCase(
      static_cast<unsigned int>(xenos::ColorFormat::k_32_32_FLOAT));
  {
    EMIdArray format_packed_32_32_float;
    for_each_eM([&](uint32_t eM_index) {
      // Two-channel float export keeps red and selects alpha for its second
      // channel when swap_rb is set.
      spv::Id value = builder_->createTriOp(
          spv::OpSelect, type_float_, swap_red_blue,
          builder_->createCompositeExtract(eM_original[eM_index], type_float_, 3),
          builder_->createCompositeExtract(eM_original[eM_index], type_float_, 1));
      format_packed_32_32_float[eM_index] = builder_->createUnaryOp(
          spv::OpBitcast, type_uint4_,
          builder_->createCompositeInsert(value, eM_original[eM_index],
                                          type_float4_, 1));
    });
    add_format_case(format_packed_32_32_float, 3);
  }

  // k_32_32_32_32_FLOAT
  format_switch.makeBeginCase(
      static_cast<unsigned int>(xenos::ColorFormat::k_32_32_32_32_FLOAT));
  {
    EMIdArray format_packed_32_32_32_32_float;
    for_each_eM([&](uint32_t eM_index) {
      format_packed_32_32_32_32_float[eM_index] = builder_->createUnaryOp(
          spv::OpBitcast, type_uint4_, eM_swapped[eM_index]);
    });
    add_format_case(format_packed_32_32_32_32_float, 4);
  }

  format_switch.makeEndSwitch();

  // Select the result and the element size based on the format.
  // Phi must be the first instructions in a block.
  EMIdArray eM_packed;
  for_each_eM([&](uint32_t eM_index) {
    auto eM_packed_phi = std::make_unique<spv::Instruction>(
        builder_->getUniqueId(), type_uint4_, spv::OpPhi);
    // Default case for an invalid format.
    eM_packed_phi->addIdOperand(const_uint4_0_);
    eM_packed_phi->addIdOperand(format_switch.getDefaultPhiParent());
    for (const FormatCase& format_case : format_cases) {
      eM_packed_phi->addIdOperand(format_case.eM_packed[eM_index]);
      eM_packed_phi->addIdOperand(format_case.phi_parent);
    }
    eM_packed[eM_index] = eM_packed_phi->getResultId();
    builder_->getBuildPoint()->addInstruction(std::move(eM_packed_phi));
  });
  spv::Id element_bytes_log2;
  {
    auto element_bytes_log2_phi = std::make_unique<spv::Instruction>(
        builder_->getUniqueId(), type_uint_, spv::OpPhi);
    // Default case for an invalid format (doesn't enter any element size
    // conditional, skipped).
    element_bytes_log2_phi->addIdOperand(builder_->makeUintConstant(5));
    element_bytes_log2_phi->addIdOperand(format_switch.getDefaultPhiParent());
    for (const FormatCase& format_case : format_cases) {
      element_bytes_log2_phi->addIdOperand(
          builder_->makeUintConstant(format_case.element_bytes_log2));
      element_bytes_log2_phi->addIdOperand(format_case.phi_parent);
    }
    element_bytes_log2 = element_bytes_log2_phi->getResultId();
    builder_->getBuildPoint()->addInstruction(
        std::move(element_bytes_log2_phi));
  }

  // Endian-swap.
  spv::Id endian =
      builder_->createTriOp(spv::OpBitFieldUExtract, type_uint_, format_info,
                            const_uint_0_, builder_->makeUintConstant(3));
  for_each_eM([&](uint32_t eM_index) {
    eM_packed[eM_index] = EndianSwap128Uint4(eM_packed[eM_index], endian);
  });

  // Load the index of eM0 in the stream.
  spv::Id eM0_index = builder_->createTriOp(
      spv::OpBitFieldUExtract, type_uint_,
      builder_->createCompositeExtract(eA_vector, type_uint_, 1), const_uint_0_,
      builder_->makeUintConstant(23));

  // The stream count validates the base index, not each eM offset. Xbox
  // exports all written elements when eM0 is in range, including elements
  // beyond the count (verified with a five-register memory allocation).
  spv::Id stream_count = builder_->createTriOp(
      spv::OpBitFieldUExtract, type_uint_,
      builder_->createCompositeExtract(eA_vector, type_uint_, 3), const_uint_0_,
      builder_->makeUintConstant(23));
  spv::Id eM_indices_to_store = builder_->createTriOp(
      spv::OpSelect, type_uint_,
      builder_->createBinOp(spv::OpULessThan, type_bool_, eM0_index,
                            stream_count),
      builder_->createLoad(var_main_memexport_data_written_, spv::NoPrecision),
      const_uint_0_);

  // Get the eM0 address in bytes.
  // Left-shift the stream base address by 2 to both convert it from dwords to
  // bytes and drop the upper bits.
  spv::Id const_uint_2 = builder_->makeUintConstant(2);
  // Masked to physical - the guest may use a mirror window.
  spv::Id eM0_address_bytes = builder_->createBinOp(
      spv::OpBitwiseAnd, type_uint_,
      builder_->createBinOp(
          spv::OpIAdd, type_uint_,
          builder_->createBinOp(
              spv::OpShiftLeftLogical, type_uint_,
              builder_->createCompositeExtract(eA_vector, type_uint_, 0),
              const_uint_2),
          builder_->createBinOp(spv::OpShiftLeftLogical, type_uint_, eM0_index,
                                element_bytes_log2)),
      builder_->makeUintConstant(0x1FFFFFFF));

  // Raw memory ALLOC sizes 0 and 1 suppress carry across the physical
  // four-element window. Larger allocations use linear element addresses.
  spv::Id wrap_elements = builder_->createBinOp(
      spv::OpULessThanEqual, type_bool_,
      builder_->createLoad(var_main_memexport_allocation_size_, spv::NoPrecision),
      builder_->makeUintConstant(1));
  auto element_address = [&](spv::Id base, uint32_t index,
                             uint32_t element_units) -> spv::Id {
    if (!index) {
      return base;
    }
    spv::Id linear = builder_->createBinOp(
        spv::OpIAdd, type_uint_, base,
        builder_->makeUintConstant(index * element_units));
    uint32_t window_mask = 4 * element_units - 1;
    spv::Id wrapped = builder_->createBinOp(
        spv::OpBitwiseOr, type_uint_,
        builder_->createBinOp(spv::OpBitwiseAnd, type_uint_, base,
                              builder_->makeUintConstant(~window_mask)),
        builder_->createBinOp(spv::OpBitwiseAnd, type_uint_, linear,
                              builder_->makeUintConstant(window_mask)));
    return builder_->createTriOp(spv::OpSelect, type_uint_, wrap_elements,
                                 wrapped, linear);
  };

  // Store based on the element size.
  auto store_needed_eM = [&](std::function<void(uint32_t eM_index)> fn) {
    for_each_eM([&](uint32_t eM_index) {
      SpirvBuilder::IfBuilder if_eM_needed(
          builder_->createBinOp(
              spv::OpINotEqual, type_bool_,
              builder_->createBinOp(spv::OpBitwiseAnd, type_uint_,
                                    eM_indices_to_store,
                                    builder_->makeUintConstant(1u << eM_index)),
              const_uint_0_),
          spv::SelectionControlDontFlattenMask, *builder_, 2, 1);
      fn(eM_index);
      if_eM_needed.makeEndIf();
    });
  };
  SpirvBuilder::SwitchBuilder element_size_switch(
      element_bytes_log2, spv::SelectionControlDontFlattenMask, *builder_);
  element_size_switch.makeBeginCase(0);
  {
    store_needed_eM([&](uint32_t eM_index) {
      spv::Id element_address_bytes =
          element_address(eM0_address_bytes, eM_index, 1);
      // replace_shift = 8 * (element_address_bytes & 3)
      spv::Id replace_shift = builder_->createQuadOp(
          spv::OpBitFieldInsert, type_uint_, const_uint_0_,
          element_address_bytes, builder_->makeUintConstant(3), const_uint_2);
      StoreUint32ToSharedMemory(
          builder_->createBinOp(spv::OpShiftLeftLogical, type_uint_,
                                builder_->createCompositeExtract(
                                    eM_packed[eM_index], type_uint_, 0),
                                replace_shift),
          builder_->createUnaryOp(
              spv::OpBitcast, type_int_,
              builder_->createBinOp(spv::OpShiftRightLogical, type_uint_,
                                    element_address_bytes, const_uint_2)),
          builder_->createBinOp(spv::OpShiftLeftLogical, type_uint_,
                                builder_->makeUintConstant(0xFFu),
                                replace_shift));
    });
  }
  element_size_switch.makeBeginCase(1);
  {
    spv::Id const_uint_1 = builder_->makeUintConstant(1);
    spv::Id eM0_address_words = builder_->createBinOp(
        spv::OpShiftRightLogical, type_uint_, eM0_address_bytes, const_uint_1);
    store_needed_eM([&](uint32_t eM_index) {
      spv::Id element_address_words =
          element_address(eM0_address_words, eM_index, 1);
      // replace_shift = 16 * (element_address_words & 1)
      spv::Id replace_shift = builder_->createQuadOp(
          spv::OpBitFieldInsert, type_uint_, const_uint_0_,
          element_address_words, builder_->makeUintConstant(4), const_uint_1);
      StoreUint32ToSharedMemory(
          builder_->createBinOp(spv::OpShiftLeftLogical, type_uint_,
                                builder_->createCompositeExtract(
                                    eM_packed[eM_index], type_uint_, 0),
                                replace_shift),
          builder_->createUnaryOp(
              spv::OpBitcast, type_int_,
              builder_->createBinOp(spv::OpShiftRightLogical, type_uint_,
                                    element_address_words, const_uint_1)),
          builder_->createBinOp(spv::OpShiftLeftLogical, type_uint_,
                                builder_->makeUintConstant(0xFFFFu),
                                replace_shift));
    });
  }
  element_size_switch.makeBeginCase(2);
  {
    spv::Id eM0_address_dwords = builder_->createBinOp(
        spv::OpShiftRightLogical, type_uint_, eM0_address_bytes, const_uint_2);
    store_needed_eM([&](uint32_t eM_index) {
      StoreUint32ToSharedMemory(
          builder_->createCompositeExtract(eM_packed[eM_index], type_uint_, 0),
          builder_->createUnaryOp(
              spv::OpBitcast, type_int_,
              element_address(eM0_address_dwords, eM_index, 1)));
    });
  }
  element_size_switch.makeBeginCase(3);
  {
    spv::Id eM0_address_dwords = builder_->createBinOp(
        spv::OpShiftRightLogical, type_uint_, eM0_address_bytes, const_uint_2);
    store_needed_eM([&](uint32_t eM_index) {
      spv::Id element_value = eM_packed[eM_index];
      spv::Id element_address_dwords_int = builder_->createUnaryOp(
          spv::OpBitcast, type_int_,
          element_address(eM0_address_dwords, eM_index, 2));
      StoreUint32ToSharedMemory(
          builder_->createCompositeExtract(element_value, type_uint_, 0),
          element_address_dwords_int);
      StoreUint32ToSharedMemory(
          builder_->createCompositeExtract(element_value, type_uint_, 1),
          builder_->createBinOp(spv::OpIAdd, type_int_,
                                element_address_dwords_int,
                                builder_->makeIntConstant(1)));
    });
  }
  element_size_switch.makeBeginCase(4);
  {
    spv::Id eM0_address_dwords = builder_->createBinOp(
        spv::OpShiftRightLogical, type_uint_, eM0_address_bytes, const_uint_2);
    store_needed_eM([&](uint32_t eM_index) {
      spv::Id element_value = eM_packed[eM_index];
      spv::Id element_address_dwords_int = builder_->createUnaryOp(
          spv::OpBitcast, type_int_,
          element_address(eM0_address_dwords, eM_index, 4));
      StoreUint32ToSharedMemory(
          builder_->createCompositeExtract(element_value, type_uint_, 0),
          element_address_dwords_int);
      for (uint32_t element_dword_index = 1; element_dword_index < 4;
           ++element_dword_index) {
        StoreUint32ToSharedMemory(
            builder_->createCompositeExtract(element_value, type_uint_,
                                             element_dword_index),
            builder_->createBinOp(spv::OpIAdd, type_int_,
                                  element_address_dwords_int,
                                  builder_->makeIntConstant(
                                      static_cast<int>(element_dword_index))));
      }
    });
  }
  element_size_switch.makeEndSwitch();

  // Close the conditionals for whether memory export is allowed in this
  // invocation.
  if_address_valid.makeEndIf();
  if (if_pixel_not_killed.has_value()) {
    if_pixel_not_killed->makeEndIf();
  }
  if (if_memexport_allowed.has_value()) {
    if_memexport_allowed->makeEndIf();
  }
}

}  // namespace gpu
}  // namespace xe
