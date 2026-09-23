/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/cpu/testing/util.h"

#include <cstdint>
#include <cstring>

#include "xenia/cpu/compiler/passes/simplification_pass.h"

using namespace xe::cpu::hir;
using namespace xe::cpu;
using namespace xe::cpu::testing;
using xe::cpu::ppc::PPCContext;

namespace {

constexpr uint64_t kOne = 0x3FF0000000000000ull;
constexpr uint64_t kZero = 0x0000000000000000ull;
constexpr uint64_t kDenormal = 0x0000000000000001ull;
constexpr uint64_t kNegDenormal = 0x800FFFFFFFFFFFFFull;
constexpr uint64_t kMinNormal = 0x0010000000000000ull;
constexpr uint64_t kInf = 0x7FF0000000000000ull;
constexpr uint64_t kNaN = 0x7FF8000000000000ull;

void SetF64Bits(PPCContext* ctx, int reg, uint64_t bits) {
  std::memcpy(&ctx->f[reg], &bits, sizeof(bits));
}

Value* F64Constant(HIRBuilder& b, uint64_t bits) {
  double d;
  std::memcpy(&d, &bits, sizeof(d));
  return b.LoadConstantFloat64(d);
}

bool HasDenormalQuirk(HIRBuilder& b) {
  for (Block* block = b.first_block(); block; block = block->next) {
    for (Instr* i = block->instr_head; i; i = i->next) {
      if (i->GetOpcodeNum() == OPCODE_DENORMAL_QUIRK) {
        return true;
      }
    }
  }
  return false;
}

bool SimplifiesAway(void (*emit)(HIRBuilder& b)) {
  HIRBuilder b;
  b.MakeCurrent();
  emit(b);
  compiler::passes::SimplificationPass pass;
  bool changed = false;
  pass.Run(&b, changed);
  const bool folded = !HasDenormalQuirk(b);
  b.RemoveCurrent();
  return folded;
}

}  // namespace

// quirk = any operand a double denormal && every operand finite.
TEST_CASE("DENORMAL_QUIRK", "[instr]") {
  TestFunction test([](HIRBuilder& b) {
    StoreGPR(b, 3,
             b.ZeroExtend(
                 b.DenormalQuirk(LoadFPR(b, 1), LoadFPR(b, 2), LoadFPR(b, 3)),
                 INT64_TYPE));
    b.Return();
  });
  struct Case {
    uint64_t a, b, c;
    uint64_t quirk;
  };
  const Case cases[] = {
      {kOne, kOne, kOne, 0},
      {kZero, kZero, kZero, 0},
      {kMinNormal, kOne, kZero, 0},
      {kDenormal, kOne, kOne, 1},
      {kOne, kNegDenormal, kOne, 1},
      {kOne, kOne, kDenormal, 1},
      {kDenormal, kDenormal, kDenormal, 1},
      {kDenormal, kInf, kOne, 0},
      {kOne, kDenormal, kNaN, 0},
      {kInf, kInf, kInf, 0},
  };
  for (const Case& c : cases) {
    INFO("operands 0x" << std::hex << c.a << " 0x" << c.b << " 0x" << c.c);
    test.Run(
        [&](PPCContext* ctx) {
          SetF64Bits(ctx, 1, c.a);
          SetF64Bits(ctx, 2, c.b);
          SetF64Bits(ctx, 3, c.c);
        },
        [&](PPCContext* ctx) { REQUIRE(ctx->r[3] == c.quirk); });
  }
}

// Constant operands take the backends' materialisation path.
TEST_CASE("DENORMAL_QUIRK_CONSTANT_OPERAND", "[instr]") {
  TestFunction test([](HIRBuilder& b) {
    StoreGPR(
        b, 3,
        b.ZeroExtend(b.DenormalQuirk(LoadFPR(b, 1), F64Constant(b, kDenormal),
                                     LoadFPR(b, 1)),
                     INT64_TYPE));
    b.Return();
  });
  test.Run([](PPCContext* ctx) { SetF64Bits(ctx, 1, kOne); },
           [](PPCContext* ctx) { REQUIRE(ctx->r[3] == 1); });
  test.Run([](PPCContext* ctx) { SetF64Bits(ctx, 1, kInf); },
           [](PPCContext* ctx) { REQUIRE(ctx->r[3] == 0); });
}

TEST_CASE("DENORMAL_QUIRK_FOLD", "[instr]") {
  // Widened singles and normal constants can never be double denormals.
  REQUIRE(SimplifiesAway([](HIRBuilder& b) {
    Value* single = b.UnpackSingle(b.Truncate(LoadGPR(b, 1), INT32_TYPE));
    StoreGPR(b, 3,
             b.ZeroExtend(
                 b.DenormalQuirk(single, b.Neg(single), F64Constant(b, kOne)),
                 INT64_TYPE));
    b.Return();
  }));
  REQUIRE(SimplifiesAway([](HIRBuilder& b) {
    Value* single =
        b.Convert(b.Convert(LoadFPR(b, 1), FLOAT32_TYPE), FLOAT64_TYPE);
    StoreGPR(b, 3,
             b.ZeroExtend(b.DenormalQuirk(single, single, single), INT64_TYPE));
    b.Return();
  }));
  // A guest FPR or a denormal constant may be one.
  REQUIRE_FALSE(SimplifiesAway([](HIRBuilder& b) {
    StoreGPR(b, 3,
             b.ZeroExtend(b.DenormalQuirk(LoadFPR(b, 1), F64Constant(b, kOne),
                                          F64Constant(b, kOne)),
                          INT64_TYPE));
    b.Return();
  }));
  REQUIRE_FALSE(SimplifiesAway([](HIRBuilder& b) {
    Value* single = b.UnpackSingle(b.Truncate(LoadGPR(b, 1), INT32_TYPE));
    StoreGPR(
        b, 3,
        b.ZeroExtend(b.DenormalQuirk(single, F64Constant(b, kDenormal), single),
                     INT64_TYPE));
    b.Return();
  }));
}
