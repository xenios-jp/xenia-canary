/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#pragma once

#include <array>
#include <cstdint>
#include <initializer_list>
#include <span>

namespace xe::net::crypto {
using Sha1Digest = std::array<uint8_t, 20>;
Sha1Digest HmacSha1(std::span<const uint8_t> key,
                    std::initializer_list<std::span<const uint8_t>> parts);
}  // namespace xe::net::crypto
