/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#pragma once

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace xe::net {
struct LanInterface {
  std::string name;
  uint32_t address = 0;
  uint32_t broadcast = 0;
  uint32_t index = 0;
};
// Pure selection also permits socket-free tests. A nonempty selector must
// match a local interface name or its IPv4 address; it never falls back.
std::optional<LanInterface> SelectLanInterface(
    std::span<const LanInterface> interfaces, std::string_view selector,
    std::string_view preferred = {});
// Active broadcast-capable adapters, in deterministic interface-index order.
std::vector<LanInterface> EnumerateLanInterfaces();
std::optional<LanInterface> ResolveLanInterface(std::string_view selector);
}  // namespace xe::net
