/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/net/lan_interface.h"
#include "third_party/catch/single_include/catch2/catch.hpp"

using namespace xe::net;

TEST_CASE("LAN selection respects explicit choices and valid primary adapters",
          "[net]") {
  const std::vector<LanInterface> adapters{
      {"inactive", 0, 0, 1},
      {"wired", 0xc0000202, 0xc00002ff, 2},
      {"wireless", 0xc6336402, 0xc63364ff, 3}};
  REQUIRE(SelectLanInterface(adapters, "", "wireless")->name == "wireless");
  REQUIRE(SelectLanInterface(adapters, "wired", "wireless")->name == "wired");
  REQUIRE(SelectLanInterface(adapters, "192.0.2.2", "wireless")->name ==
          "wired");
  REQUIRE_FALSE(SelectLanInterface(adapters, "missing", "wireless"));
  REQUIRE_FALSE(SelectLanInterface(adapters, "inactive", "wireless"));
  REQUIRE(SelectLanInterface(adapters, "", "tunnel")->name == "wired");
  REQUIRE(SelectLanInterface(adapters, "", "inactive")->name == "wired");
  REQUIRE_FALSE(SelectLanInterface({}, "", "wireless"));
}

TEST_CASE(
    "Automatic LAN selection prefers configured networks to self-assigned "
    "links",
    "[net]") {
  const std::vector<LanInterface> adapters{
      {"direct", 0xa9fe0202, 0xa9feffff, 1},
      {"configured", 0xc0000202, 0xc00002ff, 2}};
  REQUIRE(SelectLanInterface(adapters, "", "direct")->name == "configured");
  REQUIRE(SelectLanInterface(adapters, "", "missing")->name == "configured");
  REQUIRE(SelectLanInterface(adapters, "direct", "configured")->name ==
          "direct");
  REQUIRE(
      SelectLanInterface(std::span(adapters).first(1), "", "direct")->name ==
      "direct");
}
