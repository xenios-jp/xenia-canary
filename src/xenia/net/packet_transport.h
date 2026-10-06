/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#pragma once
#include <string>
#include <string_view>
#include "xenia/net/xnet_secure.h"

namespace xe::net::secure {
class PacketTransport {
 public:
  virtual ~PacketTransport() = default;
  virtual Mac identity() const = 0;
  virtual uint32_t mtu() const = 0;
  virtual bool healthy() const { return true; }
  virtual int Send(std::span<const uint8_t> frame) = 0;
  virtual bool Receive(std::vector<uint8_t>& frame, uint32_t timeout_ms) = 0;
};
// Requires packet-capture access granted by the administrator. Opening this
// transport never changes permissions or elevates the emulator.
std::unique_ptr<PacketTransport> OpenPacketInterface(std::string_view interface,
                                                     std::string& error);
}  // namespace xe::net::secure
