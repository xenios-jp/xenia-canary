/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#pragma once
#include "xenia/net/packet_transport.h"
#include "xenia/net/xnet_association.h"

namespace xe::net::secure {
class NativeDatagramTransport : public Transport {
 public:
  virtual uint16_t port() const = 0;
  virtual void Close() = 0;
  virtual int ReadHandle() const = 0;
  virtual size_t PendingBytes() const = 0;
  virtual int ReceiveGuest(std::span<uint8_t> buffer, uint32_t flags,
                           Endpoint& source, bool nonblocking) = 0;
};
class NativeLan {
 public:
  NativeLan(Lan& registry, std::shared_ptr<PacketTransport> transport,
            AssociationConfig config);
  ~NativeLan();
  std::unique_ptr<NativeDatagramTransport> Bind(uint16_t port, bool vdp);
  uint32_t Connect(uint32_t secure_address);
  uint32_t Status(uint32_t secure_address);
  void Close(uint32_t secure_address);
  void Unregister(const KeyId& id);
  void Reset();

 private:
  struct Core;
  std::shared_ptr<Core> core_;
};
}  // namespace xe::net::secure
