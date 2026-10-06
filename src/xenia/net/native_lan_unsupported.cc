/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/net/native_lan.h"
namespace xe::net::secure {
NativeLan::NativeLan(Lan&, std::shared_ptr<PacketTransport>,
                     AssociationConfig) {}
NativeLan::~NativeLan() = default;
std::unique_ptr<NativeDatagramTransport> NativeLan::Bind(uint16_t, bool) {
  return {};
}
uint32_t NativeLan::Connect(uint32_t) { return 10050; }
uint32_t NativeLan::Status(uint32_t) { return 0; }
void NativeLan::Close(uint32_t) {}
void NativeLan::Unregister(const KeyId&) {}
void NativeLan::Reset() {}
}  // namespace xe::net::secure
