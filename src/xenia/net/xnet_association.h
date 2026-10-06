/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#pragma once
#include <deque>
#include "xenia/net/xnet_secure.h"

namespace xe::net::secure {
struct AssociationConfig {
  Address identity{};
  ProtectionKeys broadcast;
  Metadata metadata{};
  std::array<uint8_t, 4>
      teardown_reason{};  // Opaque local policy; no enum inferred.
  std::function<std::optional<Key>(const KeyId&)> credential;
  std::function<uint64_t()> freshness_time;
  uint32_t mtu = 1500;
  size_t peer_limit = 32;
  // Provisional local scheduling policy, not recovered native constants.
  uint64_t retry_ms = 1000, idle_ms = 60000, keepalive_ms = 10000;
  unsigned retries = 5;
  // Provisional counter boundary; opt-in permits authenticated zero and starts
  // at zero.
  bool sequence_zero = false;
};
enum class AssociationStatus { Missing, Pending, Established };
struct AssociationEvent {
  enum class Type { Datagram, Connected, Retired } type = Type::Datagram;
  Address peer{};
  KeyId id{};
  Mac source{};
  Packet packet;
};
// The owning transport serializes calls. Queues contain values, never guest
// pointers. Authentication and target binding precede every state transition.
class Associations {
 public:
  explicit Associations(AssociationConfig config);
  ~Associations();
  bool Connect(const Address& peer, const KeyId& id, uint64_t now_ms);
  bool Send(const Address& peer, const KeyId& id, Packet packet,
            uint64_t now_ms);
  bool Discover(Packet packet);
  void Receive(const Frame& frame, uint64_t now_ms);
  void Tick(uint64_t now_ms);
  void Unregister(const KeyId& id, uint64_t now_ms);
  void Close(const Address& peer, const KeyId& id, uint64_t now_ms);
  void Reset();
  AssociationStatus Status(const Address& peer, const KeyId& id) const;
  std::vector<Frame> TakeOutbound();
  std::vector<AssociationEvent> TakeEvents();

 private:
  struct State;
  using PeerKey = std::pair<Address, KeyId>;
  AssociationConfig config_;
  Mac local_{};
  std::map<PeerKey, std::unique_ptr<State>> states_;
  std::vector<Frame> outbound_;
  std::vector<AssociationEvent> events_;
  uint16_t discovery_sequence_ = 0;
  uint64_t freshness_ = 0;
  uint32_t NewSpi();
  uint64_t NewFreshness();
  bool Emit(Mac destination, std::vector<uint8_t> payload,
            bool broadcast = false);
  bool ExchangePacket(State& state, uint64_t now_ms);
  bool Protected(State& state, const Packet& packet, uint64_t now_ms);
  void Established(State& state, uint64_t now_ms);
  void Retired(PeerKey key);
};
}  // namespace xe::net::secure
