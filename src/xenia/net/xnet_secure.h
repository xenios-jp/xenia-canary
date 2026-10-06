/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#pragma once

#include <optional>
#include "xenia/net/lan.h"
#include "xenia/net/sha1.h"

namespace xe::net::secure {
using Block = std::array<uint8_t, 8>;
using Address = std::array<uint8_t, 36>;
using DhValue = std::array<uint8_t, 96>;
using Metadata = std::array<uint8_t, 12>;
using Mac = std::array<uint8_t, 6>;
using PlatformHmac =
    std::function<bool(std::span<const uint8_t>, crypto::Sha1Digest&)>;

// Fixed-size RFC 2409 group-one key agreement. Fresh 256-bit private randomness
// per association; masked exponentiation and secret erasure on
// move/destruction.
class DhKeyPair {
 public:
  static std::optional<DhKeyPair> Generate();
  static std::optional<DhKeyPair> FromPrivate(
      std::span<const uint8_t> big_endian_exponent);
  ~DhKeyPair();
  DhKeyPair(DhKeyPair&& other) noexcept;
  DhKeyPair& operator=(DhKeyPair&& other) noexcept;
  const DhValue& public_value() const { return public_; }
  std::optional<DhValue> Shared(const DhValue& peer) const;

 private:
  explicit DhKeyPair(std::span<const uint8_t> exponent);
  void Erase();
  std::array<uint8_t, 32> exponent_{};
  DhValue public_{};
  bool active_ = true;
};

struct ProtectionKeys {
  Key authentication{};
  std::array<uint8_t, 24> cipher{};
  Block iv_seed{};
};
struct RegistrationKeys {
  Key authentication{};
  std::array<uint8_t, 24> cipher{};
};
struct TrafficKeys {
  ProtectionKeys a, b;
};
std::optional<ProtectionKeys> DeriveBroadcast(
    const Key& executable_key, const PlatformHmac& platform_hmac);
RegistrationKeys DeriveRegistration(const Key& broadcast_auth, const KeyId& id,
                                    const Key& key);
TrafficKeys DeriveTraffic(const Key& registration_auth, const DhValue& shared,
                          const Block& initiator, const Block& responder);
Block PacketIv(const Block& seed, uint32_t sequence);
Metadata MakeMetadata(uint32_t executable_id, uint32_t version, uint16_t major,
                      uint16_t minor, uint16_t build, uint8_t revision);
// Unauthenticated diagnostics only; never an admission or key-install decision.
uint32_t FailedAuthCompatibility(const Metadata& local, const Metadata& remote);
bool FreshnessNewer(uint64_t incoming, uint64_t previous, bool response);

struct Frame {
  Mac source{}, destination{};
  bool broadcast = false;
  std::vector<uint8_t> payload;
};
std::optional<std::vector<uint8_t>> EncodeFrame(const Frame& frame);
std::optional<Frame> DecodeFrame(std::span<const uint8_t> wire);
// Packet transport validation: selected NIC identity, native wire envelope,
// supported secure record geometry and physical-interface MTU only.
bool AllowedPacketFrame(std::span<const uint8_t> wire, const Mac& local,
                        uint32_t mtu, bool outgoing);

enum class Kind { Control, Udp, Vdp, Discovery };
struct Packet {
  Kind kind = Kind::Udp;
  uint16_t source_port = 1000, destination_port = 1000;
  uint16_t control_type = 0;
  std::vector<uint8_t> encrypted, clear;
  Metadata metadata{};
};
// No guest delivery on failure. Discovery metadata is supplied by a provider,
// never generated from application bytes. Sequence authentication precedes CBC.
bool Encode(const ProtectionKeys& keys, uint32_t spi, uint32_t sequence,
            const Packet& packet, std::vector<uint8_t>& wire);
std::optional<Packet> Decode(const ProtectionKeys& keys, uint32_t spi,
                             uint32_t sequence, std::span<const uint8_t> wire,
                             bool discovery = false);
std::optional<Packet> FromGuestVdp(uint16_t source, uint16_t destination,
                                   std::span<const uint8_t> guest);
std::vector<uint8_t> ToGuestVdp(const Packet& packet);

struct Exchange {
  bool response = false;
  uint16_t reserved = 0, hops = 127;
  KeyId id{};
  uint32_t initiator_spi = 0, responder_spi = 0;
  Block initiator_challenge{}, responder_challenge{};
  uint64_t freshness = 0;
  Block address_iv{};
  Address initiator_address{}, responder_address{};
  DhValue public_value{};
};
std::optional<std::array<uint8_t, 256>> EncodeExchange(
    const RegistrationKeys& keys, const Exchange& exchange);
// Authentication and record syntax only. Caller must validate DH and bind the
// authenticated addresses, challenges, identifier and SPIs to its pending
// state.
std::optional<Exchange> DecodeExchange(const RegistrationKeys& keys,
                                       const KeyId& id,
                                       std::span<const uint8_t> wire);
bool ResponseMatches(const Exchange& request, const Exchange& response);

// One instance per receive direction/generation. Candidates never mutate state;
// Commit follows authentication and validated decoding. Includes previous epoch
// candidates at wrap, rather than advancing an epoch on arrival order alone.
class ReplayWindow {
 public:
  explicit ReplayWindow(bool allow_zero = false) : allow_zero_(allow_zero) {}
  std::vector<uint32_t> Candidates(uint16_t low) const;
  bool Commit(uint32_t authenticated_sequence);
  uint32_t highest() const { return highest_; }

 private:
  bool Permitted(uint32_t sequence) const;
  uint32_t highest_ = 0, seen_ = 0;
  bool allow_zero_ = false;
};
}  // namespace xe::net::secure
