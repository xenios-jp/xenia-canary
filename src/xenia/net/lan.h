/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2024 Xenia Emulator. All rights reserved.                        *
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#pragma once

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include "xenia/base/byte_order.h"

namespace xe::net {
// Guest address ABI and mapping conventions follow Xenia Netplay. See the
// source revision and paths in README.md; native protection is separate.
namespace secure {
class NativeLan;
}

using KeyId = std::array<uint8_t, 8>;
using Key = std::array<uint8_t, 16>;

inline constexpr size_t kMaxDatagramSize = 65507;
inline constexpr size_t kMaxSidecarDatagramSize = 8192;
inline constexpr size_t kSecureDatagramHeaderSize = 52;
inline constexpr size_t kMaxSecureDatagramPayload =
    kMaxDatagramSize - kSecureDatagramHeaderSize;

// SGADDR.platform_type within abOnline (Netplay src/xenia/kernel/xnet.h).
inline constexpr size_t kPlatformOffset = 16;
inline constexpr uint8_t kXbox360Platform = 1;
inline constexpr uint16_t kDefaultOnlinePort = 36000;

// Guest ABI: IPv4 numerics and all scalar fields are big endian.
struct XnAddr {
  be<uint32_t> address;
  be<uint32_t> online_address;
  be<uint16_t> online_port;
  uint8_t mac[6];
  uint8_t online[20];
};
static_assert(sizeof(XnAddr) == 36);
struct SessionInfo {
  KeyId id;
  XnAddr host;
  Key key;
};
static_assert(sizeof(SessionInfo) == 60);

// Endpoint numerics are host order. Guest ports retain their Xbox meaning;
// only the transport boundary adds a per-instance offset.
struct Endpoint {
  uint32_t address = 0;
  uint16_t port = 0;
  bool operator==(const Endpoint&) const = default;
};
struct Datagram {
  Endpoint source;
  std::vector<uint8_t> data;
};

// Small service boundary for LAN/session directories and future relays.
// Implementations own their resources. Send returns bytes or -1; Receive
// returns false on timeout/error. No guest pointers may cross this boundary.
class Transport {
 public:
  virtual ~Transport() = default;
  virtual int Send(Endpoint destination, std::span<const uint8_t> data) = 0;
  virtual bool Receive(Datagram& datagram, uint32_t timeout_ms) = 0;
};

struct Peer {
  XnAddr address{};
  KeyId key_id{};
  uint32_t secure_address = 0;
  bool connected = false;
};
class Lan;
struct LanConfig {
  bool native_secure = false;
  std::array<uint8_t, 6> native_mac{};
  std::function<std::unique_ptr<secure::NativeLan>(Lan&)> native_factory{};
  uint32_t address = 0x7F000001;
  uint32_t machine_id = 0;  // 0: random for this instance's lifetime.
  uint16_t port_base = 0;
  uint16_t loopback_stride = 0;
  // Real interfaces use a wildcard receive bind and a directed broadcast.
  // Every Mac in this mode must use the same port_base.
  bool real_interface = false;
  uint32_t interface_index = 0;
  uint32_t broadcast_address = 0xFFFFFFFF;
  std::vector<uint32_t> broadcast_targets;
  // Public Netplay transport uses raw IPv4 handles and guest datagrams. Port
  // translations come from the service or an explicit per-process override.
  bool online = false;
  uint16_t online_port = 0;
  std::map<uint16_t, uint16_t> bind_ports;
  std::map<uint16_t, uint16_t> connect_ports;
  // Optional native hosts for virtual peers. Ports still use loopback_stride;
  // an empty map preserves the single-host loopback transport.
  std::map<uint32_t, uint32_t> virtual_peer_hosts;
  // Optional transport injection for embedded hosts and deterministic tests.
  // Empty uses the native LAN transport.
  std::function<std::unique_ptr<Transport>(Lan&, uint16_t)> transport_factory;
};

// Independent state per console, including in-process consoles in tests.
// Methods are thread safe; snapshots own their storage. Secure addresses are
// opaque, session-specific IPv4 handles, resolved before native socket calls.
class Lan {
 public:
  explicit Lan(LanConfig config);
  ~Lan();
  const XnAddr& identity() const { return identity_; }
  bool native_secure() const { return config_.native_secure; }
  secure::NativeLan* native() const { return native_.get(); }
  // Removes live and retired routes for one authenticated native generation.
  std::vector<uint32_t> RetireNative(const XnAddr& peer, const KeyId& id);
  void MarkNativeConnected(uint32_t secure);
  uint32_t interface_index() const { return config_.interface_index; }
  bool IsBroadcast(uint32_t address) const;
  uint64_t machine_id() const { return MachineId(identity_); }
  static uint64_t MachineId(const XnAddr& address);
  static void Random(std::span<uint8_t> bytes);
  static void CreateKey(KeyId& id, Key& key);
  uint32_t RegisterKey(const KeyId& id, const Key& key);
  // Retires identity mappings, retaining bounded outgoing routes for guest
  // datagrams pending at teardown. Unmap or Reset also removes those routes.
  uint32_t UnregisterKey(const KeyId& id);
  bool GetKey(const KeyId& id, Key& key) const;
  // Session ownership is separate from explicit guest registration. A guest
  // may register the key returned by XSessionCreate without a duplicate error.
  uint32_t AcquireSessionKey(const KeyId& id, const Key& key);
  void ReleaseSessionKey(const KeyId& id);
  uint32_t ReplaceKey(const KeyId& old_id, const KeyId& new_id, const Key& key);
  uint32_t Map(const XnAddr& address, const KeyId& id, uint32_t& secure);
  uint32_t Reverse(uint32_t secure, XnAddr& address, KeyId& id) const;
  uint32_t Unmap(uint32_t secure);
  uint32_t Connect(uint32_t secure);
  uint32_t ConnectStatus(uint32_t secure) const;
  bool Outgoing(Endpoint& endpoint, bool datagram = true) const;
  // Runtime LAN framing, not the retail console wire protocol. Secure sends
  // carry the exact sender identity and session id; raw sends stay unchanged.
  // Encode returns a Winsock error or zero and resolves the native destination.
  uint32_t EncodeDatagram(Endpoint& destination,
                          std::span<const uint8_t> payload,
                          std::vector<uint8_t>& wire) const;
  // Receives native endpoints, strips framing and establishes inbound mappings.
  // False means discard (malformed, unregistered key, or association failure).
  bool DecodeDatagram(Datagram& datagram);
  bool Bind(Endpoint& endpoint);
  void Incoming(Endpoint& endpoint, bool translate_secure = true,
                bool datagram = true) const;
  std::vector<Endpoint> Broadcast(uint16_t guest_port) const;
  std::vector<Peer> peers() const;
  void Reset();
  void SetLimits(size_t keys, size_t peers);

 private:
  void RetirePeers(const KeyId& id);  // Called with mutex_ held.
  bool IsVirtual(uint32_t address) const;
  bool MapPort(Endpoint& endpoint) const;
  bool ResolveOutgoing(Endpoint& endpoint, KeyId* id,
                       bool datagram = true) const;  // mutex_ held.
  const LanConfig config_;
  XnAddr identity_{};
  mutable std::mutex mutex_;
  struct RegisteredKey {
    Key key;
    bool guest_registered = false;
    size_t sessions = 0;
  };
  std::map<KeyId, RegisteredKey> keys_;
  std::map<uint32_t, Peer> peers_;
  // Oldest first, capped at peer_limit_. Not usable for identity or Connect.
  std::vector<Peer> retired_peers_;
  std::vector<uint16_t> bound_ports_;
  uint32_t next_address_ = 1;
  size_t key_limit_ = 8, peer_limit_ = 32;
  std::unique_ptr<secure::NativeLan> native_;
};

bool IsPublicIpv4(uint32_t address);

}  // namespace xe::net
