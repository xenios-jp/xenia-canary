/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/net/native_lan.h"
#include <poll.h>
#include <atomic>
#include <condition_variable>
#include <future>
#include "third_party/catch/single_include/catch2/catch.hpp"

using namespace xe::net;
using namespace xe::net::secure;
namespace {
struct Wire {
  std::mutex mutex;
  std::condition_variable changed;
  std::deque<std::vector<uint8_t>> frames[2];
};
class TestTransport : public PacketTransport {
 public:
  TestTransport(std::shared_ptr<Wire> wire, unsigned side)
      : wire_(std::move(wire)), side_(side) {}
  Mac identity() const override {
    return {2, 17, 34, 51, 68, uint8_t(side_ + 1)};
  }
  uint32_t mtu() const override { return 1500; }
  bool healthy() const override { return healthy_; }
  void Disconnect() {
    healthy_ = false;
    wire_->changed.notify_all();
  }
  int Send(std::span<const uint8_t> bytes) override {
    if (!AllowedPacketFrame(bytes, identity(), mtu(), true)) {
      return -1;
    }
    std::lock_guard lock(wire_->mutex);
    wire_->frames[1 - side_].emplace_back(bytes.begin(), bytes.end());
    wire_->changed.notify_all();
    return static_cast<int>(bytes.size());
  }
  bool Receive(std::vector<uint8_t>& bytes, uint32_t timeout_ms) override {
    std::unique_lock lock(wire_->mutex);
    wire_->changed.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
      return !healthy_ || !wire_->frames[side_].empty();
    });
    if (!healthy_) {
      return false;
    }
    if (wire_->frames[side_].empty()) {
      return false;
    }
    bytes = std::move(wire_->frames[side_].front());
    wire_->frames[side_].pop_front();
    return true;
  }

 private:
  std::shared_ptr<Wire> wire_;
  unsigned side_;
  std::atomic<bool> healthy_{true};
};
struct NativePair {
  std::shared_ptr<Wire> wire = std::make_shared<Wire>();
  KeyId id{1, 3, 5, 7, 9, 11, 13, 15};
  Key key{2, 4, 6, 8, 10, 12, 14, 16};
  std::shared_ptr<TestTransport> transports[2];
  std::unique_ptr<Lan> a, b;
  NativePair() {
    auto broadcast = DeriveBroadcast(key, [&](auto input, auto& digest) {
      digest = xe::net::crypto::HmacSha1(key, {input});
      return true;
    });
    REQUIRE(broadcast);
    a = Make(0, *broadcast);
    b = Make(1, *broadcast);
    REQUIRE(a->RegisterKey(id, key) == 0);
    REQUIRE(b->RegisterKey(id, key) == 0);
  }
  std::unique_ptr<Lan> Make(unsigned side, ProtectionKeys broadcast) {
    auto transport = std::make_shared<TestTransport>(wire, side);
    transports[side] = transport;
    LanConfig config;
    config.address =
        0xC0000201 + side;  // Documentation-only synthetic endpoints.
    config.native_secure = true;
    config.native_mac = transport->identity();
    config.native_factory = [broadcast, transport](Lan& registry) {
      AssociationConfig association;
      association.broadcast = broadcast;
      association.mtu = transport->mtu();
      memcpy(association.identity.data(), &registry.identity(),
             association.identity.size());
      association.credential =
          [&registry](const KeyId& id) -> std::optional<Key> {
        Key key;
        return registry.GetKey(id, key) ? std::optional(key) : std::nullopt;
      };
      return std::make_unique<NativeLan>(registry, transport,
                                         std::move(association));
    };
    return std::make_unique<Lan>(std::move(config));
  }
};
bool Ready(NativeDatagramTransport& port, int deadline_ms = 5000) {
  pollfd descriptor{port.ReadHandle(), POLLIN, 0};
  return poll(&descriptor, 1, deadline_ms) == 1 &&
         (descriptor.revents & POLLIN);
}
}  // namespace
TEST_CASE(
    "Native secure transport uses stable NIC identity and guest virtual port "
    "readiness",
    "[net][secure][native]") {
  for (bool reverse : {false, true}) {
    NativePair pair;
    auto& sender = reverse ? *pair.b : *pair.a;
    auto& receiver = reverse ? *pair.a : *pair.b;
    auto tx = sender.native()->Bind(1001, false);
    auto rx = receiver.native()->Bind(1001, false);
    REQUIRE(tx);
    REQUIRE(rx);
    CHECK_FALSE(receiver.native()->Bind(1001, false));
    CHECK_FALSE(Ready(*rx, 0));
    const std::vector<uint8_t> data{0, 1, 2, 3, 4, 5, 6};
    const int data_size = static_cast<int>(data.size());
    REQUIRE(tx->Send({0xFFFFFFFF, 1001}, data) == data_size);
    REQUIRE(Ready(*rx));
    CHECK(rx->PendingBytes() == data.size());
    std::array<uint8_t, 32> buffer{};
    Endpoint source;
    REQUIRE(rx->ReceiveGuest(buffer, 2, source, true) == data_size);
    CHECK(source == Endpoint{1, 1001});
    CHECK(Ready(*rx, 0));
    CHECK(rx->ReceiveGuest(std::span(buffer).first(2), 2, source, true) == -1);
    CHECK(errno == EMSGSIZE);
    CHECK(rx->PendingBytes() == data.size());
    REQUIRE(rx->ReceiveGuest(buffer, 0, source, true) == data_size);
    CHECK_FALSE(Ready(*rx, 0));
    CHECK(rx->ReceiveGuest(buffer, 0, source, true) == -1);
    CHECK(errno == EWOULDBLOCK);
    uint32_t address;
    REQUIRE(sender.Map(receiver.identity(), pair.id, address) == 0);
    REQUIRE(sender.Connect(address) == 0);
    REQUIRE(tx->Send({address, 1001}, data) == data_size);
    REQUIRE(Ready(*rx));
    REQUIRE(rx->ReceiveGuest(buffer, 0, source, true) == data_size);
    CHECK(std::equal(data.begin(), data.end(), buffer.begin()));
    CHECK(sender.ConnectStatus(address) == 2);
    REQUIRE(receiver.peers().size() == 1);
    CHECK(receiver.peers()[0].connected);
    XnAddr peer;
    KeyId id;
    REQUIRE(receiver.Reverse(source.address, peer, id) == 0);
    CHECK(memcmp(&peer, &sender.identity(), sizeof(peer)) == 0);
    CHECK(id == pair.id);
    REQUIRE(rx->Send({source.address, 1001}, data) == data_size);
    REQUIRE(Ready(*tx));
    CHECK(tx->PendingBytes() == data.size());
    // Local retirement synchronously drops the old generation's socket queue.
    REQUIRE(sender.Unmap(address) == 0);
    CHECK(tx->PendingBytes() == 0);
    CHECK_FALSE(Ready(*tx, 0));
    CHECK(sender.ConnectStatus(address) == 0);
    tx->Close();
    rx->Close();
  }
}
TEST_CASE("Native virtual ports preserve the VDP prefix and unblock close",
          "[net][secure][native]") {
  NativePair pair;
  auto tx = pair.a->native()->Bind(1000, true);
  auto rx = pair.b->native()->Bind(1000, true);
  REQUIRE(tx);
  REQUIRE(rx);
  uint32_t address;
  REQUIRE(pair.a->Map(pair.b->identity(), pair.id, address) == 0);
  std::vector<uint8_t> vdp{0, 3, 10, 20, 30, 40, 50};
  REQUIRE(tx->Send({address, 1000}, vdp) == static_cast<int>(vdp.size()));
  REQUIRE(Ready(*rx));
  Datagram packet;
  REQUIRE(rx->Receive(packet, 0));
  CHECK(packet.data == vdp);
  auto auto_bound = pair.a->native()->Bind(0, false);
  REQUIRE(auto_bound);
  CHECK(auto_bound->port() >= 49152);
  auto blocked = std::async(std::launch::async, [&] {
    uint8_t byte;
    Endpoint source;
    return rx->ReceiveGuest(std::span(&byte, 1), 0, source, false);
  });
  rx->Close();
  REQUIRE(blocked.wait_for(std::chrono::seconds(5)) ==
          std::future_status::ready);
  CHECK(blocked.get() == -1);
}
TEST_CASE(
    "Native mode without a packet transport cannot publish a synthetic "
    "identity",
    "[net][native]") {
  LanConfig config;
  config.native_secure = true;
  Lan lan(std::move(config));
  CHECK(lan.identity().address == 0);
  CHECK(lan.native() == nullptr);
  CHECK(lan.Connect(1) != 0);
}

TEST_CASE("Native registry accepts the reviewed three-bit LAN key category",
          "[net][secure][native]") {
  NativePair pair;
  auto id = pair.id;
  id[0] = 0x1F;
  REQUIRE(pair.a->RegisterKey(id, pair.key) == 0);
  REQUIRE(pair.a->AcquireSessionKey(id, pair.key) == 0);
  Key key;
  CHECK(pair.a->GetKey(id, key));
  CHECK(key == pair.key);
  REQUIRE(pair.a->UnregisterKey(id) == 0);
  CHECK(pair.a->GetKey(id, key));
  pair.a->ReleaseSessionKey(id);
  CHECK_FALSE(pair.a->GetKey(id, key));
  id[0] = 0x20;
  CHECK(pair.a->RegisterKey(id, pair.key) != 0);
  CHECK(pair.a->AcquireSessionKey(id, pair.key) != 0);
}

TEST_CASE(
    "Broker loss retires mappings and wakes native receives without discarding "
    "credentials",
    "[net][secure][native]") {
  NativePair pair;
  auto tx = pair.a->native()->Bind(1001, false);
  auto rx = pair.b->native()->Bind(1001, false);
  REQUIRE(tx);
  REQUIRE(rx);
  uint32_t address;
  REQUIRE(pair.a->Map(pair.b->identity(), pair.id, address) == 0);
  const std::array<uint8_t, 3> data{1, 3, 5};
  REQUIRE(tx->Send({address, 1001}, data) == 3);
  REQUIRE(Ready(*rx));
  Datagram packet;
  REQUIRE(rx->Receive(packet, 0));
  auto blocked = std::async(std::launch::async, [&] {
    uint8_t byte;
    Endpoint source;
    return tx->ReceiveGuest(std::span(&byte, 1), 0, source, false);
  });
  pair.transports[0]->Disconnect();
  REQUIRE(blocked.wait_for(std::chrono::seconds(5)) ==
          std::future_status::ready);
  CHECK(blocked.get() == -1);
  CHECK(pair.a->peers().empty());
  CHECK(pair.a->ConnectStatus(address) == 0);
  CHECK(tx->Send({address, 1001}, data) == -1);
  Key key;
  CHECK(pair.a->GetKey(pair.id, key));
  CHECK(key == pair.key);
}
