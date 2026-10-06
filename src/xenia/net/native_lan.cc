/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/net/native_lan.h"
#include <fcntl.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cerrno>
#include <condition_variable>
#include <thread>
#include "xenia/base/clock.h"
#include "xenia/base/logging.h"

namespace xe::net::secure {
namespace {
uint64_t Now() { return Clock::QueryHostUptimeMillis(); }
Address Serialized(const XnAddr& address) {
  Address bytes;
  memcpy(bytes.data(), &address, bytes.size());
  return bytes;
}
XnAddr GuestAddress(const Address& bytes) {
  XnAddr address;
  memcpy(&address, bytes.data(), bytes.size());
  return address;
}
struct Inbox {
  std::mutex mutex;
  std::condition_variable changed;
  std::deque<Datagram> packets;
  bool closed = false;
  // The pipe signals select readiness; packet bytes stay in the queue.
  int reader = -1, writer = -1;
  Inbox() {
    int handles[2];
    if (pipe(handles)) {
      return;
    }
    reader = handles[0];
    writer = handles[1];
    bool valid = reader < FD_SETSIZE;
    for (int handle : handles) {
      int flags = fcntl(handle, F_GETFL);
      valid &= flags >= 0 && fcntl(handle, F_SETFL, flags | O_NONBLOCK) == 0 &&
               fcntl(handle, F_SETFD, FD_CLOEXEC) == 0;
    }
    if (!valid) {
      close(reader);
      close(writer);
      reader = writer = -1;
    }
  }
  ~Inbox() {
    if (reader >= 0) {
      close(reader);
    }
    if (writer >= 0) {
      close(writer);
    }
  }
  void Signal() {
    uint8_t byte = 1;
    if (writer >= 0) {
      (void)write(writer, &byte, 1);
    }
    changed.notify_all();
  }
  void Push(Datagram packet) {
    std::lock_guard lock(mutex);
    if (closed || packets.size() >= 64) {
      return;
    }
    if (packets.empty()) {
      Signal();
    }
    packets.push_back(std::move(packet));
  }
  void Retire(uint32_t address) {
    std::lock_guard lock(mutex);
    std::erase_if(packets, [&](const Datagram& packet) {
      return packet.source.address == address;
    });
    if (packets.empty() && reader >= 0) {
      uint8_t byte;
      (void)read(reader, &byte, 1);
    }
  }
};
}  // namespace
struct NativeLan::Core {
  Lan& registry;
  std::shared_ptr<PacketTransport> transport;
  std::mutex mutex;
  Associations associations;
  std::map<std::pair<uint16_t, bool>, std::weak_ptr<Inbox>> ports;
  std::jthread worker;
  uint16_t next_port = 49152;
  Core(Lan& lan, std::shared_ptr<PacketTransport> io, AssociationConfig config)
      : registry(lan),
        transport(std::move(io)),
        associations(std::move(config)) {
    worker = std::jthread([this](std::stop_token stop) {
      while (!stop.stop_requested()) {
        std::vector<uint8_t> wire;
        bool received = transport->Receive(wire, 25);
        std::vector<Frame> outbound;
        {
          std::lock_guard lock(mutex);
          if (!transport->healthy()) {
            XELOGE(
                "Native LAN packet transport unavailable; retiring secure "
                "associations");
            for (const auto& peer : registry.peers()) {
              associations.Close(Serialized(peer.address), peer.key_id, Now());
            }
            Deliver();
            associations.Reset();
            for (auto& [port, weak] : ports) {
              if (auto inbox = weak.lock()) {
                std::lock_guard lock(inbox->mutex);
                inbox->closed = true;
                inbox->packets.clear();
                inbox->Signal();
              }
            }
            break;
          }
          if (received) {
            if (auto frame = DecodeFrame(wire)) {
              associations.Receive(*frame, Now());
            }
          }
          // Credential ownership stays in Lan. Removing its last owner retires
          // every corresponding native generation before further delivery.
          associations.Tick(Now());
          Deliver();
          outbound = associations.TakeOutbound();
        }
        for (const auto& frame : outbound) {
          if (auto encoded = EncodeFrame(frame)) {
            const auto sent = transport->Send(*encoded);
            if (sent != static_cast<int>(encoded->size())) {
              XELOGE("Native LAN packet transmission failed");
            }
          }
        }
      }
    });
  }
  ~Core() {
    worker.request_stop();
    if (worker.joinable()) {
      worker.join();
    }
  }
  void Deliver() {  // mutex held; no external guest callbacks.
    for (auto& event : associations.TakeEvents()) {
      uint32_t address = 0;
      if (event.packet.kind != Kind::Discovery) {
        auto peer = GuestAddress(event.peer);
        if (event.type == AssociationEvent::Type::Retired) {
          for (auto address : registry.RetireNative(peer, event.id)) {
            for (auto& [port, weak] : ports) {
              if (auto inbox = weak.lock()) {
                inbox->Retire(address);
              }
            }
          }
          continue;
        }
        if (registry.Map(peer, event.id, address)) {
          continue;
        }
        if (event.type == AssociationEvent::Type::Connected) {
          registry.MarkNativeConnected(address);
          continue;
        }
      } else {
        // Discovery's native source IP is virtual. Do not map application
        // membership payloads or fabricate a remote address record.
        address = 1;
      }
      bool discovery = event.packet.kind == Kind::Discovery;
      bool vdp = event.packet.kind == Kind::Vdp;
      if (!discovery) {
        XELOGD(
            "Native LAN guest datagram received: vdp={}, encrypted={}, "
            "clear={}, source port={}, destination port={}",
            vdp, event.packet.encrypted.size(), event.packet.clear.size(),
            event.packet.source_port, event.packet.destination_port);
      }
      for (auto it = ports.begin(); it != ports.end();) {
        auto inbox = it->second.lock();
        if (!inbox) {
          it = ports.erase(it);
          continue;
        }
        if (it->first.first == event.packet.destination_port &&
            (discovery || it->first.second == vdp)) {
          auto data = vdp ? ToGuestVdp(event.packet) : event.packet.encrypted;
          inbox->Push({{address, event.packet.source_port}, std::move(data)});
        }
        ++it;
      }
    }
  }
  class Port final : public NativeDatagramTransport {
   public:
    Port(std::shared_ptr<Core> core, std::shared_ptr<Inbox> inbox,
         uint16_t port, bool vdp)
        : core_(std::move(core)),
          inbox_(std::move(inbox)),
          port_(port),
          vdp_(vdp) {}
    ~Port() override { Close(); }
    uint16_t port() const override { return port_; }
    void Close() override {
      std::lock_guard lock(inbox_->mutex);
      inbox_->closed = true;
      inbox_->packets.clear();
      inbox_->Signal();
    }
    int ReadHandle() const override { return inbox_->reader; }
    size_t PendingBytes() const override {
      std::lock_guard lock(inbox_->mutex);
      return inbox_->packets.empty() ? 0 : inbox_->packets.front().data.size();
    }
    int Send(Endpoint destination, std::span<const uint8_t> bytes) override {
      {
        std::lock_guard lock(inbox_->mutex);
        if (inbox_->closed) {
          errno = EBADF;
          return -1;
        }
      }
      std::lock_guard lock(core_->mutex);
      if (!core_->transport->healthy()) {
        errno = ENETDOWN;
        return -1;
      }
      Packet packet;
      bool success;
      if (core_->registry.IsBroadcast(destination.address)) {
        packet.source_port = port_;
        packet.destination_port = destination.port;
        packet.encrypted.assign(bytes.begin(), bytes.end());
        success = core_->associations.Discover(std::move(packet));
      } else {
        XnAddr peer;
        KeyId id;
        if (core_->registry.Reverse(destination.address, peer, id)) {
          errno = EHOSTUNREACH;
          return -1;
        }
        if (vdp_) {
          auto converted = FromGuestVdp(port_, destination.port, bytes);
          if (!converted) {
            errno = EINVAL;
            return -1;
          }
          packet = std::move(*converted);
        } else {
          packet.source_port = port_;
          packet.destination_port = destination.port;
          packet.encrypted.assign(bytes.begin(), bytes.end());
        }
        success = core_->associations.Send(Serialized(peer), id,
                                           std::move(packet), Now());
        XELOGD(
            "Native LAN guest datagram sent: vdp={}, bytes={}, source port={}, "
            "destination port={}, accepted={}",
            vdp_, bytes.size(), port_, destination.port, success);
      }
      if (!success) {
        errno = EMSGSIZE;
        return -1;
      }
      return static_cast<int>(bytes.size());
    }
    bool Receive(Datagram& datagram, uint32_t timeout_ms) override {
      std::unique_lock lock(inbox_->mutex);
      inbox_->changed.wait_for(
          lock, std::chrono::milliseconds(timeout_ms),
          [&] { return inbox_->closed || !inbox_->packets.empty(); });
      if (inbox_->packets.empty()) {
        return false;
      }
      datagram = std::move(inbox_->packets.front());
      inbox_->packets.pop_front();
      DrainSignal();
      return true;
    }
    int ReceiveGuest(std::span<uint8_t> buffer, uint32_t flags,
                     Endpoint& source, bool nonblocking) override {
      const int native_flags = flags == 2 ? MSG_PEEK : 0;
      if (flags != 0 && flags != 2) {
        errno = EINVAL;
        return -1;
      }
      std::unique_lock lock(inbox_->mutex);
      if (!nonblocking) {
        inbox_->changed.wait(
            lock, [&] { return inbox_->closed || !inbox_->packets.empty(); });
      }
      if (inbox_->packets.empty()) {
        errno = inbox_->closed ? EBADF : EWOULDBLOCK;
        return -1;
      }
      const auto& packet = inbox_->packets.front();
      source = packet.source;
      size_t size = std::min(buffer.size(), packet.data.size());
      std::copy_n(packet.data.begin(), size, buffer.begin());
      bool truncated = size < packet.data.size();
      if (!(native_flags & MSG_PEEK)) {
        inbox_->packets.pop_front();
        DrainSignal();
      }
      if (truncated) {
        errno = EMSGSIZE;
        return -1;
      }
      return static_cast<int>(size);
    }

   private:
    void DrainSignal() {
      if (inbox_->packets.empty()) {
        uint8_t byte;
        (void)read(inbox_->reader, &byte, 1);
      }
    }
    std::shared_ptr<Core> core_;
    std::shared_ptr<Inbox> inbox_;
    uint16_t port_;
    bool vdp_;
  };
};
NativeLan::NativeLan(Lan& registry, std::shared_ptr<PacketTransport> transport,
                     AssociationConfig config)
    : core_(std::make_shared<Core>(registry, std::move(transport),
                                   std::move(config))) {}
NativeLan::~NativeLan() {
  core_->worker.request_stop();
  core_->worker.join();
  Reset();
}
std::unique_ptr<NativeDatagramTransport> NativeLan::Bind(uint16_t port,
                                                         bool vdp) {
  std::lock_guard lock(core_->mutex);
  if (!core_->transport->healthy()) {
    return nullptr;
  }
  std::erase_if(core_->ports, [](const auto& entry) {
    auto inbox = entry.second.lock();
    if (!inbox) {
      return true;
    }
    std::lock_guard lock(inbox->mutex);
    return inbox->closed;
  });
  if (core_->ports.size() >= 128) {
    return nullptr;
  }
  if (!port) {
    do {
      port = core_->next_port++;
      if (!core_->next_port) {
        core_->next_port = 49152;
      }
    } while (core_->ports.contains({port, vdp}));
  }
  auto& weak = core_->ports[{port, vdp}];
  if (!weak.expired()) {
    return nullptr;
  }
  auto inbox = std::make_shared<Inbox>();
  if (inbox->reader < 0) {
    return nullptr;
  }
  weak = inbox;
  return std::make_unique<Core::Port>(core_, std::move(inbox), port, vdp);
}
uint32_t NativeLan::Connect(uint32_t secure_address) {
  XnAddr peer;
  KeyId id;
  if (core_->registry.Reverse(secure_address, peer, id)) {
    return 10022;
  }
  std::lock_guard lock(core_->mutex);
  if (!core_->transport->healthy()) {
    return 10050;
  }
  return core_->associations.Connect(Serialized(peer), id, Now()) ? 0 : 10055;
}
uint32_t NativeLan::Status(uint32_t secure_address) {
  XnAddr peer;
  KeyId id;
  if (core_->registry.Reverse(secure_address, peer, id)) {
    return 0;
  }
  std::lock_guard lock(core_->mutex);
  if (!core_->transport->healthy()) {
    return 0;
  }
  return static_cast<uint32_t>(
      core_->associations.Status(Serialized(peer), id));
}
void NativeLan::Close(uint32_t secure_address) {
  XnAddr peer;
  KeyId id;
  if (core_->registry.Reverse(secure_address, peer, id)) {
    return;
  }
  std::lock_guard lock(core_->mutex);
  core_->associations.Close(Serialized(peer), id, Now());
  core_->Deliver();
}
void NativeLan::Unregister(const KeyId& id) {
  std::lock_guard lock(core_->mutex);
  core_->associations.Unregister(id, Now());
  core_->Deliver();
}
void NativeLan::Reset() {
  if (!core_) {
    return;
  }
  std::lock_guard lock(core_->mutex);
  core_->associations.Reset();
  for (auto& [port, weak] : core_->ports) {
    if (auto inbox = weak.lock()) {
      std::lock_guard lock(inbox->mutex);
      inbox->closed = true;
      inbox->packets.clear();
      inbox->Signal();
    }
  }
}
}  // namespace xe::net::secure
