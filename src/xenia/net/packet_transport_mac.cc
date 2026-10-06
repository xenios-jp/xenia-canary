/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/net/packet_transport.h"

#if XE_PLATFORM_MAC
#include <ifaddrs.h>
#include <net/bpf.h>
#include <net/if.h>
#include <net/if_dl.h>
#include <pcap/pcap.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

namespace xe::net::secure {
namespace {
class CaptureTransport final : public PacketTransport {
 public:
  CaptureTransport(pcap_t* capture, Mac mac, uint32_t mtu)
      : capture_(capture), mac_(mac), mtu_(mtu) {}
  ~CaptureTransport() override { pcap_close(capture_); }
  Mac identity() const override { return mac_; }
  uint32_t mtu() const override { return mtu_; }
  bool healthy() const override { return healthy_; }
  int Send(std::span<const uint8_t> frame) override {
    if (!AllowedPacketFrame(frame, mac_, mtu_, true)) {
      return -1;
    }
    std::lock_guard lock(mutex_);
    const int result = pcap_inject(capture_, frame.data(), frame.size());
    if (result < 0) {
      healthy_ = false;
    }
    return result;
  }
  bool Receive(std::vector<uint8_t>& frame, uint32_t timeout_ms) override {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(timeout_ms);
    do {
      {
        std::lock_guard lock(mutex_);
        pcap_pkthdr* header = nullptr;
        const u_char* bytes = nullptr;
        const int result = pcap_next_ex(capture_, &header, &bytes);
        if (result < 0) {
          healthy_ = false;
          return false;
        }
        if (result == 1) {
          if (header->caplen == header->len &&
              AllowedPacketFrame({bytes, header->caplen}, mac_, mtu_, false)) {
            frame.assign(bytes, bytes + header->caplen);
            return true;
          }
          continue;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
  }

 private:
  pcap_t* capture_;
  Mac mac_;
  uint32_t mtu_;
  std::atomic<bool> healthy_{true};
  std::mutex mutex_;
};
}  // namespace
std::unique_ptr<PacketTransport> OpenPacketInterface(std::string_view name,
                                                     std::string& error) {
  Mac mac{};
  ifaddrs* list = nullptr;
  if (getifaddrs(&list)) {
    error = "Interface enumeration failed";
    return {};
  }
  for (auto* it = list; it; it = it->ifa_next) {
    if (it->ifa_addr && name == it->ifa_name &&
        it->ifa_addr->sa_family == AF_LINK) {
      auto* link = reinterpret_cast<sockaddr_dl*>(it->ifa_addr);
      if (link->sdl_alen == mac.size()) {
        std::copy_n(reinterpret_cast<const uint8_t*>(LLADDR(link)), mac.size(),
                    mac.begin());
      }
    }
  }
  freeifaddrs(list);
  if (mac == Mac{}) {
    error = "Interface has no Ethernet identity";
    return {};
  }
  const std::string interface(name);
  if (interface.size() >= IFNAMSIZ) {
    error = "Invalid interface name";
    return {};
  }
  ifreq request{};
  std::copy(interface.begin(), interface.end(), request.ifr_name);
  const int socket_fd = socket(AF_INET, SOCK_DGRAM, 0);
  const bool mtu_ok =
      socket_fd >= 0 && ioctl(socket_fd, SIOCGIFMTU, &request) == 0;
  if (socket_fd >= 0) {
    close(socket_fd);
  }
  if (!mtu_ok || request.ifr_mtu < 576) {
    error = "Interface MTU unavailable";
    return {};
  }
  char buffer[PCAP_ERRBUF_SIZE]{};
  pcap_t* capture =
      pcap_open_live(interface.c_str(), request.ifr_mtu + 18, 0, 1, buffer);
  if (!capture) {
    error = buffer;
    return {};
  }
  // Flush each packet immediately; BPF buffering delays guest updates.
  unsigned immediate = 1;
  if (ioctl(pcap_fileno(capture), BIOCIMMEDIATE, &immediate) < 0) {
    error = "Immediate packet delivery unavailable";
    pcap_close(capture);
    return {};
  }
  if (pcap_datalink(capture) != DLT_EN10MB ||
      pcap_setnonblock(capture, 1, buffer)) {
    error = "Ethernet capture unavailable";
    pcap_close(capture);
    return {};
  }
  bpf_program filter{};
  if (pcap_compile(capture, &filter, "udp src port 3074 and udp dst port 3074",
                   1, PCAP_NETMASK_UNKNOWN) < 0) {
    error = pcap_geterr(capture);
    pcap_close(capture);
    return {};
  }
  const int filter_result = pcap_setfilter(capture, &filter);
  pcap_freecode(&filter);
  if (filter_result < 0) {
    error = pcap_geterr(capture);
    pcap_close(capture);
    return {};
  }
  return std::make_unique<CaptureTransport>(capture, mac, request.ifr_mtu);
}
}  // namespace xe::net::secure
#endif
