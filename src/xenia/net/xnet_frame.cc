/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <algorithm>
#include "xenia/net/xnet_secure.h"

namespace xe::net::secure {
namespace {
constexpr Mac kBroadcast{255, 255, 255, 255, 255, 255};
uint16_t Word(std::span<const uint8_t> bytes) {
  return uint16_t((uint16_t(bytes[0]) << 8) | bytes[1]);
}
void SetWord(std::span<uint8_t> bytes, uint16_t value) {
  bytes[0] = uint8_t(value >> 8);
  bytes[1] = uint8_t(value);
}
uint16_t Checksum(std::span<const uint8_t> bytes, uint32_t sum = 0) {
  for (size_t i = 0; i < bytes.size(); i += 2) {
    sum +=
        (uint32_t(bytes[i]) << 8) | (i + 1 < bytes.size() ? bytes[i + 1] : 0);
  }
  while (sum >> 16) {
    sum = (sum & 0xFFFF) + (sum >> 16);
  }
  return uint16_t(~sum);
}
uint32_t PseudoHeader(std::span<const uint8_t> ip, uint16_t udp_length) {
  return uint32_t(Word(ip.subspan(12, 2))) + Word(ip.subspan(14, 2)) +
         Word(ip.subspan(16, 2)) + Word(ip.subspan(18, 2)) + 17 + udp_length;
}
bool Unicast(const Mac& mac) {
  return !(mac[0] & 1) &&
         std::ranges::any_of(mac, [](uint8_t byte) { return byte != 0; });
}
}  // namespace
std::optional<std::vector<uint8_t>> EncodeFrame(const Frame& frame) {
  if (!Unicast(frame.source) ||
      (frame.broadcast ? frame.destination != kBroadcast
                       : !Unicast(frame.destination)) ||
      frame.payload.size() > kMaxDatagramSize) {
    return std::nullopt;
  }
  std::vector<uint8_t> result(42 + frame.payload.size(), 0);
  std::copy(frame.destination.begin(), frame.destination.end(), result.begin());
  std::copy(frame.source.begin(), frame.source.end(), result.begin() + 6);
  SetWord(std::span(result).subspan(12, 2), 0x0800);
  auto ip = std::span(result).subspan(14, 20);
  ip[0] = 0x45;
  SetWord(ip.subspan(2, 2), uint16_t(28 + frame.payload.size()));
  ip[8] = 128;
  ip[9] = 17;
  // Native peers use virtual IP 0.0.0.1; Ethernet MACs identify the machines.
  ip[15] = 1;
  if (frame.broadcast) {
    std::fill(ip.begin() + 16, ip.end(), 255);
  } else {
    ip[19] = 1;
  }
  SetWord(ip.subspan(10, 2), Checksum(ip));
  auto udp = std::span(result).subspan(34);
  SetWord(udp.first(2), 3074);
  SetWord(udp.subspan(2, 2), 3074);
  SetWord(udp.subspan(4, 2), uint16_t(udp.size()));
  std::copy(frame.payload.begin(), frame.payload.end(), udp.begin() + 8);
  const auto checksum = Checksum(udp, PseudoHeader(ip, uint16_t(udp.size())));
  SetWord(udp.subspan(6, 2), checksum ? checksum : 0xFFFF);
  return result;
}
std::optional<Frame> DecodeFrame(std::span<const uint8_t> wire) {
  if (wire.size() < 42 || Word(wire.subspan(12, 2)) != 0x0800) {
    return std::nullopt;
  }
  const auto ip = wire.subspan(14, 20);
  const auto length = Word(ip.subspan(2, 2));
  if (ip[0] != 0x45 || ip[9] != 17 || !ip[8] || length < 28 ||
      length > wire.size() - 14 || (Word(ip.subspan(6, 2)) & 0xBFFF) ||
      Checksum(ip) || ip[12] || ip[13] || ip[14] || ip[15] != 1) {
    return std::nullopt;
  }
  const bool broadcast =
      ip[16] == 255 && ip[17] == 255 && ip[18] == 255 && ip[19] == 255;
  if (!broadcast && (ip[16] || ip[17] || ip[18] || ip[19] != 1)) {
    return std::nullopt;
  }
  const auto udp = wire.subspan(34, length - 20);
  if (Word(udp.first(2)) != 3074 || Word(udp.subspan(2, 2)) != 3074 ||
      Word(udp.subspan(4, 2)) != udp.size() ||
      (Word(udp.subspan(6, 2)) &&
       Checksum(udp, PseudoHeader(ip, uint16_t(udp.size()))))) {
    return std::nullopt;
  }
  Frame frame;
  std::copy_n(wire.begin(), 6, frame.destination.begin());
  std::copy_n(wire.begin() + 6, 6, frame.source.begin());
  if (!Unicast(frame.source) || (broadcast ? frame.destination != kBroadcast
                                           : !Unicast(frame.destination))) {
    return std::nullopt;
  }
  frame.broadcast = broadcast;
  frame.payload.assign(udp.begin() + 8, udp.end());
  return frame;
}
bool AllowedPacketFrame(std::span<const uint8_t> wire, const Mac& local,
                        uint32_t mtu, bool outgoing) {
  if (mtu < 576 || mtu > 65535 || wire.size() > size_t(mtu) + 14) {
    return false;
  }
  const auto frame = DecodeFrame(wire);
  if (!frame ||
      (outgoing ? frame->source != local
                : frame->source == local ||
                      (frame->destination != local && !frame->broadcast))) {
    return false;
  }
  const Mac zero{};
  if (frame->source == zero || (frame->source[0] & 1) ||
      (!frame->broadcast &&
       (frame->destination == zero || (frame->destination[0] & 1)))) {
    return false;
  }
  const auto& p = frame->payload;
  if (p.size() < 16) {
    return false;
  }
  if (!p[0] && !p[1] && !p[2] && !p[3]) {
    return !frame->broadcast && p.size() == 256 &&
           (Word(std::span(p).subspan(4, 2)) == 0x0058 ||
            Word(std::span(p).subspan(4, 2)) == 0x0158) &&
           p[6] == 128 && !p[7] &&
           Word(std::span(p).subspan(132, 2)) == 0x8058 && p[134] == 100 &&
           !p[135] && Word(std::span(p).subspan(232, 2)) == 0x8158 &&
           p[234] == 24 && !p[235];
  }
  const uint8_t kind = p[0] & 15;
  if (kind >= 1 && kind <= 3) {
    return false;
  }
  const bool zero_spi = !p[1] && !p[2] && !p[3];
  return frame->broadcast ? zero_spi && kind >= 7 && kind <= 9 && p.size() >= 29
                          : !zero_spi;
}
}  // namespace xe::net::secure
