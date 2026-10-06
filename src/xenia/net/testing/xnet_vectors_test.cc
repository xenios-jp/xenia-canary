/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <algorithm>
#include <fstream>
#include <sstream>
#include "third_party/catch/single_include/catch2/catch.hpp"
#include "third_party/rapidjson/include/rapidjson/document.h"
#include "xenia/net/xnet_secure.h"
using namespace xe::net;
using namespace xe::net::secure;
namespace {
std::vector<uint8_t> Hex(const rapidjson::Value& value) {
  const std::string text(value.GetString(), value.GetStringLength());
  REQUIRE(text.size() % 2 == 0);
  std::vector<uint8_t> bytes;
  for (size_t i = 0; i < text.size(); i += 2) {
    bytes.push_back(
        static_cast<uint8_t>(std::stoul(text.substr(i, 2), nullptr, 16)));
  }
  return bytes;
}
template <size_t N>
std::array<uint8_t, N> Fixed(const rapidjson::Value& value) {
  auto bytes = Hex(value);
  REQUIRE(bytes.size() == N);
  std::array<uint8_t, N> result;
  std::copy(bytes.begin(), bytes.end(), result.begin());
  return result;
}
}  // namespace
TEST_CASE("Native codecs match independently computed synthetic packets",
          "[net][secure]") {
  std::ifstream input(XE_XNET_SYNTHETIC_VECTORS);
  REQUIRE(input.good());
  std::stringstream text;
  text << input.rdbuf();
  rapidjson::Document v;
  v.Parse(text.str().c_str());
  REQUIRE_FALSE(v.HasParseError());
  REQUIRE(std::string(v["origin"].GetString()) == "synthetic-only");
  const auto platform_key = Fixed<16>(v["inputs"]["platform_hmac_key"]);
  auto broadcast =
      DeriveBroadcast(Fixed<16>(v["inputs"]["executable_lan_key"]),
                      [&](auto bytes, auto& digest) {
                        digest = crypto::HmacSha1(platform_key, {bytes});
                        return true;
                      });
  REQUIRE(broadcast);
  const auto stream = Hex(v["derived"]["broadcast_stream"]);
  CHECK(std::ranges::equal(broadcast->authentication,
                           std::span(stream).first(16)));
  CHECK(broadcast->cipher ==
        Fixed<24>(v["derived"]["broadcast_cipher_odd_parity"]));
  CHECK(
      std::ranges::equal(broadcast->iv_seed, std::span(stream).subspan(40, 8)));
  ProtectionKeys traffic;
  const auto derived = Hex(v["derived"]["traffic_stream"]);
  std::copy_n(derived.begin(), 16, traffic.authentication.begin());
  std::copy_n(derived.begin() + 16, 24, traffic.cipher.begin());
  std::copy_n(derived.begin() + 80, 8, traffic.iv_seed.begin());
  REQUIRE(v["packets"].Size() == 107);
  for (const auto& item : v["packets"].GetArray()) {
    INFO(item["name"].GetString());
    const std::string type = item["type"].GetString();
    Packet packet;
    packet.kind = type == "discovery" ? Kind::Discovery
                  : type == "udp"     ? Kind::Udp
                  : type == "control" ? Kind::Control
                                      : Kind::Vdp;
    packet.source_port =
        item.HasMember("source_port") ? item["source_port"].GetUint() : 1000;
    packet.destination_port = item.HasMember("destination_port")
                                  ? item["destination_port"].GetUint()
                                  : 1000;
    packet.control_type =
        item.HasMember("control_type") ? item["control_type"].GetUint() : 0;
    packet.encrypted = Hex(item["body"]);
    if (item.HasMember("clear")) {
      packet.clear = Hex(item["clear"]);
    }
    const bool discovery = packet.kind == Kind::Discovery;
    if (discovery) {
      packet.metadata = Fixed<12>(item["metadata"]);
    }
    const auto& keys = discovery ? *broadcast : traffic;
    const uint32_t spi = discovery ? 0 : 0x040506;
    const uint32_t sequence = item["sequence"].GetUint();
    const auto expected = Hex(item["wire"]);
    std::vector<uint8_t> wire;
    REQUIRE(Encode(keys, spi, sequence, packet, wire));
    CHECK(wire == expected);
    auto decoded = Decode(keys, spi, sequence, expected, discovery);
    REQUIRE(decoded);
    CHECK(decoded->encrypted == packet.encrypted);
    CHECK(decoded->clear == packet.clear);
    CHECK(decoded->source_port == packet.source_port);
    CHECK(decoded->destination_port == packet.destination_port);
    CHECK(decoded->metadata == packet.metadata);
    for (size_t i = 0; i < expected.size(); ++i) {
      CHECK_FALSE(
          Decode(keys, spi, sequence, std::span(expected).first(i), discovery));
      auto mutation = expected;
      mutation[i] ^= 1;
      CHECK_FALSE(Decode(keys, spi, sequence, mutation, discovery));
    }
  }
}
