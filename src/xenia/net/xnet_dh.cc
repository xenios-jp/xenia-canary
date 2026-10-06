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
using Number = std::array<uint32_t, 24>;
constexpr Number kPrime = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xA63A3620u, 0xF44C42E9u,
                           0x625E7EC6u, 0xE485B576u, 0x6D51C245u, 0x4FE1356Du,
                           0xF25F1437u, 0x302B0A6Du, 0xCD3A431Bu, 0xEF9519B3u,
                           0x8E3404DDu, 0x514A0879u, 0x3B139B22u, 0x020BBEA6u,
                           0x8A67CC74u, 0x29024E08u, 0x80DC1CD1u, 0xC4C6628Bu,
                           0x2168C234u, 0xC90FDAA2u, 0xFFFFFFFFu, 0xFFFFFFFFu};
constexpr Number kOne{1}, kTwo{2};

constexpr Number Select(const Number& a, const Number& b, uint32_t use_b) {
  const uint32_t mask = 0u - use_b;
  Number result{};
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = (a[i] & ~mask) | (b[i] & mask);
  }
  return result;
}
constexpr Number Reduce(const Number& value, uint32_t high) {
  Number reduced{};
  uint64_t borrow = 0;
  for (size_t i = 0; i < value.size(); ++i) {
    const uint64_t difference = uint64_t(value[i]) - kPrime[i] - borrow;
    reduced[i] = uint32_t(difference);
    borrow = difference >> 63;
  }
  return Select(value, reduced, high | (uint32_t(borrow) ^ 1));
}
constexpr Number Double(const Number& value) {
  Number result{};
  uint32_t carry = 0;
  for (size_t i = 0; i < value.size(); ++i) {
    result[i] = (value[i] << 1) | carry;
    carry = value[i] >> 31;
  }
  return Reduce(result, carry);
}
constexpr Number RadixPower(size_t bits, Number result = kOne) {
  for (size_t i = 0; i < bits; ++i) {
    result = Double(result);
  }
  return result;
}
constexpr auto kRadix = RadixPower(768);
constexpr auto kRadixSquared = RadixPower(768, kRadix);

// Schoolbook product followed by fixed-count Montgomery reduction. For this
// modulus, -p[0]^-1 mod 2^32 is one. Every carry loop has a public fixed bound;
// no secret-dependent branches or table indices are used in multiplication.
Number Multiply(const Number& a, const Number& b) {
  std::array<uint32_t, 49> product{};
  for (size_t i = 0; i < 24; ++i) {
    uint64_t carry = 0;
    for (size_t j = 0; j < 24; ++j) {
      const uint64_t sum = uint64_t(a[i]) * b[j] + product[i + j] + carry;
      product[i + j] = uint32_t(sum);
      carry = sum >> 32;
    }
    product[i + 24] = uint32_t(carry);
  }
  for (size_t i = 0; i < 24; ++i) {
    const uint32_t factor = product[i];
    uint64_t carry = 0;
    for (size_t j = 0; j < 24; ++j) {
      const uint64_t sum =
          uint64_t(factor) * kPrime[j] + product[i + j] + carry;
      product[i + j] = uint32_t(sum);
      carry = sum >> 32;
    }
    for (size_t j = i + 24; j < product.size(); ++j) {
      const uint64_t sum = uint64_t(product[j]) + carry;
      product[j] = uint32_t(sum);
      carry = sum >> 32;
    }
  }
  Number result{};
  std::copy_n(product.begin() + 24, 24, result.begin());
  return Reduce(result, product[48]);
}
Number Power(const Number& base, std::span<const uint8_t> big_endian_exponent) {
  const auto converted = Multiply(base, kRadixSquared);
  auto result = kRadix;
  for (auto byte : big_endian_exponent) {
    for (int bit = 7; bit >= 0; --bit) {
      const auto squared = Multiply(result, result);
      const auto multiplied = Multiply(squared, converted);
      result = Select(squared, multiplied, (byte >> bit) & 1);
    }
  }
  return Multiply(result, kOne);
}
DhValue Serialize(const Number& number) {
  DhValue bytes{};
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = uint8_t(number[i / 4] >> ((i % 4) * 8));
  }
  return bytes;
}
Number Deserialize(const DhValue& bytes) {
  Number number{};
  for (size_t i = 0; i < bytes.size(); ++i) {
    number[i / 4] |= uint32_t(bytes[i]) << ((i % 4) * 8);
  }
  return number;
}
bool Less(const Number& a, const Number& b) {
  for (size_t i = a.size(); i-- > 0;) {
    if (a[i] != b[i]) {
      return a[i] < b[i];
    }
  }
  return false;
}
DhValue SubgroupExponent() {
  auto q = kPrime;
  --q[0];
  for (size_t i = 0; i < q.size(); ++i) {
    q[i] = (q[i] >> 1) | (i + 1 < q.size() ? q[i + 1] << 31 : 0);
  }
  auto bytes = Serialize(q);
  std::ranges::reverse(bytes);
  return bytes;
}
void Wipe(std::span<uint8_t> bytes) {
  volatile uint8_t* output = bytes.data();
  for (size_t i = 0; i < bytes.size(); ++i) {
    output[i] = 0;
  }
}
}  // namespace
DhKeyPair::DhKeyPair(std::span<const uint8_t> exponent) {
  std::copy(exponent.begin(), exponent.end(), exponent_.begin());
  public_ = Serialize(Power(kTwo, exponent_));
}
void DhKeyPair::Erase() {
  Wipe(exponent_);
  active_ = false;
}
DhKeyPair::~DhKeyPair() { Erase(); }
DhKeyPair::DhKeyPair(DhKeyPair&& other) noexcept
    : exponent_(other.exponent_),
      public_(other.public_),
      active_(other.active_) {
  other.Erase();
}
DhKeyPair& DhKeyPair::operator=(DhKeyPair&& other) noexcept {
  if (this != &other) {
    Erase();
    exponent_ = other.exponent_;
    public_ = other.public_;
    active_ = other.active_;
    other.Erase();
  }
  return *this;
}
std::optional<DhKeyPair> DhKeyPair::Generate() {
  std::array<uint8_t, 32> exponent;
  Lan::Random(exponent);
  exponent[0] |= 0x80;
  DhKeyPair result(exponent);
  Wipe(exponent);
  return result;
}
std::optional<DhKeyPair> DhKeyPair::FromPrivate(
    std::span<const uint8_t> exponent) {
  if (exponent.size() != 32 ||
      std::ranges::all_of(exponent.first(16),
                          [](uint8_t byte) { return byte == 0; })) {
    return std::nullopt;
  }
  return DhKeyPair(exponent);
}
std::optional<DhValue> DhKeyPair::Shared(const DhValue& peer) const {
  if (!active_) {
    return std::nullopt;
  }
  const auto remote = Deserialize(peer);
  auto maximum = kPrime;
  maximum[0] -= 2;
  // Peer range and subgroup checks operate only on public inputs.
  if (Less(remote, kTwo) || Less(maximum, remote) ||
      Power(remote, SubgroupExponent()) != kOne) {
    return std::nullopt;
  }
  return Serialize(Power(remote, exponent_));
}
}  // namespace xe::net::secure
