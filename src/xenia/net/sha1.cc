/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/net/sha1.h"

#include <algorithm>
#include "third_party/crypto/TinySHA1.hpp"

namespace xe::net::crypto {
Sha1Digest HmacSha1(std::span<const uint8_t> key,
                    std::initializer_list<std::span<const uint8_t>> parts) {
  sha1::SHA1 sha;
  Sha1Digest digest{};
  if (key.size() > 64) {
    sha.processBytes(key.data(), key.size());
    sha.finalize(digest.data());
    sha.reset();
    key = digest;
  }
  std::array<uint8_t, 64> inner, outer;
  inner.fill(0x36);
  outer.fill(0x5C);
  for (size_t i = 0; i < key.size(); ++i) {
    inner[i] ^= key[i];
    outer[i] ^= key[i];
  }
  sha.processBytes(inner.data(), inner.size());
  for (auto part : parts) {
    if (!part.empty()) {
      sha.processBytes(part.data(), part.size());
    }
  }
  sha.finalize(digest.data());
  sha.reset();
  sha.processBytes(outer.data(), outer.size());
  sha.processBytes(digest.data(), digest.size());
  sha.finalize(digest.data());
  return digest;
}
}  // namespace xe::net::crypto
