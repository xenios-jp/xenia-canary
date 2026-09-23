/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/metal/metal_upload_cache.h"

#include <cstdint>
#include <utility>

#include "third_party/catch/include/catch.hpp"

namespace xe {
namespace gpu {
namespace test {

TEST_CASE("Upload cache reuses only an identical key", "[metal]") {
  struct Slice {
    size_t index = 0;
  };
  using Key = std::pair<uint32_t, uint64_t>;
  metal::UploadCache<Slice, Key> cache;
  size_t uploads = 0;
  bool fail = false;
  auto upload = [&](Slice& out) {
    if (fail) {
      return false;
    }
    out.index = uploads++;
    return true;
  };
  Slice first, next;
  REQUIRE(cache.GetOrUpload({4, 1}, upload, first));
  REQUIRE(uploads == 1);
  REQUIRE(cache.GetOrUpload({4, 1}, upload, next));
  REQUIRE(next.index == first.index);
  REQUIRE(uploads == 1);
  // A different revision or size is a different upload.
  REQUIRE(cache.GetOrUpload({4, 2}, upload, next));
  REQUIRE(next.index != first.index);
  REQUIRE(cache.GetOrUpload({2, 2}, upload, next));
  REQUIRE(uploads == 3);
  // A failed upload does not replace the cached entry.
  fail = true;
  REQUIRE_FALSE(cache.GetOrUpload({8, 3}, upload, next));
  fail = false;
  Slice cached;
  REQUIRE(cache.GetOrUpload({2, 2}, upload, cached));
  REQUIRE(cached.index == next.index);
  REQUIRE(uploads == 3);
  // After a reset, the retired slice is never returned.
  cache.Reset();
  REQUIRE(cache.GetOrUpload({2, 2}, upload, next));
  REQUIRE(uploads == 4);
}

}  // namespace test
}  // namespace gpu
}  // namespace xe
