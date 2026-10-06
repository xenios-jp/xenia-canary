/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2023 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <vector>

#include "xenia/base/filesystem.h"
#include "xenia/vfs/devices/host_path_device.h"
#include "xenia/vfs/devices/stfs_xbox.h"
#include "xenia/vfs/file.h"
#include "xenia/vfs/gdfx_util.h"

#include "third_party/catch/include/catch.hpp"

namespace xe::vfs::test {

TEST_CASE("Host file deletion waits for the last open handle", "[host_file]") {
  struct TemporaryDirectory {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("xenia-vfs-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    ~TemporaryDirectory() {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
  } temporary;
  REQUIRE(std::filesystem::create_directory(temporary.path));
  const auto path = temporary.path / "shared.tmp";
  {
    std::ofstream stream(path);
    stream << "abc";
  }
  HostPathDevice device("test:", temporary.path, false);
  REQUIRE(device.Initialize());
  auto* entry = device.ResolvePath("shared.tmp");
  REQUIRE(entry);
  auto close = [](File* file) {
    if (file) {
      file->Destroy();
    }
  };
  std::unique_ptr<File, decltype(close)> first(nullptr, close),
      second(nullptr, close);
  File* opened = nullptr;
  REQUIRE(entry->Open(filesystem::FileAccess::kGenericRead, &opened) ==
          X_STATUS_SUCCESS);
  first.reset(opened);
  REQUIRE(entry->Open(filesystem::FileAccess::kGenericRead, &opened) ==
          X_STATUS_SUCCESS);
  second.reset(opened);
  entry->SetForDeletion(true);
  REQUIRE_FALSE(entry->Delete());
  first.reset();
  REQUIRE(std::filesystem::exists(path));
  REQUIRE(second->entry()->device() == &device);
  std::array<uint8_t, 3> bytes{};
  size_t read = 0;
  REQUIRE(second->ReadSync(bytes, 0, &read) == X_STATUS_SUCCESS);
  REQUIRE(read == bytes.size());
  REQUIRE(bytes == std::array<uint8_t, 3>{'a', 'b', 'c'});
  second.reset();
  REQUIRE_FALSE(std::filesystem::exists(path));
  REQUIRE(device.ResolvePath("shared.tmp") == nullptr);
}

TEST_CASE("STFS Decode date and time", "[stfs_decode]") {
  SECTION("10 June 2022 19:46:00 UTC - Decode") {
    const uint16_t date = 0x54CA;
    const uint16_t time = 0x9DBD;
    const uint64_t result = 132993639580000000;

    const uint64_t timestamp = decode_fat_timestamp(date, time);

    REQUIRE(timestamp == result);
  }
}

// Writes a directory entry at |offset| in |image|.
static void WriteGdfxEntry(std::vector<uint8_t>& image, size_t offset,
                           uint16_t node_l, uint16_t node_r, uint32_t sector,
                           uint32_t length, const char* name) {
  const uint8_t name_length = uint8_t(std::strlen(name));
  std::memcpy(&image[offset + 0], &node_l, sizeof(node_l));
  std::memcpy(&image[offset + 2], &node_r, sizeof(node_r));
  std::memcpy(&image[offset + 4], &sector, sizeof(sector));
  std::memcpy(&image[offset + 8], &length, sizeof(length));
  image[offset + 12] = 0;
  image[offset + 13] = name_length;
  std::memcpy(&image[offset + 14], name, name_length);
}

TEST_CASE("GDFX directory padding", "[gdfx]") {
  SECTION("Padding that starts a sector stands in for no entry") {
    REQUIRE(GdfxPaddedEntryOrdinal(0) == 0);
    REQUIRE(GdfxPaddedEntryOrdinal(0x200) == 0);
  }

  SECTION("Padding after an entry stands in for the next sector's first") {
    REQUIRE(GdfxPaddedEntryOrdinal(1) == 0x200);
    REQUIRE(GdfxPaddedEntryOrdinal(0x1F0) == 0x200);
    REQUIRE(GdfxPaddedEntryOrdinal(0xFFFF) == 0x10000);
  }

  SECTION("A lookup follows padding to the next sector") {
    // A two sector root directory at sector 1 whose second entry didn't fit
    // in the first sector.
    std::vector<uint8_t> image(8 * kGdfxSectorSize);
    const size_t root = kGdfxSectorSize;
    WriteGdfxEntry(image, root, 0, 0x1F0, 6, 0x10, "a.bin");
    std::memset(&image[root + 0x1F0 * 4], 0xFF, kGdfxSectorSize - 0x1F0 * 4);
    WriteGdfxEntry(image, root + kGdfxSectorSize, 0, 0, 5, 0x100,
                   "default.xex");

    const GdfxPartitionInfo partition = {0, 1, 2 * kGdfxSectorSize};
    auto location =
        GdfxFindFile(image.data(), image.size(), partition, "default.xex");
    REQUIRE(location);
    REQUIRE(location->offset == 5 * kGdfxSectorSize);
    REQUIRE(location->length == 0x100);
  }

  SECTION("A lookup in an empty directory finds nothing") {
    std::vector<uint8_t> image(4 * kGdfxSectorSize);
    std::memset(&image[kGdfxSectorSize], 0xFF, kGdfxSectorSize);
    const GdfxPartitionInfo partition = {0, 1, kGdfxSectorSize};
    REQUIRE(!GdfxFindFile(image.data(), image.size(), partition, "a.bin"));
  }
}

}  // namespace xe::vfs::test
