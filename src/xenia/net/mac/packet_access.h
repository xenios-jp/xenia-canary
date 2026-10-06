/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#pragma once

#include <string>

namespace xe::net {
// UI-thread startup only. Uses the OS administrator prompt for lasting packet
// access; never elevates the emulator or depends on a particular app build.
// Cancellation returns false with an empty error.
bool EnsurePacketAccess(std::string& error);
}  // namespace xe::net
