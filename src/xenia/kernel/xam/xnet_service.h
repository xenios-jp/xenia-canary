/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#pragma once
#include "xenia/net/lan.h"
namespace xe::kernel {
class KernelState;
}
namespace xe::kernel::xam {
bool NativeLanEnabled();
std::shared_ptr<net::Lan> LocalLan(KernelState* state);
}  // namespace xe::kernel::xam
