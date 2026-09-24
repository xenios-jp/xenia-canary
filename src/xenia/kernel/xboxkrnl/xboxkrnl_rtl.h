/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_KERNEL_XBOXKRNL_XBOXKRNL_RTL_H_
#define XENIA_KERNEL_XBOXKRNL_XBOXKRNL_RTL_H_

#include "xenia/kernel/util/kernel_fwd.h"
#include "xenia/kernel/xobject.h"
#include "xenia/xbox.h"

namespace xe {
namespace kernel {
namespace xboxkrnl {

// This structure tries to match the one on the 360 as best I can figure out.
// Unfortunately some games have the critical sections pre-initialized in
// their embedded data and InitializeCriticalSection will never be called.
#pragma pack(push, 1)
struct X_RTL_CRITICAL_SECTION {
  X_DISPATCH_HEADER header;
  int32_t lock_count;                          // 0x10 -1 -> 0 on first lock
  xe::be<int32_t> recursion_count;             // 0x14  0 -> 1 on first lock
  TypedGuestPointer<X_KTHREAD> owning_thread;  // 0x18 PKTHREAD 0 unless locked
};
#pragma pack(pop)
static_assert_size(X_RTL_CRITICAL_SECTION, 28);

void xeRtlInitializeCriticalSection(X_RTL_CRITICAL_SECTION* cs,
                                    uint32_t cs_ptr);
X_STATUS xeRtlInitializeCriticalSectionAndSpinCount(X_RTL_CRITICAL_SECTION* cs,
                                                    uint32_t cs_ptr,
                                                    uint32_t spin_count);

}  // namespace xboxkrnl
}  // namespace kernel
}  // namespace xe

#endif  // XENIA_KERNEL_XBOXKRNL_XBOXKRNL_RTL_H_
