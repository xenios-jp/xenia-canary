/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/trace_profile.h"

#include <chrono>
#include <ctime>

#include "xenia/base/platform.h"
#if XE_PLATFORM_WIN32
#include "xenia/base/platform_win.h"
#endif

namespace xe {
namespace gpu {

#if XE_PLATFORM_WIN32
namespace {
uint64_t FileTimesToNs(const FILETIME& kernel, const FILETIME& user) {
  return (((uint64_t(kernel.dwHighDateTime) << 32) | kernel.dwLowDateTime) +
          ((uint64_t(user.dwHighDateTime) << 32) | user.dwLowDateTime)) *
         100;
}
}  // namespace
#else
namespace {
uint64_t ClockNs(clockid_t clock) {
  timespec value{};
  return clock_gettime(clock, &value) == 0
             ? uint64_t(value.tv_sec) * 1000000000 + uint64_t(value.tv_nsec)
             : 0;
}
}  // namespace
#endif

uint64_t TraceThreadCpuNs() {
#if XE_PLATFORM_WIN32
  FILETIME created, exited, kernel, user;
  return GetThreadTimes(GetCurrentThread(), &created, &exited, &kernel, &user)
             ? FileTimesToNs(kernel, user)
             : 0;
#else
  return ClockNs(CLOCK_THREAD_CPUTIME_ID);
#endif
}

uint64_t TraceProcessCpuNs() {
#if XE_PLATFORM_WIN32
  FILETIME created, exited, kernel, user;
  return GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)
             ? FileTimesToNs(kernel, user)
             : 0;
#else
  return ClockNs(CLOCK_PROCESS_CPUTIME_ID);
#endif
}

uint64_t TraceWallNs() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count());
}

}  // namespace gpu
}  // namespace xe
