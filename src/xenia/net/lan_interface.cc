/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/net/lan_interface.h"

#include <algorithm>
#include <set>

#if XE_PLATFORM_MAC
#include <SystemConfiguration/SystemConfiguration.h>
#endif

#if XE_PLATFORM_WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#endif

namespace xe::net {
std::optional<LanInterface> SelectLanInterface(
    std::span<const LanInterface> interfaces, std::string_view selector,
    std::string_view preferred) {
  if (selector.empty()) {
    // Prefer configured networks to self-assigned links, including when a
    // transient primary-interface entry points at a direct cable connection.
    // Keep self-assigned links available for direct console connections.
    for (bool link_local : {false, true}) {
      const LanInterface* fallback = nullptr;
      for (const auto& interface : interfaces) {
        if (!interface.address || !interface.broadcast || !interface.index ||
            ((interface.address & 0xffff0000u) == 0xa9fe0000u) != link_local) {
          continue;
        }
        if (interface.name == preferred) {
          return interface;
        }
        if (!fallback) {
          fallback = &interface;
        }
      }
      if (fallback) {
        return *fallback;
      }
    }
    return std::nullopt;
  }
  in_addr parsed{};
  const bool numeric =
      inet_pton(AF_INET, std::string(selector).c_str(), &parsed) == 1;
  for (const auto& interface : interfaces) {
    if (!interface.address || !interface.broadcast || !interface.index) {
      continue;
    }
    if (selector == interface.name ||
        (numeric && ntohl(parsed.s_addr) == interface.address)) {
      return interface;
    }
  }
  return std::nullopt;
}
std::vector<LanInterface> EnumerateLanInterfaces() {
  std::vector<LanInterface> interfaces;
#if !XE_PLATFORM_WIN32
#if XE_PLATFORM_MAC
  // System Configuration identifies normal Ethernet/Wi-Fi adapters without
  // guessing device names; VPN tunnels and peer-to-peer radio links are
  // excluded.
  std::set<std::string> physical;
  if (CFArrayRef adapters = SCNetworkInterfaceCopyAll()) {
    for (CFIndex index = 0; index < CFArrayGetCount(adapters); ++index) {
      auto adapter = static_cast<SCNetworkInterfaceRef>(
          CFArrayGetValueAtIndex(adapters, index));
      CFStringRef type = SCNetworkInterfaceGetInterfaceType(adapter);
      CFStringRef name = SCNetworkInterfaceGetBSDName(adapter);
      char device[IFNAMSIZ]{};
      if (type && name &&
          (CFEqual(type, kSCNetworkInterfaceTypeEthernet) ||
           CFEqual(type, kSCNetworkInterfaceTypeIEEE80211)) &&
          CFStringGetCString(name, device, sizeof(device),
                             kCFStringEncodingUTF8)) {
        physical.emplace(device);
      }
    }
    CFRelease(adapters);
  }
#endif
  ifaddrs* list = nullptr;
  if (getifaddrs(&list)) {
    return interfaces;
  }
  for (const ifaddrs* it = list; it; it = it->ifa_next) {
    if (!it->ifa_addr || it->ifa_addr->sa_family != AF_INET ||
        !it->ifa_broadaddr || !(it->ifa_flags & IFF_UP) ||
        !(it->ifa_flags & IFF_BROADCAST) || (it->ifa_flags & IFF_LOOPBACK)) {
      continue;
    }
#if XE_PLATFORM_MAC
    if (!physical.contains(it->ifa_name) || !(it->ifa_flags & IFF_RUNNING)) {
      continue;
    }
#endif
    interfaces.push_back(
        {it->ifa_name,
         ntohl(reinterpret_cast<const sockaddr_in*>(it->ifa_addr)
                   ->sin_addr.s_addr),
         ntohl(reinterpret_cast<const sockaddr_in*>(it->ifa_broadaddr)
                   ->sin_addr.s_addr),
         if_nametoindex(it->ifa_name)});
  }
  freeifaddrs(list);
#endif
  std::sort(
      interfaces.begin(), interfaces.end(), [](const auto& a, const auto& b) {
        return a.index != b.index ? a.index < b.index : a.address < b.address;
      });
  return interfaces;
}
std::optional<LanInterface> ResolveLanInterface(std::string_view selector) {
  std::string preferred;
#if XE_PLATFORM_MAC
  if (SCDynamicStoreRef store = SCDynamicStoreCreate(
          nullptr, CFSTR("Xenia LAN interface selection"), nullptr, nullptr)) {
    if (CFPropertyListRef state = SCDynamicStoreCopyValue(
            store, CFSTR("State:/Network/Global/IPv4"))) {
      if (CFGetTypeID(state) == CFDictionaryGetTypeID()) {
        auto name = static_cast<CFStringRef>(
            CFDictionaryGetValue(static_cast<CFDictionaryRef>(state),
                                 kSCDynamicStorePropNetPrimaryInterface));
        char device[IFNAMSIZ]{};
        if (name && CFGetTypeID(name) == CFStringGetTypeID() &&
            CFStringGetCString(name, device, sizeof(device),
                               kCFStringEncodingUTF8)) {
          preferred = device;
        }
      }
      CFRelease(state);
    }
    CFRelease(store);
  }
#endif
  return SelectLanInterface(EnumerateLanInterfaces(), selector, preferred);
}
}  // namespace xe::net
