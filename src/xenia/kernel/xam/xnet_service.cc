/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/kernel/xam/xnet_service.h"
#include "xenia/base/clock.h"
#include "xenia/base/cvar.h"
#include "xenia/base/logging.h"
#include "xenia/kernel/user_module.h"
#include "xenia/kernel/xam/xam_module.h"
#include "xenia/net/lan_interface.h"
#include "xenia/net/native_lan.h"

DEFINE_bool(xnet_native_lan, false, "Use native console system-link transport.",
            "Network");
DEFINE_string(xnet_interface, "",
              "Native system-link interface or local IPv4 address.", "Network");

namespace xe::kernel::xboxkrnl {
bool NativePlatformHmac(std::span<const uint8_t>, net::crypto::Sha1Digest&);
}
namespace xe::kernel::xam {
bool NativeLanEnabled() { return cvars::xnet_native_lan; }
std::shared_ptr<net::Lan> LocalLan(KernelState* state) {
  if (!NativeLanEnabled()) {
    return {};
  }
  return state->GetKernelModule<XamModule>("xam.xex")->local_lan();
}
std::shared_ptr<net::Lan> XamModule::local_lan() {
  std::lock_guard lock(lan_mutex_);
  if (lan_) {
    return lan_;
  }
#if XE_PLATFORM_MAC
  const auto interface = net::ResolveLanInterface(cvars::xnet_interface);
  auto module = kernel_state_->GetExecutableModule();
  xex2_opt_lan_key* key = nullptr;
  xex2_opt_execution_info* execution = nullptr;
  if (!interface || !module ||
      !module->xex_module()->GetOptHeader(XEX_HEADER_LAN_KEY, &key) || !key ||
      !module->xex_module()->GetOptHeader(XEX_HEADER_EXECUTION_INFO,
                                          &execution) ||
      !execution) {
    XELOGE(
        "Native LAN requires an interface and executable LAN/execution "
        "headers");
    return {};
  }
  net::Key executable_key;
  std::copy_n(key->key, executable_key.size(), executable_key.begin());
  auto broadcast = net::secure::DeriveBroadcast(executable_key,
                                                xboxkrnl::NativePlatformHmac);
  if (!broadcast) {
    XELOGE("Native LAN platform authentication unavailable");
    return {};
  }
  std::string error;
  std::shared_ptr<net::secure::PacketTransport> transport =
      net::secure::OpenPacketInterface(interface->name, error);
  if (!transport) {
    XELOGE("Native LAN: {}", error);
    return {};
  }
  XELOGI("Native LAN using interface {}", interface->name);
  net::LanConfig config;
  config.native_secure = true;
  config.address = interface->address;
  config.real_interface = true;
  config.interface_index = interface->index;
  config.broadcast_address = interface->broadcast;
  config.native_mac = transport->identity();
  net::secure::AssociationConfig association;
  association.broadcast = *broadcast;
  association.freshness_time = [] { return Clock::QueryGuestSystemTime(); };
  auto* version = export_resolver_->GetExportByOrdinal("xboxkrnl.exe", 0x0158);
  if (!version || !version->variable_ptr) {
    return {};
  }
  auto* bytes =
      kernel_state_->memory()->TranslateVirtual(version->variable_ptr);
  association.metadata = net::secure::MakeMetadata(
      execution->title_id, execution->version_value,
      xe::load_and_swap<uint16_t>(bytes),
      xe::load_and_swap<uint16_t>(bytes + 2),
      xe::load_and_swap<uint16_t>(bytes + 4), bytes[6]);
  association.mtu = transport->mtu();
  config.native_factory = [transport, association](net::Lan& registry) mutable {
    std::memcpy(association.identity.data(), &registry.identity(),
                association.identity.size());
    association.credential =
        [&registry](const net::KeyId& id) -> std::optional<net::Key> {
      net::Key key;
      return registry.GetKey(id, key) ? std::optional(key) : std::nullopt;
    };
    return std::make_unique<net::secure::NativeLan>(registry, transport,
                                                    association);
  };
  lan_ = std::make_shared<net::Lan>(std::move(config));
#endif
  return lan_;
}
}  // namespace xe::kernel::xam
