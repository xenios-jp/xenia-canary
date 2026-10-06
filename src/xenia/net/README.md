The native LAN path is opt-in with `--xnet_native_lan=true`. Leaving
`xnet_interface` empty automatically selects an active physical IPv4 broadcast
interface, preferring a configured LAN over a self-assigned direct link and
favoring the primary adapter within that class;
`--xnet_interface=<interface name or local IPv4 address>` is an optional override.
Peers are discovered from native broadcasts; no console IP address is required.

The networking layer uses Xenia's byte-order types, clocks, logging, existing
crypto dependencies, and platform key service. Executable credentials and
compatibility metadata come from the loaded guest at runtime.

The macOS transport uses the system packet-capture library and requires
administrator-granted capture access. The app provides [one-time setup](mac/README.md)
when this access is missing. Immediate delivery avoids capture
buffering. The transport does not elevate the emulator or change device
permissions. Frame checks confine traffic to the selected MAC identity, physical
MTU, native wire envelope, and supported secure record geometry.

The tests use synthetic credentials, documentation-range endpoints, and local
in-memory transports.

This implementation supports discovery and UDP/VDP datagram associations on
macOS. Connected UDP send/receive and native QoS are not implemented. Interface
selection is automatic, but the selected adapter remains fixed for the guest
session; discovery across independent LANs and live adapter migration are not
implemented.

## Sources

The LAN registry's guest address ABI, platform identity, key categories,
address mapping, and public endpoint routing
follow the BSD-licensed [Xenia Netplay fork](https://github.com/AdrianCassar/xenia-canary/tree/6dbaa1fefd1e07cc3d5377e763c68fe8073cbe8c):
`src/xenia/kernel/xnet.h`, `src/xenia/kernel/xam/xam_net.cc`, and
`src/xenia/kernel/xsocket.cc`. The inherited Xenia copyright notice is retained
in `lan.h` and `lan.cc`.

The native secure transport was independently implemented from a
functional protocol specification based on console system-software analysis and
LAN interoperability observations. The specification describes
wire layouts, authentication, key derivation, and association behavior; retail
implementation code and captured packet fixtures are not included. Local retry,
timeout, and initial sequence policies are identified in `AssociationConfig`.

HMAC-SHA1 follows [RFC 2104](https://www.rfc-editor.org/rfc/rfc2104), using Xenia's
existing TinySHA1 dependency. Key agreement uses the public group in
[RFC 2409 section 6.1](https://www.rfc-editor.org/rfc/rfc2409#section-6.1).
The DES implementation is Xenia's existing `third_party/crypto/des` dependency;
its original notices remain with that source.

`xnet_vectors_test.cc` checks encoding, authentication, exchange binding, and
replay behavior against synthetic inputs in `xnet_synthetic_vectors.json`.
`xnet_association_test.cc` and `native_lan_test.cc` exercise peer lifecycle and
guest socket behavior using in-memory packet transports. These tests reproduce
the functional contracts without console binaries, real session keys, or packet
captures. Native QoS and the remaining interface limitations are listed above.
