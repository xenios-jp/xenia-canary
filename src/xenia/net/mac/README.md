# macOS console networking setup

The app uses its existing unprivileged libpcap backend. At startup, when native
LAN is enabled, it checks for installed packet-access setup and membership of
the current user in the dedicated `access_bpf` group. Existing Wireshark
ChmodBPF installations are supported. Setup is independent of app signatures,
versions, install locations, and daily updates.

When setup is absent, the app invokes its bundled setup resource through
AppleScript's `do shell script … with administrator privileges`. macOS handles
the password dialog, with networking-specific explanatory text; the app neither
receives nor stores a password. The shell argument is quoted separately from
the AppleScript string, and the command uses a clean environment. Canceling
continues startup without installing anything or enabling native packet access.

After approval, setup adds only the launching user's account to `access_bpf`
and installs these root-owned, unprivileged-user-unwritable files:

- `/Library/Application Support/Xenia-edge/PacketAccess/packet-access.sh`
- `/Library/LaunchDaemons/com.xenia.edge.packet-access.plist`

A bounded pool of BPF devices is assigned to `root:access_bpf`, mode `0660`.
The launch job runs once at boot to restore devfs permissions and exits. There
is no KeepAlive job, running relay, runtime privileged IPC, or app-build pin.
The emulator never runs as root. Group members can use packet devices from
other programs too; these permissions are account-based, not app-exclusive.
Additional accounts need their own approved setup.

The installed script provides an administrator-only `--uninstall` action. It
unregisters this job, removes this setup's two files, and restores BPF device
permissions unless a separate Wireshark ChmodBPF installation is present. The
shared capture group is preserved to avoid altering other applications' setup.

Before upstream adoption, verify approval, cancellation, normal-user packet
access immediately after setup, a second launch with no prompt, boot-time
restoration, daily app replacement, and uninstall. Build success and synthetic
protocol tests do not establish these host-integration behaviors.
