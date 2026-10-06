#!/bin/bash
# Xenia : Xbox 360 Emulator Research Project
# Copyright 2026 Ben Vanik. All rights reserved.
# Released under the BSD license - see LICENSE in the root for more details.
set -euo pipefail
PATH=/usr/bin:/bin:/usr/sbin:/sbin
export PATH
[[ $(id -u) == 0 ]] || { echo "Administrator authorization required" >&2; exit 77; }
setup_dir='/Library/Application Support/Xenia-edge/PacketAccess'
installed_script="$setup_dir/packet-access.sh"
plist=/Library/LaunchDaemons/com.xenia.edge.packet-access.plist

apply_permissions() {
  # Opening an unbound BPF device creates the next devfs node on macOS. No
  # interface is attached and no packet is read. Bound the available pool.
  for ((index = 0; index < 256; ++index)); do
    node="/dev/bpf$index"
    /usr/bin/head -c 0 "$node" 2>/dev/null || true
    if [[ -c "$node" && ! -L "$node" ]]; then
      /usr/sbin/chown root:access_bpf "$node"
      /bin/chmod 660 "$node"
    fi
  done
}

case ${1:-} in
  --apply)
    [[ $# == 1 ]] || exit 64
    apply_permissions
    ;;
  --install)
    [[ $# == 2 && $2 =~ ^[0-9]+$ && $2 -ge 501 ]] || exit 64
    account=$(/usr/bin/id -nu "$2")
    [[ $(/usr/bin/id -u "$account") == "$2" ]] || exit 67
    # Refuse symlinked or user-writable installation parents before writing.
    for directory in /Library '/Library/Application Support' \
        '/Library/Application Support/Xenia-edge' "$setup_dir" /Library/LaunchDaemons; do
      if [[ -e "$directory" || -L "$directory" ]]; then
        [[ -d "$directory" && ! -L "$directory" ]] || exit 73
        [[ $(/usr/bin/stat -f '%u' "$directory") == 0 ]] || exit 73
        mode=$(/usr/bin/stat -f '%Lp' "$directory")
        (( (8#$mode & 0022) == 0 )) || exit 73
      else
        /usr/bin/install -d -o root -g wheel -m 755 "$directory"
      fi
    done
    /usr/sbin/dseditgroup -o read -n /Local/Default access_bpf >/dev/null 2>&1 || \
      /usr/sbin/dseditgroup -o create -n /Local/Default -r 'Packet capture access' access_bpf
    /usr/sbin/dseditgroup -o edit -n /Local/Default -a "$account" -t user access_bpf
    [[ ! -L "$installed_script" && ! -L "$plist" ]] || exit 73
    /usr/bin/install -o root -g wheel -m 755 "$0" "$installed_script"
    temporary=$(/usr/bin/mktemp /Library/LaunchDaemons/.xenia-packet-access.XXXXXXXX)
    trap '/bin/rm -f "$temporary"' EXIT
    /bin/cat > "$temporary" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.xenia.edge.packet-access</string>
  <key>ProgramArguments</key><array>
    <string>/bin/bash</string>
    <string>/Library/Application Support/Xenia-edge/PacketAccess/packet-access.sh</string>
    <string>--apply</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>ProcessType</key><string>Background</string>
</dict></plist>
PLIST
    /usr/bin/plutil -lint "$temporary" >/dev/null
    /usr/bin/install -o root -g wheel -m 644 "$temporary" "$plist"
    apply_permissions
    /bin/launchctl bootout system "$plist" 2>/dev/null || true
    /bin/launchctl bootstrap system "$plist"
    ;;
  --uninstall)
    [[ $# == 1 ]] || exit 64
    /bin/launchctl bootout system "$plist" 2>/dev/null || true
    /bin/rm -f "$plist" "$installed_script"
    # Preserve a separate Wireshark installation and the shared capture group.
    if [[ ! -e /Library/LaunchDaemons/org.wireshark.ChmodBPF.plist ]]; then
      for node in /dev/bpf*; do
        [[ -c "$node" && ! -L "$node" ]] || continue
        if [[ $(/usr/bin/stat -f '%Sg' "$node") == access_bpf ]]; then
          /usr/sbin/chown root:wheel "$node"
          /bin/chmod 600 "$node"
        fi
      done
    fi
    ;;
  *) exit 64 ;;
esac
