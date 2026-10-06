/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/net/mac/packet_access.h"

#import <Foundation/Foundation.h>
#include <grp.h>
#include <membership.h>
#include <sys/stat.h>
#include <unistd.h>
#include <uuid/uuid.h>

namespace xe::net {
namespace {
bool RootOwnedFile(const char* path) {
  struct stat attributes{};
  return !lstat(path, &attributes) && S_ISREG(attributes.st_mode) && attributes.st_uid == 0 &&
         !(attributes.st_mode & 0022);
}
bool Installed() {
  bool job =
      (RootOwnedFile("/Library/LaunchDaemons/com.xenia.edge.packet-access.plist") &&
       RootOwnedFile("/Library/Application Support/Xenia-edge/PacketAccess/packet-access.sh")) ||
      RootOwnedFile("/Library/LaunchDaemons/org.wireshark.ChmodBPF.plist");
  if (!job) {
    return false;
  }
  auto* group = getgrnam("access_bpf");
  if (!group) {
    return false;
  }
  uuid_t user_uuid{}, group_uuid{};
  int member = 0;
  return !mbr_uid_to_uuid(getuid(), user_uuid) && !mbr_gid_to_uuid(group->gr_gid, group_uuid) &&
         !mbr_check_membership(user_uuid, group_uuid, &member) && member;
}
NSString* AppleScriptString(NSString* value) {
  NSMutableString* quoted = [NSMutableString stringWithString:@"\""];
  for (NSUInteger index = 0; index < value.length; ++index) {
    unichar c = [value characterAtIndex:index];
    if (c == '\\' || c == '\"') {
      [quoted appendString:@"\\"];
    }
    [quoted appendFormat:@"%C", c];
  }
  [quoted appendString:@"\""];
  return quoted;
}
}  // namespace
bool EnsurePacketAccess(std::string& error) {
  @autoreleasepool {
    error.clear();
    if (geteuid() == 0) {
      error = "Run Edge as your normal Mac user to enable console networking.";
      return false;
    }
    if (Installed()) {
      return true;
    }
    NSString* path = [[NSBundle mainBundle] pathForResource:@"packet-access" ofType:@"sh"];
    if (!path) {
      error = "This build is missing its console-networking setup resource.";
      return false;
    }
    // AppleScript's quoted form protects the shell argument independently of
    // AppleScript source quoting. The numeric UID is the sole setup argument.
    NSString* source = [NSString
        stringWithFormat:
            @"do shell script (\"/usr/bin/env -i PATH=/usr/bin:/bin:/usr/sbin:/sbin "
            @"/bin/bash \" & quoted form of %@ & \" "
            @"--install %u\") with prompt \"Enable local multiplayer networking for Edge. "
            @"This one-time setup grants your Mac account access to network packets. "
            @"App updates will not require setup again.\" with administrator privileges",
            AppleScriptString(path), getuid()];
    NSDictionary* reason = nil;
    NSAppleScript* script = [[NSAppleScript alloc] initWithSource:source];
    if (![script executeAndReturnError:&reason]) {
      if ([reason[NSAppleScriptErrorNumber] integerValue] != -128) {
        NSString* message = reason[NSAppleScriptErrorMessage];
        error = message ? message.UTF8String : "Console-networking setup failed.";
      }
      return false;
    }
    if (!Installed()) {
      error = "Console-networking access could not be verified. Restart Edge and try again.";
      return false;
    }
    return true;
  }
}
}  // namespace xe::net
