/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2024 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#ifndef XENIA_GPU_DXC_COMPILER_H_
#define XENIA_GPU_DXC_COMPILER_H_

#include <cstdint>
#include <string>
#include <vector>

// Forward-declare the DXC COM interfaces so that dxcapi.h (and its WinAdapter
// COM shims on non-Windows) only need to be included by dxc_compiler.cc.
struct IDxcCompiler3;
struct IDxcUtils;

namespace xe {
namespace gpu {

// Backend-agnostic wrapper around the DirectX Shader Compiler (DXC) for
// compiling HLSL to DXIL, enabling SM 6.x features such as barycentric
// interpolation and ResourceDescriptorHeap.
//
// The DXC compiler library (dxcompiler.dll / libdxcompiler.dylib) is loaded at
// runtime via LoadLibrary/dlopen, so the build has no link-time dependency on
// it. If the library cannot be loaded, IsAvailable() returns false and callers
// fall back to their existing shader path.
class DxcCompiler {
 public:
  DxcCompiler() = default;
  ~DxcCompiler();

  DxcCompiler(const DxcCompiler&) = delete;
  DxcCompiler& operator=(const DxcCompiler&) = delete;

  // Loads the DXC compiler library and creates the compiler/utils interfaces.
  // Returns true on success.
  bool Initialize();

  // Returns true if DXC compilation is available.
  bool IsAvailable() const { return compiler_ != nullptr && utils_ != nullptr; }

  // Compile HLSL source code to DXIL bytecode.
  // target: shader target profile, e.g. "vs_6_0", "ps_6_0", "gs_6_0", "ds_6_0".
  // Returns true on success, with DXIL bytecode in dxil_out. On failure,
  // error_message (if provided) contains the error description.
  bool Compile(const std::string& hlsl_source, const std::string& entry_point,
               const std::string& target, std::vector<uint8_t>& dxil_out,
               std::string* error_message = nullptr);

  // Get disassembly of DXIL bytecode for debugging.
  bool Disassemble(const std::vector<uint8_t>& dxil,
                   std::string& disassembly_out);

 private:
  // dlopen / LoadLibrary handle for the DXC compiler library.
  void* library_ = nullptr;
  IDxcCompiler3* compiler_ = nullptr;
  IDxcUtils* utils_ = nullptr;
};

}  // namespace gpu
}  // namespace xe

#endif  // XENIA_GPU_DXC_COMPILER_H_
