/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2024 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include "xenia/gpu/dxc_compiler.h"

#include "xenia/base/logging.h"
#include "xenia/base/platform.h"

#if XE_PLATFORM_WIN32
#include "xenia/base/platform_win.h"
#else
#include <dlfcn.h>
#endif

// Cross-platform DXC API. On non-Windows this pulls in DXC's WinAdapter COM
// shims (HRESULT, __uuidof, IID_PPV_ARGS, WCHAR, ...). Included last so its
// Windows-compatibility macros do not leak into the xenia headers above.
#include "dxc/dxcapi.h"

namespace xe {
namespace gpu {

DxcCompiler::~DxcCompiler() {
  if (utils_) {
    utils_->Release();
    utils_ = nullptr;
  }
  if (compiler_) {
    compiler_->Release();
    compiler_ = nullptr;
  }
  if (library_) {
#if XE_PLATFORM_WIN32
    FreeLibrary(reinterpret_cast<HMODULE>(library_));
#else
    dlclose(library_);
#endif
    library_ = nullptr;
  }
}

bool DxcCompiler::Initialize() {
  // Load the DXC compiler library at runtime and resolve DxcCreateInstance, so
  // the build does not depend on the library being present at link time.
  DxcCreateInstanceProc dxc_create_instance = nullptr;
#if XE_PLATFORM_WIN32
  library_ = reinterpret_cast<void*>(LoadLibraryW(L"dxcompiler.dll"));
  if (library_) {
    dxc_create_instance =
        reinterpret_cast<DxcCreateInstanceProc>(GetProcAddress(
            reinterpret_cast<HMODULE>(library_), "DxcCreateInstance"));
  }
#else
  // Candidate locations. dlopen interprets @executable_path/@loader_path/@rpath
  // and falls back to the default dyld search paths for a bare leaf name
  // (DYLD_LIBRARY_PATH, embedded rpaths, /usr/local/lib, ...).
  static const char* const kCandidates[] = {
      "libdxcompiler.dylib",
      "@executable_path/libdxcompiler.dylib",
      "@loader_path/libdxcompiler.dylib",
      "@executable_path/../Frameworks/libdxcompiler.dylib",
      "/opt/homebrew/lib/libdxcompiler.dylib",
      "/usr/local/lib/libdxcompiler.dylib",
  };
  for (const char* candidate : kCandidates) {
    library_ = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
    if (library_) {
      break;
    }
  }
  if (library_) {
    dxc_create_instance = reinterpret_cast<DxcCreateInstanceProc>(
        dlsym(library_, "DxcCreateInstance"));
  }
#endif

  if (!dxc_create_instance) {
    XELOGI(
        "DxcCompiler: dxcompiler library not found (or missing "
        "DxcCreateInstance); HLSL->DXIL compilation unavailable, falling back");
    return false;
  }

  // IDxcCompiler3 requires DXC version 1.5+ (released ~2020).
  HRESULT hr = dxc_create_instance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler_));
  if (FAILED(hr)) {
    XELOGE("DxcCompiler: Failed to create IDxcCompiler3 (HRESULT {:08X})",
           static_cast<uint32_t>(hr));
    return false;
  }

  hr = dxc_create_instance(CLSID_DxcUtils, IID_PPV_ARGS(&utils_));
  if (FAILED(hr)) {
    XELOGE("DxcCompiler: Failed to create IDxcUtils (HRESULT {:08X})",
           static_cast<uint32_t>(hr));
    compiler_->Release();
    compiler_ = nullptr;
    return false;
  }

  XELOGI("DxcCompiler: DXIL shader compilation available");
  return true;
}

bool DxcCompiler::Compile(const std::string& hlsl_source,
                          const std::string& entry_point,
                          const std::string& target,
                          std::vector<uint8_t>& dxil_out,
                          std::string* error_message) {
  if (!IsAvailable()) {
    if (error_message) {
      *error_message = "DXC compiler not available";
    }
    return false;
  }

  // Convert entry point and target to wide strings (WCHAR == wchar_t here).
  std::wstring entry_point_wide(entry_point.begin(), entry_point.end());
  std::wstring target_wide(target.begin(), target.end());

  // Set up compilation arguments.
  // Disable optimizations and keep debug info for better debugging in RenderDoc
  // / the Xcode GPU trace during bring-up.
  std::vector<LPCWSTR> arguments = {
      L"-E",
      entry_point_wide.c_str(),
      L"-T",
      target_wide.c_str(),
      L"-Od",  // Disable optimizations for debugging
      L"-Zi",  // Enable debug info
      DXC_ARG_WARNINGS_ARE_ERRORS,
  };

  // Create source buffer.
  DxcBuffer source_buffer;
  source_buffer.Ptr = hlsl_source.c_str();
  source_buffer.Size = hlsl_source.size();
  source_buffer.Encoding = DXC_CP_UTF8;

  // Compile the shader.
  IDxcResult* result = nullptr;
  HRESULT hr = compiler_->Compile(&source_buffer, arguments.data(),
                                  static_cast<UINT32>(arguments.size()),
                                  nullptr,  // No include handler
                                  IID_PPV_ARGS(&result));

  if (FAILED(hr)) {
    if (error_message) {
      *error_message = "DXC Compile call failed";
    }
    return false;
  }

  // Check compilation status.
  HRESULT status;
  result->GetStatus(&status);

  if (FAILED(status)) {
    // Retrieve error messages.
    IDxcBlobUtf8* errors = nullptr;
    if (SUCCEEDED(result->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors),
                                    nullptr)) &&
        errors && errors->GetStringLength() > 0) {
      if (error_message) {
        *error_message = errors->GetStringPointer();
      }
      errors->Release();
    } else {
      if (error_message) {
        *error_message = "Unknown compilation error";
      }
    }
    result->Release();
    return false;
  }

  // Retrieve the compiled shader object.
  IDxcBlob* shader_blob = nullptr;
  if (FAILED(result->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&shader_blob),
                               nullptr)) ||
      !shader_blob) {
    if (error_message) {
      *error_message = "Failed to retrieve compiled shader";
    }
    result->Release();
    return false;
  }

  // Copy the DXIL bytecode to the output vector.
  const uint8_t* shader_data =
      static_cast<const uint8_t*>(shader_blob->GetBufferPointer());
  size_t shader_size = shader_blob->GetBufferSize();
  dxil_out.assign(shader_data, shader_data + shader_size);

  shader_blob->Release();
  result->Release();

  return true;
}

bool DxcCompiler::Disassemble(const std::vector<uint8_t>& dxil,
                              std::string& disassembly_out) {
  if (!IsAvailable()) {
    return false;
  }

  DxcBuffer dxil_buffer;
  dxil_buffer.Ptr = dxil.data();
  dxil_buffer.Size = dxil.size();
  dxil_buffer.Encoding = DXC_CP_ACP;  // Binary data

  IDxcResult* result = nullptr;
  HRESULT hr = compiler_->Disassemble(&dxil_buffer, IID_PPV_ARGS(&result));

  if (FAILED(hr)) {
    return false;
  }

  HRESULT status;
  result->GetStatus(&status);

  if (FAILED(status)) {
    result->Release();
    return false;
  }

  IDxcBlobUtf8* disasm_blob = nullptr;
  if (FAILED(result->GetOutput(DXC_OUT_DISASSEMBLY, IID_PPV_ARGS(&disasm_blob),
                               nullptr)) ||
      !disasm_blob) {
    result->Release();
    return false;
  }

  disassembly_out = disasm_blob->GetStringPointer();

  disasm_blob->Release();
  result->Release();

  return true;
}

}  // namespace gpu
}  // namespace xe
