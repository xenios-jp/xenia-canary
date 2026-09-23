xe_test_suite(xenia-gpu-tests ${CMAKE_CURRENT_SOURCE_DIR}
  LINKS
    capstone fmt glslang-spirv imgui
    xenia-apu xenia-base xenia-core xenia-cpu xenia-gpu
    xenia-hid-portal xenia-kernel xenia-ui xenia-patcher
)

if(XE_TARGET_X86_64 AND TARGET xenia-gpu-tests)
  target_link_libraries(xenia-gpu-tests PRIVATE xenia-cpu-backend-x64)
elseif(XE_TARGET_AARCH64 AND TARGET xenia-gpu-tests)
  target_link_libraries(xenia-gpu-tests PRIVATE xenia-cpu-backend-a64)
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND TARGET xenia-gpu-tests)
  target_link_libraries(xenia-gpu-tests PRIVATE xenia-apu-sdl)
endif()
