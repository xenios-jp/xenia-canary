xe_test_suite(xenia-net-tests ${CMAKE_CURRENT_SOURCE_DIR} LINKS xenia-net)
if(WIN32)
  set_source_files_properties(native_lan_test.cc PROPERTIES HEADER_FILE_ONLY TRUE)
endif()
target_compile_definitions(xenia-net-tests PRIVATE
  XE_XNET_SYNTHETIC_VECTORS="${CMAKE_CURRENT_SOURCE_DIR}/xnet_synthetic_vectors.json")
