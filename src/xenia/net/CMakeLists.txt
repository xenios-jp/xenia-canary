add_library(xenia-net STATIC
  lan.cc lan_interface.cc sha1.cc xnet_secure.cc
  xnet_association.cc xnet_dh.cc xnet_frame.cc
  ${PROJECT_SOURCE_DIR}/third_party/crypto/des/des.cpp
)
target_link_libraries(xenia-net PUBLIC xenia-base)
xe_target_defaults(xenia-net)
if(WIN32)
  target_sources(xenia-net PRIVATE native_lan_unsupported.cc)
else()
  target_sources(xenia-net PRIVATE native_lan.cc)
endif()
if(APPLE)
  target_sources(xenia-net PRIVATE packet_transport_mac.cc mac/packet_access.mm)
  set_source_files_properties(mac/packet_access.mm
    PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
  target_link_libraries(xenia-net PUBLIC
    pcap "-framework Foundation" "-framework SystemConfiguration")
endif()

if(XENIA_BUILD_TESTS)
  add_subdirectory(testing)
endif()
