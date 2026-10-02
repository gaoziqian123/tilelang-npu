# Hexagon backend: pure C++ target-specific TileOp registrations.

file(GLOB TILE_LANG_HEXAGON_ALWAYS_SRCS CONFIGURE_DEPENDS
  src/hexagon/*.cc
  src/hexagon/op/*.cc
  src/hexagon/codegen/intrin_rule_hexagon.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_HEXAGON_ALWAYS_SRCS})

# The root includes TVM's config.cmake, which shadows command-line CACHE
# USE_HEXAGON with a normal OFF variable. Read the user's cache value without
# enabling TVM's DSP runtime in the subsequent add_subdirectory(tvm).
if(DEFINED CACHE{USE_HEXAGON})
  set(TILE_LANG_HEXAGON_SDK_ENABLED "$CACHE{USE_HEXAGON}")
else()
  set(TILE_LANG_HEXAGON_SDK_ENABLED OFF)
endif()
if(TILE_LANG_HEXAGON_SDK_ENABLED)
  set(TILELANG_HEXAGON_SDK "$ENV{HEXAGON_SDK}" CACHE PATH "Hexagon SDK for offline compilation")
  file(GLOB _TL_HEXAGON_COMPILERS
    "${TILELANG_HEXAGON_SDK}/tools/HEXAGON_Tools/*/Tools/bin/hexagon-clang")
  if(NOT _TL_HEXAGON_COMPILERS)
    message(FATAL_ERROR "USE_HEXAGON=ON requires TILELANG_HEXAGON_SDK (or HEXAGON_SDK) containing hexagon-clang")
  endif()
  message(STATUS "Hexagon SDK enabled: ${TILELANG_HEXAGON_SDK}; host source generator only (no DSP linkage)")
else()
  message(STATUS "Hexagon SDK disabled; building host-only source generator")
endif()
# Source generation never links DSP libraries: it is the host-only fallback
# even without the SDK. Compilation/loading is delegated to the callback.
set(TILE_LANG_HEXAGON_ACTIVE_SRCS
  src/hexagon/codegen/codegen_hexagon.cc
  src/hexagon/codegen/rt_mod_hexagon.cc)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_HEXAGON_ACTIVE_SRCS})
