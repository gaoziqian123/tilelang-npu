# Hexagon backend: pure C++ target-specific TileOp registrations.

file(GLOB TILE_LANG_HEXAGON_SRCS
  src/hexagon/op/*.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_HEXAGON_SRCS})
