#ifndef TVM_TL_HEXAGON_LAYOUTS_H_
#define TVM_TL_HEXAGON_LAYOUTS_H_

#include "layout/layout.h"

namespace tvm {
namespace tl {
namespace hexagon {

enum class HexagonLayoutMode { kNone = 0, kAH = 1, kWH = 2, kRM = 3, kIdentityForTest = 100 };

struct HexagonLayoutSpec {
  HexagonLayoutMode mode;
  const char *name;
  Layout (*make)(const tirx::Buffer &buffer);
  bool (*detect)(const Layout &layout, const tirx::Buffer &buffer);
};

bool RegisterHexagonLayout(const HexagonLayoutSpec &spec);

Layout MakeHexagonLayout(HexagonLayoutMode mode, const tirx::Buffer &buffer);
HexagonLayoutMode DetectHexagonLayoutMode(const Layout &layout,
                                          const tirx::Buffer &buffer);
HexagonLayoutMode HexagonLayoutModeFromString(const ffi::String &mode);
ffi::String HexagonLayoutModeToString(HexagonLayoutMode mode);

Layout MakeHexagonAHLayout(ffi::Array<PrimExpr> shape);
Layout MakeHexagonAHLayout(const tirx::Buffer &buffer);
Layout MakeHexagonWHLayout(const tirx::Buffer &buffer);
Layout MakeHexagonRMLayout(const tirx::Buffer &buffer);

HexagonLayoutMode DetectHexagonAHMode(const Layout &layout,
                                      const tirx::Buffer &buffer);
HexagonLayoutMode DetectHexagonWHMode(const Layout &layout,
                                       const tirx::Buffer &buffer);
HexagonLayoutMode DetectHexagonRMMode(const Layout &layout,
                                      const tirx::Buffer &buffer);

} // namespace hexagon
} // namespace tl
} // namespace tvm

#endif // TVM_TL_HEXAGON_LAYOUTS_H_
