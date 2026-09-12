/*!
 * \file tl/hexagon/layouts.cc
 * \brief Hexagon HMX AH/WH tile layouts.
 */

#include "hexagon/layouts.h"

#include "support/check.h"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op.h>

#include <algorithm>
#include <vector>

namespace tvm {
namespace tl {
namespace hexagon {

using namespace ffi;
using namespace tirx;

namespace {

struct HexagonTileShapeInfo {
  int64_t rows;
  int64_t cols;
};

HexagonTileShapeInfo GetHexagonTileShapeInfoChecked(Array<PrimExpr> shape) {
  ICHECK(shape.size() >= 2) << "Hexagon AH/WH layout expects rank >= 2 shape";
  size_t ndim = shape.size();
  auto rows = as_const_int(shape[ndim - 2]);
  auto cols = as_const_int(shape[ndim - 1]);
  ICHECK(rows && cols) << "Hexagon AH/WH layout requires constant last-2 dims";
  ICHECK(*rows % 32 == 0) << "Hexagon AH/WH row extent must be multiple of 32, got " << *rows;
  ICHECK(*cols % 32 == 0) << "Hexagon AH/WH column extent must be multiple of 32, got " << *cols;
  return {*rows, *cols};
}

bool TryGetHexagonTileShapeInfo(const Buffer &buffer,
                                HexagonTileShapeInfo *info) {
  if (!buffer.defined() || buffer->shape.size() < 2) {
    return false;
  }
  size_t ndim = buffer->shape.size();
  auto rows = as_const_int(buffer->shape[ndim - 2]);
  auto cols = as_const_int(buffer->shape[ndim - 1]);
  if (!rows || !cols || *rows % 32 != 0 || *cols % 32 != 0) {
    return false;
  }
  *info = {*rows, *cols};
  return true;
}

Layout ExpandLayout2D(const Layout &base, Array<PrimExpr> shape) {
  Array<PrimExpr> leading_shape;
  leading_shape.reserve(shape.size() - 2);
  for (size_t i = 0; i + 2 < shape.size(); ++i) {
    leading_shape.push_back(shape[i]);
  }
  return base->Expand(leading_shape);
}

std::vector<HexagonLayoutSpec> *HexagonLayoutRegistry() {
  static auto *registry = new std::vector<HexagonLayoutSpec>();
  return registry;
}

const HexagonLayoutSpec *FindHexagonLayoutSpec(HexagonLayoutMode mode) {
  auto *registry = HexagonLayoutRegistry();
  auto it = std::find_if(registry->begin(), registry->end(),
                         [mode](const HexagonLayoutSpec &spec) {
                           return spec.mode == mode;
                         });
  return it == registry->end() ? nullptr : &*it;
}

Layout MakeHexagonZip16Layout2D(int64_t rows, int64_t cols) {
  // HMX AH stores a row-major matrix as 32x32 fp16 tiles.  Inside each tile
  // the storage is 16 HVX vectors of 64 fp16 lanes.  Vector `rp` contains source
  // rows `2*rp` and `2*rp+1`; `Q6_Vh_vshuff_Vh` maps a lane as
  //   lane = 32 * (col % 2) + row_in_pair + 2 * floor((col % 32) / 2)
  // which is exactly the zip16 row-pair interleave used by hrt_stage_act_hvx.
  Var i = InputPlaceholder(0);
  Var j = InputPlaceholder(1);

  PrimExpr row_tile = FloorDiv(i, 32);
  PrimExpr col_tile = FloorDiv(j, 32);
  PrimExpr row = FloorMod(i, 32);
  PrimExpr col = FloorMod(j, 32);
  PrimExpr row_pair = FloorDiv(row, 2);
  PrimExpr row_in_pair = FloorMod(row, 2);
  PrimExpr lane = FloorMod(col, 2) * 32 + row_in_pair + FloorDiv(col, 2) * 2;

  return Layout(Array<PrimExpr>{Integer(rows), Integer(cols)},
                {row_tile, col_tile, row_pair, lane});
}

Layout MakeHexagonWHTileLayout2D(int64_t rows, int64_t cols) {
  // HMX WH stores the logical weight matrix W[K,N] as 32x32 fp16 tiles in
  // cb-major (N-tile outer, K-tile inner) order.  Buffers carry the logical
  // [N,K] shape (the gemm B operand as declared), so with i = n, j = k:
  //   tile      = (i/32) * (K/32) + (j/32)        (cb-major tile order)
  //   in-tile   = ((j%32)/2) * 64 + (i%32)*2 + (j%2)
  // which matches wh_convert_t (runtime/wh_convert.h) exactly.
  Var i = InputPlaceholder(0);
  Var j = InputPlaceholder(1);

  PrimExpr n_tile = FloorDiv(i, 32);
  PrimExpr k_tile = FloorDiv(j, 32);
  PrimExpr k_pair = FloorDiv(FloorMod(j, 32), 2);
  PrimExpr lane = FloorMod(i, 32) * 2 + FloorMod(j, 2);

  return Layout(Array<PrimExpr>{Integer(rows), Integer(cols)},
                {n_tile, k_tile, k_pair, lane});
}

} // namespace

bool RegisterHexagonLayout(const HexagonLayoutSpec &spec) {
  ICHECK(spec.mode != HexagonLayoutMode::kNone)
      << "Cannot register Hexagon NONE layout";
  ICHECK(spec.name != nullptr) << "Hexagon layout spec requires a name";
  ICHECK(spec.make != nullptr) << "Hexagon layout spec requires a constructor";
  ICHECK(spec.detect != nullptr) << "Hexagon layout spec requires a detector";
  auto *registry = HexagonLayoutRegistry();
  auto it = std::find_if(registry->begin(), registry->end(),
                         [&](const HexagonLayoutSpec &other) {
                           return other.mode == spec.mode;
                         });
  ICHECK(it == registry->end())
      << "Duplicate Hexagon layout registration for " << spec.name;
  registry->push_back(spec);
  return true;
}

Layout MakeHexagonLayout(HexagonLayoutMode mode, const Buffer &buffer) {
  ICHECK(mode != HexagonLayoutMode::kNone)
      << "Cannot construct Hexagon NONE layout";
  const HexagonLayoutSpec *spec = FindHexagonLayoutSpec(mode);
  ICHECK(spec != nullptr) << "Unregistered Hexagon layout mode "
                          << static_cast<int>(mode);
  return spec->make(buffer);
}

HexagonLayoutMode DetectHexagonLayoutMode(const Layout &layout,
                                          const Buffer &buffer) {
  for (const auto &spec : *HexagonLayoutRegistry()) {
    if (spec.detect(layout, buffer)) {
      return spec.mode;
    }
  }
  return HexagonLayoutMode::kNone;
}

HexagonLayoutMode HexagonLayoutModeFromString(const String &mode) {
  std::string m = mode;
  if (m == "none" || m == "NONE") {
    return HexagonLayoutMode::kNone;
  }
  if (m == "rm" || m == "RM") {
    return HexagonLayoutMode::kRM;
  }
  if (m == "ah" || m == "AH") {
    return HexagonLayoutMode::kAH;
  }
  if (m == "wh" || m == "WH") {
    return HexagonLayoutMode::kWH;
  }
  if (m == "identity_for_test" || m == "IDENTITY_FOR_TEST") {
    return HexagonLayoutMode::kIdentityForTest;
  }
  LOG(FATAL) << "Unknown Hexagon layout mode: " << m;
  return HexagonLayoutMode::kNone;
}

String HexagonLayoutModeToString(HexagonLayoutMode mode) {
  switch (mode) {
  case HexagonLayoutMode::kNone:
    return String("none");
  case HexagonLayoutMode::kAH:
    return String("ah");
  case HexagonLayoutMode::kWH:
    return String("wh");
  case HexagonLayoutMode::kRM:
    return String("rm");
  case HexagonLayoutMode::kIdentityForTest:
    return String("identity_for_test");
  }
  LOG(FATAL) << "Unknown Hexagon layout mode ordinal: "
             << static_cast<int>(mode);
  return String("none");
}

Layout MakeHexagonAHLayout(Array<PrimExpr> shape) {
  auto info = GetHexagonTileShapeInfoChecked(shape);
  return ExpandLayout2D(MakeHexagonZip16Layout2D(info.rows, info.cols), shape);
}

Layout MakeHexagonAHLayout(const Buffer &buffer) {
  ICHECK(buffer.defined()) << "Hexagon AH layout expects a defined buffer";
  ICHECK(buffer->dtype.bits() == 16) << "Hexagon AH layout expects fp16/u16/i16 elements, got " << buffer->dtype;
  return MakeHexagonAHLayout(buffer->shape);
}

Layout MakeHexagonWHLayout(const Buffer &buffer) {
  ICHECK(buffer.defined()) << "Hexagon WH layout expects a defined buffer";
  ICHECK(buffer->dtype.bits() == 16) << "Hexagon WH layout expects fp16/u16/i16 elements, got " << buffer->dtype;
  // WH differs from AH both in-tile (lane = 2*(n%32) + k%2, k-pair vectors)
  // and in tile order (cb-major).  Trailing 2-D is the logical [N, K].
  auto info = GetHexagonTileShapeInfoChecked(buffer->shape);
  return ExpandLayout2D(MakeHexagonWHTileLayout2D(info.rows, info.cols), buffer->shape);
}

Layout MakeHexagonRMLayout(const Buffer &buffer) {
  ICHECK(buffer.defined()) << "Hexagon RM layout expects a defined buffer";
  Array<PrimExpr> fwds;
  fwds.reserve(buffer->shape.size());
  for (size_t i = 0; i < buffer->shape.size(); ++i) {
    fwds.push_back(InputPlaceholder(static_cast<int>(i)));
  }
  return Layout(buffer->shape, fwds);
}

namespace {

bool DetectHexagonAHLayout(const Layout &layout, const Buffer &buffer) {
  HexagonTileShapeInfo info;
  if (!TryGetHexagonTileShapeInfo(buffer, &info) || buffer->dtype.bits() != 16) {
    return false;
  }
  return StructuralEqual()(layout, MakeHexagonAHLayout(buffer));
}

bool DetectHexagonWHLayout(const Layout &layout, const Buffer &buffer) {
  HexagonTileShapeInfo info;
  if (!TryGetHexagonTileShapeInfo(buffer, &info) || buffer->dtype.bits() != 16) {
    return false;
  }
  return StructuralEqual()(layout, MakeHexagonWHLayout(buffer));
}

bool DetectHexagonRMLayout(const Layout &layout, const Buffer &buffer) {
  if (!buffer.defined()) {
    return false;
  }
  return StructuralEqual()(layout, MakeHexagonRMLayout(buffer));
}

Layout MakeHexagonIdentityLayoutForTest(const Buffer &buffer) {
  ICHECK(buffer.defined()) << "Hexagon test identity layout expects a defined buffer";
  return MakeLinearLayout(buffer->shape);
}

bool DetectHexagonIdentityLayoutForTest(const Layout &layout, const Buffer &buffer) {
  if (!buffer.defined()) {
    return false;
  }
  return StructuralEqual()(layout, MakeHexagonIdentityLayoutForTest(buffer));
}

bool RegisterBuiltinHexagonLayouts() {
  // Extending the framework is intentionally data-only: append one row here
  // after adding the new enum value and constructor/detector functions.
  RegisterHexagonLayout(
      {HexagonLayoutMode::kAH, "ah", MakeHexagonAHLayout, DetectHexagonAHLayout});
  RegisterHexagonLayout(
      {HexagonLayoutMode::kWH, "wh", MakeHexagonWHLayout, DetectHexagonWHLayout});
  RegisterHexagonLayout(
      {HexagonLayoutMode::kRM, "rm", MakeHexagonRMLayout, DetectHexagonRMLayout});
  return true;
}

const bool builtin_hexagon_layouts_registered = RegisterBuiltinHexagonLayouts();

} // namespace

HexagonLayoutMode DetectHexagonAHMode(const Layout &layout,
                                      const Buffer &buffer) {
  return DetectHexagonAHLayout(layout, buffer) ? HexagonLayoutMode::kAH
                                              : HexagonLayoutMode::kNone;
}

HexagonLayoutMode DetectHexagonWHMode(const Layout &layout,
                                      const Buffer &buffer) {
  return DetectHexagonWHLayout(layout, buffer) ? HexagonLayoutMode::kWH
                                              : HexagonLayoutMode::kNone;
}

HexagonLayoutMode DetectHexagonRMMode(const Layout &layout,
                                      const Buffer &buffer) {
  return DetectHexagonRMLayout(layout, buffer) ? HexagonLayoutMode::kRM
                                              : HexagonLayoutMode::kNone;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef()
      .def("tl.hexagon.make_ah_layout",
            [](Array<PrimExpr> shape) { return MakeHexagonAHLayout(shape); })
      .def("tl.hexagon.make_ah_layout_buffer",
            [](const Buffer &buffer) { return MakeHexagonAHLayout(buffer); })
      .def("tl.hexagon.make_wh_layout",
             [](const Buffer &buffer) { return MakeHexagonWHLayout(buffer); })
      .def("tl.hexagon.make_rm_layout",
           [](const Buffer &buffer) { return MakeHexagonRMLayout(buffer); })
      .def("tl.hexagon.make_layout",
           [](String mode, const Buffer &buffer) {
             return MakeHexagonLayout(HexagonLayoutModeFromString(mode), buffer);
           })
      .def("tl.hexagon.detect_layout_mode",
           [](const Layout &layout, const Buffer &buffer) {
             return HexagonLayoutModeToString(
                 DetectHexagonLayoutMode(layout, buffer));
           })
      .def("tl.hexagon.detect_layout_mode_ordinal",
           [](const Layout &layout, const Buffer &buffer) {
             return static_cast<int>(DetectHexagonLayoutMode(layout, buffer));
           })
      .def("tl.hexagon.register_identity_layout_for_test",
           []() {
             const HexagonLayoutSpec *existing =
                 FindHexagonLayoutSpec(HexagonLayoutMode::kIdentityForTest);
             if (existing != nullptr) {
               return false;
             }
             return RegisterHexagonLayout({HexagonLayoutMode::kIdentityForTest,
                                           "identity_for_test",
                                           MakeHexagonIdentityLayoutForTest,
                                           DetectHexagonIdentityLayoutForTest});
           })
      .def("tl.hexagon.detect_ah_mode",
            [](const Layout &layout, const Buffer &buffer) {
              return static_cast<int>(DetectHexagonAHMode(layout, buffer));
            })
      .def("tl.hexagon.detect_wh_mode",
            [](const Layout &layout, const Buffer &buffer) {
              return static_cast<int>(DetectHexagonWHMode(layout, buffer));
            })
      .def("tl.hexagon.detect_rm_mode",
           [](const Layout &layout, const Buffer &buffer) {
             return static_cast<int>(DetectHexagonRMMode(layout, buffer));
            });
}

} // namespace hexagon
} // namespace tl
} // namespace tvm
