/* Hexagon uses the common layout-aware elementwise fill lowering. */
#include "backend/common/op/fill.h"
#include "hexagon/target_utils.h"

namespace tvm::tl {
namespace {
bool MatchHexagonFill(Target target) { return TargetIsHexagon(target); }

bool RegisterHexagonFill() {
  RegisterFillImpl(FillImpl{"hexagon.Fill", MatchHexagonFill,
                            backend::Fill::Lower});
  return true;
}
const bool hexagon_fill_registered = RegisterHexagonFill();
} // namespace
} // namespace tvm::tl
