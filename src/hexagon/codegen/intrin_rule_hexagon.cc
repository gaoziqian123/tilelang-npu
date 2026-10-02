#include "target/intrin_rule.h"
#include "support/check.h"
#include <limits>
#include <tvm/tirx/op_attr_types.h>

namespace tvm::codegen::intrin {
using tirx::FLowerIntrinsic;
TVM_REGISTER_OP("tl.infinity").set_attr<FLowerIntrinsic>(
    "hexagon.FLowerIntrinsic", [](PrimExpr expr) -> PrimExpr {
      const auto *call = expr.as<tirx::CallNode>();
      ICHECK(call && call->dtype.is_float() && call->dtype.lanes() == 1);
      return FloatImm(call->dtype, std::numeric_limits<double>::infinity());
    });
struct HexagonMath {
  std::string operator()(DataType t, std::string name) const {
    // Do not emit a scalar libm name with vector arguments. Vector math must
    // first be lowered to explicit HVX operations (approximation is not IEEE).
    if (t.lanes() != 1) return "";
    if (t.is_float() && t.bits() == 32) return name + "f";
    return name;
  }
};
#define HEX_MATH(name) \
  TVM_REGISTER_OP("tirx." name).set_attr<FLowerIntrinsic>( \
      "hexagon.FLowerIntrinsic", DispatchPureExtern<HexagonMath>);
HEX_MATH("exp")
HEX_MATH("log")
HEX_MATH("sqrt")
HEX_MATH("sin")
HEX_MATH("cos")
HEX_MATH("tanh")
HEX_MATH("fabs")
HEX_MATH("floor")
HEX_MATH("ceil")
HEX_MATH("exp2")
HEX_MATH("log2")
HEX_MATH("log10")
HEX_MATH("trunc")
HEX_MATH("round")
HEX_MATH("nearbyint")
HEX_MATH("pow")
HEX_MATH("tan")
HEX_MATH("sinh")
HEX_MATH("cosh")
HEX_MATH("erf")
#undef HEX_MATH
} // namespace tvm::codegen
