/* Layout-aware transpose uses the common loop lowering, without GPU helpers. */
#include "backend/common/op/transpose.h"
#include "hexagon/target_utils.h"
#include "hexagon/layouts.h"
#include "affine_transpose.h"
#include <tvm/tirx/builtin.h>
#include <tvm/ir/transform.h>

namespace tvm::tl {
TVM_REGISTER_PASS_CONFIG_OPTION("tl.hexagon.affine_transpose", Bool);
namespace {
bool MatchHexagonTranspose(Target target) { return TargetIsHexagon(target); }
tirx::Stmt LowerHexagonTranspose(const TransposeNode &op, const LowerArgs &args,
                                arith::Analyzer *analyzer) {
  using namespace tirx;
  ICHECK(args.thread_bounds.defined() && args.thread_index.defined())
      << "Hexagon transpose requires defined thread_bounds and thread_index";
  auto layout = args.layout_map.Get(op.dst);
  auto src_layout = args.layout_map.Get(op.src);
  // RM affine regions only. Packed layouts keep their existing lowering.
  auto rm = [&](Buffer b) {
    auto l = args.layout_map.Get(b);
    return !l || hexagon::DetectHexagonLayoutMode(l.value(), b) ==
                     hexagon::HexagonLayoutMode::kRM;
  };
  if (tvm::transform::PassContext::Current()->GetConfig<Bool>("tl.hexagon.affine_transpose").value_or(Bool(false)) &&
      op.src->shape.size() == 2 && op.dst->shape.size() == 2 &&
      op.src->dtype == op.dst->dtype &&
      (op.src->dtype == DataType::Float(16) || op.src->dtype == DataType::Float(32)) &&
      rm(op.src) && rm(op.dst)) {
    if (hexagon::LegalAffineTranspose(op.src, op.dst, op.src_range,
                                      op.dst_range, args, analyzer))
      return hexagon::EmitAffineTranspose(op.src, op.dst, op.src_range,
                                          op.dst_range, args, analyzer);
    ICHECK(op.src.scope().find("shared") != 0 && op.dst.scope().find("shared") != 0 &&
           op.src.scope().find("vtcm") != 0 && op.dst.scope().find("vtcm") != 0)
        << "Hexagon VTCM transpose requires proven bounds, strides and 128B padded allocations";
  }
  if (op.src->shape.size() == 2 && op.dst->shape.size() == 2 &&
      op.src->dtype == DataType::Float(16) && op.dst->dtype == DataType::Float(16) &&
      op.src.scope() == "shared.dyn" && op.dst.scope() == "shared.dyn" &&
      layout && hexagon::DetectHexagonLayoutMode(layout.value(), op.dst) == hexagon::HexagonLayoutMode::kWH &&
      (!src_layout || hexagon::DetectHexagonLayoutMode(src_layout.value(), op.src) == hexagon::HexagonLayoutMode::kNone) &&
      analyzer->CanProveEqual(FloorMod(op.src->shape[0], 32), 0) &&
      analyzer->CanProveEqual(FloorMod(op.src->shape[1], 32), 0)) {
    // RM[R,C] -> WH[C,R] is a zip16 of each source tile, with swapped
    // tile coordinates. No scalar VTCM transpose and no per-element barrier.
    Var t("transpose_tile");
    PrimExpr cols = FloorDiv(op.src->shape[1], 32);
    PrimExpr count = FloorDiv(op.src->shape[0], 32) * cols;
    PrimExpr ix = t * args.thread_bounds->extent + args.thread_index;
    PrimExpr row = FloorDiv(ix, cols) * 32, col = FloorMod(ix, cols) * 32;
    auto addr = [](Buffer b, PrimExpr r, PrimExpr c) {
      return Call(DataType::Handle(), builtin::address_of(), {BufferLoad(b,{r,c})});
    };
    Stmt copy = Evaluate(Call(DataType::Handle(), Op::Get("tl.hexagon.pack_ah_strided"),
        {addr(op.dst,col,row),addr(op.src,row,col),op.src->shape[1],Integer(1)}));
    Stmt sync = Evaluate(Call(DataType::Int(32), builtin::tvm_storage_sync(),{StringImm("shared")}));
    return SeqStmt({sync,For(t,0,FloorDiv(count+args.thread_bounds->extent-1,args.thread_bounds->extent),
                                ForKind::kSerial,IfThenElse(ix<count,copy)),sync});
  }
  return backend::Transpose::Lower(op,args,analyzer);
}
bool RegisterHexagonTranspose() {
  RegisterTransposeImpl(TransposeImpl{"hexagon.Transpose", MatchHexagonTranspose,
                                      LowerHexagonTranspose});
  return true;
}
const bool hexagon_transpose_registered = RegisterHexagonTranspose();
} // namespace
} // namespace tvm::tl
