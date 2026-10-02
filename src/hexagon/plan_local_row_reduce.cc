/* Physical local-row planning, deliberately distinct from local.reducer epochs.
 * Runs after LowerTileOp and FlattenBuffer: ownership and contiguous addresses
 * are known, but vectorization/codegen have not erased the reduction structure.
 * No producer is moved, recomputed, or erased by this pass.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include "target_utils.h"

namespace tvm::tl {
using namespace tirx;
using namespace ffi;

class LocalRowPlanner : public StmtExprMutator {
  arith::Analyzer az;
  static bool Local(const Buffer &b) {
    return b.scope() == "local" || b.scope() == "local.fragment";
  }
  Stmt VisitStmt_(const ForNode *loop) final {
    Stmt result = StmtExprMutator::VisitStmt_(loop);
    loop = result.as<ForNode>();
    const auto *s = loop->body.as<BufferStoreNode>();
    if ((loop->kind != ForKind::kSerial && loop->kind != ForKind::kUnrolled) ||
        !az.CanProveEqual(loop->min, 0) || !s ||
        s->predicate.defined() || s->indices.size() != 1 ||
        s->value.dtype() != DataType::Float(32) || !Local(s->buffer))
      return result;
    PrimExpr a, b;
    bool maximum = false;
    if (auto add = s->value.as<AddNode>()) { a = add->a; b = add->b; }
    if (auto mx = s->value.as<MaxNode>()) {
      a = mx->a; b = mx->b; maximum = true;
    }
    auto dst = a.defined() ? a.as<BufferLoadNode>() : nullptr;
    auto src = b.defined() ? b.as<BufferLoadNode>() : nullptr;
    if (!src || !dst || src->predicate.defined() || dst->predicate.defined() ||
        !dst->buffer.same_as(s->buffer) || !Local(src->buffer) ||
        src->buffer->data.same_as(dst->buffer->data) ||
        src->indices.size() != 1 || dst->indices.size() != 1)
      return result;
    auto base = Substitute(src->indices[0], {{loop->loop_var, Integer(0)}});
    auto di = Substitute(dst->indices[0], {{loop->loop_var, Integer(0)}});
    if (!az.CanProveEqual(src->indices[0], base + loop->loop_var) ||
        !az.CanProveEqual(dst->indices[0], di) ||
        !az.CanProveEqual(s->indices[0], di)) return result;
    // This versioned plan is a numerical ABI, not a reassociation permission.
    // Seed enters AFTER the horizontal tree, then ordered scalar tail. NaN
    // (max) / nonfinite (sum), including exceptional seeds, replay the ORIGINAL
    // stored row in index order. Stores must survive any future map fusion.
    PrimExpr ptr = Call(DataType::Handle(), builtin::address_of(),
                        {BufferLoad(src->buffer, {base})});
    PrimExpr value = Call(DataType::Float(32), Op::Get("tl.hexagon.local_row_reduce"),
        {ptr, loop->extent, BufferLoad(dst->buffer, {di}), Integer(maximum),
         Integer(32), StringImm("v1:chain32:rotate16,8,4,2,1:seed_after_tree:ordered_tail:stored_ordered_replay")});
    return BufferStore(s->buffer, value, s->indices);
  }
public:
  PrimFunc Run(PrimFunc f) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.has_value() || !TargetIsHexagon(target.value())) return f;
    f.CopyOnWrite()->body = VisitStmt(f->body);
    return f;
  }
};

namespace transform {
tvm::transform::Pass PlanLocalRowReduce() {
  auto pass = [](PrimFunc f, IRModule, tvm::transform::PassContext) {
    return LocalRowPlanner().Run(std::move(f));
  };
  return tirx::transform::CreatePrimFuncPass(pass, 0, "tl.PlanLocalRowReduce", {});
}
TVM_FFI_STATIC_INIT_BLOCK() {
  tvm::ffi::reflection::GlobalDef().def("tl.transform.PlanLocalRowReduce", PlanLocalRowReduce);
}
TVM_REGISTER_PASS_CONFIG_OPTION("tl.hexagon.plan_local_row_reduce", Bool);
} // namespace transform
} // namespace tvm::tl
