/* Run after all shared ThreadSync insertion. Preserve ordering fences and
 * device completion primitives; only cooperative storage barriers change. */
#include "../op/operator.h"
#include "../transform/common/worker_group_context.h"
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm::tl {
using namespace tirx;
using namespace ffi;
namespace {
class WorkerSync : public StmtExprMutator {
  bool in_group{false};
  bool has_groups{false};
 public:
  Stmt Run(Stmt body) {
    PostOrderVisit(body, [&](const ObjectRef &n) {
      if (auto a = n.as<AttrStmtNode>()) has_groups |= a->attr_key == "tl.workergroup_local";
    });
    return operator()(body);
  }
 private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != "tl.workergroup_local") return StmtExprMutator::VisitStmt_(op);
    CHECK(!in_group, ValueError) << "nested worker synchronization scope";
    WorkerGroupContext::From(op); // validate owner and group extent
    in_group = true;
    Stmt result = StmtExprMutator::VisitStmt_(op);
    in_group = false;
    return result;
  }
  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::tvm_storage_sync()) && has_groups) {
      CHECK(in_group, ValueError)
          << "full-team barrier outside worker ownership scope is unsafe";
      auto scope = op->args[0].as<StringImmNode>();
      CHECK(scope && (scope->value == "shared" || scope->value == "shared.dyn"), ValueError)
          << "unsupported worker storage synchronization scope";
      return Call(op->dtype, Op::Get("tl.hexagon.workergroup_barrier"), {});
    }
    return StmtExprMutator::VisitExpr_(op);
  }
};
}
TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.hexagon.transform.LowerWorkerSync", [] {
    auto pass = [](PrimFunc f, const IRModule &, tvm::transform::PassContext) {
      Stmt body = WorkerSync().Run(f->body);
      f.CopyOnWrite()->body = body;
      return f;
    };
    return tirx::transform::CreatePrimFuncPass(pass, 0, "tl.LowerWorkerSync", {});
  });
}
}
