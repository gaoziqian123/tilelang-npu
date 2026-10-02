/* Conservative whole-buffer ownership for explicit engine tokens.
 * No alias/subregion disjointness assumptions, no implicit completion. */
#include "../op/operator.h"
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <tvm/tirx/builtin.h>
#include <map>
#include <set>

namespace tvm::tl {
using namespace tirx;
using namespace ffi;
namespace {
struct Access {
  std::set<const VarNode *> read, write;
};
class VerifyAsync : public StmtExprVisitor {
 public:
  explicit VerifyAsync(bool enabled) : enabled(enabled) {}
  void Finish() { ICHECK(live.empty()) << "async token escapes function"; }
 private:
  std::map<int, Access> live;
  Access *capture{nullptr};
  bool enabled;
  void Touch(const Buffer &b, bool write) {
    auto data=b->data.get();
    for (const auto &[slot,a]:live)
      ICHECK(!a.write.count(data) && (!write || !a.read.count(data)))
          << "async buffer ownership conflict before wait, slot " << slot;
    if(capture) {
      ICHECK(b.scope()=="shared" || b.scope()=="shared.dyn")
          << "async captures require shared storage";
      (write?capture->write:capture->read).insert(data);
    }
  }
  void VisitExpr_(const BufferLoadNode *op) final { Touch(op->buffer,false); }
  void VisitStmt_(const BufferStoreNode *op) final {
    Touch(op->buffer,true); VisitExpr(op->value);
  }
  void VisitExpr_(const CallNode *op) final {
    if (op->args.size() && op->args[0].as<StringImmNode>() &&
        op->args[0].as<StringImmNode>()->value=="tl::engine_wait") {
      ICHECK(!capture) << "async wait inside producer";
      auto slot=op->args[1].as<IntImmNode>();
      ICHECK(slot && live.erase(slot->value)==1) << "async wait without live token";
      return;
    }
    auto tile=ParseOperator(GetRef<Call>(op));
    if(tile.defined()) {
      if(enabled && op->op.same_as(Op::Get("tl.tileop.gemm")))
        ICHECK(capture) << "async kernel cannot also issue synchronous HMX";
      auto regions=tile->GetAccessRegions();
      for(auto r:regions.reads) Touch(r->buffer,false);
      for(auto r:regions.writes) Touch(r->buffer,true);
      return;
    }
    ICHECK(!capture) << "async producer has unsupported non-TileOp call";
    if(!live.empty() && op->op.same_as(builtin::call_extern()))
      ICHECK(false) << "async live token crosses opaque external effect";
    StmtExprVisitor::VisitExpr_(op);
  }
  void VisitStmt_(const AttrStmtNode *op) final {
    if(op->attr_key!="hexagon.async_scope") {
      StmtExprVisitor::VisitStmt_(op); return;
    }
    ICHECK(!capture) << "nested async producer";
    auto slot=op->value.as<IntImmNode>();
    ICHECK(slot && slot->value>=0 && slot->value<8 && !live.count(slot->value))
        << "async slot reused before wait";
    Access access; capture=&access; VisitStmt(op->body); capture=nullptr;
    live.emplace(slot->value,std::move(access));
  }
  void VisitStmt_(const ForNode *op) final {
    // A loop iteration must retire everything it produces. Pipelined pairs
    // can be explicitly unrolled; loop-carried ownership is not guessed.
    ICHECK(live.empty()) << "async token crosses loop boundary";
    StmtExprVisitor::VisitStmt_(op);
    ICHECK(live.empty()) << "async token escapes loop iteration";
  }
  void VisitStmt_(const IfThenElseNode *op) final {
    ICHECK(live.empty()) << "async token crosses conditional boundary";
    VisitExpr(op->condition); VisitStmt(op->then_case);
    ICHECK(live.empty()) << "async token escapes conditional branch";
    if(op->else_case.defined()) VisitStmt(op->else_case.value());
    ICHECK(live.empty()) << "async token escapes conditional branch";
  }
};
}
TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.transform.VerifyHexagonAsync", [] {
    auto pass=[](PrimFunc f,const IRModule &,tvm::transform::PassContext) {
      bool enabled=false;
      PostOrderVisit(f->body,[&](const ObjectRef &n) {
        if(auto a=n.as<AttrStmtNode>())
          if(a->attr_key=="hexagon.async_scope") enabled=true;
      });
      if(enabled) { VerifyAsync v(true); v(f->body); v.Finish(); }
      return f;
    };
    return tirx::transform::CreatePrimFuncPass(pass,0,"tl.VerifyHexagonAsync",{});
  });
}
}
