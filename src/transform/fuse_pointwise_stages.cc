#include <tvm/arith/analyzer.h>
#include <tvm/ir/op.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <tvm/tirx/op_attr_types.h>
#include <unordered_map>
#include <unordered_set>

namespace tvm::tl {
using namespace tirx;
using namespace ffi;

// Whole-function address/view audit. Buffer data fields are not expression
// children, so visiting a handle Var means an address escaped ordinary access.
class PrivateAudit : public StmtExprVisitor {
public:
  std::unordered_set<const VarNode *> escaped;
  std::unordered_map<const VarNode *, std::unordered_set<const BufferNode *>> views;
  void VisitExpr_(const VarNode *n) final {
    if (n->dtype.is_handle()) escaped.insert(n);
  }
  void VisitExpr_(const CallNode *call) final {
    static const auto effects = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
    auto callee = call->op.as<Op>();
    bool pure = callee.has_value() && effects.count(callee.value()) &&
                effects[callee.value()]->value == static_cast<int>(CallEffectKind::kPure);
    for (auto arg : call->args) {
      PostOrderVisit(arg, [&](const ObjectRef &n) {
        if (auto v = n.as<VarNode>(); v && v->dtype.is_handle()) escaped.insert(v);
        if (auto load = n.as<BufferLoadNode>(); load && !pure)
          escaped.insert(load->buffer->data.get());
      });
    }
    StmtExprVisitor::VisitExpr_(call);
  }
  void VisitExpr_(const BufferLoadNode *n) final {
    views[n->buffer->data.get()].insert(n->buffer.get());
    StmtExprVisitor::VisitExpr_(n);
  }
  void VisitStmt_(const BufferStoreNode *n) final {
    views[n->buffer->data.get()].insert(n->buffer.get());
    StmtExprVisitor::VisitStmt_(n);
  }
};

class PointwiseFusion : public StmtExprMutator {
  PrivateAudit audit;
  arith::Analyzer az;

  bool Pure(const PrimExpr &e, bool loads = true) {
    static const auto effects = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
    bool ok = true;
    PostOrderVisit(e, [&](const ObjectRef &n) {
      bool leaf = n.as<IntImmNode>() || n.as<FloatImmNode>() || n.as<StringImmNode>() || n.as<VarNode>() || n.as<OpNode>();
      bool op = n.as<CastNode>() || n.as<SelectNode>() || n.as<AddNode>() ||
          n.as<SubNode>() || n.as<MulNode>() || n.as<DivNode>() ||
          n.as<ModNode>() || n.as<FloorDivNode>() || n.as<FloorModNode>() ||
          n.as<MinNode>() || n.as<MaxNode>() || n.as<EQNode>() ||
          n.as<NENode>() || n.as<LTNode>() || n.as<LENode>() ||
          n.as<GTNode>() || n.as<GENode>() || n.as<AndNode>() ||
          n.as<OrNode>() || n.as<NotNode>();
      auto l = n.as<BufferLoadNode>();
      bool pure_call = false;
      if (auto call = n.as<CallNode>()) {
        auto callee = call->op.as<Op>();
        pure_call = loads && callee.has_value() && effects.count(callee.value()) &&
                    effects[callee.value()]->value == static_cast<int>(CallEffectKind::kPure);
      }
      if (!(leaf || op || pure_call || (loads && l && !l->predicate.defined()))) ok = false;
    });
    return ok;
  }
  const BufferStoreNode *Nest(Stmt s, std::vector<For> &loops) {
    while (auto f = s.as<ForNode>()) {
      if (f->thread_binding.defined() || !f->annotations.empty() ||
          (f->kind != ForKind::kSerial && f->kind != ForKind::kVectorized)) return nullptr;
      loops.push_back(GetRef<For>(f));
      s = f->body;
    }
    return loops.empty() ? nullptr : s.as<BufferStoreNode>();
  }
  bool Equal(const Array<PrimExpr> &a, const Array<PrimExpr> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
      if (!az.CanProveEqual(a[i], b[i])) return false;
    return true;
  }
  Stmt Pair(const Stmt &first, const Stmt &second) {
    std::vector<For> p, c;
    auto ps = Nest(first, p);
    auto cs = Nest(second, c);
    if (!ps || !cs || p.size() != c.size()) return Stmt();
    Buffer b = ps->buffer;
    // Only thread-private storage may be forwarded across the two nests.
    // A shared/global buffer can communicate between lanes even when the
    // accesses look pointwise; thread_extent traversal does not waive this.
    if (!b.same_as(cs->buffer) ||
        (b.scope() != "local" && b.scope() != "local.fragment") ||
        !b->strides.empty() || audit.escaped.count(b->data.get()) ||
        audit.views[b->data.get()].size() != 1 ||
        ps->predicate.defined() || cs->predicate.defined()) return Stmt();
    Map<Var, PrimExpr> mapping;
    std::unordered_set<const VarNode *> vars;
    for (size_t i = 0; i < p.size(); ++i) {
      mapping.Set(c[i]->loop_var, p[i]->loop_var);
      vars.insert(p[i]->loop_var.get()); vars.insert(c[i]->loop_var.get());
    }
    for (size_t i = 0; i < p.size(); ++i) {
      auto a = p[i]; auto d = c[i];
      if (a->kind != d->kind || (a->step.defined() && !az.CanProveEqual(a->step.value(), 1)) ||
          (d->step.defined() && !az.CanProveEqual(d->step.value(), 1)) ||
          !az.CanProveEqual(a->min, Substitute(d->min, mapping)) ||
           !az.CanProveEqual(a->extent, Substitute(d->extent, mapping))) return Stmt();
       // A rectangular slice may retain enclosing-loop coordinates. Each
       // varying coordinate must be a distinct, unshifted induction variable.
       int coordinates = 0;
       for (auto index : ps->indices) {
         if (az.CanProveEqual(index, a->loop_var)) ++coordinates;
         else {
           bool dependent = false;
           PostOrderVisit(index, [&](const ObjectRef &n) {
             if (auto v = n.as<VarNode>()) dependent |= v == a->loop_var.get();
           });
           if (dependent || !Pure(index, false)) return Stmt();
         }
       }
       if (coordinates != 1) return Stmt();
      for (auto bound : {a->min, a->extent, d->min, d->extent}) {
        if (!Pure(bound, false)) return Stmt();
        bool dependent = false;
        PostOrderVisit(bound, [&](const ObjectRef &n) {
          if (auto v = n.as<VarNode>()) dependent |= vars.count(v) != 0;
        });
        if (dependent) return Stmt();
      }
    }
    auto consumer = Downcast<BufferStore>(Substitute(GetRef<Stmt>(cs), mapping));
    if (!Equal(ps->indices, consumer->indices) || !Pure(ps->value) ||
        !Pure(consumer->value)) return Stmt();
    bool valid = true;
    int uses = 0;
    // Producer may read its *own* old version at exactly the same coordinate.
    // Stencils (including in producer) would observe different memory versions.
    auto check = [&](const PrimExpr &e, bool count) {
      PostOrderVisit(e, [&](const ObjectRef &n) {
        if (auto l = n.as<BufferLoadNode>(); l && l->buffer->data.same_as(b->data)) {
          if (!l->buffer.same_as(b) || !Equal(l->indices, ps->indices)) valid = false;
          if (count) ++uses;
        }
      });
    };
    check(ps->value, false); check(consumer->value, true);
    if (!valid || uses != 1) return Stmt();
    class Forward : public StmtExprMutator {
    public:
      Buffer buffer;
      PrimExpr value;
      PrimExpr VisitExpr_(const BufferLoadNode *n) final {
        return n->buffer.same_as(buffer) ? value : StmtExprMutator::VisitExpr_(n);
      }
    } forward;
    forward.buffer = b;
    forward.value = ps->value.dtype() == b->dtype ? ps->value : Cast(b->dtype, ps->value);
    Stmt result = forward(consumer);
    for (auto it = p.rbegin(); it != p.rend(); ++it) {
      auto loop = *it;
      loop.CopyOnWrite()->body = result;
      result = loop;
    }
    return result;
  }
  Stmt VisitStmt_(const AttrStmtNode *n) final {
    if (n->attr_key != "thread_extent") return GetRef<Stmt>(n);
    return StmtExprMutator::VisitStmt_(n);
  }
  Stmt VisitStmt_(const SeqStmtNode *n) final {
    auto visited = Downcast<SeqStmt>(StmtExprMutator::VisitStmt_(n));
    Array<Stmt> out;
    bool changed = false;
    for (auto s : visited->seq) {
      auto fused = out.empty() ? Stmt() : Pair(out.back(), s);
      if (fused.defined()) { out.Set(out.size() - 1, fused); changed = true; }
      else out.push_back(s);
    }
    if (!changed) return visited;
    return out.size() == 1 ? out[0] : SeqStmt(out);
  }
public:
  Stmt Run(const Stmt &s) { audit(s); return operator()(s); }
};

TVM_REGISTER_PASS_CONFIG_OPTION("tl.disable_fuse_pointwise_stages", Bool);

tirx::transform::Pass FusePointwiseStages() {
  auto run = [](PrimFunc f, IRModule, tvm::transform::PassContext) {
    Stmt body = PointwiseFusion().Run(f->body);
    if (!body.same_as(f->body)) f.CopyOnWrite()->body = body;
    return f;
  };
  return tirx::transform::CreatePrimFuncPass(run, 0, "tl.FusePointwiseStages", {});
}
TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.transform.FusePointwiseStages", FusePointwiseStages);
}
} // namespace tvm::tl
