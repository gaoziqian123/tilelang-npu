// Ordered, bounded accumulator forwarding. No algebraic reassociation.
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <unordered_set>
#include <unordered_map>
#include "arith/ir_mutator_with_analyzer.h"

namespace tvm::tl {
using namespace tirx;
using namespace ffi;
namespace {
class Promote : public arith::IRMutatorWithAnalyzer {
  std::unordered_set<const VarNode *> allocated, parameters;
  std::unordered_map<const VarNode *, std::unordered_set<const BufferNode *>> views;
  bool noalias;
  std::vector<Buffer> exposed;
  std::unordered_set<const VarNode *> nonlocal, generated;
  int64_t budget;
  bool Uses(PrimExpr e, Var v) {
    bool found = false;
    PostOrderVisit(e, [&](const ObjectRef &n) { found |= n.same_as(v); });
    return found;
  }
  bool Loop(const ForNode *f) {
    auto n = f->extent.as<IntImmNode>();
    return n && n->value > 0 && n->value <= 4096 && is_zero(f->min) &&
           !f->thread_binding.defined() &&
           (!f->step.defined() || is_one(f->step.value()));
  }
  bool Bounds(Buffer b, Array<PrimExpr> idx) {
    if (!is_zero(b->elem_offset) || idx.size()!=b->shape.size() ||
        views[b->data.get()].size()!=1) return false;
    if (!b->strides.empty()) {
      if (b->strides.size()!=b->shape.size()) return false;
      PrimExpr stride=1;
      for (int j=int(b->shape.size())-1;j>=0;--j) {
        if (!analyzer_->CanProveEqual(b->strides[j],stride)) return false;
        stride=stride*b->shape[j];
      }
    }
    for (size_t j=0;j<idx.size();++j)
      if (!Index(idx[j]) || !Index(b->shape[j]) ||
          !analyzer_->CanProve(idx[j]>=0) || !analyzer_->CanProve(idx[j]<b->shape[j])) return false;
    return true;
  }
  bool Index(PrimExpr e) {
    bool pure=true;
    PostOrderVisit(e,[&](const ObjectRef &o) {
      if (o.as<BufferLoadNode>() || o.as<CallNode>()) pure=false;
    });
    return pure;
  }
  bool Disjoint(Buffer a, Buffer b) {
    if (a->data.same_as(b->data)) return false;
    return (allocated.count(a->data.get()) && allocated.count(b->data.get())) ||
           (allocated.count(a->data.get()) && parameters.count(b->data.get())) ||
           (allocated.count(b->data.get()) && parameters.count(a->data.get())) ||
           (noalias && parameters.count(a->data.get()) && parameters.count(b->data.get()));
  }
  bool Pure(PrimExpr e, Buffer dst) {
    if (e.as<IntImmNode>() || e.as<FloatImmNode>()) return true;
    if (auto v=e.as<VarNode>()) return !v->dtype.is_handle();
    if (auto l=e.as<BufferLoadNode>()) {
      if (!Disjoint(dst,l->buffer) || l->predicate.defined() || !Bounds(l->buffer,l->indices)) return false;
      for (auto i:l->indices) if (!Pure(i,dst)) return false;
      return true;
    }
    if (auto n=e.as<AddNode>()) return Pure(n->a,dst)&&Pure(n->b,dst);
    if (auto n=e.as<SubNode>()) return Pure(n->a,dst)&&Pure(n->b,dst);
    if (auto n=e.as<MulNode>()) return Pure(n->a,dst)&&Pure(n->b,dst);
    return false;
  }
  Stmt Try(const ForNode *red) {
    if (!Loop(red) || red->kind!=ForKind::kSerial || !red->annotations.empty()) return {};
    auto group=red->body.as<ForNode>();
    if (!group) return {};
    const ForNode *lane=group;
    bool grouped=group->kind==ForKind::kSerial && group->annotations.count("tl.vectorized_group") &&
        group->annotations.at("tl.vectorized_group").as<IntImmNode>() &&
        group->annotations.at("tl.vectorized_group").as<IntImmNode>()->value==1;
    if (grouped) lane=group->body.as<ForNode>();
    if (!lane || !Loop(lane) || lane->kind!=ForKind::kVectorized || !lane->annotations.empty() ||
        (grouped && (!Loop(group) || group->annotations.size()!=1))) return {};
    auto st=lane->body.as<BufferStoreNode>();
    auto add=st?st->value.as<AddNode>():nullptr;
    auto old=add?add->a.as<BufferLoadNode>():nullptr;
    if (!old || st->predicate.defined() || old->predicate.defined() ||
        !old->buffer.same_as(st->buffer) || old->indices.size()!=st->indices.size() ||
        st->buffer->dtype!=DataType::Float(32)) return {};
    if (generated.count(st->buffer->data.get()) || nonlocal.count(st->buffer->data.get())) return {};
    // A backend may have outstanding bounded effects on its operands. Never
    // forward any buffer that might alias such an operand, even outside this
    // nest. Unknown/unbounded escapes are rejected by the function audit.
    bool exclusive=true;
    PostOrderVisit(GetRef<For>(red), [&](const ObjectRef &o) {
      if (auto l=o.as<BufferLoadNode>()) {
        for (auto b:exposed) if (!Disjoint(l->buffer,b)) exclusive=false;
      }
    });
    if (!exclusive) return {};
    int width=lane->extent.as<IntImmNode>()->value;
    if (width*4>budget) return {};
    analyzer_->Bind(red->loop_var,Range::FromMinExtent(red->min,red->extent),true);
    if (grouped) analyzer_->Bind(group->loop_var,Range::FromMinExtent(group->min,group->extent),true);
    analyzer_->Bind(lane->loop_var,Range::FromMinExtent(lane->min,lane->extent),true);
    if (!Bounds(st->buffer,st->indices) || !Pure(add->b,st->buffer)) return {};
    PrimExpr flat=0;
    for (size_t j=0;j<st->indices.size();++j) {
      auto idx=st->indices[j];
      if (Uses(idx,red->loop_var) || !Pure(idx,st->buffer) ||
          !analyzer_->CanProveEqual(idx,old->indices[j])) return {};
      flat=flat*st->buffer->shape[j]+idx;
    }
    Map<Var,PrimExpr> zero{{lane->loop_var,0}};
    PrimExpr delta=lane->loop_var;
    if (grouped) { zero.Set(group->loop_var,0); delta=delta+group->loop_var*width; }
    if (!analyzer_->CanProveEqual(flat,Substitute(flat,zero)+delta)) return {};
    Buffer acc=decl_buffer({lane->extent},st->buffer->dtype,"ordered_acc","local");
    auto wrap=[&](Stmt body) { auto f=GetRef<For>(lane); f.CopyOnWrite()->body=body; return Stmt(f); };
    Stmt seed=wrap(BufferStore(acc,GetRef<BufferLoad>(old),{lane->loop_var}));
    Stmt update=wrap(BufferStore(acc,Add(BufferLoad(acc,{lane->loop_var}),add->b),{lane->loop_var}));
    auto r=GetRef<For>(red); r.CopyOnWrite()->body=update;
    Stmt store=wrap(BufferStore(st->buffer,BufferLoad(acc,{lane->loop_var}),st->indices));
    Stmt body=SeqStmt({seed,r,store});
    body=SBlockRealize({},Bool(true),SBlock({}, {}, {}, "ordered_accumulator",body,{}, {acc}, {}, {{"tl.ordered_accumulator",Integer(1)}}));
    if (grouped) { auto g=GetRef<For>(group); g.CopyOnWrite()->body=body; body=g; }
    return body;
  }
  Stmt VisitStmt_(const ForNode *n) final {
    Stmt promoted=Try(n);
    return promoted.defined()?promoted:IRMutatorWithAnalyzer::VisitStmt_(n);
  }
public:
  Promote(arith::Analyzer *a, bool alias, int64_t bytes):IRMutatorWithAnalyzer(a),noalias(alias),budget(bytes) {}
  Stmt Run(PrimFunc f) {
    bool unsafe=false;
    for (auto kv:f->buffer_map) {
      parameters.insert(kv.second->data.get());
      views[kv.second->data.get()].insert(kv.second.get());
    }
    PostOrderVisit(f->body,[&](const ObjectRef &o) {
      if (auto b=o.as<SBlockNode>()) {
        for (auto a:b->alloc_buffers) {
          // No shared/externally visible allocation promotion in v1.
          if (a.scope()!="local") nonlocal.insert(a->data.get());
          if (b->annotations.count("tl.ordered_accumulator")) generated.insert(a->data.get());
          allocated.insert(a->data.get());
          views[a->data.get()].insert(a.get());
        }
        if (!b->match_buffers.empty()) unsafe=true;
      }
      if (auto l=o.as<BufferLoadNode>()) views[l->buffer->data.get()].insert(l->buffer.get());
      if (auto s=o.as<BufferStoreNode>()) views[s->buffer->data.get()].insert(s->buffer.get());
      if (auto a=o.as<AttrStmtNode>()) {
        if (a->attr_key!="thread_extent" || !analyzer_->CanProveEqual(a->value,1)) unsafe=true;
      }
      if (auto l=o.as<ForNode>(); l && (l->kind==ForKind::kParallel || l->thread_binding.defined()) &&
          !analyzer_->CanProveEqual(l->extent,1)) unsafe=true;
    });
    // Walk expressions with context: address_of is legal only as an operand
    // of an audited bounded-effect operation. A naked address, unknown call,
    // pointer arithmetic/alias, or escape through a let remains fail closed.
    class Effects : public StmtExprVisitor {
      Promote *p;
      bool *unsafe;
      bool pointer_arg=false;
      bool synchronous=false;
      void VisitExpr_(const VarNode *v) final {
        if (v->dtype.is_handle()) *unsafe=true;
      }
      void VisitExpr_(const CallNode *c) final {
        auto op=c->op.as<OpNode>();
        static auto bounded=Op::GetAttrMap<Bool>("tl.region_bounded_effects");
        static auto synchronous_ops=Op::GetAttrMap<Bool>("tl.region_synchronous");
        static auto effects=Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
        if (op && op->name=="tirx.address_of" && pointer_arg && c->args.size()==1) {
          if (auto l=c->args[0].as<BufferLoadNode>()) {
            if (!synchronous) p->exposed.push_back(l->buffer);
            bool saved=pointer_arg; pointer_arg=false;
            for (auto i:l->indices) VisitExpr(i);
            pointer_arg=saved;
            return;
          }
        }
        bool bounded_call=op && bounded.get(GetRef<Op>(op),Bool(false))->value;
        bool synchronous_call=op && synchronous_ops.get(GetRef<Op>(op),Bool(false))->value;
        bool pure=op && effects.count(GetRef<Op>(op)) &&
            effects[GetRef<Op>(op)]->value==static_cast<int>(CallEffectKind::kPure) && !c->dtype.is_handle();
        bool sync=op && op->name=="tirx.tvm_storage_sync";
        if (!bounded_call && !synchronous_call && !pure && !sync) *unsafe=true;
        bool saved=pointer_arg;
        bool saved_sync=synchronous;
        for (auto a:c->args) { pointer_arg=bounded_call || synchronous_call; synchronous=synchronous_call; VisitExpr(a); }
        pointer_arg=saved;
        synchronous=saved_sync;
      }
    public:
      Effects(Promote *p, bool *unsafe):p(p),unsafe(unsafe) {}
    };
    Effects(this,&unsafe)(f->body);
    return unsafe?f->body:VisitStmt(f->body);
  }
};
}
TVM_REGISTER_PASS_CONFIG_OPTION("tl.enable_ordered_accumulator_promotion", Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("tl.ordered_accumulator_budget_bytes", Integer);
tirx::transform::Pass PromoteOrderedAccumulator() {
  auto run=[](PrimFunc f, IRModule, tvm::transform::PassContext ctx) {
    arith::Analyzer a;
    int64_t budget=ctx->GetConfig<Integer>("tl.ordered_accumulator_budget_bytes",Integer(128)).value()->value;
    bool noalias=f->GetAttr<Bool>("tir.noalias",Bool(false)).value()->value;
    f.CopyOnWrite()->body=Promote(&a,noalias,budget).Run(f);
    return f;
  };
  return tirx::transform::CreatePrimFuncPass(run,0,"tl.PromoteOrderedAccumulator",{});
}
TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.transform.PromoteOrderedAccumulator",PromoteOrderedAccumulator);
}
}
