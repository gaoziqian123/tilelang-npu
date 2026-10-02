/* Forward producer SSA into an existing physical reduction plan. This pass
 * runs after vectorization so a map vector is evaluated once, not scalarized
 * or recomputed by a callback. All producer stores remain for replay/users.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/runtime/logging.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <unordered_set>
#include <unordered_map>
#include "target_utils.h"

namespace tvm::tl {
using namespace tirx;
using namespace ffi;
class RowMapFusion : public StmtExprMutator {
  int reducer_mask_;
  arith::Analyzer az;
  std::unordered_set<const VarNode*> escaped;
  std::unordered_map<const VarNode*,std::unordered_set<const BufferNode*>> views;
  static bool Named(const CallNode* c, const char* name) {
    auto op=c?c->op.as<OpNode>():nullptr; return op && op->name==name;
  }
  PrimExpr Ptr(Buffer b, PrimExpr index) {
    return Call(DataType::Handle(),builtin::address_of(),{BufferLoad(b,{index})});
  }
  bool Pure(PrimExpr e) {
    static auto effects=Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
    bool ok=true;
    PostOrderVisit(e,[&](const ObjectRef& n) {
      if(auto c=n.as<CallNode>()) {
        auto op=c->op.as<Op>();
        if(!op.has_value() || !effects.count(op.value()) ||
           effects[op.value()]->value!=int(CallEffectKind::kPure)) ok=false;
      }
      if(auto v=n.as<VarNode>(); v && v->dtype.is_handle()) ok=false;
      if(auto l=n.as<BufferLoadNode>(); l && l->predicate.defined()) ok=false;
    }); return ok;
  }
  bool Loop(const ForNode* f) {
    return f && !f->thread_binding.defined() && f->annotations.empty() &&
      (f->kind==ForKind::kSerial || f->kind==ForKind::kUnrolled) &&
      (!f->step.defined() || az.CanProveEqual(f->step.value(),1));
  }
  Array<Stmt> Parts(Stmt s) {
    if(auto seq=s.as<SeqStmtNode>()) return seq->seq;
    return {s};
  }
  // Fuse one row's terminal map partition(s) with its reduction. The producer
  // may have pure scalar setup before the partitions, but no opaque effects.
  Stmt Row(Stmt producer, Stmt consumer) {
    auto cs=Parts(consumer);
    if(cs.empty() || cs.size()>2) return Stmt();
    auto dst=cs.back().as<BufferStoreNode>();
    auto plan=dst?dst->value.as<CallNode>():nullptr;
    if(!Named(plan,"tl.hexagon.local_row_reduce") || plan->args.size()!=6 ||
        dst->predicate.defined()) return Stmt();
    auto kind=plan->args[3].as<IntImmNode>();
    if(!kind || !(reducer_mask_ & (kind->value ? 2 : 1))) return Stmt();
    auto contract=plan->args[5].as<StringImmNode>();
    auto width=plan->args[4].as<IntImmNode>();
    if(!width || width->value!=32 || !contract || contract->value!=
        "v1:chain32:rotate16,8,4,2,1:seed_after_tree:ordered_tail:stored_ordered_replay") return Stmt();
    auto address=plan->args[0].as<CallNode>();
    auto src=address && address->args.size()==1?address->args[0].as<BufferLoadNode>():nullptr;
    if(!src || src->indices.size()!=1 || dst->indices.size()!=1) return Stmt();
    Buffer b=src->buffer;
    if(escaped.count(b->data.get()) || escaped.count(dst->buffer->data.get()) ||
       views[b->data.get()].size()!=1 || views[dst->buffer->data.get()].size()!=1)
      return Stmt();
    PrimExpr base=az.Simplify(src->indices[0]), n=plan->args[1], seed=plan->args[2];
    auto seed_load=seed.as<BufferLoadNode>();
    if(!seed_load || !seed_load->buffer.same_as(dst->buffer) ||
        seed_load->indices.size()!=1 || !az.CanProveEqual(seed_load->indices[0],dst->indices[0])) return Stmt();
    auto size=n.as<IntImmNode>();
    if(!size || size->value<=0 || !b->strides.empty()) return Stmt();
    if(cs.size()==2) {
      auto init=cs[0].as<BufferStoreNode>();
      if(!init || init->predicate.defined() || !init->buffer.same_as(dst->buffer) || init->indices.size()!=1 ||
          !az.CanProveEqual(init->indices[0],dst->indices[0]) || !Pure(init->value)) return Stmt();
      seed=init->value;
      bool loads=false;
      PostOrderVisit(seed,[&](const ObjectRef& node) { if(node.as<BufferLoadNode>()) loads=true; });
      if(loads) return Stmt();
    }
    auto ps=Parts(producer);
    // Vectorization removes a unit-trip row loop. Reconstruct only this exact
    // full-vector terminal store, so it uses the same domain/access verifier.
    if(auto terminal=ps.back().as<BufferStoreNode>()) {
      int lanes=terminal->value.dtype().lanes();
      auto ramp=terminal->indices.size()==1?terminal->indices[0].as<RampNode>():nullptr;
      if(lanes==32 && ramp && az.CanProveEqual(n,32) &&
         az.CanProveEqual(ramp->base,base) && az.CanProveEqual(ramp->stride,1)) {
        Var unit("row_map_unit",DataType::Int(32));
        ps.Set(ps.size()-1,For(unit,Integer(0),Integer(1),ForKind::kSerial,GetRef<BufferStore>(terminal)));
      }
    }
    size_t begin=ps.size();
    while(begin && ps[begin-1].as<ForNode>()) --begin;
    if(begin==ps.size()) return Stmt();
    // A complete terminal map may follow earlier full-row stages (cast/scale,
    // then Select, for example). Only forward the terminal stage: earlier
    // stages remain materialized, in their original order, outside the state
    // update. Do not mistake repeated full domains for adjacent partitions.
    auto last=ps.back().as<ForNode>();
    auto last_store=last?last->body.as<BufferStoreNode>():nullptr;
    if(Loop(last) && last_store &&
       az.CanProveEqual(last->min,0) &&
       az.CanProveEqual(last->extent*last_store->value.dtype().lanes(),n))
      begin=ps.size()-1;
    // No alias views or cross-coordinate read of the materialized row. Unknown
    // calls/handles in the candidate are rejected, not declared pure.
    bool legal=true;
    PostOrderVisit(producer,[&](const ObjectRef& node) {
      if(auto c=node.as<CallNode>(); c && !Pure(GetRef<PrimExpr>(c))) legal=false;
      if(auto l=node.as<BufferLoadNode>(); l && l->buffer->data.same_as(b->data) &&
          !l->buffer.same_as(b)) legal=false;
      if(auto s=node.as<BufferStoreNode>(); s && s->buffer->data.same_as(b->data) &&
          !s->buffer.same_as(b)) legal=false;
      // Consumer seed is evaluated at the old boundary; it must not depend on
      // the destination being overwritten during producer execution.
      if(auto l=node.as<BufferLoadNode>(); l && l->buffer->data.same_as(dst->buffer->data)) legal=false;
      if(auto s=node.as<BufferStoreNode>(); s && s->buffer->data.same_as(dst->buffer->data)) legal=false;
    });
    if(!legal) return Stmt();
    for(size_t i=0;i<begin;i++) {
      if(ps[i].as<ForNode>()) continue; // unchanged pure prefix, audited above
      if(auto attr=ps[i].as<AttrStmtNode>(); attr && attr->attr_key=="lexical_alloc_scope") continue;
      auto s=ps[i].as<BufferStoreNode>();
      if(!s || !Pure(s->value) || s->buffer->data.same_as(b->data)) return Stmt();
    }
    Buffer state=decl_buffer({Integer(64)},DataType::Float(32),"row_map_state","local");
    PrimExpr state_ptr=Ptr(state,Integer(0)), maximum=plan->args[3];
    Array<Stmt> out;
    out.push_back(AllocBuffer(state));
    out.push_back(Evaluate(Call(DataType::Int(32),Op::Get("tl.hexagon.row_map_init"),{state_ptr,maximum})));
    for(size_t i=0;i<begin;i++) out.push_back(ps[i]);
    PrimExpr cursor=Integer(0);
    for(size_t i=begin;i<ps.size();i++) {
      auto f=ps[i].as<ForNode>();
      auto s=f?f->body.as<BufferStoreNode>():nullptr;
      if(!Loop(f) || !s || !s->buffer.same_as(b) || s->indices.size()!=1 ||
          s->predicate.defined() || !Pure(s->value) ||
          !s->value.dtype().is_float() || s->value.dtype().bits()!=32) return Stmt();
      int lanes=s->value.dtype().lanes();
      PrimExpr index=s->indices[0];
      if(lanes>1) {
        auto ramp=index.as<RampNode>();
        if(!ramp || !az.CanProveEqual(ramp->stride,1) || lanes!=32) return Stmt();
        index=ramp->base;
      }
      if(!(az.CanProveEqual(index,base+f->loop_var*lanes) ||
           (az.CanProveEqual(f->min,0) && az.CanProveEqual(f->extent,1) && az.CanProveEqual(index,base))) ||
          !az.CanProveEqual(f->min*lanes,cursor) ||
          !Pure(f->min) || !Pure(f->extent) ||
          !az.CanProve(f->min>=0) || !az.CanProve(f->extent>=0)) return Stmt();
      PostOrderVisit(s->value,[&](const ObjectRef& node) {
        if(auto l=node.as<BufferLoadNode>(); l && l->buffer->data.same_as(b->data))
          if(!l->buffer.same_as(b) || l->indices.size()!=1 ||
             !az.CanProveEqual(l->indices[0].as<RampNode>() ? l->indices[0].as<RampNode>()->base : l->indices[0],index) ||
             l->dtype.lanes()!=lanes ||
             (l->indices[0].as<RampNode>() && !az.CanProveEqual(l->indices[0].as<RampNode>()->stride,1))) legal=false;
      });
      for(auto bound:{f->min,f->extent})
        PostOrderVisit(bound,[&](const ObjectRef& node) { if(node.as<BufferLoadNode>()) legal=false; });
      if(!legal) return Stmt();
      cursor=az.Simplify((f->min+f->extent)*lanes);
      Var value("row_map_value",s->value.dtype());
      Stmt update=Evaluate(Call(DataType::Int(32),Op::Get("tl.hexagon.row_map_update"),
          {state_ptr,value,az.Simplify(index-base),n,maximum}));
      Stmt body=SeqStmt({Bind(value,s->value),BufferStore(b,value,s->indices),update});
      For loop=GetRef<For>(f); loop.CopyOnWrite()->body=body;
      out.push_back(loop);
    }
    if(!az.CanProveEqual(cursor,n)) return Stmt();
    PrimExpr finish=Call(DataType::Float(32),Op::Get("tl.hexagon.row_map_finish"),
        {state_ptr,plan->args[0],n,seed,maximum});
    out.push_back(BufferStore(dst->buffer,finish,dst->indices));
    return SeqStmt(out);
  }
  Stmt Pair(Stmt producer, Stmt consumer) {
    auto p=producer.as<ForNode>(), c=consumer.as<ForNode>();
    if(Loop(p) && Loop(c) && az.CanProveEqual(p->min,c->min) &&
        az.CanProveEqual(p->extent,c->extent)) {
      // Align outer row ownership only when the consumer contains the plan.
      Stmt aligned=Substitute(c->body,{{c->loop_var,p->loop_var}});
      auto fused=Row(p->body,aligned);
      if(fused.defined()) {
        For loop=GetRef<For>(p); loop.CopyOnWrite()->body=fused; return loop;
      }
    }
    return Row(producer,consumer);
  }
  Stmt VisitStmt_(const SeqStmtNode* node) final {
    auto visited=Downcast<SeqStmt>(StmtExprMutator::VisitStmt_(node));
    Array<Stmt> out;
    for(auto s:visited->seq) {
      auto fused=out.empty()?Stmt():Pair(out.back(),s);
      if(fused.defined()) out.Set(out.size()-1,fused); else out.push_back(s);
    }
    return SeqStmt(out);
  }
public:
  explicit RowMapFusion(int reducer_mask=3):reducer_mask_(reducer_mask) {}
  PrimFunc Run(PrimFunc f) {
    auto target=f->GetAttr<Target>(tvm::attr::kTarget);
    if(!target.has_value() || !TargetIsHexagon(target.value())) return f;
    class Audit : public StmtExprVisitor {
      RowMapFusion* owner;
      void VisitExpr_(const VarNode* n) final {
        if(n->dtype.is_handle()) owner->escaped.insert(n);
      }
      void VisitExpr_(const BufferLoadNode* n) final {
        owner->views[n->buffer->data.get()].insert(n->buffer.get());
        StmtExprVisitor::VisitExpr_(n);
      }
      void VisitStmt_(const BufferStoreNode* n) final {
        owner->views[n->buffer->data.get()].insert(n->buffer.get());
        StmtExprVisitor::VisitStmt_(n);
      }
      void VisitExpr_(const CallNode* n) final {
        if(Named(n,"tl.hexagon.local_row_reduce")) {
          // The plan's own pointer use is known and remains for replay.
          auto addr=n->args[0].as<CallNode>();
          if(addr) for(auto a:addr->args) VisitExpr(a);
          for(size_t i=1;i<n->args.size();i++) VisitExpr(n->args[i]);
          return;
        }
        if(n->op.same_as(builtin::address_of())) {
          for(auto arg:n->args) PostOrderVisit(arg,[&](const ObjectRef& node) {
            if(auto l=node.as<BufferLoadNode>()) owner->escaped.insert(l->buffer->data.get());
          });
        }
        StmtExprVisitor::VisitExpr_(n);
      }
    public:
      explicit Audit(RowMapFusion* owner):owner(owner) {}
      void Run(Stmt s) { VisitStmt(s); }
    } audit(this);
    audit.Run(f->body);
    f.CopyOnWrite()->body=VisitStmt(f->body); return f;
  }
};
namespace transform {
tirx::transform::Pass FuseLocalRowMap() {
  auto run=[](PrimFunc f,IRModule,tvm::transform::PassContext ctx) {
    int mask=ctx->GetConfig<Integer>("tl.hexagon.row_map_reducers").value_or(Integer(3))->value;
    TVM_FFI_ICHECK(mask>=0 && mask<=3) << "row_map_reducers: bit0=sum, bit1=max";
    return RowMapFusion(mask).Run(f);
  };
  return tirx::transform::CreatePrimFuncPass(run,0,"tl.FuseLocalRowMap",{});
}
TVM_REGISTER_PASS_CONFIG_OPTION("tl.hexagon.fuse_local_row_map",Bool);
TVM_REGISTER_PASS_CONFIG_OPTION("tl.hexagon.row_map_reducers",Integer);
TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.transform.FuseLocalRowMap",FuseLocalRowMap);
}
}
}
