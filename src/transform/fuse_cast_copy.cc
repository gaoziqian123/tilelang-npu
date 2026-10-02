// Adjacent reaching-definition forwarding before layout lowering. Liveness is
// per memory version, not allocation single-use; unknown effects fail closed.
#include <tvm/arith/analyzer.h>
#include <tvm/ir/op.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <tvm/tirx/op_attr_types.h>
#include <unordered_set>
#include <unordered_map>

namespace tvm::tl {
using namespace tirx;
using namespace ffi;
namespace {
bool Named(const CallNode *c, const char *name) {
  auto op = c ? c->op.as<OpNode>() : nullptr;
  return op && op->name == name;
}
const CallNode *CopyCall(const Stmt &s) {
  auto e = s.as<EvaluateNode>();
  auto c = e ? e->value.as<CallNode>() : nullptr;
  return Named(c, "tl.tileop.copy") && c->args.size() == 2 ? c : nullptr;
}
const BufferLoadNode *Region(const PrimExpr &e) {
  auto c = e.as<CallNode>();
  return Named(c, "tl.region") && c->args.size() >= 3 ? c->args[0].as<BufferLoadNode>() : nullptr;
}
using Live = std::unordered_set<const VarNode *>;
class CastCopy : public StmtExprMutator {
  arith::Analyzer az;
  Live escaped;
  Live allocated;
  std::unordered_map<const VarNode *, std::unordered_set<const BufferNode *>> views;
  std::unordered_map<const StmtNode *, Live> after;
  bool Full(const PrimExpr &e) {
    auto l = Region(e); auto c = e.as<CallNode>();
    if (!l || c->args.size() != l->indices.size()+2) return false;
    for (size_t i=0;i<l->indices.size();++i)
      if (!az.CanProveEqual(l->indices[i],0) ||
          !az.CanProveEqual(c->args[i+2],l->buffer->shape[i])) return false;
    return true;
  }
  void Reads(const ObjectRef &s, Live &live) {
    PostOrderVisit(s,[&](const ObjectRef &o) {
      if (auto l=o.as<BufferLoadNode>()) live.insert(l->buffer->data.get());
    });
  }
  Live Liveness(Stmt s, Live live) {
    after[s.get()] = live;
    if (auto seq=s.as<SeqStmtNode>()) {
      for (auto it=seq->seq.rbegin();it!=seq->seq.rend();++it) live=Liveness(*it,live);
    } else if (auto f=s.as<ForNode>()) {
      // Least fixed point includes zero-trip and loop-carried paths.
      Live exit=live;
      while (true) {
        Live next=Liveness(f->body,live);
        next.insert(exit.begin(),exit.end());
        if (next==live) break;
        live=std::move(next);
      }
      Reads(f->min,live); Reads(f->extent,live);
      // A complete rectangular store nest kills its previous memory version.
      std::vector<const ForNode *> nest;
      Stmt leaf=s;
      while (auto loop=leaf.as<ForNode>()) {
        if (loop->thread_binding.defined() || !loop->annotations.empty() ||
            (loop->kind!=ForKind::kSerial && loop->kind!=ForKind::kVectorized) ||
            (loop->step.defined() && !az.CanProveEqual(loop->step.value(),1))) break;
        nest.push_back(loop); leaf=loop->body;
        if (auto seq=leaf.as<SeqStmtNode>()) leaf=seq->seq.back();
      }
      if (auto st=leaf.as<BufferStoreNode>();st && !st->predicate.defined() &&
          st->indices.size()==nest.size() && st->buffer->shape.size()==nest.size()) {
        bool full=true;
        for(size_t i=0;i<nest.size();++i)
          full &= az.CanProveEqual(nest[i]->min,0) &&
                  az.CanProveEqual(nest[i]->extent,st->buffer->shape[i]) &&
                  az.CanProveEqual(st->indices[i],nest[i]->loop_var);
        Live reads; Reads(s,reads);
        if(full && !reads.count(st->buffer->data.get())) live.erase(st->buffer->data.get());
      }
    } else if (auto branch=s.as<IfThenElseNode>()) {
      auto a=Liveness(branch->then_case,live);
      auto b=branch->else_case.defined()?Liveness(branch->else_case.value(),live):live;
      a.insert(b.begin(),b.end()); live=std::move(a); Reads(branch->condition,live);
    } else if (auto a=s.as<AttrStmtNode>()) {
      live=Liveness(a->body,live); Reads(a->value,live);
    } else if (auto b=s.as<SBlockRealizeNode>()) {
      auto entry=Liveness(b->block->body,live);
      if (!az.CanProve(b->predicate)) entry.insert(live.begin(),live.end());
      live=std::move(entry); Reads(b->predicate,live);
      if(b->block->init.defined()) Reads(b->block->init.value(),live);
    } else if (auto c=CopyCall(s)) {
      auto dst=Region(c->args[1]);
      if (dst && Full(c->args[1])) live.erase(dst->buffer->data.get());
      Reads(c->args[0],live);
      if(dst) for(auto index:dst->indices) Reads(index,live);
      if(auto region=c->args[1].as<CallNode>())
        for(size_t i=2;i<region->args.size();++i) Reads(region->args[i],live);
    } else {
      Reads(s,live);
    }
    return live;
  }
  bool Safe(Buffer b) {
    return b.scope()=="local" && allocated.count(b->data.get()) && !escaped.count(b->data.get()) &&
           views[b->data.get()].size()==1 && b->strides.empty();
  }
  // The source region's direct BufferLoad denotes the address being replaced,
  // not a scalar read. Only that occurrence is exempt: its indices, predicate,
  // and buffer metadata still execute, as do ALL other consumer arguments.
  // Compare backing data Vars, not Buffer identity, to include alias views.
  bool ConsumerIndependent(const CallNode *copy, const BufferLoadNode *anchor) {
    class Audit : public StmtExprVisitor {
    public:
      using StmtExprVisitor::VisitExpr;
      const VarNode *temp;
      bool independent = true;
      std::unordered_set<const BufferNode *> visited;
      void Metadata(Buffer b) {
        if (!visited.insert(b.get()).second) return;
        for (auto e : b->shape) VisitExpr(e);
        for (auto e : b->strides) VisitExpr(e);
        VisitExpr(b->elem_offset);
      }
      void AnchorChildren(const BufferLoadNode *n) {
        Metadata(n->buffer);
        for (auto e : n->indices) VisitExpr(e);
        if (n->predicate.defined()) VisitExpr(n->predicate.value());
      }
      void VisitExpr_(const BufferLoadNode *n) final {
        if (n->buffer->data.get() == temp) independent = false;
        AnchorChildren(n);
      }
      void VisitExpr_(const VarNode *n) final {
        if (n == temp) independent = false;
      }
    } audit;
    audit.temp = anchor->buffer->data.get();
    audit.AnchorChildren(anchor);
    const auto *source = copy->args[0].as<CallNode>();
    for (size_t i = 1; i < source->args.size(); ++i)
      audit.VisitExpr(source->args[i]);
    for (size_t i = 1; i < copy->args.size(); ++i)
      audit.VisitExpr(copy->args[i]);
    return audit.independent;
  }
  Stmt Pair(Stmt producer, Stmt consumer) {
    auto c=CopyCall(consumer);
    if (!c) return Stmt();
    auto region=Region(c->args[0]);
    // Audit before arithmetic simplification: a Select or cancelling bounds
    // expression must not hide a memory dependency from this proof.
    if (!region || !ConsumerIndependent(c,region) || !Full(c->args[0])) return Stmt();
    auto destination=Region(c->args[1]);
    if (!destination || destination->buffer->dtype!=region->buffer->dtype) return Stmt();
    Buffer temp=region->buffer;
    if (!Safe(temp) || !after.count(consumer.get()) ||
        after.at(consumer.get()).count(temp->data.get())) return Stmt();
    std::vector<For> loops;
    std::vector<Stmt> others;
    Stmt leaf=producer;
    while (true) {
      if (auto f=leaf.as<ForNode>()) {
        if (f->thread_binding.defined() || !f->annotations.empty() ||
            (f->kind!=ForKind::kSerial && f->kind!=ForKind::kVectorized) ||
            (f->step.defined() && !az.CanProveEqual(f->step.value(),1))) return Stmt();
        loops.push_back(GetRef<For>(f)); leaf=f->body;
      } else if (auto seq=leaf.as<SeqStmtNode>()) {
        // Only a terminal cast nest may move past independent scalar setup.
        for (size_t i=0;i+1<seq->seq.size();++i) others.push_back(seq->seq[i]);
        leaf=seq->seq.back();
      } else break;
    }
    auto store=leaf.as<BufferStoreNode>();
    auto cast=store?store->value.as<CastNode>():nullptr;
    auto load=cast?cast->value.as<BufferLoadNode>():nullptr;
    if (!store || !load || !store->buffer.same_as(temp) || !Safe(load->buffer) ||
        store->predicate.defined() || load->predicate.defined() ||
        loops.size()!=temp->shape.size() || load->indices.size()!=loops.size() ||
        store->indices.size()!=loops.size() || load->buffer->shape.size()!=loops.size() ||
        cast->dtype!=temp->dtype || load->buffer->data.same_as(temp->data) ||
        temp->dtype!=DataType::Float(16) || load->buffer->dtype!=DataType::Float(32)) return Stmt();
    for (size_t i=0;i<loops.size();++i) {
      if (!az.CanProveEqual(loops[i]->min,0) ||
          !az.CanProveEqual(loops[i]->extent,temp->shape[i]) ||
          !az.CanProveEqual(store->indices[i],loops[i]->loop_var) ||
          !az.CanProveEqual(load->indices[i],loops[i]->loop_var) ||
          !az.CanProveEqual(load->buffer->shape[i],temp->shape[i])) return Stmt();
    }
    // Fission must not cross effects, aliases or source/temporary dependencies.
    bool independent=true;
    for (auto s:others) {
      if (!s.as<BufferStoreNode>()) return Stmt();
      PostOrderVisit(s,[&](const ObjectRef &o) {
        if (o.as<CallNode>()) independent=false;
        Buffer b;
        if (auto st=o.as<BufferStoreNode>()) b=st->buffer;
        if (auto ld=o.as<BufferLoadNode>()) b=ld->buffer;
        if (b.defined() && (!Safe(b) || b->data.same_as(temp->data) ||
                           b->data.same_as(load->buffer->data))) independent=false;
      });
    }
    if (!independent) return Stmt();
    auto src=Downcast<Call>(c->args[0]);
    auto args=src->args;
    args.Set(0,BufferLoad(load->buffer,region->indices));
    src.CopyOnWrite()->args=args;
    auto copy=GetRef<Call>(c); auto ca=copy->args; ca.Set(0,src);
    copy.CopyOnWrite()->args=ca;
    class Remove : public StmtExprMutator {
    public:
      const StmtNode *target;
      Stmt VisitStmt_(const BufferStoreNode *s) final {
        return s==target?Evaluate(0):GetRef<Stmt>(s);
      }
    } remove;
    remove.target=store;
    return SeqStmt({remove(producer),Evaluate(copy)});
  }
  Stmt VisitStmt_(const SeqStmtNode *s) final {
    Array<Stmt> out;
    // Match original nodes so the liveness keys remain valid.
    for (size_t i=0;i<s->seq.size();++i) {
      if (i+1<s->seq.size()) {
        auto fused=Pair(s->seq[i],s->seq[i+1]);
        if (fused.defined()) { out.push_back(fused); ++i; continue; }
      }
      out.push_back(VisitStmt(s->seq[i]));
    }
    return SeqStmt(out);
  }
public:
  Stmt Run(Stmt body) {
    PostOrderVisit(body,[&](const ObjectRef &o) {
      if(auto b=o.as<SBlockNode>())
        for(auto buffer:b->alloc_buffers) allocated.insert(buffer->data.get());
      if (auto l=o.as<BufferLoadNode>()) views[l->buffer->data.get()].insert(l->buffer.get());
      if (auto s=o.as<BufferStoreNode>()) views[s->buffer->data.get()].insert(s->buffer.get());
      if (auto v=o.as<VarNode>();v && v->dtype.is_handle()) escaped.insert(v);
      if (auto c=o.as<CallNode>();c && !Named(c,"tl.tileop.copy") && !Named(c,"tl.region") &&
          !Named(c,"tl.tileop.reduce")) {
        static auto effects=Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
        auto op=c->op.as<Op>();
        bool pure=op.has_value() && effects.count(op.value()) &&
            effects[op.value()]->value==int(CallEffectKind::kPure);
        if (pure && !Named(c,"tirx.address_of")) return;
        // A region descriptor is not an address escape. Other nested loads
        // passed to opaque calls conservatively escape (including address_of).
        PostOrderVisit(GetRef<Call>(c),[&](const ObjectRef &a) {
          if (auto l=a.as<BufferLoadNode>()) escaped.insert(l->buffer->data.get());
        });
      }
    });
    Liveness(body,{});
    return operator()(body);
  }
};
} // namespace
TVM_REGISTER_PASS_CONFIG_OPTION("tl.enable_fuse_cast_copy", Bool);
tirx::transform::Pass FuseCastCopy() {
  auto run=[](PrimFunc f, IRModule, tvm::transform::PassContext) {
    f.CopyOnWrite()->body=CastCopy().Run(f->body); return f;
  };
  return tirx::transform::CreatePrimFuncPass(run,0,"tl.FuseCastCopy",{});
}
TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.transform.FuseCastCopy",FuseCastCopy);
}
} // namespace tvm::tl
