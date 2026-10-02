#include "../op/operator.h"
#include "../layout/layout.h"
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <map>
#include <set>

namespace tvm::tl {
using namespace tirx;
using namespace ffi;
namespace {
struct Uses : StmtExprVisitor {
  std::set<const VarNode *> reads, writes;
  void VisitExpr_(const BufferLoadNode *n) final { reads.insert(n->buffer->data.get()); }
  void VisitStmt_(const BufferStoreNode *n) final {
    writes.insert(n->buffer->data.get()); VisitExpr(n->value);
  }
  void VisitExpr_(const CallNode *n) final {
    if (auto op=n->op.as<OpNode>(); op && op->name=="tl.hexagon.group_reduce") {
      auto load=n->args[5].as<CallNode>()->args[0].as<BufferLoadNode>();
      CHECK(load,ValueError); writes.insert(load->buffer->data.get());
      VisitExpr(n->args[3]); VisitExpr(n->args[4]); return;
    }
    auto tile = ParseOperator(GetRef<Call>(n));
    if (tile.defined()) {
      auto regions = tile->GetAccessRegions();
      for (auto r : regions.reads) reads.insert(r->buffer->data.get());
      for (auto r : regions.writes) writes.insert(r->buffer->data.get());
      return;
    }
    if (auto op = n->op.as<OpNode>(); op &&
        (op->name == "tirx.exp" || op->name == "tirx.exp2" || op->name == "tl.infinity" ||
         op->name == "tirx.reinterpret" || op->name == "tl.hexagon.vector_leaf")) {
      StmtExprVisitor::VisitExpr_(n);
      return;
    }
    if(auto op=n->op.as<OpNode>(); op && op->name=="tl.hexagon.vector_io") {
      CHECK(!n->args.empty(),ValueError);
      auto tag=n->args[0].as<StringImmNode>(); CHECK(tag,ValueError);
      if(tag->value=="load" || tag->value=="store") {
        CHECK(n->args.size()==4,ValueError);
        auto addr=n->args[1].as<CallNode>(); CHECK(addr && addr->args.size()==1,ValueError);
        auto region=addr->args[0].as<BufferLoadNode>(); CHECK(region,ValueError);
        (tag->value=="store"?writes:reads).insert(region->buffer->data.get());
        VisitExpr(n->args[3]); return;
      }
      CHECK(tag->value=="bitcast" || tag->value=="splat16" || tag->value=="splat32" ||
            tag->value=="extract16" || tag->value=="extract32",ValueError);
      StmtExprVisitor::VisitExpr_(n); return;
    }
    CHECK(false, ValueError) << "ownership cannot prove opaque call effects";
  }
};
class Ownership : public StmtExprMutator {
  Array<Map<String, Any>> groups;
  std::map<const VarNode *, std::set<int>> writers, readers;
  std::map<const VarNode *, int> owners;
  std::set<const VarNode *> persistent_shared;
  std::set<const ForNode *> pinned_rows;
  Optional<Var> physical;
  bool in_role{false};
  int parallel_depth{0};
  int64_t role_workers{1};
  int current_role{-1};
  int Canonical(int id) {
    if (!groups[id].count("physical_group")) return id;
    auto physical = groups[id].at("physical_group").cast<Integer>()->value;
    for (int i = 0; i < id; ++i)
      if (groups[i].at("physical_group").cast<Integer>()->value == physical) return i;
    return id;
  }
  int RoleId(const AttrStmtNode *a) {
    auto spec = Downcast<Map<String, Any>>(a->node);
    for (size_t i = 0; i < groups.size(); ++i)
      if (groups[i].at("name").cast<String>() == spec.at("name").cast<String>()) return i;
    CHECK(false, ValueError) << "ownership role absent from plan";
    return -1;
  }
  Stmt Wrap(Stmt s, int id) {
    CHECK(physical.defined(), ValueError) << "ownership requires physical thread binding";
    auto g = groups[id];
    PrimExpr first = g.at("first_worker").cast<Integer>();
    g.Set("local_id", physical.value() - first);
    g.Set("local_size", g.at("worker_count"));
    PrimExpr count = g.at("worker_count").cast<Integer>();
    Stmt owned = AttrStmt(g, "tl.workergroup_local", Integer(1), s);
    return IfThenElse(physical.value() >= first && physical.value() < first + count, owned);
  }
 public:
  Stmt Run(Stmt s) {
    std::map<const VarNode *, std::set<int>> stage_users;
    std::set<const VarNode *> shared;
    PostOrderVisit(s, [&](const ObjectRef &n) {
      if (auto b = n.as<BufferNode>(); b &&
          (GetRef<Buffer>(b).scope() == "shared" || GetRef<Buffer>(b).scope() == "shared.dyn"))
        shared.insert(b->data.get());
      if (auto b = n.as<AllocBufferNode>(); b &&
          (b->buffer.scope() == "shared" || b->buffer.scope() == "shared.dyn"))
        shared.insert(b->buffer->data.get());
      if (auto block = n.as<SBlockNode>())
        for (auto b : block->alloc_buffers)
          if (b.scope() == "shared" || b.scope() == "shared.dyn") shared.insert(b->data.get());
    });
    int pipelines = 0;
    PostOrderVisit(s, [&](const ObjectRef &n) {
      if (auto f = n.as<ForNode>(); f && f->annotations.count("tl.workergroup_groups")) {
        ++pipelines;
        groups = f->annotations.at("tl.workergroup_groups").cast<Array<Map<String, Any>>>();
      }
    });
    CHECK(pipelines == 1, ValueError) << "ownership currently requires one planned pipeline";
    PostOrderVisit(s, [&](const ObjectRef &n) {
      if (auto a = n.as<AttrStmtNode>(); a && a->attr_key == "tl.pipeline_stage") {
        int id = RoleId(a); Uses uses; uses(a->body);
        for (auto b : uses.writes) stage_users[b].insert(id);
        for (auto b : uses.reads) stage_users[b].insert(id);
        for (auto b : uses.writes) writers[b].insert(Canonical(id));
        for (auto b : uses.reads) readers[b].insert(Canonical(id));
      }
    });
    for (auto &[b, ids] : writers) {
      CHECK(ids.size() == 1, ValueError) << "buffer has multiple worker owners";
      int id = *ids.begin();
      if (readers[b].empty() || (readers[b].size() == 1 && readers[b].count(id))) owners[b] = id;
      if (owners.count(b) && shared.count(b) && stage_users[b].size() > 1)
        persistent_shared.insert(b);
    }
    return operator()(s);
  }
 private:
  Stmt VisitStmt_(const AttrStmtNode *a) final {
    if (a->attr_key == tirx::attr::thread_extent) {
      auto iv = Downcast<IterVar>(a->node);
      if (iv->thread_tag == "threadIdx.x") {
        auto old = physical; physical = iv->var;
        auto result = StmtExprMutator::VisitStmt_(a); physical = old; return result;
      }
    }
    if (a->attr_key != "tl.pipeline_stage") return StmtExprMutator::VisitStmt_(a);
    CHECK(!in_role, ValueError) << "nested ownership role";
    in_role = true;
    current_role=RoleId(a);
    role_workers=groups[RoleId(a)].at("worker_count").cast<Integer>()->value;
    // Solve the row-connected component within this stage, not just loops
    // directly touching scalar recurrence state. A matrix producer (mask/scale)
    // and its row reduction consumer must have identical ownership as well.
    struct RowLoops : StmtExprVisitor {
      std::vector<const ForNode *> loops;
      void VisitStmt_(const ForNode *n) final {
        if (n->kind == ForKind::kParallel) { loops.push_back(n); return; }
        StmtExprVisitor::VisitStmt_(n);
      }
    } row_loops;
    row_loops(a->body);
    std::set<const VarNode *> connected = persistent_shared;
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto loop : row_loops.loops) {
        Uses uses; uses(GetRef<Stmt>(loop));
        bool touches = false;
        for (auto b : uses.reads) touches |= connected.count(b);
        for (auto b : uses.writes) touches |= connected.count(b);
        if (!touches) continue;
        pinned_rows.insert(loop);
        // Only non-private buffers connect different row loops. Private staging
        // is reused per row and must not connect unrelated loop domains.
        PostOrderVisit(GetRef<Stmt>(loop), [&](const ObjectRef &obj) {
          auto add = [&](Buffer b) {
            if (b.scope() != "shared" && b.scope() != "shared.dyn") return;
            CHECK(!b->shape.empty() && StructuralEqual()(b->shape[0], loop->extent), ValueError)
                << "row-connected buffer has incompatible row extent";
            changed |= connected.insert(b->data.get()).second;
          };
          auto access = [&](Buffer b, PrimExpr row) {
            if (b.scope() != "shared" && b.scope() != "shared.dyn") return;
            CHECK(row.same_as(loop->loop_var), ValueError)
                << "row-connected access must preserve the outer row index";
            add(b);
          };
          if (auto l = obj.as<BufferLoadNode>()) access(l->buffer, l->indices[0]);
          if (auto s = obj.as<BufferStoreNode>()) access(s->buffer, s->indices[0]);
          if (auto c = obj.as<CallNode>()) {
            auto tile = ParseOperator(GetRef<Call>(c));
            if (tile.defined()) {
              auto regions = tile->GetAccessRegions();
              for (auto r : regions.reads) access(r->buffer, r->region[0]->min);
              for (auto r : regions.writes) access(r->buffer, r->region[0]->min);
            }
          }
        });
      }
    }
    Stmt body = VisitStmt(a->body);
    if (role_workers > 1 && Canonical(RoleId(a)) == RoleId(a)) {
      bool has_state = false;
      for (auto b : persistent_shared) has_state |= owners.at(b) == RoleId(a);
      if (has_state) {
        auto fence = Evaluate(Call(DataType::Int(32), Op::Get("tl.hexagon.workergroup_barrier"), {}));
        body = SeqStmt({fence, body});
      }
    }
    in_role = false;
    pinned_rows.clear();
    return Wrap(body, RoleId(a));
  }
  Stmt Own(Stmt s) {
    if (in_role) return s;
    Uses uses; uses(s);
    std::set<int> candidates;
    for (auto b : uses.reads) if (owners.count(b)) candidates.insert(owners.at(b));
    for (auto b : uses.writes) if (owners.count(b)) candidates.insert(owners.at(b));
    CHECK(candidates.size() == 1, ValueError) << "outside operation has unproven/conflicting ownership";
    int id = *candidates.begin();
    for (auto b : uses.reads)
      CHECK(!writers.count(b) || (owners.count(b) && owners.at(b) == id), ValueError)
          << "outside read conflicts with pipeline shared ownership";
    for (auto b : uses.writes)
      CHECK(!writers.count(b) || (owners.count(b) && owners.at(b) == id), ValueError)
          << "outside write conflicts with pipeline shared ownership";
    // Sequential epilogues can introduce staging storage after the pipeline.
    // Its producer determines the owner of subsequent epilogue reads. Keep the
    // same conflict checks for every later write; never overwrite an owner.
    for (auto b : uses.writes) {
      CHECK(!owners.count(b) || owners.at(b) == id, ValueError)
          << "outside write conflicts with prior epilogue ownership";
      writers[b].insert(id);
      owners[b] = id;
    }
    return Wrap(s, id);
  }
  Stmt VisitStmt_(const EvaluateNode *n) final {
    if (n->value.as<IntImmNode>()) return GetRef<Stmt>(n);
    if(in_role && role_workers>1 && parallel_depth==0) {
      if(auto c=n->value.as<CallNode>()) {
        auto tile=ParseOperator(GetRef<Call>(c));
        if(tile.defined()) {
          auto name=Downcast<Op>(c->op)->name;
          if(name=="tl.tileop.copy" || name=="tl.tileop.transform") {
            auto fence=Evaluate(Call(DataType::Int(32),Op::Get("tl.hexagon.workergroup_barrier"),{}));
            return SeqStmt({fence,GetRef<Stmt>(n),fence});
          }
        }
      }
    }
    return Own(GetRef<Stmt>(n));
  }
  Stmt VisitStmt_(const ForNode *n) final {
    bool pin = pinned_rows.count(n);
    if (in_role && role_workers > 1 && parallel_depth == 0 && n->kind == ForKind::kParallel) {
      Uses uses; uses(GetRef<Stmt>(n));
      for (auto b : uses.reads) pin |= persistent_shared.count(b);
      for (auto b : uses.writes) pin |= persistent_shared.count(b);
    }
    if(n->kind==ForKind::kParallel) ++parallel_depth;
    auto result=StmtExprMutator::VisitStmt_(n);
    if(n->kind==ForKind::kParallel) --parallel_depth;
    if (pin) {
      CHECK(is_zero(n->min), ValueError) << "persistent row layout requires zero-based rows";
      // The planner has already proved a single outer row partition. A hard
      // loop layout keeps scalar-state and matrix-row stages on the same worker.
      IterVar row(Range::FromMinExtent(0, n->extent), n->loop_var, IterVarType::kDataPar);
      Var rep("row_rep", DataType::Int(32));
      Fragment fixed({row}, {floordiv(n->loop_var, Integer(role_workers))},
                     floormod(n->loop_var, Integer(role_workers)),
                     IterVar(Range::FromMinExtent(0, 1), rep, IterVarType::kDataPar));
      auto loop = Downcast<For>(result);
      CHECK(!loop->annotations.count("parallel_loop_layout"), ValueError)
          << "persistent row layout is compiler-owned; remove conflicting explicit layout";
      loop.CopyOnWrite()->annotations.Set("parallel_loop_layout", fixed);
      result = loop;
    }
    return result;
  }
  Stmt VisitStmt_(const BufferStoreNode *n) final { return Own(GetRef<Stmt>(n)); }
  PrimExpr VisitExpr_(const CallNode *n) final {
    auto call=Downcast<Call>(StmtExprMutator::VisitExpr_(n));
    if (auto op=n->op.as<OpNode>(); op && op->name=="tl.hexagon.group_reduce") {
      CHECK(in_role,ValueError);
      auto args=call->args;
      args.Set(2,groups[current_role].count("physical_group") ?
          groups[current_role].at("physical_group").cast<Integer>() : Integer(current_role));
      call.CopyOnWrite()->args=args;
    }
    return call;
  }
};
}
TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.hexagon.transform.InferWorkerOwnership", [] {
    auto pass = [](PrimFunc f, const IRModule &, tvm::transform::PassContext) {
      Stmt body = Ownership().Run(f->body); f.CopyOnWrite()->body = body; return f;
    };
    return tirx::transform::CreatePrimFuncPass(pass, 0, "tl.InferWorkerOwnership", {});
  });
}
}
