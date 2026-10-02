/* Fixed, disjoint worker-group planning. This pass deliberately preserves
 * TileOps: group-local layout/materialization must precede ordinary lowering. */
#include "../op/operator.h"
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/analysis.h>
#include <limits>
#include <map>
#include <set>
#include <vector>

namespace tvm::tl {
using namespace tirx;
using namespace ffi;
namespace {
int64_t Constant(const PrimExpr &e, const char *what) {
  auto n = e.as<IntImmNode>();
  CHECK(n, ValueError) << "worker pipeline requires static " << what;
  return n->value;
}
int64_t Product(int64_t a, int64_t b) {
  CHECK(a >= 0 && b >= 0 && (!b || a <= INT64_MAX / b), ValueError)
      << "worker pipeline resource/generation overflow";
  return a * b;
}
struct Role {
  String name, engine;
  int64_t count, weight, first{0};
  std::set<const VarNode *> reads, writes;
  String owner;
  int64_t physical_group{-1};
  Stmt body;
};

// A deliberately narrow, syntactic ownership proof: every access to the state
// uses the outer parallel row variable as its first index. Equal static row
// domains mean equal worker ownership (the same cyclic Parallel lowering),
// regardless of inner vector/serial loops. Opaque pointer accesses reject.
class RowPartitionProof : public StmtExprVisitor {
  const VarNode *data;
  const ForNode *row{nullptr};
 public:
  bool valid{true}, seen{false};
  int64_t minimum{-1}, extent{-1};
  explicit RowPartitionProof(const VarNode *d):data(d) {}
 private:
  void Access(const Buffer &b,const Array<PrimExpr> &indices) {
    if(b->data.get()!=data) return;
    seen=true;
    if(!row || indices.empty() || !indices[0].same_as(row->loop_var)) { valid=false; return; }
    auto lo=row->min.as<IntImmNode>(), n=row->extent.as<IntImmNode>();
    if(!lo || !n || lo->value!=0 || n->value<=0) { valid=false; return; }
    if(extent!=-1 && (minimum!=lo->value || extent!=n->value)) valid=false;
    minimum=lo->value; extent=n->value;
  }
  void VisitStmt_(const ForNode *n) final {
    auto saved=row;
    if(n->kind==ForKind::kParallel) {
      if(row) valid=false;
      row=n;
    }
    StmtExprVisitor::VisitStmt_(n); row=saved;
  }
  void VisitExpr_(const BufferLoadNode *n) final {
    Access(n->buffer,n->indices); StmtExprVisitor::VisitExpr_(n);
  }
  void VisitStmt_(const BufferStoreNode *n) final {
    Access(n->buffer,n->indices); StmtExprVisitor::VisitStmt_(n);
  }
  void VisitExpr_(const CallNode *n) final {
    auto tile=ParseOperator(GetRef<Call>(n));
    if(tile.defined()) {
      auto name=Downcast<Op>(n->op)->name;
      auto regions=tile->GetAccessRegions();
      auto check=[&](const BufferRegion &r) {
        if(r->buffer->data.get()!=data) return;
        seen=true;
        if((name!="tl.tileop.copy" && name!="tl.tileop.transform") || r->region.empty()) { valid=false; return; }
        auto full=[](const Range &range,const PrimExpr &shape) {
          auto lo=range->min.as<IntImmNode>();
          return lo && lo->value==0 && StructuralEqual()(range->extent,shape);
        };
        for(size_t i=1;i<r->region.size();++i)
          if(!full(r->region[i],r->buffer->shape[i])) valid=false;
        if(row) {
          auto one=r->region[0]->extent.as<IntImmNode>();
          if(!one || one->value!=1) { valid=false; return; }
          Access(r->buffer,{r->region[0]->min});
        } else {
          // Uniform full-region collective. Ownership inserts group fences
          // before and after it, so its partition may differ from row ownership.
          if(!full(r->region[0],r->buffer->shape[0])) valid=false;
        }
      };
      for(auto r:regions.reads) check(r);
      for(auto r:regions.writes) check(r);
      return;
    }
    bool touches=false;
    PostOrderVisit(GetRef<Call>(n),[&](const ObjectRef &v) {
      if(auto p=v.as<VarNode>(); p==data) touches=true;
      if(auto p=v.as<BufferLoadNode>(); p && p->buffer->data.get()==data) touches=true;
    });
    if(touches) valid=false; // includes address_of and TileOp region escape
    StmtExprVisitor::VisitExpr_(n);
  }
};

// Whole-function audit: a declaration inside the producer is not sufficient
// when an epilogue or another loop can mutate the resident value or its source.
class ResidentAudit : public StmtExprVisitor {
  std::map<const VarNode *, Buffer> views;
  std::map<const VarNode *, int> writes;
  std::set<const VarNode *> destinations, sources;
  bool opaque{false};
  void View(Buffer b) {
    auto key = b->data.get();
    if (views.count(key))
      CHECK(views.at(key).same_as(b), ValueError)
          << "resident whole-function buffer alias is unproved";
    else views[key] = b;
  }
  void VisitExpr_(const BufferLoadNode *n) final { View(n->buffer); }
  void VisitStmt_(const BufferStoreNode *n) final {
    View(n->buffer); ++writes[n->buffer->data.get()];
    StmtExprVisitor::VisitStmt_(n);
  }
  void VisitExpr_(const CallNode *n) final {
    auto tile = ParseOperator(GetRef<Call>(n));
    if (tile.defined()) {
      auto regions = tile->GetAccessRegions();
      for (auto r : regions.reads) View(r->buffer);
      for (auto r : regions.writes) { View(r->buffer); ++writes[r->buffer->data.get()]; }
      if (n->annotations.count("tl.worker_resident_loop")) {
        CHECK(regions.reads.size() == 1 && regions.writes.size() == 1, ValueError)
            << "resident declaration requires one source and destination";
        sources.insert(regions.reads[0]->buffer->data.get());
        destinations.insert(regions.writes[0]->buffer->data.get());
      }
      return; // Access regions account for tl.region/address metadata.
    }
    PostOrderVisit(GetRef<Call>(n), [&](const ObjectRef &arg) {
      if (auto v = arg.as<VarNode>(); v && v->dtype.is_handle()) opaque = true;
      if (arg.as<BufferLoadNode>()) opaque = true;
    });
    StmtExprVisitor::VisitExpr_(n);
  }
 public:
  void Run(Stmt body) {
    bool has_resident = false;
    PostOrderVisit(body, [&](const ObjectRef &n) {
      if (auto c = n.as<CallNode>()) has_resident |= c->annotations.count("tl.worker_resident_loop");
    });
    if (!has_resident) return;
    VisitStmt(body);
    CHECK(!opaque, ValueError) << "resident lifetime has unknown pointer escape/effects";
    for (auto key : sources)
      CHECK(writes[key] == 0, ValueError) << "resident source written outside or inside pipeline";
    for (auto key : destinations)
      CHECK(writes[key] == 1, ValueError) << "resident destination has writes outside initializer";
  }
};

class Planner : public StmtExprMutator {
 public:
  Planner(int64_t workers, int64_t events, int64_t bytes)
      : max_workers(workers), max_events(events), max_bytes(bytes) {
    CHECK(workers > 0 && events > 0 && bytes > 0, ValueError)
        << "worker pipeline capabilities must be positive";
  }
 private:
  int64_t max_workers, max_events, max_bytes, team{0}, depth{0};
  int64_t outer_trips{1};
  bool active{false};
  int role{-1};
  std::vector<Role> roles;
  std::map<const VarNode *, Buffer> buffers;
  std::vector<const VarNode *> buffer_order;
  std::vector<const ForNode *> outer_loops;
  std::map<const VarNode *, Var> resident;
  std::map<const VarNode *, const VarNode *> resident_sources;
  std::map<const VarNode *, int> write_count, allocation_level;
  Optional<Var> resident_guard;
  int64_t pipeline_trips{0};
  const ForNode *pipeline_loop{nullptr};
  bool terminal_guard{false};
  std::set<int> terminal_roles;
  std::set<const VarNode *> terminal_writes;
  arith::Analyzer analyzer;
  std::set<const VarNode *> initialized_private;

  void Access(const Buffer &b, bool write) {
    if (!active || role < 0) return;
    auto key = b->data.get();
    auto it = buffers.find(key);
    CHECK(it == buffers.end() || it->second.same_as(b), ValueError)
        << "worker pipeline buffer aliases are unsupported";
    if (it == buffers.end()) buffer_order.push_back(key);
    buffers[key] = b;
    (write ? roles[role].writes : roles[role].reads).insert(key);
    if (write) {
      ++write_count[key];
      if (terminal_guard) terminal_writes.insert(key);
    }
  }
  Stmt VisitStmt_(const AllocBufferNode *op) final {
    allocation_level[op->buffer->data.get()] = outer_loops.size();
    return GetRef<Stmt>(op);
  }
  Stmt VisitStmt_(const SBlockNode *op) final {
    for (auto b : op->alloc_buffers) allocation_level[b->data.get()] = outer_loops.size();
    return StmtExprMutator::VisitStmt_(op);
  }
  void Region(const BufferRegion &r, bool write) {
    CHECK(r->region.size() == r->buffer->shape.size(), ValueError)
        << "worker pipeline region rank mismatch";
    for (size_t i = 0; i < r->region.size(); ++i) {
      const auto &range = r->region[i];
      CHECK(analyzer.CanProve(range->min >= 0) &&
            analyzer.CanProve(range->extent >= 0) &&
            analyzer.CanProve(range->min + range->extent <= r->buffer->shape[i]),
            ValueError) << "worker pipeline region bounds not proven";
    }
    Access(r->buffer, write);
  }
  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    Access(op->buffer, false);
    return StmtExprMutator::VisitExpr_(op);
  }
  Stmt VisitStmt_(const BufferStoreNode *op) final {
    if (active) CHECK(role >= 0, ValueError) << "unowned pipeline store";
    Access(op->buffer, true);
    return StmtExprMutator::VisitStmt_(op);
  }
  PrimExpr VisitExpr_(const CallNode *op) final {
    if (!active) {
      auto tile = ParseOperator(GetRef<Call>(op));
      if (tile.defined() && Downcast<Op>(op->op)->name == "tl.tileop.fill") {
        for (auto r : tile->GetAccessRegions().writes) {
          bool full = r->buffer.scope() == "local" || r->buffer.scope() == "shared" || r->buffer.scope() == "shared.dyn";
          for (size_t i=0; i<r->region.size(); ++i)
            full &= analyzer.CanProveEqual(r->region[i]->min, 0) &&
                    analyzer.CanProveEqual(r->region[i]->extent, r->buffer->shape[i]);
          if (full) initialized_private.insert(r->buffer->data.get());
        }
      }
      return StmtExprMutator::VisitExpr_(op);
    }
    auto tile = ParseOperator(GetRef<Call>(op));
    if (tile.defined()) {
      CHECK(role >= 0, ValueError) << "unowned pipeline TileOp";
      auto name = Downcast<Op>(op->op)->name;
      bool gemm = name == "tl.tileop.gemm";
      CHECK((gemm && roles[role].engine == "hmx") ||
            (!gemm && roles[role].engine == "hvx"), ValueError)
          << "worker pipeline role/engine effect mismatch";
      CHECK(gemm || name == "tl.tileop.copy" || name == "tl.tileop.fill" ||
            name == "tl.tileop.transform" || name == "tl.tileop.reduce", ValueError)
          << "unsupported worker pipeline TileOp " << name;
      auto regions = tile->GetAccessRegions();
      if (resident_guard.defined()) {
        CHECK(name == "tl.tileop.copy" && regions.writes.size() == 1 &&
              regions.reads.size() == 1, ValueError)
            << "resident guard requires a single pure copy";
        auto declared = op->annotations.Get("tl.worker_resident_loop");
        CHECK(declared && declared.value().same_as(resident_guard.value()), ValueError)
            << "conditional copy requires explicit resident loop lifetime";
        auto dst = regions.writes[0], src = regions.reads[0];
        CHECK(dst->buffer.scope() == "shared" || dst->buffer.scope() == "shared.dyn", ValueError)
            << "resident buffer must be shared";
        CHECK(src->buffer.scope() == "global", ValueError) << "resident source must be immutable global storage";
        for (auto r : src->region)
          CHECK(!UsesVar(r->min, [&](const VarNode *v) { return v == resident_guard.value().get(); }) &&
                !UsesVar(r->extent, [&](const VarNode *v) { return v == resident_guard.value().get(); }), ValueError)
              << "resident source varies across reuse loop";
        for (size_t i = 0; i < dst->region.size(); ++i)
          CHECK(analyzer.CanProveEqual(dst->region[i]->min, 0) &&
                analyzer.CanProveEqual(dst->region[i]->extent, dst->buffer->shape[i]), ValueError)
              << "resident producer must initialize complete slot";
        CHECK(!resident.count(dst->buffer->data.get()), ValueError) << "multiple resident initializers";
        CHECK(allocation_level.count(dst->buffer->data.get()) &&
              allocation_level.at(dst->buffer->data.get()) < static_cast<int>(outer_loops.size()), ValueError)
            << "resident allocation must enclose the reuse loop";
        resident.emplace(dst->buffer->data.get(), resident_guard.value());
        resident_sources[dst->buffer->data.get()] = src->buffer->data.get();
      }
      for (auto r : regions.reads) Region(r, false);
      for (auto r : regions.writes) Region(r, true);
      return GetRef<Call>(op);
    }
    if (auto intrinsic=op->op.as<OpNode>(); intrinsic && intrinsic->name=="tl.hexagon.group_reduce") {
      CHECK(role>=0 && roles[role].engine=="hvx",ValueError);
      auto load=op->args[5].as<CallNode>()->args[0].as<BufferLoadNode>();
      CHECK(load,ValueError);
      Region(BufferRegion(load->buffer,{Range::FromMinExtent(0,op->args[6])}),true);
      Array<PrimExpr> args=op->args;
      args.Set(3,VisitExpr(args[3])); args.Set(4,VisitExpr(args[4]));
      return Call(op->dtype,op->op,args,op->annotations,op->span);
    }
    if (auto intrinsic=op->op.as<OpNode>(); intrinsic && intrinsic->name=="tl.hexagon.vector_leaf") {
      CHECK(role>=0 && roles[role].engine=="hvx",ValueError) << "register leaf requires HVX role";
      return StmtExprMutator::VisitExpr_(op);
    }
    if (auto intrinsic=op->op.as<OpNode>(); intrinsic && intrinsic->name=="tl.hexagon.vector_io") {
      CHECK(role>=0 && roles[role].engine=="hvx" && !op->args.empty(),ValueError);
      auto tag=op->args[0].as<StringImmNode>(); CHECK(tag,ValueError);
      if(tag->value=="load" || tag->value=="store") {
        CHECK(op->args.size()==4,ValueError);
        auto address=op->args[1].as<CallNode>();
        CHECK(address && address->op.same_as(builtin::address_of()) && address->args.size()==1,ValueError);
        auto load=address->args[0].as<BufferLoadNode>(); CHECK(load,ValueError);
        CHECK(load->indices.size()==1 && load->buffer->shape.size()==1,ValueError);
        int lanes=1024/load->buffer->dtype.bits();
        Region(BufferRegion(load->buffer,{Range::FromMinExtent(load->indices[0],lanes)}),tag->value=="store");
        Array<PrimExpr> args=op->args;
        args.Set(3,VisitExpr(args[3]));
        return Call(op->dtype,op->op,args,op->annotations,op->span);
      }
      CHECK(tag->value=="bitcast" || tag->value=="splat16" || tag->value=="splat32" ||
            tag->value=="extract16" || tag->value=="extract32",ValueError) << "unknown vector_io effect";
      return StmtExprMutator::VisitExpr_(op);
    }
    // Scalar/vector math is stage-local, but its arguments still carry buffer
    // reads. Do not return the original Call as for TileOps (whose regions
    // already account for reads): that would lose softmax dependencies.
    if (auto intrinsic = op->op.as<OpNode>(); intrinsic &&
        (intrinsic->name == "tirx.exp" || intrinsic->name == "tirx.exp2" || intrinsic->name == "tl.infinity" || intrinsic->name == "tirx.reinterpret")) {
      CHECK(role >= 0 && roles[role].engine == "hvx", ValueError)
          << "worker pipeline math requires an HVX role";
      return StmtExprMutator::VisitExpr_(op);
    }
    // Fail closed: even apparently pure externs can hide engine work/barriers.
    CHECK(false, ValueError) << "unsupported worker pipeline call/effect " << op->op;
    return GetRef<Call>(op);
  }
  Stmt VisitStmt_(const IfThenElseNode *op) final {
    if (active) {
      // A uniform last-iteration epilogue is stage work, not a resident
      // initializer. Keep the condition INSIDE the stage: callback construction
      // must still acquire/publish every edge on every logical iteration.
      if (pipeline_loop && role >= 0 && !resident_guard.defined() &&
          !terminal_guard && !op->else_case.defined() &&
          analyzer.CanProveEqual(op->condition,
              pipeline_loop->loop_var == pipeline_loop->min + pipeline_loop->extent - 1)) {
        CHECK(pipeline_trips > 0, ValueError) << "terminal stage requires a reachable last iteration";
        terminal_roles.insert(role);
        terminal_guard = true;
        auto body = VisitStmt(op->then_case);
        terminal_guard = false;
        return IfThenElse(op->condition, body);
      }
      CHECK(role >= 0 && !resident_guard.defined() && !op->else_case.defined() &&
            !outer_loops.empty() && pipeline_trips == depth, ValueError)
          << "worker pipeline conditional requires full-depth resident lifetime";
      const auto *loop = outer_loops.back();
      CHECK(analyzer.CanProveEqual(op->condition, loop->loop_var == loop->min), ValueError)
          << "resident initialization must be at first reuse iteration";
      auto eval = op->then_case.as<EvaluateNode>();
      CHECK(eval && eval->value.as<CallNode>(), ValueError)
          << "resident guard must contain exactly one copy";
      resident_guard = loop->loop_var;
      auto body = VisitStmt(op->then_case);
      resident_guard = std::nullopt;
      return IfThenElse(op->condition, body);
    }
    auto saved = initialized_private;
    auto result = StmtExprMutator::VisitStmt_(op);
    initialized_private = saved; // a conditional fill is not a dominating initializer
    return result;
  }
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      auto iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        auto old = team;
        team = Constant(op->value, "team extent");
        auto result = StmtExprMutator::VisitStmt_(op);
        team = old;
        return result;
      }
    }
    if (op->attr_key != "tl.pipeline_stage")
      return StmtExprMutator::VisitStmt_(op);
    CHECK(active, ValueError) << "pipeline_stage outside T.Pipelined";
    CHECK(role < 0, ValueError) << "nested pipeline_stage unsupported";
    auto spec = Downcast<Map<String, Any>>(op->node);
    Role r{spec.at("name").cast<String>(), spec.at("engine").cast<String>(),
           spec.at("workers").cast<Integer>()->value,
           spec.at("weight").cast<Integer>()->value};
    if (spec.count("physical_owner")) {
      r.owner = spec.at("physical_owner").cast<String>();
      CHECK(!r.owner.empty() && r.count > 0 && r.weight == 0, ValueError)
          << "explicit physical owner requires an exact worker count";
    }
    CHECK(!r.name.empty() && (r.engine == "hvx" || r.engine == "hmx"), ValueError)
        << "invalid worker role name/engine";
    CHECK((r.count > 0 && r.weight == 0) || (r.count == 0 && r.weight > 0), ValueError)
        << "worker counts/weights must be positive and exclusive";
    CHECK(r.engine != "hmx" || r.count == 1, ValueError) << "HMX singleton required";
    for (const auto &other : roles)
      CHECK(other.name != r.name, ValueError) << "duplicate pipeline role";
    role = roles.size();
    roles.push_back(r);
    Stmt body = VisitStmt(op->body);
    roles[role].body = body;
    role = -1;
    return AttrStmt(op->node, op->attr_key, op->value, body);
  }
  Stmt VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kThreadBinding && op->thread_binding.defined() &&
        op->thread_binding.value()->thread_tag == "threadIdx.x") {
      auto old = team;
      team = Constant(op->extent, "team extent");
      auto result = StmtExprMutator::VisitStmt_(op);
      team = old;
      return result;
    }
    auto it = op->annotations.find("tl.workergroup_depth");
    if (it == op->annotations.end()) {
      if (active) CHECK(role >= 0, ValueError) << "role scope must directly enclose pipeline work";
      auto old = outer_trips;
      if (!active && op->kind == ForKind::kSerial) {
        // Only impose static bounds on outer loops containing worker scopes.
        bool contains = false;
        PostOrderVisit(op->body, [&](const ObjectRef &n) {
          if (auto a = n.as<AttrStmtNode>()) contains |= a->attr_key == "tl.pipeline_stage";
        });
        if (contains) outer_trips = Product(outer_trips, Constant(op->extent, "outer trip count"));
      }
      bool push = !active && op->kind == ForKind::kSerial;
      auto initialized_before_loop = initialized_private;
      if (push) outer_loops.push_back(op);
      analyzer.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
      auto result = StmtExprMutator::VisitStmt_(op);
      if (push) outer_loops.pop_back();
      outer_trips = old;
      initialized_private = initialized_before_loop;
      return result;
    }
    CHECK(!active, ValueError) << "nested worker pipeline resource double booking";
    CHECK(team > 0 && team <= max_workers, ValueError) << "worker pipeline team exceeds capability";
    depth = (*it).second.cast<Integer>()->value;
    CHECK(depth > 0, ValueError) << "positive worker pipeline depth required";
    int64_t trips = Constant(op->extent, "pipeline trip count");
    pipeline_trips = trips;
    pipeline_loop = op;
    Product(outer_trips, trips); // bound the invocation-wide ordinal before launch
    active = true;
    roles.clear(); buffers.clear(); buffer_order.clear(); resident.clear();
    resident_sources.clear(); write_count.clear();
    terminal_roles.clear(); terminal_writes.clear();
    analyzer.Bind(op->loop_var, Range::FromMinExtent(op->min, op->extent));
    Stmt body = VisitStmt(op->body);
    active = false;
    pipeline_loop = nullptr;
    CHECK(!roles.empty(), ValueError) << "empty worker pipeline";
    for (auto id : terminal_roles)
      CHECK(id == static_cast<int>(roles.size()) - 1, ValueError)
          << "conditional terminal effects require the final pipeline stage";
    // Conditional producers cannot satisfy an unconditional consumer. A final
    // stage may update its own state/output, but must not publish such writes
    // as an initialized cross-role slot on skipped iterations.
    for (auto key : terminal_writes)
      for (size_t id = 0; id + 1 < roles.size(); ++id)
        CHECK(!roles[id].reads.count(key) && !roles[id].writes.count(key), ValueError)
            << "conditional terminal buffer has cross-role lifetime";
    // Resource identity is not dependency identity. In particular QK -> HVX ->
    // PV remains a forward logical DAG when QK and PV share an HMX owner.
    std::vector<size_t> physical_roles;
    std::map<std::string, size_t> named_owners;
    bool aliased = false;
    for (size_t i = 0; i < roles.size(); ++i) {
      auto &r = roles[i];
      if (!r.owner.empty() && named_owners.count(r.owner)) {
        auto &first = roles[named_owners.at(r.owner)];
        CHECK(r.engine == first.engine && r.count == first.count, ValueError)
            << "physical owner engine/count mismatch";
        r.physical_group = first.physical_group;
        aliased = true;
      } else {
        r.physical_group = physical_roles.size();
        physical_roles.push_back(i);
        if (!r.owner.empty()) named_owners.emplace(r.owner, i);
      }
    }
    int64_t exact = 0, weighted = 0, weight_sum = 0, hmx = 0;
    for (auto i : physical_roles) {
      const auto &r = roles[i];
      CHECK(r.count <= team && r.weight <= INT32_MAX, ValueError) << "worker role resource overflow";
      exact += r.count; weighted += r.weight > 0; weight_sum += r.weight;
      hmx += r.engine == "hmx";
    }
    CHECK(hmx <= 1, ValueError) << "multiple HMX owners";
    CHECK(exact + weighted <= team && (weighted || exact == team), ValueError)
        << "worker role counts do not fit team";
    if (weighted) {
      int64_t remaining = team - exact - weighted, assigned = 0;
      std::vector<std::pair<int64_t, size_t>> remainders;
      for (size_t i = 0; i < roles.size(); ++i) if (roles[i].weight) {
        auto share = Product(remaining, roles[i].weight);
        roles[i].count = 1 + share / weight_sum;
        assigned += roles[i].count;
        remainders.emplace_back(-(share % weight_sum), i);
      }
      std::sort(remainders.begin(), remainders.end());
      for (int64_t i = 0; i < team - exact - assigned; ++i)
        ++roles[remainders[i].second].count;
    }
    Array<Map<String, Any>> groups;
    Array<Map<String, Any>> physical_groups;
    int64_t cursor = hmx;
    for (auto i : physical_roles) {
      auto &r = roles[i];
      r.first = r.engine == "hmx" ? 0 : cursor;
      if (r.engine != "hmx") cursor += r.count;
      physical_groups.push_back({{"name", r.owner.empty() ? r.name : r.owner}, {"engine", r.engine},
                                 {"first_worker", Integer(r.first)}, {"worker_count", Integer(r.count)}});
    }
    for (auto &r : roles) {
      const auto &physical = roles[physical_roles[r.physical_group]];
      r.first = physical.first;
      r.count = physical.count;
      groups.push_back({{"name", r.name}, {"engine", r.engine},
                        {"first_worker", Integer(r.first)}, {"worker_count", Integer(r.count)}});
      if (aliased) {
        auto g = groups.back();
        g.Set("physical_group", Integer(r.physical_group));
        groups.Set(groups.size() - 1, g);
      }
    }
    Array<Map<String, Any>> edges;
    Array<Map<String, Any>> slots;
    bool merge = op->annotations.count("tl.workergroup_merge_edges") &&
        op->annotations.at("tl.workergroup_merge_edges").cast<Integer>()->value;
    int64_t bytes = 0, events = 0;
    for (const auto *key : buffer_order) {
      const auto &b = buffers.at(key);
      std::vector<size_t> writers, readers;
      for (size_t i = 0; i < roles.size(); ++i) {
        if (roles[i].writes.count(key)) writers.push_back(i);
        if (roles[i].reads.count(key)) readers.push_back(i);
      }
      // A singleton owner has identical element ownership at every stage.
      // Only true private storage can use this proof: shared/fragment layouts
      // and multi-lane ownership need a separate index-equivalence proof.
      if (!writers.empty() && (b.scope() == "local" || b.scope() == "shared" || b.scope() == "shared.dyn") &&
          roles[writers[0]].count >= 1) {
        auto owner = roles[writers[0]].physical_group;
        bool exclusive = true;
        for (auto i : writers) exclusive &= roles[i].physical_group == owner;
        for (auto i : readers) exclusive &= roles[i].physical_group == owner;
        bool cross_stage=writers.size()>1 || std::any_of(readers.begin(),readers.end(),[&](size_t i){return i!=writers[0];});
        if(exclusive && cross_stage && roles[writers[0]].count>1) {
          int64_t rows=-1;
          std::set<size_t> users(writers.begin(),writers.end());
          users.insert(readers.begin(),readers.end());
          for(auto i:users) {
            RowPartitionProof proof(key); proof(roles[i].body);
            CHECK(proof.valid && proof.seen,ValueError)
                << "same-owner state requires explicit identical parallel row partition: " << b->name;
            CHECK(rows==-1 || proof.extent==-1 || rows==proof.extent,ValueError)
                << "same-owner state row partition mismatch: " << b->name;
            if(proof.extent!=-1) rows=proof.extent;
          }
        }
        if (exclusive && (writers.size() > 1 ||
            std::any_of(readers.begin(), readers.end(), [&](size_t i) { return i != writers[0]; }))) {
          CHECK(initialized_private.count(key), ValueError)
              << "same-owner private state requires a dominating complete fill";
          continue;
        }
      }
      CHECK(writers.size() <= 1, ValueError) << "multiple pipeline writers/alias";
      if (writers.empty()) continue; // immutable input
      size_t producer = writers[0];
      std::vector<size_t> consumers;
      for (auto i : readers) if (i != producer) {
        CHECK(i > producer, ValueError) << "pipeline recurrence/back edge unsupported";
        consumers.push_back(i);
      }
      if (consumers.empty()) continue; // private or persistent accumulator
      CHECK(!roles[producer].reads.count(key), ValueError)
          << "cross-role read/write recurrence unsupported: " << b->name;
      CHECK(b.scope() == "shared" || b.scope() == "shared.dyn" || b.scope() == "local.fragment",
            ValueError) << "cross-role intermediate must have materializable shared/fragment storage";
      int64_t size = b->dtype.bytes();
      for (auto extent : b->shape) size = Product(size, Constant(extent, "buffer extent"));
      CHECK(size <= INT64_MAX - 127, ValueError) << "aligned slot overflow";
      int64_t stride = ((size + 127) / 128) * 128;
      int64_t allocation = Product(stride, depth);
      CHECK(allocation <= max_bytes - bytes, ValueError) << "worker slot byte budget exceeded";
      size_t slot = slots.size();
      slots.push_back({{"buffer", b}, {"byte_offset", Integer(bytes)},
                       {"bytes_per_slot", Integer(stride)}, {"depth", Integer(depth)},
                       {"alignment", Integer(128)}});
      bytes += allocation;
      for (auto consumer : consumers) {
        if (merge) {
          bool merged = false;
          for (size_t ei = 0; ei < edges.size(); ++ei) {
            auto edge = edges[ei];
            if (edge.at("producer_group").cast<Integer>()->value == static_cast<int64_t>(producer) &&
                edge.at("consumer_group").cast<Integer>()->value == static_cast<int64_t>(consumer)) {
              auto members = edge.at("protected_slots").cast<Array<Integer>>();
              members.push_back(Integer(slot));
              edge.Set("protected_slots", members);
              if (resident.count(key)) edge.Set("resident_loop", resident.at(key));
              edges.Set(ei, edge);
              merged = true; break;
            }
          }
          if (merged) continue;
        }
        int64_t needed = Product(2, depth);
        CHECK(needed <= max_events - events, ValueError) << "worker event budget exceeded";
        edges.push_back({{"producer_group", Integer(producer)}, {"consumer_group", Integer(consumer)},
                         {"slot_desc", Integer(slot)}, {"depth", Integer(depth)},
                         {"protected_slots", Array<Integer>{Integer(slot)}},
                         {"ready_event_base", Integer(events)}, {"free_event_base", Integer(events + depth)}});
        if (resident.count(key)) {
          auto edge = edges.back(); edge.Set("resident_loop", resident.at(key));
          edges.Set(edges.size() - 1, edge);
        }
        events += needed;
      }
    }
    for (const auto &[key, loop] : resident) {
      CHECK(write_count.at(key) == 1, ValueError) << "resident buffer has additional writes";
      for (auto r : roles)
        CHECK(!r.writes.count(resident_sources.at(key)), ValueError) << "resident source is modified by pipeline";
      CHECK(roles.size() > 1 && buffers.count(key), ValueError) << "unconsumed resident buffer";
      bool protected_slot = false;
      for (auto slot : slots) protected_slot |= slot.at("buffer").cast<Buffer>()->data.get() == key;
      CHECK(protected_slot, ValueError) << "resident lifetime requires a cross-role reader";
    }
    auto result = CopyOnWrite(op);
    result->body = body;
    result->annotations.Set("tl.workergroup_groups", groups);
    if (aliased) {
      result->annotations.Set("tl.workergroup_physical_groups", physical_groups);
      result->annotations.Set("tl.workergroup_stage_aliases", Integer(1));
    }
    result->annotations.Set("tl.workergroup_edges", edges);
    result->annotations.Set("tl.workergroup_slots", slots);
    result->annotations.Set("tl.workergroup_slot_bytes", Integer(bytes));
    result->annotations.Set("tl.workergroup_event_count", Integer(events));
    result->annotations.Set("tl.workergroup_max_ordinal", Integer(Product(outer_trips, trips)));
    String schedule = "round_sync";
    if (op->annotations.count("tl.workergroup_schedule"))
      schedule = op->annotations.at("tl.workergroup_schedule").cast<String>();
    CHECK(schedule == "round_sync" || schedule == "async", ValueError)
        << "worker schedule must be round_sync or async";
    CHECK(!aliased || schedule == "async", ValueError)
        << "physical owner aliases require logical-iteration async ordering";
    std::vector<int64_t> offsets(roles.size(), 0);
    for (size_t c = 0; c < roles.size(); ++c)
      for (auto edge : edges) if (edge.at("consumer_group").cast<Integer>()->value == static_cast<int64_t>(c)) {
        auto p = edge.at("producer_group").cast<Integer>()->value;
        offsets[c] = std::max(offsets[c], offsets[p] + 1);
      }
    if (schedule == "round_sync")
      for (auto edge : edges) {
        auto p = edge.at("producer_group").cast<Integer>()->value;
        auto c = edge.at("consumer_group").cast<Integer>()->value;
        CHECK(depth > offsets[c] - offsets[p], ValueError)
            << "round_sync depth must exceed edge lifetime to prevent same-tick overwrite";
      }
    Array<Integer> stage_offsets;
    for (auto offset : offsets) stage_offsets.push_back(Integer(offset));
    result->annotations.Set("tl.workergroup_schedule", schedule);
    result->annotations.Set("tl.workergroup_stage_offsets", stage_offsets);
    return Stmt(result);
  }
};
} // namespace
TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.hexagon.transform.RejectUnmaterializedWorkerPipeline", [] {
    auto pass = [](PrimFunc f, const IRModule &, tvm::transform::PassContext) {
      PostOrderVisit(f->body, [](const ObjectRef &n) {
        if (auto a = n.as<AttrStmtNode>())
          CHECK(a->attr_key != "tl.pipeline_stage" && a->attr_key != "tl.workergroup_local", ValueError)
              << "worker pipeline requires group-local materialization and single-launch adapter";
      });
      return f;
    };
    return tirx::transform::CreatePrimFuncPass(pass, 0, "tl.RejectUnmaterializedWorkerPipeline", {});
  });
  reflection::GlobalDef().def("tl.hexagon.transform.PlanWorkerPipeline",
      [](int64_t max_workers, int64_t max_events, int64_t max_slot_bytes) {
    auto pass = [=](PrimFunc f, const IRModule &, tvm::transform::PassContext) {
      ResidentAudit().Run(f->body);
      auto n = f.CopyOnWrite();
      n->body = Planner(max_workers, max_events, max_slot_bytes)(f->body);
      return f;
    };
    return tirx::transform::CreatePrimFuncPass(pass, 0, "tl.PlanWorkerPipeline", {});
  });
}
} // namespace tvm::tl
