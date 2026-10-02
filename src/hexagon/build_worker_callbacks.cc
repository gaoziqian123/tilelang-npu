#include "../op/operator.h"
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <tvm/tirx/builtin.h>
#include <map>

namespace tvm::tl {
using namespace tirx;
using namespace ffi;
namespace {
// ABI4 permits collectives only during an active logical stage. Outside the
// pipeline a proven singleton needs no rendezvous; engine completion calls
// (notably dma_copy_2d_wait) remain untouched. Larger groups need a separately
// modelled stage rather than silently bypassing runtime stage authorization.
class OutsideStageSync : public StmtExprMutator {
 public:
  explicit OutsideStageSync(int64_t count) : count_(count) {}
 private:
  int64_t count_;
  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (auto call = op->value.as<CallNode>()) {
      if (auto name = call->op.as<OpNode>(); name &&
          name->name == "tl.hexagon.workergroup_barrier") {
        CHECK(count_ == 1, ValueError)
            << "v4 collective outside active stage requires an explicit epilogue/initializer stage for multi-member owner";
        return Evaluate(Integer(0));
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};
class CallbackBody : public StmtExprMutator {
 public:
  bool v4{false};
  Array<Map<String, Any>> groups, edges, slots;
  Array<Map<String, Any>> physical_groups;
  int64_t iterations{0};
  int64_t round_count{0};
  String schedule{"async"};
  int64_t initial_alignment{1};
  Var initial{"wg_initial", DataType::UInt(64)};
 private:
  PrimExpr outer = IntImm(DataType::UInt(64), 0), ordinal;
  int64_t depth{0};
  bool pipeline{false};
  bool round{false};
  int64_t outer_count{1};
  Array<Integer> offsets;
  Var tick{"wg_tick", DataType::Int(64)};
  Optional<Var> logical_var;
  PrimExpr logical_min, logical_extent;
  std::map<const VarNode *, PrimExpr> outer_mins;
  Stmt Sync(const char *name, int64_t event, PrimExpr generation) {
    return Evaluate(Call(DataType::Int(32), Op::Get(name),
                         {Integer(event) + floormod(ordinal, make_const(DataType::UInt(64), depth)), generation}));
  }
  Stmt VisitStmt_(const ForNode *op) final {
    auto found = op->annotations.find("tl.workergroup_groups");
    if (found != op->annotations.end()) {
      CHECK(!pipeline && groups.empty(), ValueError) << "callback requires one nonnested worker pipeline";
      groups = (*found).second.cast<Array<Map<String, Any>>>();
      if (op->annotations.count("tl.workergroup_physical_groups"))
        physical_groups = op->annotations.at("tl.workergroup_physical_groups").cast<Array<Map<String, Any>>>();
      edges = op->annotations.at("tl.workergroup_edges").cast<Array<Map<String, Any>>>();
      slots = op->annotations.at("tl.workergroup_slots").cast<Array<Map<String, Any>>>();
      iterations = op->annotations.at("tl.workergroup_max_ordinal").cast<Integer>()->value;
      depth = op->annotations.at("tl.workergroup_depth").cast<Integer>()->value;
      if (!physical_groups.empty()) {
        CHECK(depth == 2 && groups.size() <= 64 &&
              op->annotations.at("tl.workergroup_event_count").cast<Integer>()->value <= 64,
              ValueError) << "v4 requires depth two and at most 64 stages/events";
        for (auto edge : edges)
          CHECK(!edge.count("resident_loop"), ValueError)
              << "v4 resident drains violate exactly-once stage event coverage";
      }
      for (auto edge : edges) if (edge.count("resident_loop")) initial_alignment = depth;
      schedule = op->annotations.at("tl.workergroup_schedule").cast<String>();
      CHECK(schedule == "round_sync" || schedule == "async", ValueError) << "unknown worker schedule";
      round = schedule == "round_sync";
      offsets = op->annotations.at("tl.workergroup_stage_offsets").cast<Array<Integer>>();
      logical_var = op->loop_var; logical_min = op->min; logical_extent = op->extent;
      ordinal = initial + outer * Cast(DataType::UInt(64), op->extent) + Cast(DataType::UInt(64), op->loop_var - op->min);
      pipeline = true;
      auto body = VisitStmt(op->body);
      pipeline = false;
      if (round) {
        int64_t drain = 0;
        for (auto offset : offsets) drain = std::max(drain, offset->value);
        auto trips = op->extent.as<IntImmNode>();
        CHECK(trips && trips->value >= 0, ValueError) << "round schedule requires static nonnegative trip count";
        uint64_t ticks = trips->value ? uint64_t(trips->value) + uint64_t(drain) : 0;
        CHECK(ticks <= UINT32_MAX && (!ticks || uint64_t(outer_count) <= UINT32_MAX / ticks), ValueError)
            << "worker round_count exceeds UINT32_MAX";
        round_count = ticks * outer_count;
        // Every lane executes the same tick loop, including prologue/drain.
        body = SeqStmt({body, Evaluate(Call(DataType::Int(32), Op::Get("tl.hexagon.workergroup_team_barrier"), {}))});
        auto count = Select(op->extent > 0, Cast(DataType::Int(64), op->extent) + make_const(DataType::Int(64), drain),
                            make_const(DataType::Int(64), 0));
        return For(tick, 0, count, ForKind::kSerial, body);
      }
      body = AttrStmt(Integer(0), "tl.workergroup_ordinal", ordinal, body);
      auto n = CopyOnWrite(op); n->body = body; return Stmt(n);
    }
    auto saved = outer;
    auto saved_count = outer_count;
    if (!pipeline && op->kind == ForKind::kSerial) {
      auto trips = op->extent.as<IntImmNode>();
      CHECK(trips && trips->value >= 0, ValueError) << "worker outer loop requires static trip count";
      CHECK(!trips->value || outer_count <= INT64_MAX / trips->value, ValueError) << "worker outer count overflow";
      outer_count *= trips->value;
      outer_mins[op->loop_var.get()] = op->min;
      outer = outer * Cast(DataType::UInt(64), op->extent) + Cast(DataType::UInt(64), op->loop_var - op->min);
    }
    auto result = StmtExprMutator::VisitStmt_(op); outer = saved; outer_count = saved_count; return result;
  }
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != "tl.workergroup_local") return StmtExprMutator::VisitStmt_(op);
    if (!pipeline) {
      if (!v4) return StmtExprMutator::VisitStmt_(op);
      auto spec = Downcast<Map<String, Any>>(op->node);
      auto count = spec.at("worker_count").cast<Integer>()->value;
      return AttrStmt(op->node, op->attr_key, op->value, OutsideStageSync(count)(op->body));
    }
    auto local = Downcast<Map<String, Any>>(op->node);
    int id = -1;
    for (size_t i = 0; i < groups.size(); ++i)
      if (groups[i].at("name").cast<String>() == local.at("name").cast<String>()) id = i;
    CHECK(id >= 0, ValueError) << "unplanned callback group";
    if (round) {
      PrimExpr iteration = tick - offsets[id];
      Map<Var, PrimExpr> replace{{logical_var.value(), Cast(logical_var.value().dtype(), iteration + logical_min)}};
      Stmt body = Substitute(op->body, replace);
      PrimExpr q = Substitute(ordinal, replace);
      body = AttrStmt(Integer(0), "tl.workergroup_ordinal", q, body);
      body = IfThenElse(iteration >= 0 && iteration < logical_extent, body);
      return AttrStmt(op->node, op->attr_key, op->value, body);
    }
    PrimExpr epoch = floordiv(ordinal, make_const(DataType::UInt(64), depth)) + 1;
    Array<Stmt> sequence;
    if (!physical_groups.empty())
      sequence.push_back(Evaluate(Call(DataType::Int(32), Op::Get("tl.hexagon.workergroup_stage_enter"), {Integer(id)})));
    for (auto edge : edges) {
      bool producer = edge.at("producer_group").cast<Integer>()->value == id;
      bool consumer = edge.at("consumer_group").cast<Integer>()->value == id;
      if (producer) {
        sequence.push_back(Sync("tl.hexagon.workergroup_wait", edge.at("free_event_base").cast<Integer>()->value, epoch - 1));
        if (edge.count("resident_loop")) {
          auto reuse = edge.at("resident_loop").cast<Var>();
          CHECK(outer_mins.count(reuse.get()), ValueError) << "resident loop lost before callback construction";
          Array<Stmt> drain;
          // A resident panel can be read in a later reuse iteration after the
          // ordinary per-slot free was published. Before replacing ANY panel,
          // acquire the last generation of EVERY slot from the preceding nest.
          for (int64_t slot = 0; slot < depth; ++slot)
            drain.push_back(Evaluate(Call(DataType::Int(32), Op::Get("tl.hexagon.workergroup_wait"),
                {Integer(edge.at("free_event_base").cast<Integer>()->value + slot), epoch - 1})));
          sequence.push_back(IfThenElse(reuse == outer_mins.at(reuse.get()) &&
              logical_var.value() == logical_min && ordinal > initial, SeqStmt(drain)));
        }
      }
      if (consumer) sequence.push_back(Sync("tl.hexagon.workergroup_wait", edge.at("ready_event_base").cast<Integer>()->value, epoch));
    }
    sequence.push_back(VisitStmt(op->body));
    for (auto edge : edges) {
      if (edge.at("producer_group").cast<Integer>()->value == id)
        sequence.push_back(Sync("tl.hexagon.workergroup_publish", edge.at("ready_event_base").cast<Integer>()->value, epoch));
      if (edge.at("consumer_group").cast<Integer>()->value == id)
        sequence.push_back(Sync("tl.hexagon.workergroup_publish", edge.at("free_event_base").cast<Integer>()->value, epoch));
    }
    if (!physical_groups.empty())
      sequence.push_back(Evaluate(Call(DataType::Int(32), Op::Get("tl.hexagon.workergroup_stage_exit"), {Integer(id)})));
    return AttrStmt(op->node, op->attr_key, op->value, SeqStmt(sequence));
  }
};
}
TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.hexagon.transform.BuildWorkerCallbacks", [] {
    auto pass = [](PrimFunc f, const IRModule &, tvm::transform::PassContext) {
      CHECK(f->attrs.defined() && f->attrs->dict.count("tl.workergroup_physical_allocations"), ValueError)
          << "callback construction requires final physical plan";
      CallbackBody builder;
      PostOrderVisit(f->body, [&](const ObjectRef &node) {
        if (auto loop = node.as<ForNode>())
          builder.v4 |= loop->annotations.count("tl.workergroup_physical_groups");
      });
      Stmt body = builder(f->body);
      CHECK(!builder.groups.empty(), ValueError) << "missing worker pipeline";
      Var worker("wg_worker", DataType::Handle()), vtcm("wg_vtcm", DataType::Handle());
      auto n = f.CopyOnWrite();
      Array<Var> parameters;
      for (auto p : f->params)
        parameters.push_back(f->buffer_map.count(p) ? f->buffer_map.at(p)->data : p);
      n->params = parameters;
      n->buffer_map = {};
      n->params.push_back(worker); n->params.push_back(vtcm); n->params.push_back(builder.initial);
      n->ret_type = PrimType(DataType::Int(32));
      n->body = SeqStmt({body, Evaluate(Call(DataType::Int(32), Op::Get("tl.hexagon.workergroup_complete"), {}))});
      f = WithAttr(f, "tl.workergroup_callback_worker", worker);
      f = WithAttr(f, "tl.workergroup_callback_vtcm", vtcm);
      f = WithAttr(f, "tl.workergroup_groups", builder.groups);
      if (!builder.physical_groups.empty())
        f = WithAttr(f, "tl.workergroup_physical_groups", builder.physical_groups);
      f = WithAttr(f, "tl.workergroup_edges", builder.edges);
      f = WithAttr(f, "tl.workergroup_abi_slots", builder.slots);
      f = WithAttr(f, "tl.workergroup_entry", Integer(1));
      f = WithAttr(f, "tl.workergroup_schedule", builder.schedule);
      f = WithAttr(f, "tl.workergroup_round_count", Integer(builder.round_count));
      f = WithAttr(f, "tl.workergroup_initial_alignment", Integer(builder.initial_alignment));
      return WithAttr(f, "tl.workergroup_iteration_count", Integer(builder.iterations));
    };
    return tirx::transform::CreatePrimFuncPass(pass, 0, "tl.BuildWorkerCallbacks", {});
  });
}
}
