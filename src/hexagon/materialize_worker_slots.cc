/* Address-level materialization of a planned worker pipeline. The resulting
 * group attributes still require group-local TileOp/layout lowering. */
#include "../op/operator.h"
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <map>

namespace tvm::tl {
using namespace tirx;
using namespace ffi;
namespace {
class Slots : public StmtExprMutator {
 public:
  Stmt Run(Stmt body) {
    // Allocation nodes enclose the loops, so establish all replacements first.
    PostOrderVisit(body, [&](const ObjectRef &n) {
      auto loop = n.as<ForNode>();
      if (!loop) return;
      auto found = loop->annotations.find("tl.workergroup_slots");
      if (found == loop->annotations.end()) return;
      for (auto spec : (*found).second.cast<Array<Map<String, Any>>>()) {
        Buffer old = spec.at("buffer").cast<Buffer>();
        CHECK(!replacements.count(old.get()), ValueError)
            << "buffer shared between worker pipelines is unsupported";
        int64_t depth = spec.at("depth").cast<Integer>()->value;
        int64_t stride = spec.at("bytes_per_slot").cast<Integer>()->value;
        Buffer expanded = old;
        auto b = expanded.CopyOnWrite();
        Array<PrimExpr> shape{Integer(depth)};
        for (auto s : old->shape) shape.push_back(s);
        b->shape = shape;
        // Preserve rank/physical layout within each version and pad the outer
        // stride to the planner's actual 128-byte allocation quantum.
        Array<PrimExpr> strides;
        strides.push_back(Integer(stride / old->dtype.bytes()));
        if (!old->strides.empty()) {
          for (auto s : old->strides) strides.push_back(s);
        } else {
          PrimExpr step = Integer(1);
          std::vector<PrimExpr> reversed;
          for (auto i = old->shape.rbegin(); i != old->shape.rend(); ++i) {
            reversed.push_back(step); step = step * *i;
          }
          for (auto i = reversed.rbegin(); i != reversed.rend(); ++i) strides.push_back(*i);
        }
        b->strides = strides;
        b->data_alignment = 128;
        replacements.emplace(old.get(), expanded);
      }
    });
    return operator()(body);
  }
 private:
  std::map<const BufferNode *, Buffer> replacements;
  PrimExpr ordinal = IntImm(DataType::Int(64), 0);
  PrimExpr slot;
  std::vector<std::tuple<Var, PrimExpr, PrimExpr>> outer;
  std::map<std::string, Map<String, Any>> groups;
  Optional<Var> physical;
  bool active{false};
  Array<PrimExpr> Indices(const Array<PrimExpr> &old) {
    CHECK(active && slot.defined(), ValueError)
        << "versioned buffer access outside owning pipeline";
    Array<PrimExpr> result{slot};
    for (auto i : old) result.push_back(VisitExpr(i));
    return result;
  }
  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto it = replacements.find(op->buffer.get());
    if (it == replacements.end()) return StmtExprMutator::VisitExpr_(op);
    return BufferLoad(it->second, Indices(op->indices), op->predicate);
  }
  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto it = replacements.find(op->buffer.get());
    if (it == replacements.end()) return StmtExprMutator::VisitStmt_(op);
    return BufferStore(it->second, VisitExpr(op->value), Indices(op->indices), op->predicate);
  }
  PrimExpr VisitExpr_(const CallNode *op) final {
    auto name = op->op.as<OpNode>();
    if (name && name->name == "tl.region") {
      auto load = op->args[0].as<BufferLoadNode>();
      if (load && replacements.count(load->buffer.get())) {
        Array<PrimExpr> args{VisitExpr(op->args[0]), op->args[1], Integer(1)};
        for (size_t i = 2; i < op->args.size(); ++i) args.push_back(VisitExpr(op->args[i]));
        return Call(op->dtype, op->op, args);
      }
    }
    return StmtExprMutator::VisitExpr_(op);
  }
  Stmt VisitStmt_(const SBlockNode *op) final {
    auto result = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));
    auto n = result.CopyOnWrite();
    Array<Buffer> alloc;
    for (auto b : n->alloc_buffers) {
      auto it = replacements.find(b.get());
      alloc.push_back(it == replacements.end() ? b : it->second);
    }
    n->alloc_buffers = alloc;
    return result;
  }
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != "tl.pipeline_stage") return StmtExprMutator::VisitStmt_(op);
    CHECK(active && physical.defined(), ValueError) << "worker role missing launch binding";
    auto spec = Downcast<Map<String, Any>>(op->node);
    auto group = groups.at(spec.at("name").cast<String>());
    PrimExpr first = group.at("first_worker").cast<Integer>();
    PrimExpr count = group.at("worker_count").cast<Integer>();
    Stmt body = VisitStmt(op->body);
    Map<String, Any> local = group;
    local.Set("local_id", physical.value() - first);
    local.Set("local_size", count);
    local.Set("ordinal", ordinal);
    // This attribute is not optional: ordinary LowerTileOp must not consume
    // the body until it has adopted these local thread bounds.
    body = AttrStmt(local, "tl.workergroup_local", Integer(1), body);
    return IfThenElse(physical.value() >= first && physical.value() < first + count, body);
  }
  Stmt VisitStmt_(const ForNode *op) final {
    if (op->kind == ForKind::kThreadBinding && op->thread_binding.defined() &&
        op->thread_binding.value()->thread_tag == "threadIdx.x") {
      auto saved = physical;
      physical = op->loop_var;
      auto result = StmtExprMutator::VisitStmt_(op);
      physical = saved;
      return result;
    }
    auto found = op->annotations.find("tl.workergroup_depth");
    if (found == op->annotations.end()) {
      bool is_outer = !active && op->kind == ForKind::kSerial;
      if (is_outer) outer.emplace_back(op->loop_var, op->min, op->extent);
      auto result = StmtExprMutator::VisitStmt_(op);
      if (is_outer) outer.pop_back();
      return result;
    }
    CHECK(!active && op->annotations.count("tl.workergroup_groups"), ValueError)
        << "worker slots require a validated plan";
    // Rectangular outer loops: calculate the lexical linear ordinal by
    // collecting each enclosing loop's extent from its actual binder.
    ordinal = IntImm(DataType::Int(64), 0);
    for (const auto &[var, min, extent] : outer)
      ordinal = ordinal * Cast(DataType::Int(64), extent) + Cast(DataType::Int(64), var - min);
    ordinal = ordinal * Cast(DataType::Int(64), op->extent) +
              Cast(DataType::Int(64), op->loop_var - op->min);
    slot = floormod(ordinal, (*found).second.cast<Integer>());
    groups.clear();
    for (auto g : op->annotations.at("tl.workergroup_groups").cast<Array<Map<String, Any>>>())
      groups.emplace(g.at("name").cast<String>(), g);
    active = true;
    auto result = StmtExprMutator::VisitStmt_(op);
    active = false;
    return result;
  }
};
}
TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.hexagon.transform.MaterializeWorkerSlots", [] {
    auto pass = [](PrimFunc f, const IRModule &, tvm::transform::PassContext) {
      Stmt body = Slots().Run(f->body);
      f.CopyOnWrite()->body = body;
      return f;
    };
    return tirx::transform::CreatePrimFuncPass(pass, 0, "tl.MaterializeWorkerSlots", {});
  });
}
}
