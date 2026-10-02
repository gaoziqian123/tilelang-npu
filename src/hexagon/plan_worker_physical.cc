/* Post-layout allocation planning. Logical TileOp ranks/layouts are unchanged;
 * versions live exclusively in the backing allocation descriptor. */
#include "../op/operator.h"
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <map>
#include <set>
#include "arith/ir_mutator_with_analyzer.h"

namespace tvm::tl {
using namespace tirx;
using namespace ffi;
namespace {
class PhysicalPlan : public StmtExprVisitor {
 public:
  Array<Map<String, Any>> allocations;
  int64_t total{0};
  explicit PhysicalPlan(int64_t budget) : budget_(budget) {
    CHECK(budget > 0, ValueError) << "physical worker allocation budget must be positive";
  }
  void SetSlots(const Array<Map<String, Any>> &slots) {
    for (auto spec : slots) {
      auto b = spec.at("buffer").cast<Buffer>();
      versions_[b->data.get()] = spec.at("depth").cast<Integer>()->value;
    }
  }
  void Run(Stmt body) {
    // TileOp lowering may create additional scale/pack scratch. Include every
    // physical allocation, not just the original A/B/partial buffers.
    PostOrderVisit(body, [&](const ObjectRef &n) {
      if (auto loop = n.as<ForNode>()) {
        auto found = loop->annotations.find("tl.workergroup_slots");
        if (found != loop->annotations.end()) {
          for (auto spec : (*found).second.cast<Array<Map<String, Any>>>()) {
            auto b = spec.at("buffer").cast<Buffer>();
            auto depth = spec.at("depth").cast<Integer>()->value;
            CHECK(depth > 0, ValueError) << "invalid physical slot depth";
            CHECK(!versions_.count(b->data.get()) || versions_.at(b->data.get()) == depth, ValueError)
                << "conflicting physical version counts";
            versions_[b->data.get()] = depth;
          }
        }
      }
    });
    VisitStmt(body);
    for (const auto &[data, depth] : versions_)
      CHECK(seen_.count(data), ValueError)
          << "versioned buffer lost its physical allocation during lowering";
  }
 private:
  int64_t budget_;
  std::map<const VarNode *, int64_t> versions_;
  std::set<const VarNode *> seen_;
  int64_t Static(PrimExpr e) {
    arith::Analyzer analyzer;
    e = analyzer.Simplify(e);
    auto n = e.as<IntImmNode>();
    CHECK(n && n->value >= 0, ValueError) << "physical allocation must be statically bounded";
    return n->value;
  }
  int64_t Mul(int64_t a, int64_t b) {
    CHECK(!b || a <= INT64_MAX / b, ValueError) << "physical allocation overflow";
    return a * b;
  }
  void Allocate(Buffer b) {
    if (b.scope() != "shared" && b.scope() != "shared.dyn") return;
    CHECK(seen_.insert(b->data.get()).second, ValueError)
        << "duplicate/aliased physical worker allocation";
    CHECK(Static(b->elem_offset) == 0, ValueError) << "physical allocation offset must be zero";
    int64_t elements = 1;
    if (b->strides.empty()) {
      for (auto dim : b->shape) elements = Mul(elements, Static(dim));
    } else {
      CHECK(b->strides.size() == b->shape.size(), ValueError) << "physical stride rank mismatch";
      // Highest reachable element, not the logical shape product.
      int64_t highest = 0;
      for (size_t i = 0; i < b->shape.size(); ++i) {
        int64_t extent = Static(b->shape[i]);
        CHECK(extent > 0, ValueError) << "empty physical worker allocation";
        int64_t term = Mul(extent - 1, Static(b->strides[i]));
        CHECK(term <= INT64_MAX - highest, ValueError) << "physical stride overflow";
        highest += term;
      }
      CHECK(highest < INT64_MAX, ValueError) << "physical extent overflow";
      elements = highest + 1;
    }
    int64_t bytes = Mul(elements, b->dtype.bytes());
    CHECK(bytes > 0 && bytes <= INT64_MAX - 127, ValueError) << "invalid physical allocation size";
    int64_t alignment = std::max<int64_t>(128, b->data_alignment);
    CHECK((alignment & (alignment - 1)) == 0, ValueError) << "physical alignment must be power of two";
    CHECK(bytes <= INT64_MAX - (alignment - 1) && total <= INT64_MAX - (alignment - 1), ValueError)
        << "physical alignment overflow";
    int64_t stride = (bytes + alignment - 1) & -alignment;
    int64_t offset = (total + alignment - 1) & -alignment;
    int64_t depth = versions_.count(b->data.get()) ? versions_.at(b->data.get()) : 1;
    int64_t extent = Mul(stride, depth);
    CHECK(offset <= budget_ && extent <= budget_ - offset, ValueError)
        << "physical worker VTCM budget exceeded";
    allocations.push_back({{"buffer", b}, {"data", b->data},
        {"byte_offset", Integer(offset)}, {"bytes_per_slot", Integer(stride)},
        {"depth", Integer(depth)}, {"alignment", Integer(alignment)},
        {"physical_shape", b->shape}, {"physical_strides", b->strides}});
    total = offset + extent;
  }
  void VisitStmt_(const AllocBufferNode *op) final { Allocate(op->buffer); }
  void VisitStmt_(const SBlockNode *op) final {
    for (auto b : op->alloc_buffers) Allocate(b);
    StmtExprVisitor::VisitStmt_(op);
  }
};
}
TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.hexagon.transform.StageWorkerReads", [] {
    auto pass = [](PrimFunc f, const IRModule &, tvm::transform::PassContext) {
      // Deliberately opt-in: preserve the ordinary numerical lowering chain.
      if (!f->GetAttr<Integer>("tl.workergroup_stage_reads").value_or(Integer(0))->value) return f;
      class Stage : public arith::IRMutatorWithAnalyzer {
       public:
        arith::Analyzer analyzer;
        Stage() : arith::IRMutatorWithAnalyzer(&analyzer) {}
        Array<Map<String, Any>> allocations;
        Map<String, Any> group;
        int64_t bytes{0};
        std::map<const VarNode *, Buffer> replacements;
        PrimExpr VisitExpr_(const BufferLoadNode *op) final {
          auto n = Downcast<BufferLoad>(arith::IRMutatorWithAnalyzer::VisitExpr_(op));
          auto it = replacements.find(op->buffer->data.get());
          if (it != replacements.end()) n.CopyOnWrite()->buffer = it->second;
          return n;
        }
        Stmt VisitStmt_(const AttrStmtNode *op) final {
          if (op->attr_key == "tl.workergroup_local") {
            auto saved = group; group = Downcast<Map<String, Any>>(op->node);
            auto result = arith::IRMutatorWithAnalyzer::VisitStmt_(op); group = saved; return result;
          }
          if (op->attr_key != "tl.workergroup_ordinal" || group.empty() ||
              group.at("engine").cast<String>() != "hvx")
            return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
          // Only ordinary read-only buffer consumers. Opaque address escapes,
          // calls and writes to a candidate make that region ineligible.
          std::vector<Buffer> reads;
          std::set<const VarNode *> written, escaped;
          PostOrderVisit(op->body, [&](const ObjectRef &n) {
            if (auto v = n.as<VarNode>(); v && v->dtype.is_handle()) escaped.insert(v);
            if (auto call = n.as<CallNode>()) {
              // Includes address_of(BufferLoad): it is an address, not a read.
              PostOrderVisit(GetRef<Call>(call), [&](const ObjectRef &arg) {
                if (auto load = arg.as<BufferLoadNode>()) escaped.insert(load->buffer->data.get());
              });
            }
            if (auto s = n.as<BufferStoreNode>()) written.insert(s->buffer->data.get());
            if (auto l = n.as<BufferLoadNode>(); l && l->dtype.lanes() == 1) {
              bool found = false;
              for (auto b : reads) {
                if (b->data.same_as(l->buffer->data)) {
                  found = true;
                  if (!b.same_as(l->buffer)) escaped.insert(b->data.get());
                }
              }
              if (!found) reads.push_back(l->buffer);
            }
          });
          Array<Map<String, Any>> stages;
          for (auto b : reads) {
            if (written.count(b->data.get()) || escaped.count(b->data.get())) continue;
            for (auto a : allocations) {
              if (!a.at("data").cast<Var>().same_as(b->data) ||
                  a.at("depth").cast<Integer>()->value <= 1) continue;
              // Flattened contiguous physical backing; no layout rewrite:
              // every consumer index remains exactly the same after staging.
              if (b->shape.size() != 1 || !b->strides.empty() ||
                  !is_zero(b->elem_offset) || b->dtype.lanes() != 1) continue;
              auto extent = b->shape[0].as<IntImmNode>();
              if (!extent || extent->value <= 0 || extent->value > INT64_MAX / b->dtype.bytes()) continue;
              int64_t size = extent->value * b->dtype.bytes();
              if (size % 128 || size > a.at("bytes_per_slot").cast<Integer>()->value) continue;
              auto count = group.at("worker_count").cast<Integer>()->value;
              CHECK(count > 0 && size <= (INT64_MAX - bytes) / count, ValueError) << "DDR staging size overflow";
              Buffer copy = decl_buffer(b->shape, b->dtype, "wg_read_stage", "global");
              stages.push_back({{"source", b}, {"target", copy}, {"offset", Integer(bytes)},
                                {"bytes", Integer(size)}, {"group", group}});
              replacements[b->data.get()] = copy;
              bytes += size * count;
            }
          }
          auto body = arith::IRMutatorWithAnalyzer::VisitStmt(op->body);
          replacements.clear();
          for (int i = int(stages.size()) - 1; i >= 0; --i)
            body = AttrStmt(stages[i], "tl.workergroup_read_stage", Integer(1), body);
          return AttrStmt(op->node, op->attr_key, op->value, body);
        }
      } stage;
      stage.allocations = f->GetAttr<Array<Map<String, Any>>>("tl.workergroup_physical_allocations").value();
      auto body = stage(f->body);
      auto budget = f->GetAttr<Integer>("tl.workergroup_max_ddr_bytes");
      CHECK(budget.defined() && stage.bytes <= budget.value()->value, ValueError)
          << "worker read staging requires explicit sufficient DDR budget";
      auto n = f.CopyOnWrite(); n->body = body;
      if (stage.bytes) {
        Var ddr("wg_ddr", DataType::Handle()); n->params.push_back(ddr);
        f = WithAttr(f, "tl.workergroup_callback_ddr", ddr);
      }
      return WithAttr(f, "tl.workergroup_ddr_bytes", Integer(stage.bytes));
    };
    return tirx::transform::CreatePrimFuncPass(pass, 0, "tl.StageWorkerReads", {});
  });
  reflection::GlobalDef().def("tl.hexagon.transform.PlanWorkerPhysicalAllocations", [](int64_t vtcm_budget) {
    auto pass = [=](PrimFunc f, const IRModule &, tvm::transform::PassContext) {
      PhysicalPlan planner(vtcm_budget);
      if (auto slots = f->GetAttr<Array<Map<String, Any>>>("tl.workergroup_abi_slots"))
        planner.SetSlots(slots.value());
      planner.Run(f->body);
      f = WithAttr(f, "tl.workergroup_physical_allocations", planner.allocations);
      return WithAttr(f, "tl.workergroup_vtcm_bytes", Integer(planner.total));
    };
    return tirx::transform::CreatePrimFuncPass(pass, 0, "tl.PlanWorkerPhysicalAllocations", {});
  });
}
}
