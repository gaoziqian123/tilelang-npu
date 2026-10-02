/* Portable scalar TIR worker callback emission. This is an opt-in semantic
 * execution backend, not an HMX implementation or a performance backend. */
#include "../op/operator.h"
#include <tvm/tirx/stmt_functor.h>
#include <sstream>
#include <map>

namespace tvm::tl {
using namespace tirx;
using namespace ffi;
namespace {
class HostWorkerEmitter {
  std::ostringstream out;
  const ForNode *pipeline{nullptr};
  Array<Map<String, Any>> groups, edges, slots;
  Array<Map<String, Any>> physical;
  bool v4{false};
  std::map<const VarNode *, std::string> vars;
  std::map<const VarNode *, std::string> buffers;
  size_t next_var{0};
  bool in_role{false}, partitioned{false};
  bool in_pipeline{false};
  std::string VarName(const Var &v) {
    auto [it, inserted] = vars.emplace(v.get(), "v" + std::to_string(next_var));
    if (inserted) ++next_var;
    return it->second;
  }
  std::string Type(DataType t) {
    CHECK(t.lanes() == 1, ValueError) << "host worker emitter requires scalar TIR";
    if (t == DataType::Float(32)) return "float";
    if (t == DataType::Float(64)) return "double";
    if (t.is_int()) return "int" + std::to_string(t.bits()) + "_t";
    if (t.is_uint()) return "uint" + std::to_string(t.bits()) + "_t";
    CHECK(false, ValueError) << "unsupported host worker dtype " << t;
    return "";
  }
  std::string Address(Buffer b, Array<PrimExpr> indices) {
    CHECK(buffers.count(b->data.get()), ValueError) << "unbound worker buffer";
    CHECK(indices.size() == b->shape.size(), ValueError) << "buffer rank mismatch";
    PrimExpr offset = b->elem_offset;
    PrimExpr stride = Integer(1);
    for (int i = indices.size() - 1; i >= 0; --i) {
      offset = offset + indices[i] * (b->strides.empty() ? stride : b->strides[i]);
      stride = stride * b->shape[i];
    }
    return buffers.at(b->data.get()) + "[" + Expr(offset) + "]";
  }
  std::string Expr(PrimExpr e) {
    if (auto n = e.as<IntImmNode>()) return std::to_string(n->value) + "LL";
    if (auto n = e.as<FloatImmNode>()) { std::ostringstream s; s.precision(17); s << n->value; return s.str(); }
    if (auto n = e.as<VarNode>()) {
      CHECK(vars.count(n), ValueError) << "unbound worker variable";
      return vars.at(n);
    }
    if (auto n = e.as<CastNode>()) return "((" + Type(n->dtype) + ")(" + Expr(n->value) + "))";
    if (auto n = e.as<BufferLoadNode>()) {
      CHECK(!n->predicate.defined(), ValueError) << "predicated load unsupported";
      return Address(n->buffer, n->indices);
    }
#define BINARY(Node, Op) if (auto n = e.as<Node>()) return "(" + Expr(n->a) + Op + Expr(n->b) + ")";
    BINARY(AddNode, "+") BINARY(SubNode, "-") BINARY(MulNode, "*")
    BINARY(LTNode, "<") BINARY(LENode, "<=") BINARY(GENode, ">=")
    BINARY(EQNode, "==") BINARY(AndNode, "&&")
    // Ordinal/slot expressions are nonnegative, statically bounded by planner.
    BINARY(FloorModNode, "%") BINARY(FloorDivNode, "/")
#undef BINARY
    CHECK(false, ValueError) << "unsupported host worker expression " << e;
    return "";
  }
  int64_t I(Map<String, Any> m, const char *key) { return m.at(key).cast<Integer>()->value; }
  void Sync(size_t role, bool before) {
    for (auto e : edges) {
      if (I(e, "producer_group") == static_cast<int64_t>(role))
        out << "WG_TRY(tl_wg_group_" << (before ? "wait" : "publish")
            << "(w," << I(e, before ? "free_event_base" : "ready_event_base")
            << "+slot," << (before ? "epoch-1" : "epoch") << "));\n";
      if (I(e, "consumer_group") == static_cast<int64_t>(role))
        out << "WG_TRY(tl_wg_group_" << (before ? "wait" : "publish")
            << "(w," << I(e, before ? "ready_event_base" : "free_event_base")
            << "+slot,epoch));\n";
    }
  }
  void StmtCode(Stmt s) {
    if (auto n = s.as<SeqStmtNode>()) { for (auto child : n->seq) StmtCode(child); return; }
    if (auto n = s.as<ForNode>()) {
      if (n->kind == ForKind::kThreadBinding) {
        CHECK(n->thread_binding.defined() && n->thread_binding.value()->thread_tag == "threadIdx.x", ValueError)
            << "host worker emitter supports one thread binding";
        vars[n->loop_var.get()] = "w->physical_id"; StmtCode(n->body); return;
      }
      bool parallel = n->kind == ForKind::kParallel;
      CHECK(n->kind == ForKind::kSerial || parallel, ValueError) << "unsupported worker loop";
      CHECK(!parallel || (in_role && !partitioned), ValueError) << "nested/unowned parallel loop";
      CHECK(!n->step.defined(), ValueError) << "explicit worker loop step unsupported";
      CHECK(!in_pipeline || in_role, ValueError) << "role repetition inside pipeline unsupported";
      auto v = VarName(n->loop_var);
      out << "for(int64_t " << v << "=" << Expr(n->min);
      if (parallel) out << "+w->local_id";
      out << ";" << v << "<" << Expr(n->min + n->extent) << ";" << v << "+="
          << (parallel ? "w->local_size" : "1") << "){\n";
      bool saved = partitioned; partitioned |= parallel;
      if (n == pipeline) {
        in_pipeline = true;
        out << "const uint64_t slot=q%" << I(pipeline->annotations, "tl.workergroup_depth")
            << ", epoch=q/" << I(pipeline->annotations, "tl.workergroup_depth") << "+1;\n";
      }
      StmtCode(n->body);
      if (n == pipeline) out << "++q;\n";
      if (n == pipeline) in_pipeline = false;
      partitioned = saved; out << "}\n"; return;
    }
    if (auto n = s.as<AttrStmtNode>()) {
      CHECK(n->attr_key == "tl.pipeline_stage", ValueError) << "unsupported worker attribute";
      auto spec = Downcast<Map<String, Any>>(n->node);
      size_t id = 0;
      while (id < groups.size() && groups[id].at("name").cast<String>() != spec.at("name").cast<String>()) ++id;
      CHECK(id < groups.size() && !in_role, ValueError) << "unplanned/nested worker role";
      out << "if(w->group_id==" << (v4 ? I(groups[id], "physical_group") : id) << "){\n";
      if (v4) out << "WG_TRY(tl_wg_stage_enter(w," << id << "));\n";
      Sync(id, true); in_role = true; StmtCode(n->body); in_role = false;
      Sync(id, false);
      if (v4) out << "WG_TRY(tl_wg_stage_exit(w," << id << "));\n";
      out << "}\n"; return;
    }
    if (auto n = s.as<BufferStoreNode>()) {
      CHECK(in_role && partitioned && !n->predicate.defined(), ValueError)
          << "host worker store needs explicit role-local Parallel ownership";
      out << Address(n->buffer, n->indices) << "=" << Expr(n->value) << ";\n"; return;
    }
    if (auto n = s.as<EvaluateNode>()) {
      CHECK(n->value.as<IntImmNode>(), ValueError) << "TileOp/extern requires group-aware target lowering";
      return;
    }
    CHECK(false, ValueError) << "unsupported worker statement " << s;
  }
 public:
  String Emit(PrimFunc f) {
    PostOrderVisit(f->body, [&](const ObjectRef &n) {
      if (auto loop = n.as<ForNode>(); loop && loop->annotations.count("tl.workergroup_groups")) {
        CHECK(!pipeline, ValueError) << "one worker pipeline per callback currently supported";
        pipeline = loop;
      }
    });
    CHECK(pipeline, ValueError) << "host worker emission requires planned pipeline";
    groups = pipeline->annotations.at("tl.workergroup_groups").cast<Array<Map<String, Any>>>();
    v4 = pipeline->annotations.count("tl.workergroup_physical_groups");
    physical = v4 ? pipeline->annotations.at("tl.workergroup_physical_groups").cast<Array<Map<String, Any>>>() : groups;
    edges = pipeline->annotations.at("tl.workergroup_edges").cast<Array<Map<String, Any>>>();
    slots = pipeline->annotations.at("tl.workergroup_slots").cast<Array<Map<String, Any>>>();
    int64_t team = 0;
    if (v4) {
      CHECK(I(pipeline->annotations,"tl.workergroup_depth") == 2 && groups.size() <= 64 &&
            I(pipeline->annotations,"tl.workergroup_event_count") <= 64, ValueError)
          << "v4 requires depth two and at most 64 stages/events";
      for (auto e : edges) CHECK(!e.count("resident_loop"), ValueError) << "v4 resident drain unsupported";
    }
    for (auto g : physical) {
      CHECK(g.at("engine").cast<String>() == "hvx", ValueError) << "portable host backend does not implement HMX";
      team += I(g, "worker_count");
    }
    out << "#include <stdint.h>\n#include \"workergroup_abi.h\"\n"
           "#define WG_TRY(x) do { int rc=(x); if(rc) return rc; } while(0)\n"
           "typedef struct {void **buffers; unsigned char *scratch; uint64_t initial;} wg_args;\n"
           "static int wg_callback(const tl_wg_worker *w, void *opaque){\n"
           "wg_args *args=(wg_args*)opaque; uint64_t q=args->initial;\n";
    size_t param = 0;
    for (auto p : f->params) {
      CHECK(f->buffer_map.count(p), ValueError) << "host callback expects buffer parameters";
      Buffer b = f->buffer_map.at(p);
      auto name = "b" + std::to_string(param);
      buffers[b->data.get()] = name;
      out << Type(b->dtype) << " *" << name << "=(" << Type(b->dtype) << "*)args->buffers[" << param++ << "];\n";
    }
    for (size_t i = 0; i < slots.size(); ++i) {
      auto spec = slots[i]; Buffer b = spec.at("buffer").cast<Buffer>();
      // Use the source Buffer's indices unchanged; add its version base at
      // every access. The generated pointer expression is evaluated inside q.
      buffers[b->data.get()] = "((" + Type(b->dtype) + "*)(args->scratch+" +
          std::to_string(I(spec, "byte_offset")) + "+(q%" + std::to_string(I(spec, "depth")) +
          ")*" + std::to_string(I(spec, "bytes_per_slot")) + "))";
    }
    StmtCode(f->body);
    out << "return tl_wg_group_complete(w);}\n";
    out << "int tl_generated_worker_run(const tl_wg_caps *caps, void **buffers, void *scratch, void *sync, uint64_t sync_bytes, uint64_t initial){\n";
    out << "const tl_wg_group_desc groups[]={";
    for (auto g : physical) out << "{1," << I(g,"first_worker") << "," << I(g,"worker_count") << ",0},";
    out << "};\n";
    if (v4) {
      std::vector<int64_t> order(physical.size(), 0);
      out << "const tl_wg_stage_desc stages[]={";
      for (size_t i=0; i<groups.size(); ++i) {
        auto owner=I(groups[i],"physical_group");
        out << "{" << i << "," << owner << "," << order.at(owner)++ << ",0},";
      }
      out << "};\n";
    }
    out << "const " << (v4 ? "tl_wg_edge_v4_desc" : "tl_wg_edge_desc") << " edges[]={";
    if (edges.empty()) out << "{0}";
    for (auto e : edges) out << "{" << I(e,"producer_group") << "," << I(e,"consumer_group") << "," << I(e,"slot_desc") << "," << I(e,"depth") << "," << I(e,"ready_event_base") << "," << I(e,"free_event_base") << ",0,0},";
    out << "};\nconst tl_wg_slot_desc slots[]={";
    if (slots.empty()) out << "{0}";
    for (auto s : slots) out << "{" << I(s,"byte_offset") << "," << I(s,"bytes_per_slot") << "," << I(s,"depth") << ",128,TL_WG_MEMORY_DDR,0},";
    out << "};\n" << (v4 ? "tl_wg_plan_v4_desc plan={{TL_WG_ABI_VERSION_V4,sizeof(tl_wg_plan_v4_desc)," : "tl_wg_plan_desc plan={TL_WG_ABI_VERSION,sizeof(tl_wg_plan_desc),")
        << team << "," << physical.size() << "," << edges.size() << "," << slots.size() << ","
        << I(pipeline->annotations,"tl.workergroup_event_count") << ",0,"
        << I(pipeline->annotations,"tl.workergroup_slot_bytes") << ",0,sync_bytes,initial,"
        << I(pipeline->annotations,"tl.workergroup_max_ordinal") << ",TL_WG_EFFECT_EXACTLY_ONCE,0"
        << (v4 ? "}," + std::to_string(groups.size()) + ",TL_WG_V4_ASYNC,0,0};\n" : "};\n")
        << "wg_args args={buffers,(unsigned char*)scratch,initial};\nreturn "
        << (v4 ? "tl_wg_run_v4(caps,&plan,groups,stages," + std::to_string(groups.size()) + "," : "tl_wg_run(caps,&plan,groups,")
        << "edges,slots,wg_callback,&args,scratch,0,sync);}\n";
    return out.str();
  }
};
}
TVM_FFI_STATIC_INIT_BLOCK() {
  reflection::GlobalDef().def("tl.hexagon.EmitWorkerHost", [](PrimFunc f) { return HostWorkerEmitter().Emit(f); });
}
}
