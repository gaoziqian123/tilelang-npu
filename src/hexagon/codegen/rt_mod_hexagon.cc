#include "codegen_hexagon.h"
#include "support/check.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/module.h>
#include <tvm/target/target.h>
#include <sstream>

namespace tvm::codegen {
namespace {
using Spec = ffi::Map<ffi::String, ffi::Any>;
int64_t Number(const Spec &s, const char *key) {
  return s.at(key).cast<Integer>()->value;
}
std::string ParameterType(const Var &p) {
  auto t = p.dtype();
  if (t.is_handle()) {
    auto ptr = p->type_annotation.as<PointerTypeNode>();
    if (!ptr) return "void*";
    auto element = ptr->element_type.as<PrimTypeNode>();
    CHECK(element, ValueError) << "worker pointer parameter needs primitive element type";
    t = element->dtype;
  }
  CHECK(t.lanes() == 1, ValueError) << "worker argpack requires scalar/pointer parameters";
  std::string type;
  if (t.is_float16()) type = "__fp16";
  else if (t == DataType::Float(32)) type = "float";
  else if (t == DataType::Float(64)) type = "double";
  else if (t.is_int()) type = "int" + std::to_string(t.bits()) + "_t";
  else if (t.is_uint()) type = "uint" + std::to_string(t.bits()) + "_t";
  else CHECK(false, ValueError) << "unsupported worker parameter type";
  if (p.dtype().is_handle()) type += "*";
  return type;
}
std::string WorkerWrapper(const PrimFunc &f, const std::string &symbol) {
  auto groups = f->GetAttr<ffi::Array<Spec>>("tl.workergroup_groups").value();
  auto stages = groups;
  auto physical = f->GetAttr<ffi::Array<Spec>>("tl.workergroup_physical_groups");
  bool v4 = physical.defined();
  if (v4) groups = physical.value();
  auto edges = f->GetAttr<ffi::Array<Spec>>("tl.workergroup_edges").value();
  auto schedule = f->GetAttr<ffi::String>("tl.workergroup_schedule").value();
  CHECK(schedule == "round_sync" || schedule == "async", ValueError) << "unknown worker effect/schedule";
  bool round = schedule == "round_sync";
  CHECK(!v4 || !round, ValueError) << "v4 requires async stage scheduling";
  auto rounds = f->GetAttr<Integer>("tl.workergroup_round_count").value()->value;
  CHECK(rounds >= 0 && uint64_t(rounds) <= UINT32_MAX && (round || rounds == 0), ValueError)
      << "invalid worker round_count";
  if (round) edges = {};
  auto allocations = f->GetAttr<ffi::Array<Spec>>("tl.workergroup_physical_allocations").value();
  auto logical_slots = f->GetAttr<ffi::Array<Spec>>("tl.workergroup_abi_slots").value();
  CHECK(f->params.size() >= 3, ValueError) << "worker internal helper missing ABI parameters";
  bool staged = f->GetAttr<Var>("tl.workergroup_callback_ddr").defined();
  auto ddr_bytes = f->GetAttr<Integer>("tl.workergroup_ddr_bytes").value_or(Integer(0))->value;
  size_t users = f->params.size() - (staged ? 4 : 3);
  int64_t team = 0, events = 0;
  std::ostringstream s;
  s << "struct " << symbol << "_args {\n";
  for (size_t i = 0; i < users; ++i) s << ParameterType(f->params[i]) << " p" << i << ";\n";
  s << "void *vtcm; uint64_t initial; void *ddr; };\n";
  s << "static int " << symbol << "_adapter(const tl_wg_worker *w, void *opaque) {\nauto *a = static_cast<"
    << symbol << "_args*>(opaque);\nreturn " << symbol << "(";
  for (size_t i = 0; i < users; ++i) s << "a->p" << i << ",";
  s << "(void*)w,a->vtcm,a->initial" << (staged ? ",a->ddr" : "") << ");\n}\n";
  s << "extern \"C\" int " << symbol << "_owner(const tl_wg_caps *caps, void *ddr, void *vtcm, void *sync, uint64_t sync_bytes, uint64_t initial";
  for (size_t i = 0; i < users; ++i) s << "," << ParameterType(f->params[i]) << " p" << i;
  s << ") {\n";
  if (v4) s << "if (!caps || caps->abi_version != TL_WG_ABI_VERSION_V4) return TL_WG_INVALID;\n";
  auto iterations = f->GetAttr<Integer>("tl.workergroup_iteration_count").value()->value;
  s << "if (initial > UINT64_MAX - " << iterations << "ull) return TL_WG_INVALID;\n";
  auto initial_alignment = f->GetAttr<Integer>("tl.workergroup_initial_alignment").value_or(Integer(1))->value;
  if (initial_alignment > 1)
    s << "if (initial % " << initial_alignment << "u) return TL_WG_INVALID;\n";
  if (staged) {
    auto vtcm_bytes = f->GetAttr<Integer>("tl.workergroup_vtcm_bytes").value()->value;
    s << "if (!ddr || ((uintptr_t)ddr & 127u) || !vtcm) return TL_WG_INVALID;\n"
      << "if ((uintptr_t)ddr >= (uintptr_t)vtcm ? (uintptr_t)ddr-(uintptr_t)vtcm < "
      << vtcm_bytes << "u : (uintptr_t)vtcm-(uintptr_t)ddr < " << ddr_bytes
      << "u) return TL_WG_INVALID;\n";
  }
  s << "const tl_wg_group_desc groups[]={";
  for (auto g : groups) {
    team += Number(g, "worker_count");
    s << "{" << (g.at("engine").cast<ffi::String>() == "hmx" ? 2 : 1) << ","
      << Number(g,"first_worker") << "," << Number(g,"worker_count") << ",0},";
  }
  s << "};\n";
  if (v4) {
    std::vector<int64_t> order(groups.size(), 0);
    s << "const tl_wg_stage_desc stages[]={";
    for (size_t i = 0; i < stages.size(); ++i) {
      auto owner = Number(stages[i], "physical_group");
      CHECK(owner >= 0 && size_t(owner) < groups.size(), ValueError) << "invalid physical stage owner";
      s << "{" << i << "," << owner << "," << order[owner]++ << ",0},";
    }
    s << "};\n";
  }
  s << "const " << (v4 ? "tl_wg_edge_v4_desc" : "tl_wg_edge_desc") << " edges[]={";
  if (edges.empty()) s << "{0}";
  for (auto e : edges) {
    s << "{" << Number(e,"producer_group") << "," << Number(e,"consumer_group") << ","
      << Number(e,"slot_desc") << "," << Number(e,"depth") << "," << Number(e,"ready_event_base")
      << "," << Number(e,"free_event_base") << ",0,0},";
    events = std::max(events, Number(e,"free_event_base") + Number(e,"depth"));
  }
  s << "};\nconst tl_wg_slot_desc slots[]={";
  if (logical_slots.empty()) s << "{0}";
  for (auto logical : logical_slots) {
    auto b = logical.at("buffer").cast<Buffer>();
    bool matched = false;
    for (auto a : allocations) if (a.at("data").cast<Var>().same_as(b->data)) {
      CHECK(!matched, ValueError) << "ambiguous physical ABI slot"; matched = true;
      s << "{" << Number(a,"byte_offset") << "," << Number(a,"bytes_per_slot") << ","
        << Number(a,"depth") << "," << Number(a,"alignment") << ",TL_WG_MEMORY_VTCM,0},";
    }
    CHECK(matched, ValueError) << "ABI slot missing final physical allocation";
  }
  s << "};\n";
  s << (v4 ? "const tl_wg_plan_v4_desc plan={{TL_WG_ABI_VERSION_V4,sizeof(tl_wg_plan_v4_desc)," :
              "const tl_wg_plan_desc plan={TL_WG_ABI_VERSION,sizeof(tl_wg_plan_desc),")
    << team << "," << groups.size() << "," << edges.size() << "," << logical_slots.size()
    << "," << events << ",0," << ddr_bytes << "," << f->GetAttr<Integer>("tl.workergroup_vtcm_bytes").value()->value
    << ",sync_bytes,initial," << f->GetAttr<Integer>("tl.workergroup_iteration_count").value()->value
    << (round ? ",TL_WG_EFFECT_ROUND_SYNC," : ",TL_WG_EFFECT_EXACTLY_ONCE,") << rounds
    << (v4 ? "}," + std::to_string(stages.size()) + ",TL_WG_V4_ASYNC,0,0};\n" : "};\n") << symbol << "_args args={";
  for (size_t i = 0; i < users; ++i) s << "p" << i << ",";
  s << "vtcm,initial,ddr};\nreturn " << (v4 ? "tl_wg_run_v4" : "tl_wg_run") << "(caps,&plan,groups,";
  if (v4) s << "stages," << stages.size() << ",";
  s << "edges,slots," << symbol
    << "_adapter,&args,ddr,vtcm,sync);\n}\n";
  return s.str();
}
}
ffi::Module BuildTileLangHexagonWithoutCompile(IRModule mod, Target target) {
  CodeGenTileLangHexagon cg;
  cg.Init(false);
  ffi::Array<ffi::String> names;
  std::string wrappers;
  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>());
    auto f = Downcast<PrimFunc>(kv.second);
    auto symbol = f->GetAttr<ffi::String>(tvm::attr::kGlobalSymbol).value();
    if (f->GetAttr<Integer>("tl.workergroup_entry").value_or(Integer(0))->value) {
      wrappers += WorkerWrapper(f, symbol);
      names.push_back(std::string(symbol) + "_owner");
    } else {
      ICHECK(f->GetAttr<Integer>(tvm::attr::kCallingConv) == CallingConv::kDeviceKernelLaunch);
      names.push_back(symbol);
    }
    cg.AddFunction(kv.first, f);
  }
  return CSourceModuleCreate(cg.Finish() + wrappers, "c", names);
}
ffi::Module BuildTileLangHexagon(IRModule mod, Target target) {
  auto source = BuildTileLangHexagonWithoutCompile(mod, target);
  if (auto compile = ffi::Function::GetGlobal("tilelang_callback_hexagon_compile")) {
    // The callback owns object loading and returns an executable runtime module.
    return (*compile)(source, target).cast<ffi::Module>();
  }
  return source;
}
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("target.build.tilelang_hexagon", BuildTileLangHexagon)
      .def("target.build.tilelang_hexagon_without_compile",
           BuildTileLangHexagonWithoutCompile);
}
} // namespace tvm::codegen
