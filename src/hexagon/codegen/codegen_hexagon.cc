#include "codegen_hexagon.h"
#include "support/check.h"
#include <tvm/tirx/stmt_functor.h>
#include <tvm/arith/analyzer.h>
#include <cmath>

namespace tvm::codegen {
void CodeGenTileLangHexagon::InitFuncState(const PrimFunc &f) {
  CodeGenC::InitFuncState(f);
  vtcm_bytes_ = 0;
  worker_callback_arg_ = f->GetAttr<Var>("tl.workergroup_callback_worker");
  worker_vtcm_arg_ = f->GetAttr<Var>("tl.workergroup_callback_vtcm");
  worker_ddr_arg_ = f->GetAttr<Var>("tl.workergroup_callback_ddr");
  worker_ordinal_arg_ = f->GetAttr<Var>("tl.workergroup_callback_ordinal");
  worker_allocations_.clear();
  if (auto plan = f->GetAttr<ffi::Array<ffi::Map<ffi::String, ffi::Any>>>("tl.workergroup_physical_allocations")) {
    CHECK(worker_callback_arg_.defined() && worker_vtcm_arg_.defined(), ValueError)
        << "physical allocation plan requires callback and VTCM arguments";
    for (auto entry : plan.value()) {
      auto data = entry.at("data").cast<Var>();
      CHECK(worker_allocations_.emplace(data.get(), entry).second, ValueError)
          << "duplicate physical allocation descriptor";
    }
  }
  if (worker_callback_arg_.defined()) {
    bool is_parameter = false;
    for (auto p : f->params) is_parameter |= p.same_as(worker_callback_arg_.value());
    CHECK(is_parameter, ValueError) << "worker callback context must be an explicit parameter";
    auto ret = f->ret_type.as<PrimTypeNode>();
    CHECK(ret && ret->dtype == DataType::Int(32), ValueError)
        << "worker callback must return int32 status";
  }
  for (auto &extent : job_extents_) extent = 1;
  tirx::PostOrderVisit(f->body, [&](const ffi::ObjectRef &node) {
    const auto *attr = node.as<AttrStmtNode>();
    if (!attr || attr->attr_key != tirx::attr::thread_extent) return;
    auto iv = Downcast<IterVar>(attr->node);
    std::string tag = iv->thread_tag;
    if (tag == "blockIdx.x" || tag == "blockIdx.y" || tag == "blockIdx.z") {
      const auto *extent = attr->value.as<IntImmNode>();
      ICHECK(extent && extent->value > 0);
      job_extents_[tag.back() - 'x'] = extent->value;
    }
  });
}
std::string CodeGenTileLangHexagon::Finish() {
  if (stream.str().find("tl::dma_copy_2d_wait") != std::string::npos)
    decl_stream << "#include <tl_templates/hexagon/dma_copy.h>\n";
  if (need_worker_abi_) decl_stream << "#include <tl_templates/hexagon/workergroup_abi.h>\n";
  decl_stream << "#include <stdint.h>\n#include <math.h>\n"
                 "#include <tl_templates/hexagon/common.h>\n";
  // Adapter ABI: returns an aligned, bounds-checked slice of the current
  // cooperative job's HAP-reserved VTCM slab, stable for the entire launch.
  decl_stream << "extern \"C\" void* tl_hex_vtcm_slice(size_t, size_t, size_t);\n";
  if (need_hmx_h_) decl_stream << "#include <tl_templates/hexagon/hmx.h>\n";
  if (need_gemm_h_) decl_stream << "#include <tl_templates/hexagon/gemm.h>\n";
  if (need_hvx_h_) decl_stream << "#include <tl_templates/hexagon/hvx.h>\n";
  if (need_reduce_h_) decl_stream << "#include <tl_templates/hexagon/reduce.h>\n";
  for (const auto &entry : vector_types_) {
    decl_stream << "typedef ";
    PrintType(entry.second.element_of(), decl_stream);
    decl_stream << ' ' << entry.first << " __attribute__((ext_vector_type("
                << entry.second.lanes() << ")));\n";
  }
  return CodeGenC::Finish();
}
void CodeGenTileLangHexagon::PrintFuncPrefix(std::ostream &os) {
  os << "extern \"C\" ";
}
void CodeGenTileLangHexagon::PrintType(DataType t, std::ostream &os) {
  if (t.lanes() > 1) {
    ICHECK(!t.is_scalable_vector());
    const auto name = "tl_hex_vec_" + std::to_string(t.code()) + "_" +
                      std::to_string(t.bits()) + "_" + std::to_string(t.lanes());
    vector_types_.emplace(name, t);
    os << name;
  } else if (t.is_float16()) {
    os << "__fp16";
  } else {
    CodeGenC::PrintType(t, os);
  }
}
void CodeGenTileLangHexagon::BindThreadIndex(const IterVar &iv) {
  std::string tag = iv->thread_tag;
  if (worker_callback_arg_.defined() && tag == "threadIdx.x") {
    std::ostringstream id;
    id << "((const tl_wg_worker*)";
    PrintExpr(worker_callback_arg_.value(), id);
    id << ")->physical_id";
    var_idmap_[iv->var.get()] = id.str();
    return;
  }
  if (worker_callback_arg_.defined() && tag == "blockIdx.x") {
    CHECK(job_extents_[0] == 1 && job_extents_[1] == 1 && job_extents_[2] == 1, ValueError)
        << "worker owner launcher currently requires one logical job";
    var_idmap_[iv->var.get()] = "0";
    return;
  }
  if (tag == "blockIdx.x" || tag == "blockIdx.y" || tag == "blockIdx.z") {
    int axis = tag.back() - 'x';
    int64_t stride = 1;
    for (int i = 0; i < axis; ++i) stride *= job_extents_[i];
    auto id = AllocVarID(iv->var.get());
    PrintIndent();
    stream << "const int " << id << " = ((tl_hex_job_id() / " << stride
           << ") % " << job_extents_[axis] << ");\n";
    return;
  }
  if (tag == "threadIdx.y" || tag == "threadIdx.z") {
    ICHECK(iv->dom.defined() && is_one(iv->dom->extent))
        << "Hexagon only supports unit extent on " << tag;
    var_idmap_[iv->var.get()] = "0";
    return;
  }
  ICHECK(tag == "threadIdx.x" || tag == "blockIdx.x")
      << "Hexagon worker/job launch is one-dimensional: " << tag;
  auto id = AllocVarID(iv->var.get());
  PrintIndent();
  stream << "const int " << id << " = tl_hex_worker_id();\n";
}
void CodeGenTileLangHexagon::PrintStorageSync(const CallNode *op) {
  CHECK(!worker_callback_arg_.defined(), ValueError)
      << "unlowered full-team storage barrier in worker callback";
  if (async_region_) return; // one HMX owner; publication is token completion
  auto scope = op->args[0].as<StringImmNode>();
  ICHECK(scope && (scope->value == "shared" || scope->value == "shared.dyn"));
  PrintIndent();
  stream << "tl_hex_barrier();\n";
}
void CodeGenTileLangHexagon::VisitStmt_(const EvaluateNode *op) {
  if(auto c=op->value.as<CallNode>()) {
    auto n=c->op.as<OpNode>();
    if(n && (n->name=="tl.hexagon.dma_submit" || n->name=="tl.hexagon.dma_wait")) {
      bool submit=n->name=="tl.hexagon.dma_submit";
      CHECK(c->args.size()==(submit?7:1),ValueError);
      auto id=c->args[0].as<IntImmNode>(); CHECK(id && id->value>=0 && id->value<1024,ValueError);
      decl_stream << "#include <tl_native_dma.h>\n";
      PrintIndent();
      if(submit) {
        stream << "{ tl_dma_ticket tl_dma_ticket_" << id->value << "{}; int tl_dma_status=tl_hex_dma_copy_2d_submit(";
        for(size_t i=1;i<c->args.size();++i) { if(i!=1) stream << ','; PrintExpr(c->args[i],stream); }
        stream << ",&tl_dma_ticket_" << id->value << "); if(tl_dma_status) __builtin_trap();\n";
      } else {
        // Error does not prove drain. Never return/publish/free on any error;
        // terminating this PD is the native API's fail-closed recovery boundary.
        stream << "if(tl_hex_dma_wait(tl_dma_ticket_" << id->value << ")) __builtin_trap(); }\n";
      }
      return;
    }
  }
  if (auto call = op->value.as<CallNode>()) {
    auto name = call->op.as<OpNode>();
    if (name && name->name == "tl.hexagon.dma_copy_2d_wait") {
      CHECK(worker_callback_arg_.defined(), ValueError) << "DMA status requires worker callback return ABI";
      PrintIndent(); stream << "{ const int dma_status = tl::dma_copy_2d_wait(";
      for (size_t i=0;i<call->args.size();++i) {
        if(i) stream << ", ";
        PrintExpr(call->args[i],stream);
      }
      stream << "); if (dma_status) return dma_status; }\n";
      return;
    }
  }
  if (auto call = op->value.as<CallNode>()) {
    auto name = call->op.as<OpNode>();
    if (name && (name->name == "tl.hexagon.workergroup_barrier" ||
                 name->name == "tl.hexagon.workergroup_wait" ||
                 name->name == "tl.hexagon.workergroup_publish" ||
                 name->name == "tl.hexagon.workergroup_complete" ||
                 name->name == "tl.hexagon.workergroup_stage_enter" ||
                 name->name == "tl.hexagon.workergroup_stage_exit" ||
                 name->name == "tl.hexagon.workergroup_team_barrier")) {
      CHECK(worker_callback_arg_.defined(), ValueError)
          << "group barrier requires an explicit ABI worker callback context";
      CHECK(!async_region_, ValueError)
          << "group barrier cannot execute inside asynchronous HMX leaf";
      need_worker_abi_ = true;
      PrintIndent();
      std::string primitive = name->name == "tl.hexagon.workergroup_barrier" ? "barrier" :
          name->name == "tl.hexagon.workergroup_wait" ? "wait" :
          name->name == "tl.hexagon.workergroup_publish" ? "publish" : "complete";
      if (name->name == "tl.hexagon.workergroup_team_barrier") primitive = "team_barrier";
      if (name->name == "tl.hexagon.workergroup_stage_enter") primitive = "stage_enter";
      if (name->name == "tl.hexagon.workergroup_stage_exit") primitive = "stage_exit";
      stream << "{ const int wg_status = " << (primitive == "team_barrier" || primitive == "stage_enter" || primitive == "stage_exit" ? "tl_wg_" : "tl_wg_group_") << primitive << "((const tl_wg_worker*)";
      PrintExpr(worker_callback_arg_.value(), stream);
      for (auto arg : call->args) { stream << ", "; PrintExpr(arg, stream); }
      stream << "); if (wg_status != 0) return wg_status; }\n";
      if (primitive == "complete") { PrintIndent(); stream << "return 0;\n"; }
      return;
    }
  }
  CodeGenC::VisitStmt_(op);
}
void CodeGenTileLangHexagon::PrintStorageScope(const std::string &scope,
                                             std::ostream &os) {
  // A linker-placed VTCM section is deliberately not inferred from alignment:
  // aligned DDR is not VTCM. Shared allocations need the runtime slab ABI.
  ICHECK(scope == "global" || scope == "local" || scope.empty())
      << "Hexagon shared allocation requires a runtime VTCM slab pointer";
}
void CodeGenTileLangHexagon::PrintVecBinaryOp(const std::string &op, DataType t,
                                            PrimExpr a, PrimExpr b,
                                            std::ostream &os) {
  if (t.is_float() && (t.bits() == 16 || t.bits() == 32) &&
      t.bits() * t.lanes() % 1024 == 0 && (op == "+" || op == "*" || op == "-")) {
    // The clang vector is the lane-addressable carrier. Transfer bits (never
    // numeric casts) to B's typed HVX wrapper, one 128B register at a time.
    // In particular float32x64 is TWO HVX registers, not one integer vector.
    need_hvx_h_ = true;
    os << "([]("; PrintType(t, os); os << " a, "; PrintType(t, os);
    os << " b) { "; PrintType(t, os); os << " r; ";
    const int chunks = t.bits() * t.lanes() / 1024;
    for (int i = 0; i < chunks; ++i) {
      os << "{ tl::hvx_vec<" << (t.bits() == 16 ? "half_t" : "float")
         << "> x, y; __builtin_memcpy(&x.raw, (const char*)&a + " << i * 128
         << ", 128); __builtin_memcpy(&y.raw, (const char*)&b + " << i * 128
          << ", 128); auto z = tl::" << (op == "+" ? "hvx_add" : op == "-" ? "hvx_sub" : "hvx_mul")
         << "(x, y); __builtin_memcpy((char*)&r + " << i * 128
         << ", &z.raw, 128); } ";
    }
    os << "return r; })("; PrintExpr(a, os); os << ", "; PrintExpr(b, os);
    os << ')';
    return;
  }
  os << '('; PrintExpr(a, os); os << ' ' << op << ' '; PrintExpr(b, os); os << ')';
}
void CodeGenTileLangHexagon::PrintVecElemLoad(const std::string &v, DataType,
                                            int i, std::ostream &os) {
  os << '(' << v << ")[" << i << ']';
}
void CodeGenTileLangHexagon::PrintVecElemStore(const std::string &v, DataType,
                                             int i, const std::string &value) {
  PrintIndent(); stream << v << '[' << i << "] = " << value << ";\n";
}
void CodeGenTileLangHexagon::PrintVecElemLoadExpr(DataType t, int i,
                                                const std::string &v,
                                                std::ostream &os) {
  PrintVecElemLoad(v, t, i, os);
}
std::string CodeGenTileLangHexagon::CastFromTo(std::string v, DataType from,
                                             DataType to) {
  if (from.lanes() > 1 || to.lanes() > 1) {
    ICHECK_EQ(from.lanes(), to.lanes())
        << "Hexagon numeric vector cast requires equal lane counts";
    if (from == to) return v;
    if (from.is_float16() && to.is_float() && to.bits() == 32 && from.lanes() % 64 == 0) {
      decl_stream << "#include <tl_templates/hexagon/cast_layout.h>\n";
      std::ostringstream os;
      os << "([]("; PrintType(from,os); os << " value) { ";
      PrintType(to,os); os << " result; ";
      for (int i = 0; i < from.lanes(); i += 32)
        os << "{ auto bits=tl::detail::float_bits_from_half((const char*)&value+"
           << (i/64)*128 << "," << (i%64)/32 << "); __builtin_memcpy((char*)&result+" << i*4 << ",&bits,128); } ";
      os << "return result; })(" << v << ")";
      return os.str();
    }
    if (from.is_float() && from.bits()==32 && to.is_float() &&
        to.bits()==16 && from.lanes()%32==0) {
      decl_stream << "#include <tl_templates/hexagon/cast_layout.h>\n";
      std::ostringstream os;
      os << "([]("; PrintType(from,os); os << " value) { ";
      PrintType(to,os); os << " result; ";
      os << "tl::cast_linear_f32_f16<" << from.lanes()
         << ">(&result, (const float*)&value); ";
      os << "return result; })(" << v << ")";
      return os.str();
    }
    if (from.is_float() && to.is_float() &&
        (from.bits() == 16 || from.bits() == 32) &&
        (to.bits() == 16 || to.bits() == 32)) {
      std::ostringstream os;
      os << "__builtin_convertvector(" << v << ", ";
      PrintType(to, os); os << ')';
      return os.str();
    }
    // Numeric conversion, not a bitcast. A C vector cast requires equal byte
    // widths and is wrong for fp32 -> fp16. Evaluate the operand exactly once
    // and construct the result from scalar conversions of every TIR lane.
    std::ostringstream os;
    os << "([](";
    PrintType(from, os);
    os << " value) { return (";
    PrintType(to, os);
    os << "){";
    for (int i = 0; i < to.lanes(); ++i) {
      if (i) os << ", ";
      os << '(';
      PrintType(to.element_of(), os);
      os << ")value[" << i << ']';
    }
    os << "}; })(" << v << ')';
    return os.str();
  }
  return CodeGenC::CastFromTo(v, from, to);
}
#define HEX_EXPR(Node) \
  void CodeGenTileLangHexagon::VisitExpr_(const Node *op, std::ostream &os) { \
    CodeGenC::VisitExpr_(op, os); \
  }
HEX_EXPR(CastNode)
HEX_EXPR(NotNode)
#undef HEX_EXPR
void CodeGenTileLangHexagon::VisitExpr_(const SelectNode *op, std::ostream &os) {
  if (op->condition.dtype().lanes() == 1) {
    CodeGenC::VisitExpr_(op, os);
    return;
  }
  // Clang vector ternaries require predicate elements as wide as the result.
  // TIR bool lanes have no such physical width: an i32 comparison may select
  // fp16 values. Canonicalize truth to an all-bits mask before numeric width
  // conversion, preserving truth even for non-comparison boolean expressions.
  os << "(__builtin_convertvector((";
  PrintExpr(op->condition, os);
  os << ") != 0, ";
  PrintType(DataType::Int(op->dtype.bits(), op->dtype.lanes()), os);
  os << ") ? ";
  PrintExpr(op->true_value, os);
  os << " : ";
  PrintExpr(op->false_value, os);
  os << ')';
}
void CodeGenTileLangHexagon::VisitExpr_(const FloatImmNode *op, std::ostream &os) {
  if (std::isinf(op->value)) { os << (op->value < 0 ? "(-INFINITY)" : "INFINITY"); return; }
  if (std::isnan(op->value)) { os << "NAN"; return; }
  CodeGenC::VisitExpr_(op, os);
}
#define HEX_MINMAX(Node, cmp) \
void CodeGenTileLangHexagon::VisitExpr_(const Node *op, std::ostream &os) { \
  os << "([]("; PrintType(op->dtype, os); os << " a, "; \
  PrintType(op->dtype, os); os << " b) { return a " cmp " b ? a : b; })("; \
  PrintExpr(op->a, os); os << ", "; PrintExpr(op->b, os); os << ')'; \
}
HEX_MINMAX(MaxNode, ">")
HEX_MINMAX(MinNode, "<")
#undef HEX_MINMAX
void CodeGenTileLangHexagon::VisitExpr_(const CallNode *op, std::ostream &os) {
  const auto *builtin = op->op.as<OpNode>();
  if (builtin) {
    if (op->op.same_as(tirx::builtin::reinterpret())) {
      CHECK(op->args.size()==1,ValueError);
      auto from=op->args[0].dtype(), to=op->dtype;
      CHECK(from.bits()*from.lanes()==to.bits()*to.lanes(),ValueError);
      // CodeGenC's reinterpret uses SSAGetID keyed by printed expression.
      // A buffer load is not immutable SSA: a store through ANY alias can
      // change it. Keep evaluation local to this expression, without the
      // base generator's function-scope expression cache.
      os << "([](const "; PrintType(from,os); os << "& x) { ";
      PrintType(to,os); os << " y; __builtin_memcpy(&y,&x,sizeof(y)); return ";
      // Hexagon forbids scalar __fp16 return ABI. Reconstruct the half value
      // inside the lambda and promote only at its return boundary.
      if(to.is_float16() && to.lanes()==1) os << "(float)";
      os << "y; })("; PrintExpr(op->args[0],os); os << ')'; return;
    }
    if (builtin->name == "tl.hexagon.group_reduce") {
      CHECK(worker_callback_arg_.defined() && !async_region_,ValueError)
          << "group_reduce requires worker callback";
      CHECK(op->args.size()==7,ValueError);
      decl_stream << "#include <tl_templates/hexagon/group_reduce.h>\n";
      os << "tl::group_reduce_f32<"; PrintExpr(op->args[1],os);
      os << "," << (op->args[0].as<StringImmNode>()->value=="max" ? "true":"false")
         << ">((const tl_wg_worker*)";
      PrintExpr(worker_callback_arg_.value(),os);
      for (int i : {2,3,4}) { os << ","; PrintExpr(op->args[i],os); }
      os << ",(float*)"; PrintExpr(op->args[5],os);
      os << ","; PrintExpr(op->args[6],os); os << ")"; return;
    }
    if (builtin->name == "tl.hexagon.vector_io") {
      CHECK(op->args.size()>=2, ValueError);
      auto tag=op->args[0].as<StringImmNode>();
      CHECK(tag, ValueError); std::string n=tag->value;
      decl_stream << "#include <tl_templates/hexagon/vector_leaf.h>\n";
      auto vector128=[](DataType t) { return t.bits()*t.lanes()==1024; };
      if(n=="bitcast") {
        CHECK(op->args.size()==2 && vector128(op->dtype) && vector128(op->args[1].dtype()), ValueError);
        os << "([]("; PrintType(op->args[1].dtype(),os); os << " x) { "; PrintType(op->dtype,os);
        os << " y; __builtin_memcpy(&y,&x,128); return y; })("; PrintExpr(op->args[1],os); os << ')'; return;
      }
      if(n=="load" || n=="store") {
        bool store=n=="store";
        CHECK(op->args.size()==4 && op->args[1].dtype().is_handle(),ValueError);
        const auto* address=op->args[1].as<CallNode>();
        CHECK(address && address->op.same_as(tirx::builtin::address_of()) && address->args.size()==1,ValueError)
            << "vector memory requires a buffer address with provable extent";
        const auto* region=address->args[0].as<BufferLoadNode>();
        CHECK(region && region->indices.size()==1 && region->buffer->shape.size()==1 &&
              region->buffer->strides.empty(),ValueError) << "vector memory requires compact rank-one region";
        auto extent=region->buffer->shape[0].as<IntImmNode>();
        auto offset=region->indices[0].as<IntImmNode>();
        auto base=region->buffer->elem_offset.as<IntImmNode>();
        int bits=region->buffer->dtype.bits();
        CHECK((bits==16 || bits==32) && extent && offset && base && base->value==0,ValueError)
            << "vector memory bounds must be static";
        CHECK(offset->value>=0 && offset->value%(1024/bits)==0 &&
              offset->value<=extent->value && extent->value-offset->value>=1024/bits,ValueError)
            << "full aligned 128B allocation region not proven";
        auto valid=op->args[2].as<IntImmNode>();
        CHECK(valid && valid->value>=0 && valid->value<=128,ValueError);
        CHECK(vector128(op->args[3].dtype()),ValueError);
        CHECK(store ? op->dtype==DataType::Int(32) : vector128(op->dtype),ValueError);
        os << "([](" << (store?"void*":"const void*") << " p, "; PrintType(op->args[3].dtype(),os);
        os << " x) { HVX_Vector v; __builtin_memcpy(&v,&x,128); ";
        if(store) os << "tl::hvx_leaf::store(p,v," << valid->value << "); return 0; ";
        else { os << "v=tl::hvx_leaf::load(p," << valid->value << ",v); "; PrintType(op->dtype,os);
          os << " y; __builtin_memcpy(&y,&v,128); return y; "; }
        os << "})("; PrintExpr(op->args[1],os); os << ','; PrintExpr(op->args[3],os); os << ')'; return;
      }
      if(n=="splat16" || n=="splat32") {
        CHECK(op->args.size()==2 && op->args[1].dtype()==DataType::UInt(32) && vector128(op->dtype),ValueError);
        os << "([](uint32_t x) { HVX_Vector v=tl::hvx_leaf::" << n << "(x); "; PrintType(op->dtype,os);
        os << " y; __builtin_memcpy(&y,&v,128); return y; })("; PrintExpr(op->args[1],os); os << ')'; return;
      }
      if(n=="extract16" || n=="extract32") {
        CHECK(op->args.size()==3 && vector128(op->args[1].dtype()) && op->dtype==DataType::UInt(32),ValueError);
        auto lane=op->args[2].as<IntImmNode>(); CHECK(lane && lane->value>=0 && lane->value<(n=="extract16"?64:32),ValueError);
        os << "([]("; PrintType(op->args[1].dtype(),os); os << " x) { HVX_Vector v; __builtin_memcpy(&v,&x,128); return tl::hvx_leaf::"
           << n << "(v," << lane->value << "); })("; PrintExpr(op->args[1],os); os << ')'; return;
      }
      CHECK(false, ValueError) << "unsupported vector_io " << n;
    }
    if (builtin->name == "tl.hexagon.vector_leaf") {
      CHECK(op->args.size() >= 4, ValueError) << "malformed HVX leaf";
      auto name = op->args[0].as<StringImmNode>();
      auto mode = op->args[1].as<StringImmNode>();
      auto imm = op->args[2].as<IntImmNode>();
      CHECK(name && mode && imm, ValueError) << "HVX leaf contract must be static";
      const std::string n = name->value;
      const std::unordered_map<std::string, int> arity = {
        {"rotate",1},{"align",2},{"interleave16",1},{"deal16",1},{"select",3},
        {"abs16",1},{"neg16",1},{"abs32",1},{"neg32",1},
        {"add16",2},{"sub16",2},{"mul16",2},{"min16",2},{"max16",2},
        {"add32",2},{"sub32",2},{"mul32",2},{"min32",2},{"max32",2},
        {"sum32_asc",1},{"max32_asc",1},{"widen_linear",1},
        {"narrow_evenodd",2},{"narrow_linear",2},{"widen_evenodd",1},
        {"broadcast16",1},{"broadcast32",1},{"exp2_16_bounded",1},{"exp2_16_nonpositive",1},
        {"div32_nr2",2},{"rcp32_nr2",1},{"fma16",3},{"compare16",2},{"compare32",2},
        {"exp32_v1",1},{"pred_and",2},{"pred_or",2},{"pred_not",1}};
      auto it=arity.find(n);
      CHECK(it != arity.end(), ValueError) << "unsupported HVX leaf " << n;
      CHECK(op->args.size() == size_t(it->second+3), ValueError) << "HVX leaf arity";
      std::string expected="bits_v1";
      if (n=="widen_linear" || n=="widen_evenodd") expected="strict_exact_v1";
      else if(n=="narrow_evenodd" || n=="narrow_linear") expected="rne_gradual_quiet_payload_v1";
      else if(n=="rcp32_nr2" || n=="div32_nr2") expected="nr2_positive_bounded_v1";
      else if(n=="exp2_16_bounded") expected="qf_poly6_neg14_0_v1";
      else if(n=="exp2_16_nonpositive") expected="ggml_qf16_clamp24_v1";
      else if(n=="exp32_v1") expected="hvx_math32_v1";
      else if(n=="fma16") expected="native_target_v1";
      else if(n=="compare16" || n=="compare32") expected="ieee_bits_v1";
      else if(n=="sum32_asc" || n=="max32_asc" ||
              n.substr(0,3)=="add" || n.substr(0,3)=="sub" ||
              n.substr(0,3)=="mul" || n.substr(0,3)=="min" || n.substr(0,3)=="max")
        expected="native_target_v1";
      CHECK(mode->value==expected, ValueError) << "unsupported HVX numerical mode";
      CHECK((n=="rotate" || n=="align") ? (imm->value>=0 && imm->value<128) :
            (n=="widen_linear" || n=="widen_evenodd") ? (imm->value==0 || imm->value==1) :
            n=="broadcast16" ? (imm->value>=0 && imm->value<64) :
            n=="broadcast32" ? (imm->value>=0 && imm->value<32) :
            (n=="compare16" || n=="compare32") ? (imm->value>=0 && imm->value<12) : imm->value==0,
            ValueError) << "invalid HVX immediate";
      CHECK(op->dtype==DataType::UInt(8,128), ValueError) << "HVX requires uint8x128 bit view";
      for(size_t i=3;i<op->args.size();++i)
        CHECK(op->args[i].dtype()==op->dtype, ValueError) << "HVX operand width/type mismatch";
      decl_stream << "#include <tl_templates/hexagon/vector_leaf.h>\n";
      os << "([](";
      for(int i=0;i<it->second;++i) { if(i) os << ", "; PrintType(op->dtype,os); os << " a" << i; }
      os << ") { ";
      for(int i=0;i<it->second;++i) os << "HVX_Vector v" << i << "; __builtin_memcpy(&v" << i << ",&a" << i << ",128); ";
      os << "HVX_Vector r=tl::hvx_leaf::";
      if(n=="compare16" || n=="compare32") os << "compare<" << (n=="compare16"?16:32) << ">";
      else os << n;
      os << "(";
      for(int i=0;i<it->second;++i) { if(i) os << ','; os << 'v' << i; }
      if(n=="rotate" || n=="align" || n=="widen_linear" || n=="widen_evenodd" ||
         n=="broadcast16" || n=="broadcast32") os << ',' << imm->value;
      if(n=="compare16" || n=="compare32") os << ',' << imm->value%6 << ',' << imm->value/6;
      os << "); "; PrintType(op->dtype,os);
      os << " out; __builtin_memcpy(&out,&r,128); return out; })(";
      for(size_t i=3;i<op->args.size();++i) { if(i!=3) os << ','; PrintExpr(op->args[i],os); }
      os << ')'; return;
    }
    if (builtin->name == "tl.hexagon.exp2_hf" || builtin->name == "tl.hexagon.sub_hf") {
      const bool sub = builtin->name == "tl.hexagon.sub_hf";
      ICHECK(op->dtype.is_float() && op->dtype.bits() == 16);
      const int lanes=op->dtype.lanes();
      ICHECK(lanes == 1 || lanes == 32 || lanes == 64)
          << "explicit HF operations require scalar, 32 or 64 lanes";
      for (auto arg : op->args) ICHECK(arg.dtype() == op->dtype);
      decl_stream << "#include <tl_templates/hexagon/hf_math.h>\n";
      if (lanes == 1) {
        // Hexagon forbids __fp16 function parameters/returns. Promote at the
        // ABI boundary only; recover the original IEEE half bits before HVX.
        os << "([](float af";
        if (sub) os << ", float bf";
        os << ") -> float { __fp16 a=(__fp16)af; HVX_Vector va=Q6_V_vzero(); "
              "__builtin_memcpy(&va,&a,2); ";
        if (sub) os << "__fp16 b=(__fp16)bf; HVX_Vector vb=Q6_V_vzero(); __builtin_memcpy(&vb,&b,2); ";
        os << "HVX_Vector vr=tl::" << (sub?"sub_hf(va,vb)":"exp2_hf(va)")
           << "; __fp16 result; __builtin_memcpy(&result,&vr,2); return (float)result; })(";
        for (size_t i=0;i<op->args.size();++i) {
          if(i) os << ", ";
          os << "(float)("; PrintExpr(op->args[i],os); os << ")";
        }
        os << ')'; return;
      }
      os << "([]("; PrintType(op->dtype,os); os << " a";
      if(sub) { os << ", "; PrintType(op->dtype,os); os << " b"; }
      os << ") { HVX_Vector va=Q6_V_vzero(), vb=Q6_V_vzero(); ";
      os << "__builtin_memcpy(&va,&a," << lanes*2 << "); ";
      if(sub) os << "__builtin_memcpy(&vb,&b," << lanes*2 << "); ";
      os << "HVX_Vector vr=tl::" << (sub?"sub_hf(va,vb)":"exp2_hf(va)") << "; ";
      PrintType(op->dtype,os); os << " result; __builtin_memcpy(&result,&vr," << lanes*2 << "); return result; })(";
      for(size_t i=0;i<op->args.size();++i) { if(i) os << ", "; PrintExpr(op->args[i],os); }
      os << ')'; return;
    }
    if (builtin->name == "tl.hexagon.row_map_init" ||
        builtin->name == "tl.hexagon.row_map_update" ||
        builtin->name == "tl.hexagon.row_map_finish") {
      decl_stream << "#include <tl_templates/hexagon/row_map_reduce.h>\n";
      auto maximum = op->args.back().as<IntImmNode>();
      ICHECK(maximum && (maximum->value==0 || maximum->value==1));
      std::string name = builtin->name == "tl.hexagon.row_map_init" ? "row_map_init" :
          builtin->name == "tl.hexagon.row_map_update" ? "row_map_update" : "row_map_finish";
      if (name == "row_map_update") {
        auto t=op->args[1].dtype();
        os << "([](float* s, "; PrintType(t,os);
        os << " value, int offset, int n) { tl::row_map_update<"
           << (maximum->value?"true":"false")
           << ">(s, reinterpret_cast<const float*>(&value), offset, " << t.lanes()
           << ", n); })(";
      } else os << "tl::" << name << "<" << (maximum->value?"true":"false") << ">(";
      for(size_t i=0;i+1<op->args.size();i++) {
        if(i) os << ", "; PrintExpr(op->args[i],os);
      }
      os << ')'; return;
    }
    if (builtin->name == "tl.hexagon.local_row_reduce") {
      ICHECK_EQ(op->args.size(), 6);
      const auto *maximum = op->args[3].as<IntImmNode>();
      const auto *lanes = op->args[4].as<IntImmNode>();
      const auto *contract = op->args[5].as<StringImmNode>();
      ICHECK(maximum && (maximum->value == 0 || maximum->value == 1));
      ICHECK(lanes && lanes->value == 32);
      ICHECK(contract && contract->value ==
          "v1:chain32:rotate16,8,4,2,1:seed_after_tree:ordered_tail:stored_ordered_replay");
      decl_stream << "#include <tl_templates/hexagon/row_reduce.h>\n";
      os << "tl::row_reduce_f32<" << (maximum->value ? "true" : "false") << ">(";
      PrintExpr(op->args[0], os); os << ", ";
      PrintExpr(op->args[1], os); os << ", ";
      PrintExpr(op->args[2], os); os << ')';
      return;
    }
    if (builtin->name == "tl.hexagon.exp_f16_softmax") {
      ICHECK(op->dtype.is_float() && op->dtype.bits() == 16);
      ICHECK(op->dtype.lanes() == 32 || op->dtype.lanes() == 64)
          << "exp_f16_softmax requires 32 or 64 lanes";
      decl_stream << "#include <tl_templates/hexagon/exp.h>\n";
      os << "([]("; PrintType(op->dtype, os); os << " x) { ";
      PrintType(op->dtype, os); os << " y; HVX_Vector v=Q6_V_vzero(); ";
      int bytes = op->dtype.lanes()*2;
      os << "__builtin_memcpy(&v,&x," << bytes << "); "
         << "v=tl::exp_f16_softmax(v); __builtin_memcpy(&y,&v," << bytes
         << "); return y; })(";
      PrintExpr(op->args[0],os); os << ')';
      return;
    }
    if (builtin->name == "tirx.exp" && op->dtype.lanes() > 1) {
      // Lane-wise libm reference lowering; does not change the scalar exp
      // contract (in particular exp(-inf)=0 for masked online softmax).
      // This is deliberately not an approximate HVX exp implementation.
      ICHECK(op->dtype.is_float() && op->dtype.bits() == 32);
      if (op->dtype.lanes() % 32 == 0) {
        decl_stream << "#include <tl_templates/hexagon/exp.h>\n";
        os << "([]("; PrintType(op->dtype, os); os << " x) { ";
        PrintType(op->dtype, os); os << " y; ";
        for (int i=0;i<op->dtype.lanes()/32;++i)
          os << "{ HVX_Vector v; __builtin_memcpy(&v,(const char*)&x+" << i*128
             << ",128); v=tl::native_exp32(v); __builtin_memcpy((char*)&y+" << i*128 << ",&v,128); }";
        os << "return y; })("; PrintExpr(op->args[0],os); os << ')';
        return;
      }
      os << "([]("; PrintType(op->dtype, os); os << " x) { return (";
      PrintType(op->dtype, os); os << "){";
      for (int i = 0; i < op->dtype.lanes(); ++i) {
        if (i) os << ", ";
        os << "expf(x[" << i << "])";
      }
      os << "}; })("; PrintExpr(op->args[0], os); os << ')';
      return;
    }
    if (builtin->name == "tl.hexagon.scatter_release") {
      need_gemm_h_ = true;
      os << "tl::gemm_scatter_release(";
      PrintExpr(op->args[0], os);
      os << ')';
      return;
    }
    static const std::map<std::string, std::string> pack_names = {
        {"tl.hexagon.pack_ah_tile", "tl::gemm_pack_ah_tiles"},
        {"tl.hexagon.pack_wh_tile", "tl::gemm_pack_wh_tiles"},
        {"tl.hexagon.unpack_ah_tile", "tl::gemm_unpack_ah_tiles"},
        {"tl.hexagon.pack_ah_strided", "tl::gemm_pack_ah_strided"},
        {"tl.hexagon.pack_ah_pair_strided", "tl::gemm_pack_ah_pair_strided"},
        {"tl.hexagon.pack_wh_nt_strided", "tl::gemm_pack_wh_nt_strided"},
        {"tl.hexagon.pack_wh_nt_pair_strided", "tl::gemm_pack_wh_nt_pair_strided"},
        {"tl.hexagon.pack_wh_nt_strip", "tl::gemm_pack_wh_nt_strip"},
        {"tl.hexagon.pack_ah_strip", "tl::gemm_pack_ah_strip"},
        {"tl.hexagon.unpack_ah_strided", "tl::gemm_unpack_ah_strided"},
        {"tl.hexagon.unpack_ah_pair_strided", "tl::gemm_unpack_ah_pair_strided"}};
    auto pack = pack_names.find(builtin->name);
    if (pack != pack_names.end()) {
      const bool strided = builtin->name == "tl.hexagon.pack_ah_strided" ||
                           builtin->name == "tl.hexagon.pack_ah_pair_strided" ||
                           builtin->name == "tl.hexagon.pack_wh_nt_strided" ||
                           builtin->name == "tl.hexagon.pack_wh_nt_pair_strided" ||
                           builtin->name == "tl.hexagon.pack_wh_nt_strip" ||
                           builtin->name == "tl.hexagon.pack_ah_strip" ||
                           builtin->name == "tl.hexagon.unpack_ah_strided" ||
                           builtin->name == "tl.hexagon.unpack_ah_pair_strided";
      ICHECK_EQ(op->args.size(), strided ? 4 : 3);
      ICHECK(op->args[0].dtype().is_handle() && op->args[1].dtype().is_handle())
          << "Hexagon pack operands must be pointers";
      if (strided) {
        auto stride_type = op->args[2].dtype();
        ICHECK(stride_type.lanes() == 1 &&
               (stride_type.is_int() || stride_type.is_uint()))
            << "Hexagon pack row_stride must be a scalar integer expression";
      }
      auto tiles = op->args[strided ? 3 : 2].as<IntImmNode>();
      ICHECK(tiles && tiles->value > 0 && tiles->value <= INT32_MAX)
          << "Hexagon pack requires a positive compile-time tile count";
      need_gemm_h_ = true;
      os << pack->second << '<' << tiles->value << ">(";
      if (strided) os << "reinterpret_cast<half_t*>(";
      PrintExpr(op->args[0], os);
      if (strided) os << ')';
      os << ", ";
      if (strided) os << "reinterpret_cast<const half_t*>(";
      PrintExpr(op->args[1], os);
      if (strided) os << ')';
      if (strided) {
        os << ", ";
        PrintExpr(op->args[2], os);
      }
      os << ')';
      return;
    }
    static const std::map<std::string, std::string> names = {
        {"tl.hexagon.profile_mark", "tl::profile_mark"},
        {"tl.hexagon.transpose_2d_2", "tl::transpose_2d<2>"},
        {"tl.hexagon.transpose_2d_4", "tl::transpose_2d<4>"},
        {"tl.hexagon.require", "tl::hex_require"},
        {"tl.hexagon.hmx_mma_deep", "hmx_mma_deep"},
        {"tl.hexagon.hmx_mma_f16", "hmx_mma_f16"},
        {"tl.hexagon.hmx_clear_acc", "hmx_clear_f16"},
        {"tl.hexagon.hmx_init_scale", "hmx_init_scale"},
        {"tl.hexagon.hmx_store_after", "hmx_store_after_f16"}};
    auto it = names.find(builtin->name);
    if (it != names.end()) {
      PrintCallExtern(PrimType(op->dtype), it->second, op->args, false, os);
      return;
    }
  }
  CodeGenC::VisitExpr_(op, os);
}
void CodeGenTileLangHexagon::VisitExpr_(const ShuffleNode *op, std::ostream &os) {
  CHECK(op->vectors.size() >= 1 && op->vectors.size() <= 2 &&
        op->dtype == DataType::Float(16, 64), ValueError)
      << "Hexagon Shuffle supports only one/two half HVX vectors";
  for (auto v : op->vectors)
    CHECK(v.dtype() == op->dtype, ValueError) << "Hexagon Shuffle vector width mismatch";
  bool identity = true, even = op->vectors.size() == 2, odd = even;
  bool interleave_lo = even, interleave_hi = even;
  for (size_t i = 0; i < op->indices.size(); ++i) {
    auto n = op->indices[i].as<IntImmNode>();
    identity &= n && n->value == static_cast<int64_t>(i);
    even &= n && n->value == static_cast<int64_t>(2*i);
    odd &= n && n->value == static_cast<int64_t>(2*i+1);
    interleave_lo &= n && n->value == static_cast<int64_t>(i/2 + (i%2)*64);
    interleave_hi &= n && n->value == static_cast<int64_t>(32+i/2 + (i%2)*64);
  }
  if (identity) { PrintExpr(op->vectors[0], os); return; }
  if (interleave_lo || interleave_hi) {
    need_hvx_h_ = true;
    os << "([]("; PrintType(op->dtype, os); os << " a, ";
    PrintType(op->dtype, os); os << " b) { auto c = __builtin_shufflevector(a,b";
    for (int i=0; i<64; ++i)
      os << ", " << (i < 32 ? i : i+32) + (interleave_hi ? 32 : 0);
    os << "); HVX_Vector x; __builtin_memcpy(&x,&c,128); x=Q6_Vh_vshuff_Vh(x); "
          "__builtin_memcpy(&c,&x,128); return c; })(";
    PrintExpr(op->vectors[0], os); os << ", "; PrintExpr(op->vectors[1], os); os << ")";
    return;
  }
  CHECK(even || odd, ValueError)
      << "Hexagon Shuffle: unsupported permutation (no scalar fallback)";
  need_hvx_h_ = true;
  os << "([]("; PrintType(op->dtype, os); os << " a, ";
  PrintType(op->dtype, os); os << " b) { HVX_Vector x, y; "
      "__builtin_memcpy(&x, &a, 128); __builtin_memcpy(&y, &b, 128); "
      "x = Q6_Vh_vdeal_Vh(x); y = Q6_Vh_vdeal_Vh(y); "
      "__builtin_memcpy(&a, &x, 128); __builtin_memcpy(&b, &y, 128); "
      "return __builtin_shufflevector(a, b";
  for (int i = 0; i < 64; ++i)
    os << ", " << ((i < 32 ? i : i + 32) + (odd ? 32 : 0));
  os << "); })(";
  PrintExpr(op->vectors[0], os); os << ", "; PrintExpr(op->vectors[1], os);
  os << ")";
  return;
}
void CodeGenTileLangHexagon::VisitExpr_(const RampNode *op, std::ostream &os) {
  if ((op->dtype.is_int() || op->dtype.is_uint()) && op->dtype.bits() >= 32) {
    // Keep affine lane construction in vector arithmetic. Expanding base+i*s
    // into scalar expressions makes LLVM build long vinsert/valign chains for
    // a dynamic base. A constant lane vector plus splats is target-independent
    // integer Ramp semantics, with no memory/alignment or FP assumptions.
    os << '(';
    PrintExpr(Broadcast(op->base, op->lanes), os);
    os << " + ";
    PrintExpr(Broadcast(op->stride, op->lanes), os);
    os << " * ("; PrintType(op->dtype, os); os << "){";
    for (int i = 0; i < op->dtype.lanes(); ++i) {
      if (i) os << ", ";
      os << i;
    }
    os << "})";
    return;
  }
  os << '('; PrintType(op->dtype, os); os << "){";
  for (int i = 0; i < op->dtype.lanes(); ++i) {
    if (i) os << ", ";
    PrintExpr(op->base + op->stride * i, os);
  }
  os << '}';
}
void CodeGenTileLangHexagon::VisitExpr_(const BroadcastNode *op, std::ostream &os) {
  // A uniform 32-bit scalar is already in logical lane order. Constructing a
  // long Clang vector initializer can introduce a redundant vdelta shuffle.
  // Evaluate once, preserve float payload bits, and use native HVX splat.
  if (op->dtype.bits() == 32 && op->dtype.lanes() % 32 == 0 &&
      (op->dtype.is_float() || op->dtype.is_int() || op->dtype.is_uint())) {
    decl_stream << "#include <hexagon_types.h>\n#include <hvx_hexagon_protos.h>\n";
    os << "([]("; PrintType(op->value.dtype(), os);
    os << " value) { unsigned bits; __builtin_memcpy(&bits, &value, 4); ";
    PrintType(op->dtype, os);
    os << " result; HVX_Vector splat = Q6_V_vsplat_R(bits); ";
    for (int i = 0; i < op->dtype.lanes() / 32; ++i)
      os << "__builtin_memcpy((char*)&result + " << i * 128
         << ", &splat, 128); ";
    os << "return result; })("; PrintExpr(op->value, os); os << ')';
    return;
  }
  os << '('; PrintType(op->dtype, os); os << "){";
  for (int i = 0; i < op->dtype.lanes(); ++i) {
    if (i) os << ", ";
    PrintExpr(op->value, os);
  }
  os << '}';
}
#define HEX_STMT(Node) \
  void CodeGenTileLangHexagon::VisitStmt_(const Node *op) { \
    CodeGenC::VisitStmt_(op); \
  }
#undef HEX_STMT
void CodeGenTileLangHexagon::VisitStmt_(const ForNode *op) {
  const auto* store=op->body.as<BufferStoreNode>();
  if(store && store->indices.size()==1 && is_zero(op->min) &&
     store->value.dtype()==DataType::Float(32) && GetPtrStorageScope(store->buffer->data)=="local") {
    const auto* sel=store->value.as<SelectNode>();
    const auto* le=sel?sel->condition.as<LENode>():nullptr;
    const auto* load=sel?sel->true_value.as<BufferLoadNode>():nullptr;
    if(le && load && load->buffer.same_as(store->buffer) && load->indices.size()==1) {
      arith::Analyzer az;
      auto zero=[&](PrimExpr e){return tirx::Substitute(e,{{op->loop_var,Integer(0)}});};
      if(az.CanProveEqual(store->indices[0],zero(store->indices[0])+op->loop_var) &&
         az.CanProveEqual(load->indices[0],store->indices[0]) &&
         az.CanProveEqual(le->a,zero(le->a)+op->loop_var) &&
         az.CanProveEqual(le->b,zero(le->b)) &&
         (sel->false_value.same_as(zero(sel->false_value)) ||
          az.CanProveEqual(sel->false_value,zero(sel->false_value)))) {
        decl_stream << "#include <tl_templates/hexagon/mask.h>\n";
        PrintIndent(); stream << "tl::mask_prefix_f32(&" << GetBufferRef(DataType::Float(32),store->buffer.get(),zero(store->indices[0])) << ", ";
        PrintExpr(op->extent,stream); stream << ", "; PrintExpr(az.Simplify(le->b-zero(le->a)),stream);
        stream << ", "; PrintExpr(sel->false_value,stream); stream << ");\n";
        return;
      }
    }
  }
  if (store && store->value.dtype()==DataType::Float(32) &&
      store->indices.size()==1 && is_zero(op->min)) {
    PrimExpr a,b; bool maximum=false;
    if (const auto* add=store->value.as<AddNode>()) { a=add->a; b=add->b; }
    if (const auto* mx=store->value.as<MaxNode>()) { a=mx->a; b=mx->b; maximum=true; }
    const auto* dst=a.defined()?a.as<BufferLoadNode>():nullptr;
    const auto* src=b.defined()?b.as<BufferLoadNode>():nullptr;
    arith::Analyzer analyzer;
    if (dst && src && src->indices.size()==1 && dst->indices.size()==1 &&
        dst->buffer.same_as(store->buffer) && !src->buffer.same_as(dst->buffer)) {
      auto scope=GetPtrStorageScope(src->buffer->data);
      auto dscope=GetPtrStorageScope(dst->buffer->data);
      auto base=tirx::Substitute(src->indices[0], {{op->loop_var, Integer(0)}});
      auto di=tirx::Substitute(dst->indices[0], {{op->loop_var, Integer(0)}});
      if ((scope=="local" || scope=="local.fragment") &&
          (dscope=="local" || dscope=="local.fragment") &&
          analyzer.CanProveEqual(src->indices[0],base+op->loop_var) &&
          analyzer.CanProveEqual(dst->indices[0],di) &&
          analyzer.CanProveEqual(store->indices[0],di)) {
        decl_stream << "#include <tl_templates/hexagon/row_reduce.h>\n";
        auto ref=GetBufferRef(DataType::Float(32),dst->buffer.get(),di);
        PrintIndent(); stream << ref << " = tl::row_reduce_f32<"
            << (maximum?"true":"false") << ">(&"
            << GetBufferRef(DataType::Float(32),src->buffer.get(),base) << ", ";
        PrintExpr(op->extent,stream); stream << ", " << ref << ");\n";
        return;
      }
    }
  }
  // Pure elementwise select loops have no accumulator dependency. Preserve
  // their general broadcast/index expressions for LLVM's HVX vectorizer.
  if (store && store->value.as<SelectNode>() && store->indices.size()==1 &&
      store->value.dtype()==DataType::Float(32) &&
      GetPtrStorageScope(store->buffer->data)=="local") {
    PrintIndent(); stream << "#pragma clang loop vectorize(enable) vectorize_width(32)\n";
  }
  CodeGenC::VisitStmt_(op);
}
void CodeGenTileLangHexagon::VisitExpr_(const BufferLoadNode *op, std::ostream &os) {
  if (op->dtype.lanes() == 1) {
    CodeGenC::VisitExpr_(op, os);
    return;
  }
  ICHECK_EQ(op->indices.size(), 1);
  const auto *ramp = op->indices[0].as<RampNode>();
  ICHECK(ramp) << "Hexagon vector load requires a Ramp index";
  auto scope = GetPtrStorageScope(op->buffer->data);
  const int bytes = op->dtype.bits() * op->dtype.lanes() / 8;
  // Contiguous ordinary-memory vectors must not be reconstructed lane by lane.
  // memcpy carries no alignment promise and is safe for private/DDR addresses.
  // VTCM retains the explicit aligned HVX path below.
  if (scope != "shared" && scope != "shared.dyn" && is_one(ramp->stride)) {
    os << "([](const void* p) { "; PrintType(op->dtype, os);
    os << " result; __builtin_memcpy(&result,p," << bytes << "); return result; })(&"
       << GetBufferRef(op->dtype.element_of(), op->buffer.get(), ramp->base) << ')';
    return;
  }
  const auto *constant_base = ramp->base.as<IntImmNode>();
  const bool known_unaligned = constant_base &&
      (constant_base->value * (op->dtype.bits() / 8)) % 128 != 0;
  int64_t buffer_bytes = op->buffer->dtype.bytes();
  for (const auto &dim : op->buffer->shape) {
    const auto *extent = dim.as<IntImmNode>();
    if (!extent || extent->value <= 0 || buffer_bytes > INT64_MAX / extent->value) {
      buffer_bytes = 0;
      break;
    }
    buffer_bytes *= extent->value;
  }
  if ((scope == "shared" || scope == "shared.dyn") && is_one(ramp->stride) &&
      op->dtype.is_float() && bytes == 64 && buffer_bytes > 0 && buffer_bytes % 128 == 0) {
    // A half-register read is safe only inside a complete aligned allocation.
    // Explicitly load its enclosing register and select the requested half;
    // never rely on hardware's silent address rounding or overread a 64B slab.
    need_hvx_h_ = true;
    const char *element = op->dtype.bits() == 16 ? "half_t" : "float";
    os << "([](const " << element << "* base, const " << element
       << "* p) { tl::hex_require(tl::is_aligned<128>(base)); "
          "tl::hex_require(tl::is_aligned<64>(p)); "
          "uintptr_t address = reinterpret_cast<uintptr_t>(p); "
          "uintptr_t start = reinterpret_cast<uintptr_t>(base); "
          "tl::hex_require(address >= start && address - start <= "
       << buffer_bytes - 64 << "); auto chunk = tl::hvx_load(reinterpret_cast<const "
       << element << "*>(address & ~uintptr_t(127))); ";
    PrintType(op->dtype, os);
    os << " result; __builtin_memcpy(&result, (const char*)&chunk.raw + (address & 127), 64); return result; })(reinterpret_cast<const "
       << element << "*>(&" << GetBufferRef(op->dtype.element_of(), op->buffer.get(), Integer(0))
       << "), reinterpret_cast<const " << element << "*>(&"
       << GetBufferRef(op->dtype.element_of(), op->buffer.get(), ramp->base) << "))";
    return;
  }
  if ((scope == "shared" || scope == "shared.dyn") &&
      !known_unaligned && is_one(ramp->stride) && op->dtype.is_float() &&
      (op->dtype.bits() == 16 || op->dtype.bits() == 32) && bytes % 128 == 0) {
    // Shared storage (including aliases produced by storage rewrite) is a
    // VTCM slab. Check the actual effective address, not just the slab base:
    // a dynamic offset must never silently round down on HVX.
    need_hvx_h_ = true;
    const char *element = op->dtype.bits() == 16 ? "half_t" : "float";
    os << "([](const " << element << "* p) { tl::hex_require(tl::is_aligned<128>(p)); ";
    PrintType(op->dtype, os); os << " result; ";
    for (int offset = 0; offset < bytes; offset += 128) {
      os << "{ auto chunk = tl::hvx_load(p + " << offset / (op->dtype.bits() / 8)
         << "); __builtin_memcpy((char*)&result + " << offset
         << ", &chunk.raw, 128); } ";
    }
    os << "return result; })(reinterpret_cast<const " << element << "*>(&"
       << GetBufferRef(op->dtype.element_of(), op->buffer.get(), ramp->base) << "))";
    return;
  }
  os << '('; PrintType(op->dtype, os); os << "){";
  for (int i = 0; i < op->dtype.lanes(); ++i) {
    if (i) os << ", ";
    os << GetBufferRef(op->dtype.element_of(), op->buffer.get(), ramp->base + ramp->stride * i);
  }
  os << '}';
}
void CodeGenTileLangHexagon::VisitStmt_(const BufferStoreNode *op) {
  auto dtype = op->value.dtype();
  if (dtype.lanes() == 1) {
    CodeGenC::VisitStmt_(op);
    return;
  }
  ICHECK_EQ(op->indices.size(), 1);
  const auto *ramp = op->indices[0].as<RampNode>();
  ICHECK(ramp) << "Hexagon vector store requires a Ramp index";
  auto value = PrintExpr(op->value);
  auto temp = name_supply_->FreshName("hex_store");
  PrintIndent(); PrintType(dtype, stream);
  stream << ' ' << temp << " = " << value << ";\n";
  auto scope = GetPtrStorageScope(op->buffer->data);
  const int bytes = dtype.bits() * dtype.lanes() / 8;
  if (scope != "shared" && scope != "shared.dyn" && scope != "global" &&
      !scope.empty() && is_one(ramp->stride)) {
    PrintIndent();
    stream << "__builtin_memcpy(&"
           << GetBufferRef(dtype.element_of(), op->buffer.get(), ramp->base)
           << ", &" << temp << ", " << bytes << ");\n";
    return;
  }
  const auto *constant_base = ramp->base.as<IntImmNode>();
  const bool known_unaligned = constant_base &&
      (constant_base->value * (dtype.bits() / 8)) % 128 != 0;
  // DDR writeback has the same effective-address contract as VTCM stores.
  // Keep partial, strided, and statically unaligned writes on the lane fallback.
  if ((scope == "shared" || scope == "shared.dyn" || scope == "global" || scope.empty()) &&
      !known_unaligned && is_one(ramp->stride) &&
      dtype.is_float() && (dtype.bits() == 16 || dtype.bits() == 32) && bytes % 128 == 0) {
    need_hvx_h_ = true;
    const char *element = dtype.bits() == 16 ? "half_t" : "float";
    auto ptr = name_supply_->FreshName("hex_store_ptr");
    PrintIndent(); stream << "auto* " << ptr << " = reinterpret_cast<" << element << "*>(&"
        << GetBufferRef(dtype.element_of(), op->buffer.get(), ramp->base) << ");\n";
    PrintIndent(); stream << "tl::hex_require(tl::is_aligned<128>(" << ptr << "));\n";
    for (int offset = 0; offset < bytes; offset += 128) {
      PrintIndent(); stream << "{ tl::hvx_vec<" << element
          << "> chunk; __builtin_memcpy(&chunk.raw, (const char*)&" << temp
          << " + " << offset << ", 128); tl::hvx_store(" << ptr << " + "
          << offset / (dtype.bits() / 8) << ", chunk); }\n";
    }
    return;
  }
  for (int i = 0; i < dtype.lanes(); ++i) {
    PrintIndent();
    stream << GetBufferRef(dtype.element_of(), op->buffer.get(), ramp->base + ramp->stride * i)
           << " = " << temp << '[' << i << "];\n";
  }
}
void CodeGenTileLangHexagon::VisitStmt_(const AllocBufferNode *op) {
  const auto &b = op->buffer;
  auto scope = GetPtrStorageScope(b->data);
  if (scope != "shared" && scope != "shared.dyn") {
    // Preserve explicit stack-buffer alignment used by vector memory primitives.
    if (scope == "local" && b->data_alignment >= 128)
      stream << "alignas(" << b->data_alignment << ") ";
    CodeGenC::VisitStmt_(op);
    return;
  }
  if (worker_callback_arg_.defined()) {
    auto found = worker_allocations_.find(b->data.get());
    CHECK(found != worker_allocations_.end() && worker_vtcm_arg_.defined(), ValueError)
        << "callback shared allocation missing physical budget descriptor";
    auto entry = found->second;
    int64_t offset = entry.at("byte_offset").cast<Integer>()->value;
    int64_t stride = entry.at("bytes_per_slot").cast<Integer>()->value;
    int64_t depth = entry.at("depth").cast<Integer>()->value;
    CHECK(depth > 0 && stride > 0 && offset >= 0, ValueError)
        << "invalid callback allocation descriptor";
    auto id = AllocVarID(b->data.get());
    RegisterHandleType(b->data.get(), b->dtype);
    alloc_storage_scope_[b->data.get()] = scope;
    std::ostringstream base;
    base << "((unsigned char*)";
    PrintExpr(worker_vtcm_arg_.value(), base);
    base << " + " << offset;
    if (depth > 1) {
      base << " + ((uint64_t)(";
      if (worker_ordinal_arg_.defined()) PrintExpr(worker_ordinal_arg_.value(), base);
      else base << "wg_ordinal";
      base << ") % " << depth << ") * " << stride;
    }
    base << ")";
    // A view expression, not a pointer hoisted out of the iteration: q changes
    // while logical rank/shape and the AH/WH address transform remain fixed.
    std::ostringstream view;
    view << "(("; PrintType(b->dtype, view); view << "*)" << base.str() << ")";
    var_idmap_[b->data.get()] = view.str();
    return;
  }
  // Like hexagon_rt.h's HRT_VTCM_BASE(), the runtime supplies real HAP
  // reserved VTCM. A section/aligned stack array alone does not reserve VTCM.
  // All workers of a job see the same slab; concurrent jobs need distinct slabs.
  // No reuse across lexical lifetimes: monotonic offsets are conservative.
  int64_t bytes = b->dtype.bytes();
  for (const auto &dim : b->shape) {
    const auto *n = dim.as<IntImmNode>();
    ICHECK(n && n->value > 0) << "VTCM slab allocations need static extents";
    ICHECK_LE(n->value, INT64_MAX / bytes);
    bytes *= n->value;
  }
  ICHECK_LE(vtcm_bytes_, INT64_MAX - 127);
  const int64_t offset = (vtcm_bytes_ + 127) & ~int64_t(127);
  ICHECK_LE(bytes, INT64_MAX - offset);
  vtcm_bytes_ = offset + bytes;
  const auto id = AllocVarID(b->data.get());
  RegisterHandleType(b->data.get(), b->dtype);
  alloc_storage_scope_[b->data.get()] = scope;
  PrintIndent(); PrintType(b->dtype, stream);
  stream << "* " << id << " = (";
  PrintType(b->dtype, stream);
  stream << "*)tl_hex_vtcm_slice(" << offset << ", " << bytes << ", 128);\n";
  // Runtime allocation failure must trap before deriving any slab addresses.
  PrintIndent();
  stream << "tl::hex_require(" << id << " != nullptr);\n";
  PrintIndent();
  stream << "tl::hex_require(tl::is_aligned<128>(" << id << "));\n";
}
void CodeGenTileLangHexagon::VisitStmt_(const AttrStmtNode *op) {
  if (op->attr_key == "tl.workergroup_read_stage") {
    CHECK(worker_ddr_arg_.defined(), ValueError) << "staging needs distinct DDR backing";
    auto spec = Downcast<ffi::Map<ffi::String, ffi::Any>>(op->node);
    auto source = spec.at("source").cast<Buffer>();
    auto target = spec.at("target").cast<Buffer>();
    auto group = spec.at("group").cast<ffi::Map<ffi::String, ffi::Any>>();
    auto bytes = spec.at("bytes").cast<Integer>()->value;
    auto id = AllocVarID(target->data.get());
    RegisterHandleType(target->data.get(), target->dtype);
    need_hvx_h_ = true;
    PrintIndent(); PrintType(target->dtype, stream); stream << "* " << id << " = (";
    PrintType(target->dtype, stream); stream << "*)((unsigned char*)";
    PrintExpr(worker_ddr_arg_.value(), stream);
    stream << " + " << spec.at("offset").cast<Integer>()->value << " + " << bytes
           << " * (((const tl_wg_worker*)";
    PrintExpr(worker_callback_arg_.value(), stream);
    stream << ")->physical_id - " << group.at("first_worker").cast<Integer>()->value << "));\n";
    PrintIndent(); stream << "for (int wg_copy = 0; wg_copy < " << bytes / 128
                         << "; ++wg_copy) ((HVX_Vector*)" << id << ")[wg_copy] = ((const HVX_Vector*)";
    PrintExpr(source->data, stream); stream << ")[wg_copy];\n";
    PrintStmt(op->body);
    return;
  }
  if (op->attr_key == "tl.workergroup_ordinal") {
    CHECK(worker_callback_arg_.defined(), ValueError) << "ordinal outside callback";
    PrintIndent(); stream << "{ const uint64_t wg_ordinal = ";
    PrintExpr(op->value, stream); stream << ";\n";
    PrintStmt(op->body); PrintIndent(); stream << "}\n";
    return;
  }
  if (op->attr_key == "tl.workergroup_local") {
    CHECK(worker_callback_arg_.defined(), ValueError)
        << "unbound worker group reached production Hexagon codegen";
    PrintStmt(op->body);
    return;
  }
  if (op->attr_key == "hexagon.async_scope") {
    ICHECK(!async_region_) << "nested asynchronous engine regions are forbidden";
    const auto *slot = op->value.as<IntImmNode>();
    ICHECK(slot && slot->value >= 0 && slot->value < 8) << "async slot must be static [0,8)";
    decl_stream << "#include <tl_templates/hexagon/async.h>\n";
    PrintIndent(); stream << "if (tl_hex_worker_id() == 0) {\n";
    PrintIndent(); stream << "tl::engine_submit<" << slot->value << ">([=]() {\n";
    async_region_ = true;
    PrintStmt(op->body);
    async_region_ = false;
    PrintIndent(); stream << "});\n}\n";
    return;
  }
  if (op->attr_key == tirx::attr::thread_extent) {
    auto iv = Downcast<IterVar>(op->node);
    auto tag = iv->thread_tag;
    if (!worker_callback_arg_.defined() && (tag == "threadIdx.x" || tag == "blockIdx.x")) {
      PrintIndent();
      stream << "tl::hex_require(" << (tag == "threadIdx.x" ? "tl_hex_num_workers()" : "tl_hex_num_jobs()") << " == ";
      if (tag == "blockIdx.x")
        stream << job_extents_[0] * job_extents_[1] * job_extents_[2];
      else
        PrintExpr(op->value, stream);
      stream << ");\n";
    }
    if (tag == "blockIdx.y" || tag == "blockIdx.z") {
      BindThreadIndex(iv);
      PrintStmt(op->body);
      return;
    }
    if (tag == "threadIdx.y" || tag == "threadIdx.z") {
      ICHECK(is_one(op->value)) << "Hexagon only supports unit extent on " << tag;
      var_idmap_[iv->var.get()] = "0";
      PrintStmt(op->body);
      return;
    }
  }
  CodeGenC::VisitStmt_(op);
}
std::string CodeGenTileLangHexagon::GetBufferRef(DataType t,
                                               const BufferNode *b, PrimExpr i) {
  return CodeGenC::GetBufferRef(t, b, i);
}
void CodeGenTileLangHexagon::PrintCallExtern(Type t, ffi::String name,
    const ffi::Array<PrimExpr> &args, bool skip, std::ostream &os) {
  // Generic row-map/copy lowering emits these externs without a pack intrinsic.
  // Register their header at the use, independent of shape or kernel identity.
  if (name == "tl::gemm_unpack_ah_row" ||
      name == "tl::gemm_unpack_ah_rows" || name == "tl::gemm_pack_ah_rows")
    need_gemm_h_ = true;
  if (name == "tl::engine_wait") decl_stream << "#include <tl_templates/hexagon/async.h>\n";
  if (name == "tl::transpose_2d<2>" || name == "tl::transpose_2d<4>")
    decl_stream << "#include <tl_templates/hexagon/transpose.h>\n";
  if (name == "tl::cast_pack_f32_f16<true>" || name == "tl::cast_pack_f32_f16<false>")
    decl_stream << "#include <tl_templates/hexagon/cast_layout.h>\n";
  if (std::string(name).find("tl::AllReduce<") == 0) need_reduce_h_ = true;
  if (name == "tl::profile_region" || name == "tl::profile_mark") decl_stream << "#include <tl_templates/hexagon/profile.h>\n";
  if (name == "hmx_mma_deep") need_hmx_h_ = true;
  const size_t first = skip ? 1 : 0;
  if (name == "tl::warp_reduce_sum" || name == "tl::warp_reduce_max" ||
      name == "tl::warp_reduce_min" || name == "tl::warp_reduce_bitand" ||
      name == "tl::warp_reduce_bitor") {
    need_reduce_h_ = true;
    ICHECK_EQ(args.size() - first, 1);
    auto dtype = args[first].dtype();
    ICHECK((name != "tl::warp_reduce_bitand" && name != "tl::warp_reduce_bitor") || dtype.is_int() || dtype.is_uint())
        << "bitwise warp reduction requires integer lanes";
    if (dtype.lanes() > 1) {
      ICHECK(dtype.bits() == 32 && dtype.lanes() == 32 &&
             (dtype.is_float() || dtype.is_int()))
          << "logical warp reduction requires 32 four-byte lanes";
      os << "([]("; PrintType(dtype, os); os << " v) { tl::hvx_vec<";
      // Hexagon's int32_t is long, while B specializes hvx_vec<int>.
      os << (dtype.is_float() ? "float" : "int");
      os << "> x; __builtin_memcpy(&x.raw, &v, 128); auto r = "
         << name << "(x); __builtin_memcpy(&v, &r.raw, 128); return v; })(";
      PrintExpr(args[first], os); os << ')';
      return;
    }
  }
  if (name == "hmx_mma_deep" || name == "hmx_mma_f16") {
    need_hmx_h_ = true;
    ICHECK_EQ(args.size() - first, 3);
    auto tiles = args[first + 2].as<IntImmNode>();
    ICHECK(tiles && tiles->value > 0);
    os << (name == "hmx_mma_deep" ? "tl::hmx_mma_deep_f16<" : "tl::hmx_mma_f16<")
       << tiles->value << ">(";
    PrintExpr(args[first], os); os << ", "; PrintExpr(args[first + 1], os);
    os << ')';
    return;
  }
  // Both shared- and fragment-output GEMM use native scale/bias binding
  // and store-after. The retired readout intrinsic is not registered.
  if (name == "hmx_clear_f16" || name == "hmx_init_scale" ||
      name == "hmx_store_after_f16") {
    need_hmx_h_ = true;
    os << "tl::" << name << '(';
    for (size_t i = first; i < args.size(); ++i) {
      if (i != first) os << ", ";
      PrintExpr(args[i], os);
    }
    os << ')';
    return;
  }
  if (name == "hexagon.gemm_hmx") {
    ICHECK(false) << "hexagon.gemm_hmx must be lowered by tileop";
    return;
  }
  CodeGenC::PrintCallExtern(t, name, args, skip, os);
}
} // namespace tvm::codegen
