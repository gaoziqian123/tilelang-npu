/* Native Hexagon reductions use the common fragment ownership planner.
 * Worker collectives are implemented by the Hexagon runtime, not GPU shuffles.
 */
#include "backend/common/op/reduce.h"
#include "hexagon/target_utils.h"
#include <sstream>

namespace tvm::tl {
namespace hexagon {
struct Reduce : backend::ReduceLowerer<Reduce> {
  static tirx::Stmt Lower(const ReduceOpNode &op, const LowerArgs &args,
                          arith::Analyzer *analyzer) {
    using namespace tirx;
    if (!IsLocalBuffer(op.src) || !IsLocalBuffer(op.dst, true))
      return backend::ReduceLowerer<Reduce>::Lower(op, args, analyzer);
    ICHECK_EQ(op.batch, 1);
    ICHECK(!op.nan_propagate || op.dst->dtype.bits() == 32);
    auto remap = [&](Buffer b) {
      auto it = args.buffer_remap.find(b);
      return it == args.buffer_remap.end() ? b : (*it).second;
    };
    Buffer src = remap(op.src), dst = remap(op.dst);
    Array<Var> vars;
    Array<PrimExpr> di, si;
    for (size_t i = 0; i < op.dst->shape.size(); ++i) {
      Var v("i" + std::to_string(i)); vars.push_back(v); di.push_back(v);
    }
    Var rv("rv");
    for (size_t i = 0; i < op.src->shape.size(); ++i)
      si.push_back(i == size_t(op.dim) ? PrimExpr(rv) : PrimExpr(vars[
          op.dst->shape.size() == op.src->shape.size() ? i : i < size_t(op.dim) ? i : i - 1]));
    Stmt body = For(rv, 0, op.src->shape[op.dim], ForKind::kSerial,
        BufferStore(dst, backend::reduce::MakeReduce(op, 1,
          BufferLoad(dst, di), BufferLoad(src, si)), di));
    if (op.clear) body = SeqStmt::Flatten(BufferStore(dst,
        backend::reduce::MakeInitValue(op), di), body);
    for (int i = int(vars.size()) - 1; i >= 0; --i)
      body = For(vars[i], 0, op.dst->shape[i], ForKind::kSerial, body);
    return body;
  }
  static bool SupportsFp16Bf16NanReduce(Target) { return false; }
  static int GetPreferredVectorizedSize(const ReduceOpNode &, Target) { return 1; }
  static std::string MakeBatchAllReduce(std::string reducer, int threads,
      int scale, PrimExpr offset, PrimExpr, int batch, int stride, Target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << threads << ", " << scale
       << ", " << offset << ", " << batch << ", " << stride << ">::run_batch";
    return ss.str();
  }
  static std::string MakeScalarAllReduce(std::string reducer, int threads,
      int scale, PrimExpr offset, PrimExpr, Target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << threads << ", " << scale
       << ", " << offset << ">::run";
    return ss.str();
  }
};
} // namespace hexagon
namespace {
const bool registered = [] {
  RegisterReduceImpl(ReduceImpl{"hexagon.Reduce", TargetIsHexagon,
                                hexagon::Reduce::Lower});
  return true;
}();
} // namespace
} // namespace tvm::tl
