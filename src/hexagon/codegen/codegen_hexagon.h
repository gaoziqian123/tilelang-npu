#ifndef TILELANG_CODEGEN_HEXAGON_H_
#define TILELANG_CODEGEN_HEXAGON_H_

#include "backend/common/codegen/codegen_c_line_directives.h"
#include <map>

namespace tvm::codegen {
class CodeGenTileLangHexagon final : public CodeGenCWithLineDirectives {
public:
  std::string Finish() final;
  void PrintFuncPrefix(std::ostream &os) final;
  void PrintExtraAttrs(const PrimFunc &, std::ostream &) final {}
  void PrintStorageSync(const CallNode *op) final;
  void PrintStorageScope(const std::string &, std::ostream &) final;
  void PrintType(DataType, std::ostream &) final;
  void BindThreadIndex(const IterVar &) final;
  void InitFuncState(const PrimFunc &f) final;
  void PrintVecBinaryOp(const std::string &, DataType, PrimExpr, PrimExpr,
                        std::ostream &) final;
  void PrintVecElemLoad(const std::string &, DataType, int, std::ostream &) final;
  void PrintVecElemStore(const std::string &, DataType, int,
                        const std::string &) final;
  void PrintVecElemLoadExpr(DataType, int, const std::string &,
                            std::ostream &) final;
  std::string CastFromTo(std::string, DataType, DataType) final;
#define HEX_EXPR(Node) void VisitExpr_(const Node *op, std::ostream &os) final;
  HEX_EXPR(RampNode)
  HEX_EXPR(BufferLoadNode)
  HEX_EXPR(BroadcastNode)
  HEX_EXPR(FloatImmNode)
  HEX_EXPR(MaxNode)
  HEX_EXPR(MinNode)
  HEX_EXPR(CallNode)
  HEX_EXPR(CastNode)
  HEX_EXPR(NotNode)
  HEX_EXPR(SelectNode)
  HEX_EXPR(ShuffleNode)
#undef HEX_EXPR
#define HEX_STMT(Node) void VisitStmt_(const Node *op) final;
  HEX_STMT(ForNode)
  HEX_STMT(AllocBufferNode)
  HEX_STMT(AttrStmtNode)
  HEX_STMT(BufferStoreNode)
  HEX_STMT(EvaluateNode)
#undef HEX_STMT
protected:
  std::string GetBufferRef(DataType, const BufferNode *, PrimExpr) final;
  void PrintCallExtern(Type, ffi::String, const ffi::Array<PrimExpr> &, bool,
                       std::ostream &) final;
private:
  void HandleVolatileLoads(const std::string &value, const BufferLoadNode *,
                           std::ostream &os) final { os << value; }
  bool IsScopePartOfType() const final { return false; }
  bool need_hmx_h_{false};
  bool need_gemm_h_{false};
  bool need_hvx_h_{false};
  bool need_reduce_h_{false};
  std::map<std::string, DataType> vector_types_;
  int64_t vtcm_bytes_{0};
  bool async_region_{false};
  ffi::Optional<Var> worker_callback_arg_;
  ffi::Optional<Var> worker_vtcm_arg_;
  ffi::Optional<Var> worker_ddr_arg_;
  ffi::Optional<Var> worker_ordinal_arg_;
  bool need_worker_abi_{false};
  std::map<const VarNode *, ffi::Map<ffi::String, ffi::Any>> worker_allocations_;
  int64_t job_extents_[3]{1, 1, 1};
};
} // namespace tvm::codegen
#endif
