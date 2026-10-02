/* Internal physical-affine transpose recipe shared by copy and transpose. */
#pragma once
#include "op/operator.h"
#include <tvm/tirx/builtin.h>

namespace tvm::tl::hexagon {
using namespace tirx;
using namespace ffi;

inline PrimExpr TransposeStride(Buffer b) {
  return b->strides.empty() ? b->shape[1] : b->strides[0];
}
inline PrimExpr TransposeBytes(Buffer b) {
  return ((b->shape[0] - 1) * TransposeStride(b) + b->shape[1]) * b->dtype.bytes();
}

// Buffers and ranges here are physical, not logical layout coordinates. Never
// round up the footprint: peripheral HVX accesses need real allocation padding.
inline bool LegalAffineTranspose(Buffer src, Buffer dst, Array<Range> sr,
                                 Array<Range> dr, const LowerArgs &args,
                                 arith::Analyzer *a) {
  ICHECK(args.thread_bounds.defined() && args.thread_index.defined())
      << "Hexagon transpose requires defined thread_bounds and thread_index";
  if (src->shape.size() != 2 || dst->shape.size() != 2 ||
      sr.size() != 2 || dr.size() != 2 || src->dtype != dst->dtype ||
      (src->dtype != DataType::Float(16) && src->dtype != DataType::Float(32)))
    return false;
  ICHECK(!src->data.same_as(dst->data)) << "Hexagon transpose cannot alias";
  for (Buffer b : {src, dst}) {
    if (!b->strides.empty() && b->strides.size() != 2)
      return false;
    if (!(a->CanProve(b->shape[0] > 0) && a->CanProve(b->shape[1] > 0) &&
          a->CanProve(TransposeBytes(b) <= Integer(2147483647)) &&
          (b->strides.empty() || (b->strides.size() == 2 && a->CanProveEqual(b->strides[1], 1))) &&
          a->CanProve(TransposeStride(b) >= b->shape[1]) &&
          a->CanProveEqual(b->elem_offset, 0) &&
          a->CanProveEqual(FloorMod(TransposeBytes(b), 128), 0) &&
          (b->data_alignment >= 128 || b.scope() == "global" ||
           (b.scope() == "shared.dyn" && args.require_smem_alignment))))
      return false;
  }
  for (int i = 0; i < 2; ++i) {
    if (!(a->CanProveEqual(sr[i]->extent, dr[1-i]->extent) &&
          a->CanProve(sr[i]->extent > 0) && a->CanProve(sr[i]->min >= 0) &&
          a->CanProve(sr[i]->min + sr[i]->extent <= src->shape[i]) &&
          a->CanProve(dr[1-i]->min >= 0) &&
          a->CanProve(dr[1-i]->min + dr[1-i]->extent <= dst->shape[1-i])))
      return false;
  }
  return true;
}

inline Stmt EmitAffineTranspose(Buffer src, Buffer dst, Array<Range> sr,
                                Array<Range> dr, const LowerArgs &args,
                                arith::Analyzer *a) {
  ICHECK(args.thread_bounds.defined() && args.thread_index.defined());
  for (Buffer b : {src, dst})
    if (b.scope() == "shared.dyn" && args.require_smem_alignment)
      args.require_smem_alignment(b->data, 128);
  auto ptr = [](Buffer b) {
    return Call(DataType::Handle(), builtin::address_of(),
                {BufferLoad(b, {Integer(0), Integer(0)})});
  };
  PrimExpr call = Call(DataType::Bool(), Op::Get(src->dtype.bytes() == 2 ?
      "tl.hexagon.transpose_2d_2" : "tl.hexagon.transpose_2d_4"), {
      ptr(dst), ptr(src), TransposeBytes(dst), TransposeBytes(src),
      TransposeStride(dst), TransposeStride(src), dr[0]->min, dr[1]->min,
      sr[0]->min, sr[1]->min, sr[0]->extent, sr[1]->extent});
  Stmt body = Evaluate(Call(DataType::Handle(), Op::Get("tl.hexagon.require"), {call}));
  bool local = src.scope() == "local" || dst.scope() == "local";
  ICHECK(!local || a->CanProveEqual(args.thread_bounds->extent, 1))
      << "Local/shared transpose requires a single worker";
  if (!local) {
    Stmt sync = Evaluate(Call(DataType::Int(32), builtin::tvm_storage_sync(),
                              {StringImm("shared")}));
    body = SeqStmt({sync, IfThenElse(args.thread_index == args.thread_bounds->min, body), sync});
  }
  return body;
}
} // namespace tvm::tl::hexagon
