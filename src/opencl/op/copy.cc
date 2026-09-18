/*!
 * \file tl/opencl/op/copy.cc
 * \brief OpenCL implementation for tl.copy lowering.
 */

#include "op/copy.h"

#include <tvm/tirx/op.h>

namespace tvm {
namespace tl {

using namespace tirx;

namespace opencl {

// Whether the buffer lives in an OpenCL image/texture ("global.texture*").
static bool IsTextureBuffer(const Buffer &buf) {
  return std::string(buf.scope()).find("texture") != std::string::npos;
}

// Texture-source copy: map one work-item to one texel (the innermost axis is
// the RGBA channel and must have extent 4), so each work-item issues a single
// vector (half4/float4) load — later rewritten by TextureFlatten into one
// READ_IMAGEH — and one vector store into the destination.  The generic SIMT
// copy maps work-items to *elements* instead, which makes 4 neighbouring
// lanes redundantly read the same texel (measured on texture_staging).
static Stmt LowerTextureCopy(const CopyNode &op, const LowerArgs &lower_args,
                             arith::Analyzer *analyzer) {
  size_t nd = op.src_range.size();
  ICHECK_GE(nd, 2) << "opencl texture copy expects at least 2D src region";
  auto channel = as_const_int(op.src_range[nd - 1]->extent);
  ICHECK(channel && *channel == 4)
      << "opencl texture copy expects innermost channel extent 4, got "
      << op.src_range[nd - 1]->extent;

  // number of texels = product of leading extents
  PrimExpr texels = make_const(DataType::Int(32), 1);
  for (size_t i = 0; i + 1 < nd; i++) {
    texels = texels * op.src_range[i]->extent;
  }
  texels = analyzer->Simplify(texels);

  PrimExpr tid = cast(DataType::Int(32), lower_args.thread_index);
  PrimExpr textent = cast(DataType::Int(32), lower_args.thread_bounds->extent);

  Var li("tex_i", DataType::Int(32));
  PrimExpr texel = li * textent + tid;

  // linear texel id -> leading-dim indices (innermost leading dim fastest).
  // The channel dim stays a scalar loop var in a kVectorized inner For: after
  // TextureFlatten rewrites the loads into texture2d_load calls, VectorizeLoop
  // (which special-cases texture2d_load) merges the 4 channel lanes into one
  // vector-wide READ_IMAGEH per work-item.
  auto map_indices = [&](const Array<Range> &range, const Var &c) {
    size_t nr = range.size();
    std::vector<PrimExpr> idx(nr);
    PrimExpr rem = texel;
    for (int k = static_cast<int>(nr) - 2; k >= 0; --k) {
      PrimExpr e = cast(DataType::Int(32), range[k]->extent);
      idx[k] = range[k]->min + floormod(rem, e);
      rem = floordiv(rem, e);
    }
    idx[nr - 1] = range[nr - 1]->min + c;
    return idx;
  };

  Var c("tex_c", DataType::Int(32));
  Stmt store =
      BufferStore(op.dst, BufferLoad(op.src, map_indices(op.src_range, c)),
                  map_indices(op.dst_range, c));
  Stmt inner = For(c, 0, 4, ForKind::kVectorized, store);
  Stmt body = IfThenElse(texel < texels, inner);
  PrimExpr iters =
      analyzer->Simplify(floordiv(texels + textent - 1, textent));
  return For(li, 0, iters, ForKind::kSerial, body);
}

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op,
                               const LayoutInferArgs &layout_args,
                               InferLevel level) {
    return op.InferSIMTLayout(layout_args, level);
  }

  static Stmt Lower(const CopyNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    if (IsTextureBuffer(op.src)) {
      return LowerTextureCopy(op, lower_args, analyzer);
    }
    return LowerNormalCopy(op, lower_args, analyzer);
  }
};

} // namespace opencl

namespace {

bool MatchOpenCLCopyTarget(Target target) {
  return target.defined() && target->kind.defined() &&
         target->kind->name == "opencl";
}

bool RegisterOpenCLCopy() {
  RegisterCopyImpl(CopyImpl{
      "opencl.Copy",
      MatchOpenCLCopyTarget,
      100,
      opencl::Copy::InferLayout,
      opencl::Copy::Lower,
  });
  return true;
}

const bool opencl_copy_registered = RegisterOpenCLCopy();

} // namespace

} // namespace tl
} // namespace tvm
