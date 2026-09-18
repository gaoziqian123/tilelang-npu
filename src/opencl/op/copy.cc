/*!
 * \file tl/opencl/op/copy.cc
 * \brief OpenCL implementation for tl.copy lowering.
 */

#include "op/copy.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace opencl {

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op,
                               const LayoutInferArgs &layout_args,
                               InferLevel level) {
    return op.InferSIMTLayout(layout_args, level);
  }

  static Stmt Lower(const CopyNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
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
