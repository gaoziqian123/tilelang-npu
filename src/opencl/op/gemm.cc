/*!
 * \file tl/opencl/op/gemm.cc
 * \brief OpenCL implementation for tl.gemm instruction selection.
 */

#include "op/gemm.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace opencl {

namespace {

constexpr const char *kOpenCLFMA = "opencl.fma";

} // namespace

struct Gemm {
  static String SelectInst(const GemmNode &op, int block_size, Target target) {
    (void)op;
    (void)block_size;
    (void)target;
    return kOpenCLFMA;
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst) {
    (void)M;
    (void)N;
    (void)block_size;
    (void)target;
    (void)gemm_inst;
    policy.m_warp = 1;
    policy.n_warp = 1;
    return {1, 1};
  }

  static bool ReuseExistingSharedLayout(String gemm_inst) {
    (void)gemm_inst;
    return false;
  }
};

} // namespace opencl

namespace {

bool MatchOpenCLGemmTarget(Target target) {
  return target.defined() && target->kind.defined() &&
         target->kind->name == "opencl";
}

bool RegisterOpenCLGemm() {
  RegisterGemmImpl(GemmImpl{
      "opencl.Gemm",
      MatchOpenCLGemmTarget,
      opencl::Gemm::SelectInst,
      opencl::Gemm::ComputeWarpPartition,
      opencl::Gemm::ReuseExistingSharedLayout,
  });
  return true;
}

const bool opencl_gemm_registered = RegisterOpenCLGemm();

} // namespace

} // namespace tl
} // namespace tvm
