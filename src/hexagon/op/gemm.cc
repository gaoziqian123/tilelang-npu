/*!
 * \file tl/hexagon/op/gemm.cc
 * \brief Hexagon implementation for tl.gemm instruction selection.
 */

#include "op/gemm.h"

#include "hexagon/target_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace hexagon {

namespace {

constexpr const char *kHexagonHMX = "hexagon.hmx";

} // namespace

struct Gemm {
  static String SelectInst(const GemmNode &op, int block_size, Target target) {
    (void)op;
    (void)block_size;
    (void)target;
    return kHexagonHMX;
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

} // namespace hexagon

namespace {

bool MatchHexagonGemmTarget(Target target) { return TargetIsHexagon(target); }

bool RegisterHexagonGemm() {
  RegisterGemmImpl(GemmImpl{
      "hexagon.Gemm",
      MatchHexagonGemmTarget,
      hexagon::Gemm::SelectInst,
      hexagon::Gemm::ComputeWarpPartition,
      hexagon::Gemm::ReuseExistingSharedLayout,
  });
  return true;
}

const bool hexagon_gemm_registered = RegisterHexagonGemm();

} // namespace

} // namespace tl
} // namespace tvm
