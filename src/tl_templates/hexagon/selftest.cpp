// Compile-only ABI/instruction probe. Define TL_HEADER=1..10 for standalone
// header checks; omit it to compile all headers together. No runtime stubs.
#if !defined(TL_HEADER) || TL_HEADER == 1
#include "common.h"
unsigned check_common(half_t a, half_t b, int *p) {
  bfloat16_t bf((float(a)));
  return __pack_half2(a, b) + tl::Any(p, 3) + tl::All(p, 3) +
         tl::align_up<128>(3) + tl::is_aligned<128>(p) + (float(bf) != 0);
}
#endif
#if !defined(TL_HEADER) || TL_HEADER == 2
#include "hvx.h"
float check_hvx(float *p, half_t *h) {
  auto f = tl::hvx_load(p);
  tl::hvx_store(p, tl::hvx_add(f, tl::hvx_mul(f, tl::hvx_max(f, f))));
  auto v = tl::hvx_load(h);
  tl::hvx_store(h, tl::hvx_exp(tl::hvx_add(v, tl::hvx_mul(v, tl::hvx_max(v, v)))));
  return tl::hvx_reduce_sum(f) + tl::hvx_reduce_max(f);
}
void check_hvx_extended(float *p, half_t *h, int lane) {
  auto a = tl::hvx_load(p);
  tl::hvx_store(p, tl::hvx_div(tl::hvx_sub(a, a), a));
  tl::hvx_store(p, tl::hvx_div_approx(a, a));
  tl::hvx_store(p, tl::hvx_set_lane(a, lane, tl::hvx_lane(a, lane)));
  auto b = tl::hvx_load(h);
  tl::hvx_store(h, tl::hvx_div(tl::hvx_sub(b, b), b));
  auto c = tl::hvx_load64(p);
  tl::hvx_store64(p, tl::hvx_div(tl::hvx_sub(tl::hvx_mul(c, c), c), tl::hvx_add(c, c)));
}
#endif
#if !defined(TL_HEADER) || TL_HEADER == 3
#include "hmx.h"
void check_hmx(void *a, void *b, void *c) {
  tl::hmx_clear_f16();
  tl::hmx_mma_f16<1>(a, b);
  tl::hmx_mma_deep_f16<32>(a, b);
  tl::hmx_mma_deep_f16<33>(a, b);
  tl::hmx_mma_deep_f16<80>(a, b);
  tl::hmx_acc_read_f16(c);
}
#endif
#if !defined(TL_HEADER) || TL_HEADER == 4
#include "reduce.h"
int check_reduce(int x, int *p) {
  x = tl::AllReduce<tl::SumOp, 4, 1>::run(x, p);
  tl::AllReduce<tl::MaxOp, 4, 1, 0, 2, 4>::run_batch(p);
  x += tl::MinOp()(x, x) + tl::BitXorOp()(x, x);
  return tl::warp_reduce_sum(x) + tl::warp_reduce_max(x) +
         tl::warp_reduce_min(x) + tl::warp_reduce_bitand(x) + tl::warp_reduce_bitor(x);
}
void check_reduce_hvx(float *f, int *p) {
  auto v = tl::hvx_load(f);
  tl::hvx_store(f, tl::warp_reduce_sum(v));
  tl::hvx_store(f, tl::warp_reduce_max(v));
  tl::hvx_store(f, tl::warp_reduce_min(v));
  auto w = tl::hvx_load(p);
  tl::hvx_store(p, tl::warp_reduce_sum(w));
  tl::hvx_store(p, tl::warp_reduce_max(w));
  tl::hvx_store(p, tl::warp_reduce_min(w));
  tl::hvx_store(p, tl::warp_reduce_bitand(w));
  tl::hvx_store(p, tl::warp_reduce_bitor(w));
}
#endif
#if !defined(TL_HEADER) || TL_HEADER == 5
#include "copy.h"
void check_copy(void *a, void const *b, bool c) {
  tl::cp_async_gs<4>(a, b);
  tl::cp_async_gs<8>(static_cast<void const *>(a), b);
  tl::cp_async_gs<16>(a, b);
  tl::cp_async_gs<128>(a, b);
  tl::cp_async_gs_conditional<16>(a, b, c);
  tl::cp_async_gs_conditional<4>(static_cast<void const *>(a), b, c);
  tl::cp_async_commit(); tl::cp_async_wait<>(); tl::cp_async_wait<2>();
}
#endif
#if !defined(TL_HEADER) || TL_HEADER == 6
#include "scan.h"
void check_scan(float const *a, float *b) {
  tl::InclusiveScan1D<tl::ScanSumOp, 64>::run_auto(a, b, 65);
  tl::InclusiveScan2D<tl::ScanMaxOp, 64>::run_auto(a, b, 7, 9, 9, 9);
  tl::CumSum1D<64>::run(a, b, 65);
  tl::CumMax1D<64, true>::run(a, b, 65);
  tl::CumSum2D<64, 1>::run(a, b, 7, 9, 9, 9);
  tl::CumMax2D<64, 0, true>::run(a, b, 7, 9, 9, 9);
}
#endif
#if !defined(TL_HEADER) || TL_HEADER == 7
#include "atomic.h"
float check_atomic(float *p) {
  AtomicAdd(p, 1); AtomicAdd(*p, 1);
  AtomicMax(p, 1); AtomicMax(*p, 1);
  AtomicMin(p, 1); AtomicMin(*p, 1);
  return AtomicAddRet(p, 1);
}
#endif
#if !defined(TL_HEADER) || TL_HEADER == 8
#include "threadblock_swizzle.h"
dim3 check_swizzle() {
  auto a = tl::rasterization2DRow<3>();
  auto b = tl::rasterization2DColumn<3>();
  return {a.x, b.y, a.z};
}
#endif
#if !defined(TL_HEADER) || TL_HEADER == 9
#include "debug.h"
void check_debug(int *p) {
  debug_print_var("integer", 3);
  debug_print_var("pointer", p);
  debug_print_buffer_value("buffer", "p", 0, p[0]);
  debug_print_buffer_value("pointer", "p", 0, p);
  debug_print_msg("probe");
}
#endif
#if !defined(TL_HEADER) || TL_HEADER == 10
#include "gemm.h"
void check_gemm_helpers(void *a, void *w, void *rm, void *out) {
  tl::gemm_pack_ah_tiles<33>(a, rm);
  tl::gemm_pack_wh_tiles<33>(w, rm);
  tl::gemm_accumulator_f16 acc;
  acc.mma<33>(a, w);
  acc.finish(out);
  tl::gemm_unpack_ah_tiles<1>(rm, out);
}
void check_strided(half_t *a, half_t *w, half_t *rm, int stride) {
  tl::gemm_pack_ah_strided<2>(a, rm, stride);
  tl::gemm_pack_wh_nt_strided<2>(w, rm, stride);
  tl::gemm_unpack_ah_strided<2>(rm, a, stride);
}
#endif
