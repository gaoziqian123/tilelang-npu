#include "op/builtin_registry.h"

namespace tvm::tl {
using namespace tirx;
TVM_REGISTER_OP("tl.hexagon.vector_io")
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));
// Physical register-only leaf. Name, numerical version and immediate are
// explicit IR operands; codegen validates even manually constructed calls.
TVM_REGISTER_OP("tl.hexagon.vector_leaf")
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque));
// Explicit rounded elementwise operations, not changes to tirx.exp/reduce.
TVM_REGISTER_OP("tl.hexagon.exp2_hf")
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);
TVM_REGISTER_OP("tl.hexagon.sub_hf")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);
TVM_REGISTER_OP("tl.hexagon.exp_f16_softmax")
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true);
// Dotted backend names cannot be C++ function identifiers in
// TIR_DEFINE_TL_BUILTIN; use its underlying registration with the same attrs.
#define HEXAGON_BUILTIN(name, inputs) \
  TVM_REGISTER_OP("tl.hexagon." #name) \
      .set_num_inputs(inputs) \
      .set_attr<TScriptPrinterName>("TScriptPrinterName", "hexagon." #name) \
      .set_attr<TCallEffectKind>("TCallEffectKind", Integer(CallEffectKind::kOpaque))
HEXAGON_BUILTIN(hmx_mma_deep, 3);
HEXAGON_BUILTIN(hmx_mma_f16, 3);
HEXAGON_BUILTIN(hmx_clear_acc, 0);
HEXAGON_BUILTIN(hmx_init_scale, 2);
HEXAGON_BUILTIN(hmx_store_after, 2);
// Collective barrier on the callback's ABI worker group, never the full team.
HEXAGON_BUILTIN(workergroup_barrier, 0);
HEXAGON_BUILTIN(workergroup_wait, 2);
HEXAGON_BUILTIN(workergroup_publish, 2);
HEXAGON_BUILTIN(workergroup_complete, 0);
HEXAGON_BUILTIN(workergroup_stage_enter, 1);
HEXAGON_BUILTIN(workergroup_stage_exit, 1);
HEXAGON_BUILTIN(workergroup_team_barrier, 0);
HEXAGON_BUILTIN(dma_copy_2d_wait, 8);
HEXAGON_BUILTIN(dma_submit, 7);
HEXAGON_BUILTIN(dma_wait, 1);
// (destination pointer, source pointer, positive constant tile count).
// Pointers address compact, contiguous sequences of 32x32 fp16 tiles, not
// arbitrary parent-matrix views. Lowering owns strides and tile ordering.
HEXAGON_BUILTIN(pack_ah_tile, 3);
HEXAGON_BUILTIN(pack_wh_tile, 3);
HEXAGON_BUILTIN(unpack_ah_tile, 3);
// (destination pointer, source pointer, scalar integer row stride,
// positive constant tile count). Strides and offsets are owned by lowering.
HEXAGON_BUILTIN(pack_ah_strided, 4);
HEXAGON_BUILTIN(pack_wh_nt_strided, 4);
HEXAGON_BUILTIN(unpack_ah_strided, 4);
HEXAGON_BUILTIN(unpack_ah_pair_strided, 4);
HEXAGON_BUILTIN(scatter_release, 1);
HEXAGON_BUILTIN(pack_wh_nt_pair_strided, 4);
HEXAGON_BUILTIN(pack_ah_pair_strided, 4);
HEXAGON_BUILTIN(pack_wh_nt_strip, 4);
HEXAGON_BUILTIN(pack_ah_strip, 4);
// Reads a materialized row; opaque prevents CSE/motion across producer stores.
// (pointer, extent, seed, max?, lanes, versioned order/replay contract).
HEXAGON_BUILTIN(local_row_reduce, 6);
HEXAGON_BUILTIN(row_map_init, 2);
HEXAGON_BUILTIN(row_map_update, 5);
HEXAGON_BUILTIN(row_map_finish, 5);
HEXAGON_BUILTIN(profile_mark, 2);
HEXAGON_BUILTIN(transpose_2d_2, 12);
HEXAGON_BUILTIN(transpose_2d_4, 12);
HEXAGON_BUILTIN(require, 1);
// Closed, synchronous full-active-group collective. Verified before planning.
HEXAGON_BUILTIN(group_reduce, 7);
#undef HEXAGON_BUILTIN
// Memory effects are confined to address_of operands (no pointer publication,
// callbacks, or hidden access to unrelated buffers). This is NOT a pure-call
// attribute and does NOT assert scatter/HMX completion. Consumers must exclude
// all possibly aliasing operands from a candidate region across these calls.
// See tl_templates/hexagon/{hmx,gemm}.h for the concrete implementations.
TVM_FFI_STATIC_INIT_BLOCK() {
  // These helpers finish every memory access before returning, never retain
  // an address or invoke a callback. Unlike bounded HMX/scatter operations,
  // their operands may be forwarded in a following candidate region.
  for (const char *name : {"transpose_2d_2", "transpose_2d_4", "require", "profile_mark"})
    OpRegEntry::RegisterOrGet(std::string("tl.hexagon.")+name)
        .set_attr<Bool>("tl.region_synchronous", Bool(true));
  for (const char *name : {"hmx_mma_deep", "hmx_mma_f16", "hmx_clear_acc",
       "hmx_init_scale", "hmx_store_after", "pack_ah_strip",
       "pack_wh_nt_strip", "unpack_ah_pair_strided", "scatter_release", "profile_mark"}) {
    OpRegEntry::RegisterOrGet(std::string("tl.hexagon.")+name)
        .set_attr<Bool>("tl.region_bounded_effects", Bool(true));
  }
}
} // namespace tvm::tl
