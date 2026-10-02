#ifndef TILELANG_TRANSFORM_WORKER_GROUP_CONTEXT_H_
#define TILELANG_TRANSFORM_WORKER_GROUP_CONTEXT_H_

#include <tvm/tirx/stmt.h>
#include <tvm/ir/cast.h>
#include "../../support/check.h"

namespace tvm::tl {
// Shared by layout collection and TileOp lowering. The physical launch binder
// remains intact: HMX physical ownership must not be inferred from a local ID.
struct WorkerGroupContext {
  PrimExpr local_id;
  Range bounds;
  static WorkerGroupContext From(const tirx::AttrStmtNode *op) {
    auto spec = Downcast<ffi::Map<ffi::String, ffi::Any>>(op->node);
    auto size = spec.at("local_size").cast<Integer>();
    auto first = spec.at("first_worker").cast<Integer>();
    auto engine = spec.at("engine").cast<ffi::String>();
    CHECK(size->value > 0 && first->value >= 0, ValueError)
        << "invalid worker group extent";
    CHECK(engine == "hvx" || engine == "hmx", ValueError)
        << "invalid worker group engine";
    CHECK(engine != "hmx" || (first->value == 0 && size->value == 1), ValueError)
        << "HMX group must own physical worker zero exclusively";
    return {spec.at("local_id").cast<PrimExpr>(), Range::FromMinExtent(0, size)};
  }
};
}
#endif
