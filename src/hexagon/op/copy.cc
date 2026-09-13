/*!
 * \file tl/hexagon/op/copy.cc
 * \brief Hexagon implementation for tl.copy lowering.
 */

#include "op/copy.h"

#include "hexagon/layouts.h"
#include "hexagon/target_utils.h"
#include "support/check.h"

#include <tvm/tirx/builtin.h>
#include <tvm/runtime/logging.h>

#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace hexagon {

namespace {

bool IsVTCM(const Buffer &buf) {
  std::string scope = buf.scope();
  return scope.rfind("vtcm", 0) == 0 || scope.rfind("shared", 0) == 0;
}

bool IsAcc(const Buffer &buf) {
  std::string scope = buf.scope();
  return scope == "hmx.acc" || scope.find("fragment") != std::string::npos;
}

std::string LayoutOf(const Buffer &buf, const Map<String, ObjectRef> &ann,
                      const char *key) {
  if (auto val = ann.Get(key)) {
    if (const auto *s = val.value().as<StringImmNode>()) {
      return s->value;
    }
  }
  std::string scope = buf.scope();
  if (scope.size() >= 3 && scope.substr(scope.size() - 3) == ".ah") {
    return "ah";
  }
  if (scope.size() >= 3 && scope.substr(scope.size() - 3) == ".wh") {
    return "wh";
  }
  if (IsAcc(buf)) {
    return "ah";
  }
  return "rm";
}

HexagonLayoutMode FallbackLayoutModeOf(const Buffer &buf,
                                       const Map<String, ObjectRef> &ann,
                                       const char *key) {
  return HexagonLayoutModeFromString(String(LayoutOf(buf, ann, key)));
}

HexagonLayoutMode LayoutModeOf(const Buffer &buf, const LowerArgs &lower_args,
                               const Map<String, ObjectRef> &ann,
                               const char *key) {
  if (auto layout = lower_args.layout_map.Get(buf)) {
    HexagonLayoutMode mode = DetectHexagonLayoutMode(layout.value(), buf);
    if (mode != HexagonLayoutMode::kNone) {
      return mode;
    }
  }
  return FallbackLayoutModeOf(buf, ann, key);
}

bool IsAHLike(HexagonLayoutMode mode) {
  return mode == HexagonLayoutMode::kAH || mode == HexagonLayoutMode::kWH;
}

bool IsRM(HexagonLayoutMode mode) { return mode == HexagonLayoutMode::kRM; }

bool IsGlobal(const Buffer &buf) { return buf.scope() == "global"; }

bool SameDType(const Buffer &a, const Buffer &b) { return a->dtype == b->dtype; }

bool IsFP16(const Buffer &buf) { return buf->dtype == DataType::Float(16); }

bool IsFP32(const Buffer &buf) { return buf->dtype == DataType::Float(32); }

const char *ModeName(HexagonLayoutMode mode) {
  switch (mode) {
  case HexagonLayoutMode::kRM: return "rm";
  case HexagonLayoutMode::kAH: return "ah";
  case HexagonLayoutMode::kWH: return "wh";
  case HexagonLayoutMode::kIdentityForTest: return "identity_for_test";
  case HexagonLayoutMode::kNone: return "none";
  }
  return "unknown";
}

std::string CopyRoute(const Buffer &src, const Buffer &dst,
                      HexagonLayoutMode src_layout,
                      HexagonLayoutMode dst_layout) {
  std::ostringstream os;
  os << "src(scope=" << src.scope() << ", dtype=" << src->dtype
     << ", layout=" << ModeName(src_layout) << ") -> dst(scope="
     << dst.scope() << ", dtype=" << dst->dtype
     << ", layout=" << ModeName(dst_layout) << ")";
  return os.str();
}

PrimExpr I32(int value) { return IntImm(DataType::Int(32), value); }

// Point-view convention (mirrors the hexagon gemm view rule): a T.copy
// operand written as a point BufferLoad such as buf[i0, i1, 0, 0] denotes the
// trailing 2-D tile rooted at that point, not a single element.  The copy op
// encodes point accesses as all-ones extents; expand the trailing two extents
// to the buffer's trailing 2-D shape so recipes see the real tile.
Array<Range> ExpandPointViewRange(const Buffer &buf, const Array<Range> &ranges) {
  size_t n = ranges.size();
  if (n < 2 || buf->shape.size() < 2 || buf->shape.size() != n) return ranges;
  for (const auto &r : ranges) {
    const auto *e = r->extent.as<IntImmNode>();
    if (e == nullptr || e->value != 1) return ranges;
  }
  const auto *r0 = buf->shape[buf->shape.size() - 2].as<IntImmNode>();
  const auto *r1 = buf->shape[buf->shape.size() - 1].as<IntImmNode>();
  if (r0 == nullptr || r1 == nullptr || (r0->value == 1 && r1->value == 1)) return ranges;
  Array<Range> out = ranges;
  out.Set(n - 2, Range::FromMinExtent(ranges[n - 2]->min, I32(static_cast<int>(r0->value))));
  out.Set(n - 1, Range::FromMinExtent(ranges[n - 1]->min, I32(static_cast<int>(r1->value))));
  return out;
}

Stmt MakeExtern(const char *name, const CopyNode &op) {
  Array<PrimExpr> args;
  Array<Range> src_range = ExpandPointViewRange(op.src, op.src_range);
  Array<Range> dst_range = ExpandPointViewRange(op.dst, op.dst_range);
  args.push_back(StringImm(name));
  args.push_back(op.src->data);
  args.push_back(op.dst->data);
  args.push_back(I32(static_cast<int>(src_range.size())));
  for (const auto &r : src_range) {
    args.push_back(r->min);
    args.push_back(r->extent);
  }
  args.push_back(I32(static_cast<int>(dst_range.size())));
  for (const auto &r : dst_range) {
    args.push_back(r->min);
    args.push_back(r->extent);
  }
  if (auto val = op.annotations.Get("hexagon.copy.trans")) {
    if (const auto *i = val.value().as<IntImmNode>()) {
      args.push_back(I32(static_cast<int>(i->value)));
    }
  }
  return Evaluate(Call(DataType::Handle(), builtin::call_pure_extern(), args));
}

} // namespace

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op,
                               const LayoutInferArgs &layout_args,
                               InferLevel level) {
    (void)op;
    (void)layout_args;
    (void)level;
    return {};
  }

  static Stmt Lower(const CopyNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    (void)analyzer;
    const HexagonLayoutMode src_layout = LayoutModeOf(
        op.src, lower_args, op.annotations, "hexagon.copy.src_layout");
    const HexagonLayoutMode dst_layout = LayoutModeOf(
        op.dst, lower_args, op.annotations, "hexagon.copy.dst_layout");

    if (IsGlobal(op.src) && IsVTCM(op.dst) && IsFP16(op.src) &&
        IsRM(src_layout) && IsAHLike(dst_layout)) {
      return MakeExtern("hexagon.copy_rm_ah", op);
    }
    if ((IsGlobal(op.src) || IsVTCM(op.src)) && IsVTCM(op.dst) && IsFP32(op.src) &&
        IsFP16(op.dst) && IsRM(src_layout) &&
        dst_layout == HexagonLayoutMode::kAH) {
      return MakeExtern("hexagon.copy_f32_ah", op);
    }
    if ((IsGlobal(op.src) || IsVTCM(op.src)) && IsVTCM(op.dst) && IsFP32(op.src) &&
        IsFP16(op.dst) && IsRM(src_layout) &&
        dst_layout == HexagonLayoutMode::kWH) {
      return MakeExtern("hexagon.copy_f32_wh", op);
    }
    if (IsVTCM(op.src) && IsGlobal(op.dst) && IsAHLike(src_layout) &&
        IsRM(dst_layout)) {
      return MakeExtern("hexagon.copy_ah_rm", op);
    }
    if (IsAcc(op.src) && IsGlobal(op.dst) && IsRM(dst_layout)) {
      return MakeExtern("hexagon.copy_acc_rm", op);
    }
    if (IsAcc(op.src) && IsVTCM(op.dst) && IsFP16(op.dst) &&
        IsRM(dst_layout)) {
      return MakeExtern("hexagon.copy_acc_rm", op);
    }
    if (IsRM(src_layout) && IsRM(dst_layout) && SameDType(op.src, op.dst) &&
        ((IsGlobal(op.src) && IsGlobal(op.dst)) ||
         (IsGlobal(op.src) && IsVTCM(op.dst)) ||
         (IsVTCM(op.src) && IsGlobal(op.dst)))) {
      return MakeExtern("hexagon.copy_ddr", op);
    }
    if ((IsVTCM(op.src) || IsVTCM(op.dst) || IsAcc(op.src) || IsAcc(op.dst)) &&
        !(IsVTCM(op.src) && IsVTCM(op.dst) && IsRM(src_layout) && IsRM(dst_layout) &&
          SameDType(op.src, op.dst))) {
      LOG(FATAL) << "Unsupported Hexagon copy route at lowering: "
                 << CopyRoute(op.src, op.dst, src_layout, dst_layout)
                 << "; no verified in-tile primitive exists for this layout/dtype/scope combination";
    }
    return tl::LowerNormalCopy(op, lower_args, analyzer);
  }
};

} // namespace hexagon

namespace {

bool MatchHexagonCopyTarget(Target target) { return TargetIsHexagon(target); }

bool RegisterHexagonCopy() {
  RegisterCopyImpl(CopyImpl{
      "hexagon.Copy",
      MatchHexagonCopyTarget,
      100,
      hexagon::Copy::InferLayout,
      hexagon::Copy::Lower,
  });
  return true;
}

const bool hexagon_copy_registered = RegisterHexagonCopy();

} // namespace

} // namespace tl
} // namespace tvm
