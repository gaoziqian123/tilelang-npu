/*!
 * \file tl/hexagon/op/copy.cc
 * \brief Hexagon implementation for tl.copy lowering.
 */

#include "op/copy.h"

#include "hexagon/target_utils.h"
#include "support/check.h"

#include <tvm/tirx/builtin.h>

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
  if (IsAcc(buf)) {
    return "ah";
  }
  return "rm";
}

bool IsGlobal(const Buffer &buf) { return buf.scope() == "global"; }

PrimExpr I32(int value) { return IntImm(DataType::Int(32), value); }

Stmt MakeExtern(const char *name, const CopyNode &op) {
  Array<PrimExpr> args;
  args.push_back(StringImm(name));
  args.push_back(op.src->data);
  args.push_back(op.dst->data);
  args.push_back(I32(static_cast<int>(op.src_range.size())));
  for (const auto &r : op.src_range) {
    args.push_back(r->min);
    args.push_back(r->extent);
  }
  args.push_back(I32(static_cast<int>(op.dst_range.size())));
  for (const auto &r : op.dst_range) {
    args.push_back(r->min);
    args.push_back(r->extent);
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
    (void)lower_args;
    (void)analyzer;
    const std::string src_layout =
        LayoutOf(op.src, op.annotations, "hexagon.copy.src_layout");
    const std::string dst_layout =
        LayoutOf(op.dst, op.annotations, "hexagon.copy.dst_layout");

    if (IsGlobal(op.src) && IsVTCM(op.dst) && src_layout == "rm" &&
        dst_layout == "ah") {
      return MakeExtern("hexagon.copy_rm_ah", op);
    }
    if (IsVTCM(op.src) && IsGlobal(op.dst) && src_layout == "ah" &&
        dst_layout == "rm") {
      return MakeExtern("hexagon.copy_ah_rm", op);
    }
    if (IsAcc(op.src) && IsGlobal(op.dst) && dst_layout == "rm") {
      return MakeExtern("hexagon.copy_acc_rm", op);
    }
    if (src_layout == "rm" && dst_layout == "rm" &&
        ((IsGlobal(op.src) && IsGlobal(op.dst)) ||
         (IsGlobal(op.src) && IsVTCM(op.dst)) ||
         (IsVTCM(op.src) && IsGlobal(op.dst)))) {
      return MakeExtern("hexagon.copy_ddr", op);
    }
    TVM_FFI_ICHECK(false) << "Hexagon tl.copy unsupported scope/layout: "
                          << op.src.scope() << "(" << src_layout << ")->"
                          << op.dst.scope() << "(" << dst_layout << ")";
    return Evaluate(0);
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
