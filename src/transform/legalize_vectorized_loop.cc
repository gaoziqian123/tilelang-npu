/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file layout_inference.cc
 * \brief infer the fragment/shared memory layout
 */

#include "support/check.h"
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/utils.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <tvm/target/target.h>

#include <queue>
#include <cmath>

#include "../op/parallel.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "loop_partition.h"
#include "loop_vectorize.h"

namespace tvm {
namespace tl {

using namespace tirx;
using arith::IRMutatorWithAnalyzer;

// A lazy exp may become a vector value mask only when its input is safe to
// evaluate over the ENTIRE loop domain, not just under the original guard.
// Keep this audit deliberately small: unknown calls, pointer operations and
// volatile functions are not candidates for speculative evaluation.
class HexagonMaskedExpLegalizer : public IRMutatorWithAnalyzer {
public:
  explicit HexagonMaskedExpLegalizer(arith::Analyzer *analyzer)
      : IRMutatorWithAnalyzer(analyzer) {}
  Stmt Rewrite(Stmt body) { return VisitStmt(body); }

private:
  bool SafeInput(const PrimExpr &expr) {
    if (expr.as<FloatImmNode>() || expr.as<IntImmNode>() || expr.as<VarNode>())
      return true;
    if (const auto *load = expr.as<BufferLoadNode>()) {
      if (load->predicate.defined() || load->indices.size() != load->buffer->shape.size())
        return false;
      for (size_t i = 0; i < load->indices.size(); ++i) {
        // No branch constraint is installed while this proof runs.
        if (!analyzer_->CanProve(load->indices[i] >= 0) ||
            !analyzer_->CanProve(load->indices[i] < load->buffer->shape[i]))
          return false;
        if (!SafeInput(load->indices[i])) return false;
      }
      return load->dtype == DataType::Float(32);
    }
    if (const auto *cast = expr.as<CastNode>())
      return cast->dtype == DataType::Float(32) && SafeInput(cast->value);
    if (const auto *n = expr.as<AddNode>()) return SafeInput(n->a) && SafeInput(n->b);
    if (const auto *n = expr.as<SubNode>()) return SafeInput(n->a) && SafeInput(n->b);
    if (const auto *n = expr.as<MulNode>()) return SafeInput(n->a) && SafeInput(n->b);
    return false;
  }

  PrimExpr Mask(const PrimExpr &condition, const PrimExpr &yes,
                const PrimExpr &no) {
    const auto *exp = yes.as<CallNode>();
    const auto *zero = no.as<FloatImmNode>();
    if (!exp || !exp->op.same_as(Op::Get("tirx.exp")) || exp->args.size() != 1 ||
        exp->dtype != DataType::Float(32) || !zero || zero->value != 0 ||
        std::signbit(zero->value) || no.dtype() != exp->dtype ||
        !SafeInput(exp->args[0])) return PrimExpr();
    // Do not duplicate predicates with observable effects, including loads.
    bool pure = true;
    PostOrderVisit(condition, [&](const ObjectRef &n) {
      if (n.as<CallNode>() || n.as<BufferLoadNode>()) pure = false;
    });
    if (!pure) return PrimExpr();
    // Express inactive inputs as zero, but downstream simplification may remove
    // this inner Select and feed raw inputs to exp. Inactive NaN/Inf lanes may
    // therefore still invoke cold exceptional repair. The outer Select, not
    // multiplication, preserves active IEEE values and exact inactive +zero;
    // this guarantees output semantics, not avoidance of the cold path.
    PrimExpr input = Select(condition, exp->args[0], no);
    return Select(condition, Call(exp->dtype, exp->op, {input}), no);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.same_as(builtin::if_then_else()) && call->args.size() == 3) {
        PrimExpr value = Mask(call->args[0], call->args[1], call->args[2]);
        if (value.defined())
          return BufferStore(op->buffer, value, op->indices, op->predicate, op->span);
      }
    }
    return GetRef<Stmt>(op);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    const auto *yes = op->then_case.as<BufferStoreNode>();
    const auto *no = op->else_case.as<BufferStoreNode>();
    if (yes && no && !yes->predicate.defined() && !no->predicate.defined() &&
        yes->buffer.same_as(no->buffer) && yes->indices.size() == no->indices.size()) {
      bool same = true;
      for (size_t i = 0; i < yes->indices.size(); ++i)
        same &= SafeInput(yes->indices[i]) && SafeInput(no->indices[i]) &&
                analyzer_->CanProveEqual(yes->indices[i], no->indices[i]);
      PrimExpr value = Mask(op->condition, yes->value, no->value);
      if (same && value.defined())
        return BufferStore(yes->buffer, value, yes->indices, {}, op->span);
    }
    // In particular do not descend under a guard and accidentally use that
    // guard as the proof that speculative loads are in bounds.
    return GetRef<Stmt>(op);
  }
};

// Class to legalize vectorized loops by transforming them appropriately
class LoopVectorizedLegalizer : IRMutatorWithAnalyzer {
public:
  // Static method to substitute and transform the given PrimFunc
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    // Create an instance of the legalizer with the analyzer
    LoopVectorizedLegalizer substituter(&analyzer);
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    substituter.hexagon_ = target.defined() && target.value()->kind->name == "hexagon";
    PostOrderVisit(f->body, [&](const ObjectRef &n) {
      if (const auto *attr = n.as<AttrStmtNode>())
        if (attr->attr_key == "volatile_scope") substituter.hexagon_ = false;
    });
    // Get a mutable copy of the function node
    PrimFuncNode *fptr = f.CopyOnWrite();
    // Apply the legalizer to the function body
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

private:
  bool hexagon_ = false;
  // Constructor initializing the base class with the analyzer
  LoopVectorizedLegalizer(arith::Analyzer *analyzer)
      : arith::IRMutatorWithAnalyzer(analyzer) {}

  bool IsNonTrivialVectorizedLoop(const For &loop) {
    PrimExpr extent = analyzer_->Simplify(loop->extent);
    auto extent_as_int = as_const_int(extent);
    return !extent_as_int || *extent_as_int > 1;
  }

  // Override the VisitStmt_ method to handle ForNode (loop statements)
  Stmt VisitStmt_(const ForNode *op) final {
    if (hexagon_ && op->kind == ForKind::kVectorized) {
      // Bind enclosing loops through the normal analyzer visitor, then let a
      // separate visitor bind this loop without any conditional assumptions.
      HexagonMaskedExpLegalizer masks(analyzer_);
      For rewritten = Downcast<For>(masks.Rewrite(GetRef<Stmt>(op)));
      return Legalize(rewritten.get());
    }
    return Legalize(op);
  }

  Stmt Legalize(const ForNode *op) {
    // Visit and potentially modify the loop node
    For for_node = Downcast<For>(IRMutatorWithAnalyzer::VisitStmt_(op));
    // If the loop is not vectorized, proceed with the default behavior
    if (for_node->kind != ForKind::kVectorized) {
      return for_node;
    }
    // Change the loop kind from vectorized to serial
    for_node.CopyOnWrite()->kind = ForKind::kSerial;
    int vectorize_size = GetVectorizeSize(for_node, analyzer_);
    if (vectorize_size <= 1 && IsNonTrivialVectorizedLoop(for_node)) {
      LOG(WARNING) << "T.vectorized loop over `" << for_node->loop_var
                   << "` with extent " << for_node->extent
                   << " is lowered as a serial loop because TileLang could "
                   << "not find a valid vectorization plan. Scalar accumulator "
                   << "updates inside the loop are a common cause; move "
                   << "reductions to T.unroll or T.serial if this is intended.";
    }
    // Apply vectorization transformation to the loop
    For result = VectorizeLoop(for_node, analyzer_, {}, vectorize_size);
    // Provenance only: consumers must still prove injectivity and independence.
    if (tvm::transform::PassContext::Current()->GetConfig<Bool>(
            "tl.enable_ordered_accumulator_promotion", Bool(false)).value()->value &&
        result->kind == ForKind::kSerial && vectorize_size > 1)
      result.CopyOnWrite()->annotations.Set("tl.vectorized_group", Bool(true));
    return result;
  }
};

// Create a pass that legalizes vectorized loops in the IRModule
tvm::transform::Pass LegalizeVectorizedLoop() {
  using namespace tirx::transform;
  // Define the transformation function to be applied
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return LoopVectorizedLegalizer::Substitute(std::move(f));
  };
  // Create and return a PrimFunc pass with the transformation function
  return CreatePrimFuncPass(pass_func, 0, "tl.LegalizeVectorizedLoop", {});
}

// Register the pass globally so it can be used in the compilation pipeline
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LegalizeVectorizedLoop",
                        LegalizeVectorizedLoop);
}

} // namespace tl
} // namespace tvm
