/*!
 * \file tl/hexagon/op/copy.cc
 * \brief Hexagon implementation for tl.copy lowering.
 */

#include "op/copy.h"
#include "affine_transpose.h"

#include "hexagon/layouts.h"
#include "hexagon/target_utils.h"
#include "support/check.h"

#include <tvm/tirx/builtin.h>
#include <tvm/runtime/logging.h>
#include <tvm/ir/transform.h>


namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace hexagon {

TVM_REGISTER_PASS_CONFIG_OPTION("tl.hexagon.typed_layout_copy", Bool);

namespace {
bool IsGlobal(const Buffer &buf);
bool IsVTCM(const Buffer &buf);

// Prove the lane permutation from the annotated maps. Never infer a layout
// from its name or silently fall back to scalar VTCM accesses.
Stmt LogicalTransform(const CopyNode &op, const LowerArgs &args,
                      arith::Analyzer *a) {
  CHECK(IsVTCM(op.dst) && (IsVTCM(op.src) || op.src.scope() == "local" ||
        op.src.scope() == "local.fragment"), ValueError)
      << "transform requires local/fragment/shared source and shared destination";
  CHECK((op.src->dtype == DataType::Float(16) || op.src->dtype == DataType::Float(32)) &&
        (op.dst->dtype == DataType::Float(16) || op.dst->dtype == DataType::Float(32)) &&
        (op.src->dtype == DataType::Float(16) || op.dst->dtype == DataType::Float(16)), ValueError)
      << "transform supports half layout conversion and half/float conversion";
  CHECK(op.src_range.size() == 2 && op.dst_range.size() == 2, ValueError)
      << "transform requires rank-two explicit regions";
  CHECK(!op.src->data.same_as(op.dst->data), ValueError)
      << "transform: overlapping storage requires an in-place permutation proof";
  auto sl = args.layout_map.Get(op.src), dl = args.layout_map.Get(op.dst);
  CHECK(sl && dl, ValueError) << "transform requires explicit layout annotations";
  CHECK(dl.value()->DetectInjective()->errors.empty(), ValueError)
      << "transform destination layout must be injective";
  CHECK(op.src->strides.empty() && op.dst->strides.empty() &&
        a->CanProveEqual(op.src->elem_offset, 0) &&
        a->CanProveEqual(op.dst->elem_offset, 0), ValueError)
      << "transform requires compact storage";
  int64_t extent[2];
  for (int d = 0; d < 2; ++d) {
    auto n = op.dst_range[d]->extent.as<IntImmNode>();
    CHECK(n && n->value > 0, ValueError) << "transform requires static extents";
    extent[d] = n->value;
    CHECK(a->CanProveEqual(op.src_range[d]->extent, Integer(n->value)), ValueError)
        << "transform logical shape mismatch";
    for (auto pair : {std::make_pair(op.src, op.src_range), std::make_pair(op.dst, op.dst_range)})
      CHECK(a->CanProve(pair.second[d]->min >= 0) &&
            a->CanProve(pair.second[d]->min + pair.second[d]->extent <= pair.first->shape[d]), ValueError)
          << "transform bounds unproved";
  }
  CHECK(extent[0] <= 65536 / extent[1] && extent[1] % 64 == 0, ValueError)
      << "transform requires complete 128-byte rows and at most 65536 elements";
  auto constant = [&](PrimExpr x) -> int64_t {
    auto s = a->Simplify(x); auto n = s.as<IntImmNode>();
    CHECK(n, ValueError) << "transform physical map must be statically provable";
    return n->value;
  };
  auto linear = [&](Layout l, Array<PrimExpr> ij) {
    auto mapped = l->Forward(ij); auto shape = l->OutputShape();
    PrimExpr offset = Integer(0);
    for (size_t d = 0; d < mapped.size(); ++d) offset = offset * shape[d] + mapped[d];
    return constant(offset);
  };
  auto physical = [&](Buffer b) {
    if (auto remap = args.buffer_remap.Get(b)) b = remap.value();
    PrimExpr size = Integer(1);
    for (auto x : b->shape) size *= x;
    // Fresh key prevents LowerTileOp from applying the layout a second time.
    return Buffer(b->data, b->dtype, {size}, {}, Integer(0), b->name,
                  b->data_alignment, b->offset_factor, b->buffer_type, {}, b->span);
  };
  Buffer src = physical(op.src), dst = physical(op.dst);
  const auto *fragment = sl.value().as<FragmentNode>();
  CHECK(op.src.scope() != "local.fragment" || fragment, ValueError)
      << "transform fragment ownership annotation missing";
  if (fragment) CHECK(a->CanProveEqual(fragment->ReplicateExtent(), 1), ValueError)
      << "transform replicated fragment unsupported";
  if (!fragment && op.src.scope() == "local")
    CHECK(a->CanProveEqual(args.thread_bounds->extent, 1), ValueError)
        << "transform local source requires single-worker ownership";
  if (args.require_smem_alignment) args.require_smem_alignment(dst->data, 128);
  Array<Stmt> stores;
  struct VectorCopy {
    int64_t destination, owner;
    std::vector<int64_t> bases, indices;
  };
  std::vector<VectorCopy> copies;
  // Enumerate in destination physical order, not logical row order. This also
  // supports the inverse (row-major -> blocked) permutation without scatter or
  // scalar VTCM stores. Every emitted store is one aligned contiguous vector.
  std::vector<std::pair<int64_t, std::pair<int64_t, int64_t>>> order;
  for (int64_t r=0; r<extent[0]; ++r) for (int64_t c=0; c<extent[1]; ++c)
    order.push_back({linear(dl.value(), {op.dst_range[0]->min + Integer(r),
                                       op.dst_range[1]->min + Integer(c)}), {r,c}});
  std::sort(order.begin(), order.end());
  for (size_t chunk = 0; chunk < order.size(); chunk += 64) {
    std::vector<int64_t> offsets, bases;
    int64_t owner = -1, destination = -1;
    for (int lane = 0; lane < 64; ++lane) {
      auto [r,c] = order[chunk+lane].second;
      Array<PrimExpr> si{op.src_range[0]->min + Integer(r), op.src_range[1]->min + Integer(c)};
      Array<PrimExpr> di{op.dst_range[0]->min + Integer(r), op.dst_range[1]->min + Integer(c)};
      int64_t s = linear(sl.value(), si), d = linear(dl.value(), di);
      CHECK(s >= 0 && s < constant(src->shape[0]) && d >= 0 && d < constant(dst->shape[0]), ValueError)
          << "transform physical bounds unproved";
      if (!lane) destination = d;
      CHECK(d == destination + lane, ValueError) << "transform destination must be row-contiguous";
      if (fragment) {
        auto t = constant(fragment->ForwardThread(si, Integer(0)));
        CHECK(t >= 0 && t < constant(args.thread_bounds->extent), ValueError)
            << "transform fragment owner outside worker group";
        if (!lane) owner = t;
        CHECK(t == owner, ValueError) << "transform vector crosses fragment owners";
      }
      offsets.push_back(s);
      int64_t base = s / 64 * 64;
      if (std::find(bases.begin(), bases.end(), base) == bases.end()) bases.push_back(base);
    }
    CHECK(destination % 64 == 0 && bases.size() <= 2, ValueError)
        << "transform requires aligned destination and at most two source vectors";
    VectorCopy copy{destination, owner >= 0 ? owner :
        constant(FloorMod(Integer(chunk / 64), args.thread_bounds->extent)), bases, {}};
    for (auto base : bases) {
      CHECK(base + 64 <= constant(src->shape[0]), ValueError) << "transform vector overread";
    }
    for (auto offset : offsets) {
      auto block = std::find(bases.begin(), bases.end(), offset / 64 * 64) - bases.begin();
      copy.indices.push_back(block * 64 + offset % 64);
    }
    copies.push_back(std::move(copy));
  }
  // Partition by exact lane permutation and physical owner. Fit maximal affine
  // runs of vector bases, checking EVERY enumerated vector against the fit.
  // Thus compaction changes neither addresses, casts, ownership nor operation
  // count; irregular maps retain length-one vector runs (never scalar fallback).
  std::vector<std::vector<VectorCopy>> classes;
  for (auto &copy : copies) {
    size_t group = 0;
    for (; group < classes.size(); ++group)
      if (classes[group][0].owner == copy.owner &&
          classes[group][0].indices == copy.indices &&
          classes[group][0].bases.size() == copy.bases.size()) break;
    if (group == classes.size()) classes.emplace_back();
    classes[group].push_back(std::move(copy));
  }
  for (const auto &group : classes) {
    Array<Stmt> runs;
    for (size_t start=0; start<group.size();) {
      const auto &first = group[start];
      size_t end = start + 1;
      int64_t dd = 0;
      std::vector<int64_t> ds(first.bases.size(), 0);
      if (end < group.size()) {
        dd = group[end].destination - first.destination;
        for (size_t b=0; b<ds.size(); ++b) ds[b] = group[end].bases[b] - first.bases[b];
        for (; end < group.size(); ++end) {
          bool matches = group[end].destination == first.destination + int64_t(end-start)*dd;
          for (size_t b=0; b<ds.size(); ++b)
            matches &= group[end].bases[b] == first.bases[b] + int64_t(end-start)*ds[b];
          if (!matches) break;
        }
      }
      Var index("transform_vector", DataType::Int(32));
      PrimExpr iteration = end-start == 1 ? PrimExpr(Integer(0)) : PrimExpr(index);
      Array<PrimExpr> vectors, indices;
      for (size_t b=0; b<first.bases.size(); ++b) {
        PrimExpr vector = BufferLoad(src, {Ramp(Integer(first.bases[b]) + iteration*Integer(ds[b]), Integer(1), 64)});
        if (op.src->dtype == DataType::Float(32)) vector = Cast(DataType::Float(16,64), vector);
        vectors.push_back(vector);
      }
      for (auto i : first.indices) indices.push_back(Integer(i));
      PrimExpr value = Shuffle(vectors, indices);
    if (op.dst->dtype == DataType::Float(32)) value = Cast(DataType::Float(32, 64), value);
      Stmt store = BufferStore(dst, value, {Ramp(Integer(first.destination)+iteration*Integer(dd), Integer(1), 64)});
      if (end-start > 1)
        store = For(index, 0, Integer(end-start), ForKind::kSerial, store,
                    std::nullopt, {{"pragma_unroll_explicit", Integer(0)}});
      runs.push_back(store);
      start = end;
    }
    stores.push_back(IfThenElse(args.thread_index - args.thread_bounds->min == Integer(group[0].owner), SeqStmt::Flatten(runs)));
  }
  return SeqStmt::Flatten(stores);
}

Optional<Stmt> DirectPhysicalCopy(const CopyNode &op, const LowerArgs &args,
                                  arith::Analyzer *analyzer) {
  if (!op.annotations.count("hexagon.dma_direct")) return std::nullopt;
   CHECK((IsGlobal(op.src) && IsVTCM(op.dst)) ||
         (IsVTCM(op.src) && IsGlobal(op.dst)), ValueError)
       << "direct DMA requires global/shared endpoints";
  CHECK(op.src->dtype == op.dst->dtype && !op.src->data.same_as(op.dst->data), ValueError)
      << "direct DMA requires disjoint same-dtype storage";
  CHECK(!op.src_oob_safe_value.defined() && !op.dst_block.defined(), ValueError)
      << "direct DMA cannot implement padding predicates";
  auto sl = args.layout_map.Get(op.src), dl = args.layout_map.Get(op.dst);
  CHECK(sl.has_value() && dl.has_value(), ValueError) << "direct DMA requires explicit physical layouts";
  CHECK(sl.value()->DetectInjective()->errors.empty() && dl.value()->DetectInjective()->errors.empty(), ValueError)
      << "direct DMA layout must be injective";
  CHECK(op.src_range.size() == 2 && op.dst_range.size() == 2 &&
        analyzer->CanProveEqual(op.src->elem_offset, 0) && analyzer->CanProveEqual(op.dst->elem_offset, 0), ValueError)
      << "direct DMA requires compact rank-two logical storage";
  for (auto b : {op.src,op.dst}) {
    if (!b->strides.empty())
      CHECK(b->strides.size()==2 && analyzer->CanProveEqual(b->strides[0],b->shape[1]) &&
            analyzer->CanProveEqual(b->strides[1],1), ValueError) << "DMA noncompact logical strides unsupported";
  }
  int64_t ext[2];
  for (int axis=0; axis<2; ++axis) {
    auto n = op.dst_range[axis]->extent.as<IntImmNode>();
    CHECK(n && n->value > 0, ValueError) << "direct DMA needs static panel extents";
    ext[axis] = n->value;
    CHECK(analyzer->CanProveEqual(op.src_range[axis]->extent, Integer(n->value)), ValueError) << "DMA extent mismatch";
    for (auto pair : {std::make_pair(op.src, op.src_range), std::make_pair(op.dst, op.dst_range)})
      CHECK(analyzer->CanProve(pair.second[axis]->min >= 0) &&
            analyzer->CanProve(pair.second[axis]->min + pair.second[axis]->extent <= pair.first->shape[axis]), ValueError)
          << "DMA region bounds unproved";
  }
  CHECK(ext[0] <= 1048576 / ext[1], ValueError) << "direct DMA proof panel exceeds compiler capacity";
  auto linear = [&](Layout layout, Array<PrimExpr> indices) {
    auto mapped = layout->Forward(indices); auto shape = layout->OutputShape();
    PrimExpr value = Integer(0);
    for (size_t i=0; i<mapped.size(); ++i) value = value*shape[i]+mapped[i];
    return analyzer->Simplify(value * op.src->dtype.bytes());
  };
  PrimExpr sb = linear(sl.value(), {op.src_range[0]->min, op.src_range[1]->min});
  PrimExpr db = linear(dl.value(), {op.dst_range[0]->min, op.dst_range[1]->min});
  std::vector<std::pair<int64_t,int64_t>> points;
  for (int64_t i=0;i<ext[0];++i) for(int64_t j=0;j<ext[1];++j) {
    auto s=analyzer->Simplify(linear(sl.value(), {op.src_range[0]->min+Integer(i),op.src_range[1]->min+Integer(j)})-sb);
    auto d=analyzer->Simplify(linear(dl.value(), {op.dst_range[0]->min+Integer(i),op.dst_range[1]->min+Integer(j)})-db);
    auto si=s.as<IntImmNode>(), di=d.as<IntImmNode>();
    CHECK(si && di && si->value>=0 && di->value>=0, ValueError) << "DMA physical relative mapping unproved";
    points.emplace_back(di->value,si->value);
  }
  std::sort(points.begin(), points.end());
  struct Span {int64_t dst,src,width;}; std::vector<Span> spans;
  for(auto [d,s]:points) {
    if(!spans.empty() && d==spans.back().dst+spans.back().width && s==spans.back().src+spans.back().width)
      spans.back().width += op.src->dtype.bytes();
    else spans.push_back({d,s,op.src->dtype.bytes()});
  }
  Array<Stmt> calls;
  for(size_t i=0;i<spans.size();) {
    auto p=spans[i]; int64_t rows=1,ss=p.width,ds=p.width;
    if(i+1<spans.size() && spans[i+1].width==p.width) {
      ss=spans[i+1].src-p.src; ds=spans[i+1].dst-p.dst;
      if(ss>=p.width && ds>=p.width)
        while(i+rows<spans.size() && spans[i+rows].width==p.width &&
              spans[i+rows].src==p.src+rows*ss && spans[i+rows].dst==p.dst+rows*ds) ++rows;
    }
    CHECK(p.width%128==0 && ss%128==0 && ds%128==0 &&
          analyzer->CanProveEqual(FloorMod(sb+Integer(p.src),128),0) &&
          analyzer->CanProveEqual(FloorMod(db+Integer(p.dst),128),0), ValueError)
        << "DMA descriptor alignment unproved; conversion required";
    CHECK(p.width<=0xFFFFFF && ss<=0xFFFFFF && ds<=0xFFFFFF && rows<=65535, ValueError)
        << "DMA descriptor exceeds v79 limits";
    calls.push_back(Evaluate(Call(DataType::Int(32), Op::Get("tl.hexagon.dma_copy_2d_wait"),
        {op.dst->data, db+Integer(p.dst), op.src->data, sb+Integer(p.src),
         Integer(p.width),Integer(rows),Integer(ss),Integer(ds)})));
    i+=rows;
  }
  // One leader submits and synchronously waits, then all group lanes acquire.
  auto sync=Evaluate(Call(DataType::Int(32),builtin::tvm_storage_sync(),{StringImm("shared")}));
  Stmt submit = IfThenElse(args.thread_index==args.thread_bounds->min,
                          calls.size()==1 ? calls[0] : SeqStmt(calls));
  // A shared source can have multiple writers. Publish the entire tile before
  // the sole DMA owner reads it; synchronous completion precedes slot reuse.
  if (IsVTCM(op.src)) return SeqStmt({sync, submit, sync});
  return SeqStmt({submit, sync});
}

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
  if (auto val = ann.Get(key)) {
    if (const auto *s = val.value().as<StringImmNode>()) {
      return HexagonLayoutModeFromString(String(s->value));
    }
  }
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


bool IsGlobal(const Buffer &buf) {
  std::string scope = buf.scope();
  return scope == "global" || scope.rfind("global.", 0) == 0;
}


bool IsFP16(const Buffer &buf) { return buf->dtype == DataType::Float(16); }


PrimExpr I32(int value) { return IntImm(DataType::Int(32), value); }

// Resolve the actual two-dimensional address map, not the logical shape.
// Only identity and axis permutation are supported. In particular, a layout
// annotation that was not materialized by LowerTileOp is not a storage proof.
bool PhysicalTransposeView(Buffer logical, Array<Range> region,
                           const LowerArgs &args, arith::Analyzer *a,
                           Buffer *physical, Array<Range> *ranges,
                           bool *permuted) {
  if (logical->shape.size() != 2 || region.size() != 2)
    return false;
  Buffer b = logical;
  bool swap = false;
  if (auto layout = args.layout_map.Get(logical)) {
    auto remap = args.buffer_remap.Get(logical);
    if (!remap) return false;
    b = remap.value();
    Var i("copy_i"), j("copy_j");
    auto indices = layout.value()->Forward({i, j});
    if (indices.size() != 2) return false;
    if (a->CanProveEqual(indices[0], i) && a->CanProveEqual(indices[1], j)) {
      swap = false;
    } else if (a->CanProveEqual(indices[0], j) && a->CanProveEqual(indices[1], i)) {
      swap = true;
    } else {
      return false;
    }
  } else if (args.buffer_remap.count(logical)) {
    return false;
  }
  if (b->shape.size() != 2 || (!b->strides.empty() && b->strides.size() != 2))
    return false;
  Array<Range> r = swap ? Array<Range>{region[1], region[0]} : region;
  // A standard strided buffer view can encode the same permutation without
  // a layout annotation. Canonicalize it to physical row/column order.
  if (!b->strides.empty() && a->CanProveEqual(b->strides[0], 1) &&
      a->CanProve(b->strides[1] >= b->shape[0])) {
    b = Buffer(b->data, b->dtype, {b->shape[1], b->shape[0]},
               {b->strides[1], b->strides[0]}, b->elem_offset, b->name,
               b->data_alignment, b->offset_factor, b->buffer_type, {}, b->span);
    r = {r[1], r[0]};
    swap = !swap;
  }
  // Clone even identity views: LowerTileOp revisits returned BufferLoads and
  // applies Forward to keys in buffer_remap. Physical indices must not be
  // transformed a second time.
  *physical = Buffer(b->data, b->dtype, b->shape, b->strides, b->elem_offset,
                     b->name, b->data_alignment, b->offset_factor,
                     b->buffer_type, {}, b->span);
  *ranges = r;
  *permuted = swap;
  return true;
}

Optional<Stmt> TryAffineTransposeCopy(const CopyNode &op, const LowerArgs &args,
                                      arith::Analyzer *a) {
  if (!transform::PassContext::Current()->GetConfig<Bool>(
          "tl.hexagon.affine_transpose").value_or(Bool(false)) ||
      op.src_oob_safe_value.defined() || op.dst_block.defined() ||
      !op.annotations.empty() || op.src->dtype != op.dst->dtype ||
      (op.src->dtype != DataType::Float(16) && op.src->dtype != DataType::Float(32)) ||
      op.src_range.size() != 2 || op.dst_range.size() != 2)
    return std::nullopt;
  for (int i = 0; i < 2; ++i)
    if (!a->CanProveEqual(op.src_range[i]->extent, op.dst_range[i]->extent))
      return std::nullopt;
  // Local permutation copies cannot become cooperative merely because their
  // allocation/alignment proof fails and the ordinary-copy fallback is taken.
  for (Buffer b : {op.src, op.dst}) {
    if (b.scope() != "local") continue;
    bool permutation = b->strides.size() == 2 &&
                       a->CanProveEqual(b->strides[0], 1) &&
                       a->CanProve(b->strides[1] > 1);
    if (auto layout = args.layout_map.Get(b)) {
      Var i("local_i"), j("local_j");
      auto index = layout.value()->Forward({i, j});
      permutation |= index.size() == 2 && a->CanProveEqual(index[0], j) &&
                     a->CanProveEqual(index[1], i);
    }
    if (permutation) {
      ICHECK(args.thread_bounds.defined() && args.thread_index.defined())
          << "Hexagon transpose requires defined thread_bounds and thread_index";
      ICHECK(a->CanProveEqual(args.thread_bounds->extent, 1))
          << "Local/shared transpose requires a single worker";
    }
  }
  Buffer src, dst;
  Array<Range> sr, dr;
  bool sp, dp;
  if (!PhysicalTransposeView(op.src, op.src_range, args, a, &src, &sr, &sp) ||
      !PhysicalTransposeView(op.dst, op.dst_range, args, a, &dst, &dr, &dp) || sp == dp)
    return std::nullopt;
  if (LegalAffineTranspose(src, dst, sr, dr, args, a))
    return EmitAffineTranspose(src, dst, sr, dr, args, a);
  ICHECK(!IsVTCM(src) && !IsVTCM(dst))
      << "Hexagon VTCM transpose requires proven bounds, strides and 128B padded allocations";
  return std::nullopt;
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
    if (op.annotations.count("tl.logical_transform"))
      return LogicalTransform(op, lower_args, analyzer);
    if (auto direct = DirectPhysicalCopy(op, lower_args, analyzer)) return direct.value();
    if (auto transpose = TryAffineTransposeCopy(op, lower_args, analyzer))
      return transpose.value();
    // Retain real memory effects and apply the standard AH/WH layout maps.
      auto src_mode = LayoutModeOf(op.src, lower_args, op.annotations,
                                   "hexagon.copy.src_layout");
      auto dst_mode = LayoutModeOf(op.dst, lower_args, op.annotations,
                                    "hexagon.copy.dst_layout");
      if (transform::PassContext::Current()->GetConfig<Bool>(
              "tl.hexagon.typed_layout_copy").value_or(Bool(false)) &&
          op.src.scope() == "local" && IsVTCM(op.dst) &&
          src_mode == HexagonLayoutMode::kRM && IsAHLike(dst_mode) &&
          op.src->dtype == DataType::Float(32) && IsFP16(op.dst) &&
          op.src->shape.size() == 2 && op.dst->shape.size() == 2 &&
          op.src->strides.empty() && op.dst->strides.empty() &&
          !op.src->data.same_as(op.dst->data)) {
        bool wh = dst_mode == HexagonLayoutMode::kWH;
        bool legal = analyzer->CanProveEqual(op.dst->elem_offset, 0);
        // Do not silently interpret an unrecognized source layout as RM.
        if (auto layout = lower_args.layout_map.Get(op.src))
          legal &= DetectHexagonLayoutMode(layout.value(),op.src) == HexagonLayoutMode::kRM;
        // VTCM/shared allocations are provided by the native allocator with
        // 128-byte alignment; external views require explicit alignment proof.
        legal &= op.dst->data_alignment >= 128 ||
                 (op.dst.scope() == "shared.dyn" &&
                  lower_args.require_smem_alignment != nullptr);
        for (int axis = 0; axis < 2; ++axis) {
          auto s = op.src_range[axis], d = op.dst_range[axis];
          int quantum = (axis == 0) == wh ? 32 : 2;
          legal &= analyzer->CanProveEqual(s->extent, d->extent);
          legal &= analyzer->CanProve(s->min >= 0) &&
                   analyzer->CanProve(s->min + s->extent <= op.src->shape[axis]);
          legal &= analyzer->CanProve(d->min >= 0) &&
                   analyzer->CanProve(d->min + d->extent <= op.dst->shape[axis]);
          legal &= analyzer->CanProveEqual(FloorMod(d->min, quantum), 0) &&
                   analyzer->CanProveEqual(FloorMod(d->extent, quantum), 0);
          legal &= analyzer->CanProveEqual(FloorMod(op.dst->shape[axis], 32), 0);
        }
        if (legal) {
          if (lower_args.require_smem_alignment)
            lower_args.require_smem_alignment(op.dst->data,128);
          auto ptr = [](Buffer b, PrimExpr r, PrimExpr c) {
            return Call(DataType::Handle(), builtin::address_of(), {BufferLoad(b, {r,c})});
          };
          return Evaluate(Call(DataType::Handle(), builtin::call_extern(), {
            StringImm(wh ? "tl::cast_pack_f32_f16<true>" : "tl::cast_pack_f32_f16<false>"),
            ptr(op.dst, I32(0), I32(0)),
            ptr(op.src, op.src_range[0]->min, op.src_range[1]->min),
            op.dst->shape[1], op.src->shape[1], op.dst_range[0]->min,
            op.dst_range[1]->min, op.dst_range[0]->extent, op.dst_range[1]->extent}));
        }
      }
      bool row_read=IsVTCM(op.src) && src_mode==HexagonLayoutMode::kAH && op.dst.scope()=="local";
      bool row_write=IsVTCM(op.dst) && dst_mode==HexagonLayoutMode::kAH && op.src.scope()=="local";
      if ((row_read || row_write) && IsFP16(op.src) && IsFP16(op.dst) &&
          op.src->shape.size()==2 && op.dst->shape.size()==2) {
        auto shared=row_read?op.src:op.dst;
        auto local=row_read?op.dst:op.src;
        auto region=row_read?op.src_range:op.dst_range;
        auto lr=row_read?op.dst_range:op.src_range;
        ICHECK(analyzer->CanProveEqual(region[1]->min,0));
        ICHECK(analyzer->CanProveEqual(region[1]->extent,shared->shape[1]));
        ICHECK(analyzer->CanProveEqual(local->shape[1],shared->shape[1]));
        ICHECK(analyzer->CanProveEqual(FloorMod(shared->shape[1],32),0));
        ICHECK(analyzer->CanProveEqual(FloorMod(shared->shape[0],32),0));
        ICHECK(shared->strides.empty() && local->strides.empty());
        ICHECK(!shared->data.same_as(local->data)) << "AH row copy cannot alias";
        ICHECK(analyzer->CanProveEqual(shared->elem_offset,0));
        ICHECK(analyzer->CanProveEqual(local->elem_offset,0));
        ICHECK(shared->data_alignment >= 128 ||
               (shared.scope()=="shared.dyn" && lower_args.require_smem_alignment));
        if(lower_args.require_smem_alignment)
          lower_args.require_smem_alignment(shared->data,128);
        ICHECK(analyzer->CanProveEqual(lr[0]->min,0));
        ICHECK(analyzer->CanProveEqual(lr[1]->min,0));
        ICHECK(analyzer->CanProveEqual(region[0]->extent,local->shape[0]));
        if(row_write) {
          ICHECK(analyzer->CanProveEqual(FloorMod(region[0]->min,2),0));
          ICHECK(analyzer->CanProveEqual(FloorMod(region[0]->extent,2),0));
        }
        auto ptr=[](Buffer b) { return Call(DataType::Handle(),builtin::address_of(),
            {BufferLoad(b,{I32(0),I32(0)})}); };
        return Evaluate(Call(DataType::Handle(),builtin::call_extern(),
          {StringImm(row_read?"tl::gemm_unpack_ah_rows":"tl::gemm_pack_ah_rows"),
           ptr(op.dst),ptr(op.src),shared->shape[1],region[0]->min,region[0]->extent}));
      }
      if (IsVTCM(op.src) && IsGlobal(op.dst) && IsFP16(op.src) &&
          IsFP16(op.dst) && src_mode == HexagonLayoutMode::kAH) {
        ICHECK_EQ(op.src->shape.size(), 2);
        ICHECK_EQ(op.dst->shape.size(), 2);
        if (!op.dst->strides.empty()) {
          ICHECK(analyzer->CanProveEqual(op.dst->strides[0], op.dst->shape[1]));
          ICHECK(analyzer->CanProveEqual(op.dst->strides[1], 1));
        }
        for (int axis = 0; axis < 2; ++axis) {
          ICHECK(analyzer->CanProveEqual(op.src_range[axis]->min, 0));
          ICHECK(analyzer->CanProveEqual(op.src_range[axis]->extent, op.src->shape[axis]));
          ICHECK(analyzer->CanProveEqual(op.dst_range[axis]->extent, op.src->shape[axis]));
          ICHECK(analyzer->CanProveEqual(FloorMod(op.dst_range[axis]->min, 32), 0));
          ICHECK(analyzer->CanProveEqual(FloorMod(op.src->shape[axis], 32), 0));
          ICHECK(analyzer->CanProveEqual(FloorMod(op.dst->shape[axis], 32), 0));
        }
        Var tile("unpack_tile");
        bool paired = analyzer->CanProveEqual(FloorMod(op.src->shape[1], 64), 0) &&
                      analyzer->CanProveEqual(FloorMod(op.dst->shape[1], 64), 0) &&
                      analyzer->CanProveEqual(FloorMod(op.dst_range[1]->min, 64), 0);
        int width = paired ? 64 : 32;
        PrimExpr columns = FloorDiv(op.src->shape[1], width);
        PrimExpr tiles = FloorDiv(op.src->shape[0], 32) * columns;
        PrimExpr per_worker = FloorDiv(tiles + lower_args.thread_bounds->extent - 1,
                                      lower_args.thread_bounds->extent);
        PrimExpr index = tile + per_worker *
                         (lower_args.thread_index - lower_args.thread_bounds->min);
        PrimExpr row = FloorDiv(index, columns), col = FloorMod(index, columns);
        auto address = [](Buffer buffer, PrimExpr r, PrimExpr c) {
          return Call(DataType::Handle(), builtin::address_of(), {BufferLoad(buffer, {r,c})});
        };
        Stmt body = Evaluate(Call(DataType::Handle(), Op::Get(paired ? "tl.hexagon.unpack_ah_pair_strided" : "tl.hexagon.unpack_ah_strided"),
          {address(op.dst, op.dst_range[0]->min + row*32, op.dst_range[1]->min + col*width),
           address(op.src, row*32, col*width), op.dst->shape[1], I32(1)}));
        body = For(tile, 0, FloorDiv(tiles + lower_args.thread_bounds->extent - 1,
                                   lower_args.thread_bounds->extent), ForKind::kSerial,
                   IfThenElse(index < tiles, body));
        // All readers must finish before the next GEMM reuses this shared tile buffer.
        return SeqStmt({body, Evaluate(Call(DataType::Int(32), builtin::tvm_storage_sync(),
                                          {StringImm("shared")}))});
      }
      auto mode = LayoutModeOf(op.dst, lower_args, op.annotations,
                               "hexagon.copy.dst_layout");
      bool shared_rm = IsVTCM(op.src) && src_mode == HexagonLayoutMode::kRM;
      if ((IsGlobal(op.src) || shared_rm) && IsVTCM(op.dst) && IsFP16(op.src) &&
          IsFP16(op.dst) && IsAHLike(mode)) {
        // The HVX strided helpers read aligned peripheral 128-byte blocks.
        // Restrict this path to compact allocations and 32-aligned tile
        // regions so peripheral blocks stay inside the source allocation.
        ICHECK_EQ(op.src->shape.size(), 2);
        ICHECK_EQ(op.dst->shape.size(), 2);
        if (!op.src->strides.empty()) {
          ICHECK(analyzer->CanProveEqual(op.src->strides[0], op.src->shape[1]));
          ICHECK(analyzer->CanProveEqual(op.src->strides[1], 1));
        }
        for (int axis = 0; axis < 2; ++axis) {
          ICHECK(analyzer->CanProveEqual(FloorMod(op.src_range[axis]->min, 32), 0));
          ICHECK(analyzer->CanProveEqual(op.dst_range[axis]->min, 0));
          ICHECK(analyzer->CanProveEqual(op.src_range[axis]->extent, op.dst->shape[axis]));
          ICHECK(analyzer->CanProveEqual(op.dst_range[axis]->extent, op.dst->shape[axis]));
          ICHECK(analyzer->CanProveEqual(FloorMod(op.dst->shape[axis], 32), 0));
          ICHECK(analyzer->CanProveEqual(FloorMod(op.src->shape[axis], 32), 0));
        }
        Var tile("pack_tile");
        bool paired =
                      analyzer->CanProveEqual(FloorMod(op.dst->shape[1], 64), 0) &&
                      analyzer->CanProveEqual(FloorMod(op.src->shape[1], 64), 0) &&
                      analyzer->CanProveEqual(FloorMod(op.src_range[1]->min, 64), 0);
        int width = paired ? 64 : 32;
        PrimExpr columns = FloorDiv(op.dst->shape[1], width);
        bool strip = paired && op.dst->shape[1].as<IntImmNode>();
        if (strip) { width = op.dst->shape[1].as<IntImmNode>()->value; columns = I32(1); }
        PrimExpr tiles = FloorDiv(op.dst->shape[0], 32) * columns;
        PrimExpr per_worker = FloorDiv(tiles + lower_args.thread_bounds->extent - 1,
                                      lower_args.thread_bounds->extent);
        PrimExpr index = tile + per_worker * (lower_args.thread_index - lower_args.thread_bounds->min);
        PrimExpr row = FloorDiv(index, columns), col = FloorMod(index, columns);
        auto address = [](Buffer buffer, PrimExpr r, PrimExpr c) {
          return Call(DataType::Handle(), builtin::address_of(),
                      {BufferLoad(buffer, {r, c})});
        };
        const char *name = mode == HexagonLayoutMode::kAH
                               ? (strip ? "tl.hexagon.pack_ah_strip" : (paired ? "tl.hexagon.pack_ah_pair_strided" : "tl.hexagon.pack_ah_strided"))
                               : (strip ? "tl.hexagon.pack_wh_nt_strip" : (paired ? "tl.hexagon.pack_wh_nt_pair_strided" : "tl.hexagon.pack_wh_nt_strided"));
        Stmt body = Evaluate(Call(DataType::Handle(), Op::Get(name),
            {address(op.dst, row * 32, col * width),
             address(op.src, op.src_range[0]->min + row * 32,
                      op.src_range[1]->min + col * width), op.src->shape[1], I32(strip ? width : 1)}));
        body = IfThenElse(index < tiles, body);
        // Cyclic cooperative staging, one Tiles=1 helper per physical tile.
        // The GEMM prologue barrier publishes all workers' writes.
        Stmt loop = For(tile, 0, FloorDiv(tiles + lower_args.thread_bounds->extent - 1,
                                     lower_args.thread_bounds->extent),
                   ForKind::kSerial, body);
        if (shared_rm)
          loop = SeqStmt({Evaluate(Call(DataType::Int(32), builtin::tvm_storage_sync(),
                             {StringImm("shared")})), loop});
        if (mode == HexagonLayoutMode::kWH)
          return SeqStmt({loop, Evaluate(Call(DataType::Handle(), Op::Get("tl.hexagon.scatter_release"),
                             {address(op.dst, I32(0), I32(0))}))});
        return loop;
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
