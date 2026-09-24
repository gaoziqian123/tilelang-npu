# Hexagon NPU 后端 v2 设计：从 statement emitter 到 target codegen

> 目标：调研 TileLang 上游 CUDA 后端的架构，并把可迁移的编译器结构落到 Hexagon HMX/HVX/VTCM 后端重写方案。本文只定义设计，不改变代码。

## 0. 背景与设计原则

当前 Hexagon 后端位于 `tilelang/hexagon/`，本质是 **Python/AOT statement-level emitter**：

```text
TileLang TIR
  └─ HexagonPassPipelineBody
       ├─ LayoutInference + LowerTileOp
       ├─ HexagonCopyPartition / StoragePlan / WScratchPlan / Verify
       └─ remember_lowered_mod
  └─ build_hexagon_without_compile
       └─ emit_hexagon_c(...)  # 逐语句打印 DSP-side C
            └─ hexagon.gemm_hmx / copy_* / GDN leaf -> hexagon_rt.h recipe
```

它已经验证了大量设备事实（HMX lock、VTCM 预算、worker pool、FastRPC ABI、profiling slot），也接入了 TileLang 的一部分 pass。但是核心 HMX 行为仍被压缩成 `hexagon.gemm_hmx` extern 调用，`T.gemm` 不是编译器级 tile op，而是被 lowering 成 `hexagon_rt.h` 的 recipe 调用。

v2 的目标是把 Hexagon 改成与 CUDA 类似的 **target codegen**：

- `T.gemm` lowering 产生显式的 HMX fragment load / deep-chain MMA / acc read/store IR，而不是整块 recipe。
- AH/WH/VTCM layout、tile binding、worker/pipeline schedule 是 pass 的产物，而不是 emitter 内的字符串约定。
- 内联汇编 deep-chain HMX 作为 codegen intrinsic 发射，支持最多 32 个 `32x32x32 fp16` tile 一次 chain。
- `hexagon_rt.h` 退化为 runtime support layer：FastRPC ABI、session/lock、slab、profiling、少量不可表达的 helper；不再承担主要算子 recipe。

本文把 CUDA 后端的关键机制逐项映射到 Hexagon，并给出 v2 pass pipeline、codegen 层划分、实施路线与验证策略。

## 1. CUDA 后端调研

### 1.1 模块结构

CUDA 后端的入口不是单个 emitter，而是完整 backend manifest：

```text
tilelang/cuda/__init__.py
  ├─ target / transform / language / intrinsics / op
  ├─ backend.py
  │    ├─ BackendModule(name="cuda", target_kinds=("cuda",))
  │    ├─ pipelines={"cuda": CUDA_PIPELINE}
  │    ├─ device_codegens={"cuda": target.build.tilelang_cuda}
  │    └─ callbacks: validate + nvcc compile/cache
  ├─ pipeline.py
  │    └─ CUDAPassPipelineBody
  ├─ op/gemm/*.py
  │    └─ T.gemm target implementation: mma / wgmma / tcgen05 / fma
  ├─ intrinsics/layout/*.py
  │    └─ fragment/shared/tensorcore layout maps
  ├─ intrinsics/macro/*.py
  │    └─ TensorCoreIntrinEmitter / WGMMA emitter / TCGEN05 emitter
  └─ src/cuda/codegen/*.cc
       ├─ rt_mod_cuda.cc: 注册 target.build.tilelang_cuda
       ├─ codegen_cuda.cc: CodeGenTileLangCUDA C++ source printer
       └─ intrin_rule_cuda.cc: CUDA intrinsic lowering rules
```

关键观察：

1. Python 层只负责 **target-specific op lowering / layout description / pass orchestration**；最终 CUDA C++ 源码由 C++ `CodeGenTileLangCUDA` 发射。
2. `T.gemm` 不是在 final codegen 才识别，而是在 `LowerTileOp` 前后通过 target op implementation 变成 low-level `T.ptx_*` intrinsic。
3. PTX inline asm 没散落在 Python string 模板里，而是被封装成 TileLang/TIR intrinsic（如 `tl::ptx_mma`, `tl::ptx_ldmatrix_x*`），再由 C++ codegen 统一打印。

### 1.2 `T.gemm` 到 MMA 的 lowering 链路

以 `tilelang/cuda/op/gemm/gemm_mma.py` 的 `GemmMMA` 为代表：

```text
user T.gemm(A, B, C)
  └─ target op registry 选中 cuda.mma 实现
       ├─ infer_layout(target, thread_nums)
       │    ├─ shared operand -> make_swizzled_layout
       │    ├─ fragment operand -> mma_emitter.make_mma_load_layout
       │    └─ C fragment -> mma_emitter.make_mma_store_layout
       └─ lower(layout_map, target, thread_bounds, thread_index)
            ├─ TensorCoreIntrinEmitter(... thread_var=local_thread_id)
            ├─ alloc A_local / B_local fragment
            ├─ for ki in block_K / micro_size_k:
            │    ├─ emitter.ldmatrix_a(...)
            │    ├─ emitter.ldmatrix_b(...)
            │    └─ emitter.mma(...)
            └─ 返回一个简化后的 PrimFunc macro
  └─ LayoutInference 冻结 layout map
  └─ LowerTileOp 展开 tile op 与 remap layout-sensitive access
  └─ CodeGenTileLangCUDA 发射 ldmatrix / mma intrinsic 调用或 inline asm wrapper
```

`GemmMMA` 的四种组合很重要：

| 输入位置 | CUDA 行为 | Hexagon-v2 启示 |
|---|---|---|
| shared + shared | ldmatrix A/B -> local fragment -> mma | DDR/global 先 copy 到 VTCM AH/WH，再 HMX deep-chain |
| shared + fragment | ldmatrix A，B 已在 fragment | 支持 A stage + B 常驻 WH/fragment |
| fragment + shared | A 已在 fragment，ldmatrix B | 支持 A AH 常驻 + B stage |
| fragment + fragment | 直接 mma | HMX acc/deep-chain 内 loop，仅变 K slot |

CUDA 的 `TensorCoreIntrinEmitter` 把 warp partition、micro tile 形状、fragment local size、thread lane 到 fragment 元素的映射全封装为编译器对象。Hexagon-v2 需要同等对象，例如 `HMXIntrinEmitter`，但它的 lane 概念不是 CUDA warp lane，而是 **HMX tile / worker / K-chain slot**。

### 1.3 Layout inference

CUDA layout 体系由三层组成：

1. **layout 函数**：`tilelang/cuda/intrinsics/layout/mma_layout.py` 定义从 `(thread_id, local_id)` 到 shared/fragment 坐标的 index map，例如 `shared_16x16_to_mma_*`、`mma_store_*`。
2. **op-specific infer_layout**：`GemmMMA.infer_layout()` 返回 `{buffer: Layout}`，告诉 `LayoutInference` 哪些 shared/fragment buffer 需要 swizzle 或 fragment layout。
3. **LowerTileOp remap**：`src/transform/lower_tile_op.cc` 会在 `ptx_ldmatrix`、`tl.access_ptr`、`BufferLoad/Store` 等访问点查 layout map，重写地址表达式。

Hexagon v1 目前有 `tilelang/hexagon/language/layout.py` 与 `make_ah_layout/make_wh_layout`，但还有大量 scope suffix（`vtcm.ah` / `global.wh`）约定。v2 应迁移到 CUDA 模式：scope 表示 memory space，layout map 表示物理排布；scope suffix 只作为兼容层。

### 1.4 Thread/tile binding 表达方式

CUDA 的线程绑定来自 `T.Kernel(..., threads=...)` / `threadIdx` / `blockIdx`，在 `TensorCoreIntrinEmitter.get_thread_binding()` 中取得，并通过：

- `thread_bounds` / `thread_index` 表示当前绑定范围；
- `extract_thread_binding()` 把 `thread_id` 分解成 `(tx, warp_m, warp_n)`；
- `compute_warp_partition()` 决定 block row/col warps；
- `warp_row_tiles` / `warp_col_tiles` / `chunk` 决定每个 warp 的 fragment tile。

Hexagon 没有 CUDA warp，但有三个必须显式化的层级：

```text
FastRPC executor thread（HMX lock 所属线程）
  └─ worker pool，最多 6 个 HVX workers：copy/unpack/leaf/vector phase
      └─ HMX issue stream：RPC/HMX owner 串行发射 deep-chain MMA
          └─ HMX atom：32x32x32 fp16 tile，K-chain ≤ 32
```

因此 v2 不应照搬 `threadIdx`，而应定义 Hexagon target 的 tile binding：

- `blockIdx`：host/FastRPC ABI 层面的 grid，通常对应 M/N panel 或 head slice。
- `workerIdx`：HVX worker pool parallel loop，范围 `0..<=6`，只能包 copy/unpack/vector leaf，不能包 HMX issue。
- `hmxTileM/N/K`：编译器内部 schedule 轴，映射到 AH/WH tile offset 与 deep-chain slot。
- `chainSlot`：一次 deep-chain 的 K tile 编号，`0..31`。

### 1.5 软件流水与 async copy

CUDA pipeline 关键 pass：

```text
CUDAPassPipelineBodyPrologue
  ├─ PipelinePlanning
  ├─ InjectSoftwarePipeline
  ├─ LayoutInference
  ├─ LowerTileOp
  ├─ LowerL2Persistent / DecoupleTypeCast / LegalizeSafeMemoryAccess
  └─ ...

CUDAPassPipelineBody
  ├─ LowerSharedTmem / LowerSharedBarrier
  ├─ LowerLDGSTG
  ├─ LowerHopperIntrin
  ├─ SplitHostDevice
  ├─ InjectFenceProxy / ThreadSync / Tcgen05Fence
  └─ LowerDeviceKernelLaunch / PersistThreadblock
```

`src/cuda/transform/ptx_async_copy_injector.cc` 体现了 CUDA async copy 的思路：先分析普通 global->shared copy 的源/目标/宽度，再注入 `tl.ptx_cp_async`、`ptx_commit_group`、`ptx_wait_group`，必要时保持同步语义。

Hexagon 对应物不是 `cp.async`，而是：

- `Q6_dcfetch_A(src + distance)` prefetch hint（实测 8KB 距离平顶）。
- worker pool pooled copy（4 线程常见最佳，最多 6）。
- 双/四 buffer staging（例如 packed panel PA/PB ↔ PC/PD）。
- HMX issue 与 worker pool copy/unpack overlap。

因此 v2 需要 `LowerHexagonAsyncCopy`/`HexagonPipelineMaterialize`：把 copy tile op + pipeline annotation lowered 成 `pool_start(stage_next)`、HMX deep-chain compute、`pool_join(stage_current)` 的显式 IR，而不是 emitter 看 SeqStmt 顺序临时拼接。

### 1.6 PTX inline asm 如何混进 codegen

CUDA 采用两级封装：

1. Python macro emitter 产生 `T.ptx_ldmatrix`、`T.ptx_mma`、`T.ptx_wgmma_*` 等 TIR intrinsic。
2. C++ `CodeGenTileLangCUDA::VisitExpr_(CallNode*)` 识别这些 intrinsic，打印 `tl::ptx_ldmatrix_x*`、`tl::ptx_mma(...)` 等 C++ helper；部分路径在 `codegen_cuda.cc` 直接生成 `asm volatile`。

Hexagon-v2 应仿照这个模型：

- Python `HMXIntrinEmitter` 只产生 `T.hexagon_hmx_mma_deep(...)`、`T.hexagon_ah_load(...)`、`T.hexagon_acc_read(...)` 等 canonical intrinsic。
- C++ `CodeGenTileLangHexagon` 统一打印 `hexkl_micro_hmx_mm_f16_deep*` inline asm 或 `static inline` wrapper。
- 不允许 Python emitter 直接拼内联汇编字符串。

## 2. CUDA -> Hexagon 概念映射表

| CUDA 机制 | CUDA 语义/位置 | Hexagon 对应物 | 迁移方式 | 特化点/风险 |
|---|---|---|---|---|
| `mma fragment` | warp-private A/B/C register fragment；`alloc_local`/`fragment` scope | HMX accumulator / AH tile / WH tile / optional vreg staging | 直译为 `hmx.acc` + `vtcm.ah/wh` layout + C accum materialization | HMX acc 不是普通寄存器文件；acc read/cvt 写回有固定 AH/RM 约束 |
| `ldmatrix` | shared swizzle -> fragment load | RM->AH、RM->WH、WH/AH VTCM load | 特化成 `hexagon.copy_rm_ah`、`copy_rm_wh` 或 vectorized layout op | VTCM scalar access 禁止；AH/WH 32x32 字节级 layout 已验证但多维 view 要小心 |
| `mma.sync` | single warp MMA atom | HMX `32x32x32 fp16` atom | 直译为 `hexagon.hmx_mma` intrinsic | 发射线程必须是 HMX lock 所属 executor；不能在 worker pool 内调用 |
| `wgmma/tcgen05` async MMA | warpgroup/TMEM/async barriers | HMX deep-chain（一次最多 32 K tiles） | 特化为 `hexagon.hmx_mma_deep(chain_len<=32)` | chain 长度、acc clear/read、K tail 必须由 schedule 保证 |
| shared memory | SM shared/L1 可同步 | VTCM 8MB scratch | 概念直译：`shared` -> `vtcm` | VTCM 标量 load/store 是红线；预算静态规划；128B 对齐 |
| shared swizzle | bank conflict avoidance / tensorcore layout | AH/WH permute | 特化 layout maps | AH==WH for 32x32 rm vshuff；但转置/packed/MXFP4 要独立 layout |
| `cp.async` | global->shared async copy + commit/wait | worker pool copy + `dcfetch` + async pool start/join | 不直译，设计 `hexagon.async_copy` dialect | `l2fetch` 禁用；`dcfetch` only hint；copy 和 HMX overlap 需要 runtime barrier |
| TMA | tensor-map bulk async copy | 无直接硬件等价 | 暂不支持；可把大 tile copy 降成 pooled copy | FastRPC slab/rpcmem 访问与 VTCM rolling buffer 需显式管理 |
| warp tile | warp 负责 MxN tile | HMX 32x32 output tile / M,N panel | 直译为 `hmx tile` schedule 轴 | HMX issue 单流；并行性来自 copy workers + host-level task，不是多个 warps 同时 mma |
| block tile | CTA tile with multiple warps | FastRPC kernel invocation / grid block | 部分直译 | blockIdx 可能冻结 ABI；v1 已有 `_has_blockidx_launch` 规避 |
| pipeline stages | software pipeline over K tiles | VTCM stage buffers over K/N panels | 概念直译，实现特化 | VTCM 8MB 限制比 CUDA shared 更硬；stage 数要参与 StoragePlan |
| `__syncthreads` / mbarrier | CTA/warpgroup sync | worker pool join / HMX owner sequencing | 特化 | HMX 不能跨线程；worker 内不能触发 HMX |
| L2 persistent / prefetch | L2 residency hints | `Q6_dcfetch_A` | 特化 | 距离 8192B 平顶；`l2fetch` 会崩 PD，禁止 |
| inline PTX | C++ codegen intrinsic | inline Hexagon/HMX asm / hexkl micro wrapper | 直译 codegen 模式 | 需集中 ABI/register constraint，不能散落 recipe |
| profiling | CUDA event/ptxas stats | qtimer/prof slots + host wall | 保留 runtime support | qtimer 在部分 kernel 不可信；仍需 host wall/ABL 消融 |
| launch bounds | `__launch_bounds__` | worker count / VTCM plan / executor ABI | 特化为 func attrs | worker ≤6；HMX owner 线程固定 |

结论：**layout/intrinsic/codegen 的架构可以照 CUDA；执行并行模型、async copy、barrier、资源预算必须 Hexagon 特化。**

## 3. OpenCL 后端与 CUDA 后端的差距

`tilelang/opencl/codegen.py` 是我们为了 Adreno 走通 TileLang OpenCL 的实践版本。它依赖：

- TVM/OpenCL 原生 build (`target.build.opencl`)。
- Python source wrapper (`_SourceModule`) 让 caller 可以 inspect patched source。
- 一组 source-level peephole：private accumulator SROA、half staging promotion、typed shared pointer patch、GDN/FA/GEMM 特定结构匹配等。

OpenCL 路线的优点：

- 快速验证，能在通用 codegen 后做设备编译器友好的文本修补。
- 对 Adreno 编译器病理（private array spill、generic shared pointer、vload8 形式）反应快。
- 已经建立 tuner/cache/反作弊全量校验工作流。

与 CUDA 后端的差距：

| 维度 | CUDA | OpenCL 当前 | 对 Hexagon-v2 的启示 |
|---|---|---|---|
| tile op lowering | target op -> canonical intrinsic | 多数靠 generic TIR + peephole | Hexagon 必须向 CUDA 看齐，不能靠 source regex 维持 HMX |
| layout inference | first-class Layout + LowerTileOp remap | 部分 layout 仍靠 codegen patch/结构匹配 | AH/WH 必须进入 LayoutInference，而不是 scope/string 后处理 |
| async/pipeline | pass 级 PipelinePlanning/InjectSoftwarePipeline + target lower | 主要靠 kernel 手写结构 | Hexagon 需要 pass 级 pooled copy/deep-chain overlap |
| intrinsic emission | C++ codegen 集中处理 | OpenCL C 文本 patch | HMX inline asm 必须集中在 C++ codegen，避免不可审计 |
| correctness gate | upstream tests + ptx intrinsic contract | 逐字节复现 + 真机全量校验 | Hexagon 继续采用逐字节复现门禁，但 IR 层也要有 verifier |

Hexagon-v2 应以 CUDA 架构为主，吸收 OpenCL 的两个经验：

1. 保留 **peephole 作为最后保险**，但只允许结构化、可证明、逐字节可复现的 patch；主路径必须由 IR/intrinsic 表达。
2. 保留现有 **tuner + 真机批量测量 + champion full-check** 工作流，用于搜索 HMX tile/panel/stage 参数。

## 4. Hexagon-v2 总体架构

### 4.1 目标架构图

```text
TileLang user program
  │
  ▼
BackendContext(target="hexagon")
  │
  ▼
HexagonV2PassPipeline
  ├─ Prologue: BindTarget / MaterializeKernelLaunch / Simplify / VerifyBufferInit
  ├─ HexagonNormalizeScopes
  │    └─ scope=vtcm/global/wscratch/hmx.acc; layout metadata moved to LayoutMap
  ├─ PipelinePlanning + InjectSoftwarePipeline
  ├─ HexagonLayoutInference
  │    ├─ AH/WH layout inference for T.copy/T.gemm operands
  │    └─ hmx.acc store/load layout inference
  ├─ HexagonLowerTileOp
  │    ├─ T.copy -> hexagon.copy / hexagon.async_copy
  │    └─ T.gemm -> hmx intrinsic sequence
  ├─ HexagonPipelineMaterialize
  │    └─ async pooled copy + join + HMX compute ordering
  ├─ HexagonStoragePlan
  │    └─ VTCM offset/stage buffer budget, acc materialization scratch
  ├─ HexagonWorkerPlan
  │    └─ worker pool regions, <=6, no HMX in workers
  ├─ HexagonABIPlan
  │    └─ slab/wscratch/prof slots/write_set attrs
  ├─ HexagonVerify
  └─ SplitHostDevice / MakePackedAPI-compatible lowering
  │
  ▼
CodeGenTileLangHexagon (C++)
  ├─ function shell / ABI stubs
  ├─ VTCM pointer materialization from attrs
  ├─ vector/HVX intrinsic printing
  ├─ HMX deep-chain inline asm/helper printing
  └─ calls to hexagon_rt.h support layer
  │
  ▼
DSP-side C + generated header fragments
  │
  ▼
External AOT build into FastRPC skel
```

### 4.2 代码层划分

建议目录结构：

```text
tilelang/hexagon/
  backend.py                 # manifest，不变但切换 v2 pipeline/codegen
  pipeline.py                # HexagonV2PassPipelineBody
  target.py                  # vtcm-capacity / hmx-chain-max / worker-max 等 target attrs
  transform/__init__.py      # Python FFI wrappers
  op/gemm/gemm_hmx_v2.py     # GemmHMXv2: infer_layout + lower to HMX intrinsics
  intrinsics/layout/hmx_layout.py
  intrinsics/macro/hmx_macro_generator.py
  language/intrinsics.py     # public T.hexagon.* constructors
src/hexagon/
  codegen/codegen_hexagon.{h,cc}
  codegen/rt_mod_hexagon.cc
  codegen/intrin_rule_hexagon.cc
  transform/lower_hmx_intrin.cc
  transform/lower_hexagon_async_copy.cc
  transform/hexagon_verify.cc
src/tl_templates/hexagon/
  hexagon_hmx.h              # small inline wrappers, register constraints
  hexagon_hvx.h              # vector helper wrappers if needed
runtime/hexagon_rt.h          # existing support layer, reduced role
```

原则：

- Python op emitter 只能构造 TIR intrinsic，不拼 C。
- C++ codegen 是唯一把 Hexagon intrinsic/asm 打印成 C 的地方。
- `hexagon_rt.h` 只保存运行时支持、不可由编译器表达的 ABI/helper，不保存整算子 recipe。

## 5. Pass 流水线设计

### 5.1 Pipeline 草案

```python
def HexagonV2PassPipelineBody(mod, target):
    mod = BindTarget(target)(mod)
    mod = MaterializeKernelLaunch()(mod)
    mod = LegalizeNegativeIndex()(mod)
    mod = InjectAssumes()(mod)
    mod = Simplify()(mod)
    mod = IfStmtBinding()(mod)

    # Canonical resource model
    mod = HexagonNormalizeScopes()(mod)
    mod = HexagonEarlyVerify()(mod)          # forbid scalar VTCM early

    # User schedule -> final pipelined tile-op structure
    mod = PipelinePlanning()(mod)
    mod = InjectSoftwarePipeline()(mod)
    mod = Simplify()(mod)

    # Layout before tile op lowering, CUDA 同构
    with target:
        mod = HexagonLayoutInference()(mod)  # or generic LayoutInference + target infer_layout
        mod = LowerTileOp()(mod)             # T.gemm/T.copy lower to hexagon intrinsics

    # Target-specific canonicalization
    mod = HexagonLowerAsyncCopy()(mod)
    mod = HexagonLowerHMXIntrin()(mod)       # optional normalize chain/deep forms
    mod = HexagonWorkerPlan()(mod)
    mod = HexagonStoragePlanV2()(mod)
    mod = HexagonABIPlan()(mod)
    mod = HexagonProfileConfig(...)(mod)
    mod = HexagonVerifyV2()(mod)

    # Standard finalization as much as possible
    mod = PlanAndUpdateBufferAllocationLocation()(mod)
    mod = FlattenBuffer()(mod)
    mod = ConfigIndexBitwidth()(mod)
    mod = Simplify()(mod)
    mod = SplitHostDevice()(mod) or mark_device_kernel_compat(mod)
    mod = MakePackedAPI()(mod)
    mod = LowerDeviceKernelLaunch()(mod)
    return mod
```

### 5.2 `T.gemm` lowering 设计

`GemmHMXv2` 对齐 CUDA `GemmMMA`：

```text
class GemmHMXv2(GemmBase):
  infer_layout:
    A in VTCM/shared -> AH layout
    B in VTCM/shared/global.wh -> WH layout
    C fragment -> hmx_acc_store layout

  lower:
    emitter = HMXIntrinEmitter(
      tile_m=32, tile_n=32, tile_k=32,
      chain_max=32,
      thread_var/worker_var/schedule axes,
      trans_A/trans_B, accum_dtype
    )
    if clear_accum: T.hexagon.hmx_acc_clear(C)
    for kc in serial(0, K_tiles, step=chain_len):
       emitter.load_or_bind_AH(A, kc)
       emitter.load_or_bind_WH(B, kc)
       emitter.mma_deep(AH, WH, C_acc, kc, chain_len)
    # acc read/store由上层 T.copy(C_fragment, C_global) 或 epilogue 触发
```

需要显式区分三类 buffer：

- `vtcm.ah` / `vtcm.wh`：HMX 可读 operand tile；由 layout inference 决定物理地址。
- `hmx.acc`：逻辑 accumulator fragment；不占 VTCM，但 acc read materialization 需要 VTCM scratch，纳入 StoragePlan。
- `global.wh`：权重已经 host-side 预转 WH 或 runtime WCache 转 WH；codegen 可直接 staged copy WH，不再 RM->WH。

### 5.3 Layout inference 设计

Hexagon layout 对象最少要支持：

| Layout | 逻辑 shape | 物理含义 | 用途 |
|---|---|---|---|
| `rm` | row-major | DDR/global 或 scalar-safe memory | 普通输入输出 |
| `ah32` | `[*, M, K]` | HMX activation tile，32x32 row-pair shuffled | A operand |
| `wh32` | `[*, N, K]` 或 `[K,N]` panel | HMX weight tile，32x32 shuffled | B operand |
| `acc32` | `[M,N]` tiles | HMX acc logical layout | C fragment |
| `packed_mxfp4_wh` | packed nibble + scale | MXFP4 WH panel | MoE/GEMM future |

Layout pass 必须替代 v1 的 scope suffix 依赖：

- `T.alloc_shared(..., layout="ah")` 暂兼容，但内部转换成 `buffer -> Layout(ah32)`。
- `T.copy(src, dst, layout=("rm","ah"))` 改为 copy op annotation，供 `HexagonLowerCopy` 选择转换 intrinsic。
- 多维 view 的 base folding（v1 `hexagon_fold_view_base_rc`）改为 layout address lowering 的统一规则。

### 5.4 Worker/pipeline pass

v1 已有 `HexagonCopyPartition`：大 copy 自动包成 `hexagon.pool` parallel loop，worker ≤6。v2 应把它从 recipe-friendly pass 升级成 scheduler primitive：

```text
hexagon.async_copy(dst_stage[s], src_panel[p], bytes, layout_convert, workers=4, prefetch=8192)
hexagon.pool_start(token, async_copy)
... hmx_mma_deep(stage[prev]) ...
hexagon.pool_join(token)
```

约束：

- HMX intrinsic 禁止出现在 worker pool 内。
- worker pool 默认最多 6，copy 带宽经验上 4 线程常为甜点，但 pass 应从 target attr/annotation 读取。
- `dcfetch` 只用于 DDR/rpcmem source；不要生成 `l2fetch`。
- stage buffer 数量参与 VTCM budget。

## 6. Codegen 层设计

### 6.1 `CodeGenTileLangHexagon`

职责类似 `CodeGenTileLangCUDA`：

- 打印 DSP-side C function shell，包含 `extern "C"`/skel 所需签名或内部 kernel body。
- 根据 PrimFunc attrs 打印 VTCM pointer aliases：`uint8_t *vtcm = ...; A_ah = vtcm + offset`。
- 识别并打印：
  - `hexagon.copy_*`：HVX vector copy/permute helper call 或 inline vector loop。
  - `hexagon.hmx_acc_clear/read/cvt`。
  - `hexagon.hmx_mma_deep`：deep-chain inline asm/helper。
  - `hexagon.pool_start/join`：调用 runtime worker pool support。
  - `hexagon.dcfetch_hint`：`Q6_dcfetch_A`。
  - scalar math / HVX vector math intrinsic。
- 统一管理 includes：`hexagon_types.h`、`hexagon_protos.h`、`hexkl` headers、generated `hexagon_hmx.h`。

### 6.2 HMX intrinsic surface

建议 IR intrinsic：

```text
hexagon.hmx_lock_required()       # verifier-only marker, not printed in inner loop
hexagon.hmx_acc_clear(acc, tile_m, tile_n)
hexagon.hmx_mma(acc, ah_ptr, wh_ptr, k_slot)
hexagon.hmx_mma_deep(acc, ah_base, wh_base, chain_len, ah_stride, wh_stride)
hexagon.hmx_acc_read(acc, dst_ah)
hexagon.hmx_cvt_store(acc/dst_ah, dst_rm, scale/bias/shape flags)
```

其中 `hmx_mma_deep` 是 v2 的核心。codegen 需要保证：

- `chain_len <= 32`。
- AH/WH base 128B 对齐，tile stride 静态可证明。
- acc clear 与 read/write 的生命周期可由 verifier 检查。
- 生成物在同一 HMX owner thread 上运行；如果 kernel 被 worker pool 包围，直接报错。

### 6.3 `hexagon_rt.h` 的角色演变

保留在 runtime support：

- FastRPC/skel ABI：slab base/offset、rpcmem/static map、shape 参数、入口函数。
- HMX session/open/lock discipline：`attnops_open` 所在线程 + executor call 约束。
- worker pool implementation：start/join、最多 6 worker、per-worker context。
- profiling slot ABI：prof array layout、host 读取、fsync/日志策略。
- 设备不可见或跨 kernel 资源：WCache/WH conversion host 侧支持。
- 少量 leaf helper：复杂 HVX exp/silu/reduce 等可先保留，但应逐步变成 codegen intrinsic。

迁出到 v2 codegen/pass：

- GEMM_NT main loop、panel loop、K chain、acc read/write。
- AH/WH copy layout selection。
- VTCM static offsets/stage allocation。
- async pool copy 与 HMX overlap schedule。
- deep-chain HMX instruction emission。

## 7. 现有设备知识归属清单

| 设备知识 | 当前位置/形态 | v2 归属 | 说明 |
|---|---|---|---|
| HMX open/invoke 必须同线程 | runtime/runbook 硬规则 | runtime executor + verifier attr | 编译器标记 kernel uses_hmx；host 只能投递给 NPU executor |
| worker pool ≤6 | `HexagonVerify` / runtime | pass + runtime | `workerIdx` extent 静态检查；runtime enforce |
| VTCM 8MB 预算 | `HexagonStoragePlan` | pass | stage buffer、acc materialization、profiling tail padding 全计入 |
| 禁止 VTCM scalar access | emitter verifier | early + final verifier | 必须在 Layout/Lower 后再查一次 |
| AH/WH 32x32 permute | `layout.py` / `hexagon_rt.h` recipe | layout map + copy lowering | 作为 first-class layout，不靠 scope suffix |
| non-aligned HVX load 向下对齐 | 手册 | verifier/codegen | 生成 128B aligned load；否则报错或 peel |
| dcfetch 8192B sweet spot | emitter 常数 | target attr + async copy pass | 默认 prefetch distance，可调 |
| `l2fetch` 会崩 PD | 手册 | verifier 禁止 intrinsic | 不提供 public intrinsic |
| deep-chain max 32 | 新能力 | HMXIntrinEmitter + verifier | schedule 切分 K tiles，每 chain ≤32 |
| slab/FastRPC ABI | emitter/runtime | runtime support | codegen 只消费 ABI plan attrs |
| wscratch per-worker DDR | `WScratchPlan` | ABIPlan + runtime | 保留，逐步泛化，不绑定 GDN 固定表 |
| profiling slots | emitter `_prof_next_slot` | ABIPlan + codegen hooks | slot 分配 pass 化，codegen 只打印 probes |
| rms-scaled tolerance | tests/runbook | verification strategy | 真机对拍继续沿用 |
| qtimer 部分不可信 | 手册 | profiling doc/test policy | 性能判定用 host wall + ABL 消融交叉验证 |

## 8. 分阶段实施路线

### Phase 0：只重构边界，不改生成语义

- 新建 `src/hexagon/codegen` skeleton，注册 `target.build.tilelang_hexagon_v2_without_compile`。
- C++ codegen 先打印与 Python emitter 等价的 C skeleton，或桥接调用现有 `emit_hexagon_c` 的结果，建立 build/test 通路。
- 把 v1 `HexagonVerify/StoragePlan/ProfileConfig` 的 attrs 规范化，形成 v2 ABI attrs 文档。
- 门禁：v1/v2 对同一 lowered IR 生成 C 逐字节或语义等价（允许 header 顺序差异）。

### Phase 1：GEMM 垂直切片（优先）

范围只做 `C[M,N]=A[M,K] @ B[N,K]^T` fp16/fp32-acc/fp16，`M,N,K` 静态且 32 对齐。

1. 新增 `HMXIntrinEmitter`，支持：acc clear、deep-chain MMA、acc read/store。
2. `GemmHMXv2.lower()` 不再发 `hexagon.gemm_hmx`，而是显式 HMX intrinsic sequence。
3. `CodeGenTileLangHexagon` 打印 deep-chain helper/inline asm。
4. 保留 v1 `hexagon_rt.h` copy/ABI helper，先不做 async overlap。
5. 对比现有 TL NPU GEMM：逐字节 WH conversion 一致、输出 max_rel 通过、性能至少不低于 v1 recipe；deep-chain 目标达到真机 2.1-2.35x per-tile 调用收益。

验收形状：`M=1024,N=12288,K=2560`、`M=1024,N=2560,K=4096`、`K=9216 fallback`。

### Phase 2：Layout first-class 化

- 把 `vtcm.ah/wh` scope suffix 从主路径移出，改成 `LayoutMap`。
- `LowerTileOp` 支持 Hexagon layout address remap，包含 multi-dim view base folding。
- `T.copy(layout=(rm,ah/wh))` lowering 成 vectorized copy intrinsic。
- 门禁：GEMM/GDN/FA 生成物与旧版 skel 对 AH/WH tile 逐字节一致。

### Phase 3：Pooled copy + software pipeline

- 设计 `hexagon.async_copy`、`pool_start`、`pool_join` intrinsic。
- `InjectSoftwarePipeline` 后新增 Hexagon materialization，支持 double/quad buffer。
- 默认 dcfetch distance 8192B，worker count 可 annotation/tuner 控制。
- GEMM panel stage 与 HMX deep-chain overlap；用 ABL 消融确认 copy 残量下降。

### Phase 4：推广到 FA/GDN/FFN/MoE

- FA/GDN：先把现有 TL standard NPU kernel 的 generated C 纳入 v2 codegen discipline（逐字节复现），再逐步替换 leaf helper。
- FFN：保持 fused NPU kernel，不拆成 GEMM；但内部 gate/up/down HMX loops 可复用 v2 HMX emitter。
- MoE/MXFP4：把 packed WH/MXFP4 unpack layout 作为 layout/intrinsic，而不是 recipe 私有逻辑。

### Phase 5：删除 statement emitter 主路径

- `tilelang/hexagon/emitter.py` 降级为 legacy/debug backend。
- 新 backend 默认走 v2；保留环境变量 `TILELANG_HEXAGON_LEGACY_EMITTER=1` 回退一段时间。

## 9. 验证策略

### 9.1 编译器级门禁

- **IR golden**：`T.gemm` lowering 后的 HMX intrinsic sequence golden，检查 chain_len、tile offsets、acc lifecycle。
- **layout golden**：AH/WH layout index map 对 32x32 tile 逐元素比对；multi-dim view base folding 单测。
- **verifier negative tests**：
  - VTCM scalar load/store 必须报错。
  - worker pool 内 HMX 必须报错。
  - chain_len>32 必须报错或自动 split。
  - VTCM budget>8MB 必须报错。
  - unaligned HVX load/store 必须报错或生成安全路径。

### 9.2 生成物复现门禁

沿用 TileLang 近期“零容忍审计”原则：

- 所有 generated skel source 由 generator 逐字节复现；禁止手改生成物。
- C++ codegen 输出加入 stable formatting；避免 Python dict/random order。
- 对 legacy recipe 迁移的第一阶段，允许“结构不同但 leaf helper 调用等价”，但必须提供 AST/regex 级结构门禁。

### 9.3 真机 correctness 门禁

- fp64 reference + rms-scaled tolerance：`|got-ref| / (|ref| + 0.02*rms)`。
- GEMM/GDN/FA/FFN 保持现有 sample/full-check 两级：tuner 快检只用于筛选，冠军必须 full-check。
- 对 HMX deep-chain：要求与逐 tile HMX 调用 bit-identical（新能力已验证，应固化为单测）。
- NaN/denorm/对抗数据沿用 OpenCL FA 反作弊策略：不能出现快速路径自证。

### 9.4 性能门禁

- 只用 host wall 作为最终性能口径；qtimer/prof slot 仅做分相参考。
- GEMM Phase 1 目标：单 NPU 不低于当前 TL NPU GEMM；deep-chain microbench 达到已验证的 2.1-2.35x per-tile 优势。
- Phase 3 目标：ABL 跳 copy/跳 compute 消融能解释 wall；pooled copy overlap 后 copy 残量明显下降。
- 每次改 pipeline/layout 后重跑：`test_compiler.py` / `test_driver.py` / 对应真机 plan 驱动测试。

## 10. 关键风险与决策点

1. **HMX 单流 vs CUDA 多 warp 并行模型不同**：不能机械照搬 warp tile；Hexagon 的收益来自 deep-chain + copy overlap + host-level hetero task。
2. **VTCM budget 是编译期硬约束**：pipeline stage 数、acc scratch、profiling tail padding 都必须进入 StoragePlan。
3. **HMX lock 纪律不能交给用户**：v2 codegen 只生成 DSP kernel；host runtime 必须保证所有 HMX invoke 从 executor/open 线程发起。
4. **不要恢复 helper/chunk 级包装路径**：GDN helper 版已被零容忍整改删除；v2 必须走标准 codegen/intrinsic 路线。
5. **OpenCL peephole 经验只作为保险**：Hexagon 不应构建在 source regex 上；regex patch 只允许用于临时兼容，且必须逐字节可复现。

## 11. 与 CUDA 路径的一一对应关系

| CUDA 文件/机制 | Hexagon-v2 对应 |
|---|---|
| `tilelang/cuda/backend.py` BackendModule | `tilelang/hexagon/backend.py` 注册 v2 pipeline/codegen |
| `tilelang/cuda/pipeline.py` CUDAPassPipelineBody | `HexagonV2PassPipelineBody` |
| `cuda/op/gemm/gemm_mma.py` | `hexagon/op/gemm/gemm_hmx_v2.py` |
| `TensorCoreIntrinEmitter` | `HMXIntrinEmitter` |
| `cuda/intrinsics/layout/mma_layout.py` | `hexagon/intrinsics/layout/hmx_layout.py` |
| `T.ptx_ldmatrix` | `T.hexagon.copy_rm_ah/wh` or `T.hexagon.hmx_load_*` |
| `T.ptx_mma` / `T.ptx_wgmma` | `T.hexagon.hmx_mma_deep` |
| `ptx_async_copy_injector` | `lower_hexagon_async_copy` + pooled copy materializer |
| `LowerLDGSTG` | HVX vector load/store lowering（可选） |
| `LowerHopperIntrin` | `LowerHMXIntrin` |
| `CodeGenTileLangCUDA` | `CodeGenTileLangHexagon` |
| `src/tl_templates/cuda/instruction/*.h` | `src/tl_templates/hexagon/hexagon_hmx.h` |
| nvcc compile callback/cache | external skel AOT build callback/cache（后续） |

## 12. 建议的第一批任务清单

1. 写 `HMXIntrinEmitter` 的纯 Python prototype，只生成 TIR intrinsic，不接 codegen。
2. 写 `CodeGenTileLangHexagon` skeleton，能打印空 kernel + scalar/HVX extern call。
3. 给 `hexagon.hmx_mma_deep` 定 IR schema 与 verifier。
4. GEMM v2 lowering golden：同一 `T.gemm` 输出 HMX intrinsic seq。
5. deep-chain C helper 纳入 `tl_templates/hexagon/hexagon_hmx.h`，用已有 microbench 对 bit-identical 与速度。
6. 把 v1 GEMM recipe 与 v2 GEMM 在同一 FastRPC ABI 下 A/B，先不动 GDN/FA/FFN。

---

本设计的核心取舍：**Hexagon-v2 学 CUDA 的 compiler architecture，不学 CUDA 的 execution model。** 也就是：layout inference、tile op lowering、intrinsic IR、C++ target codegen 要对齐 CUDA；worker pool、VTCM、HMX lock/deep-chain、FastRPC/slab/profiling 必须作为 Hexagon target 的一等约束进入 pass 与 runtime support。
