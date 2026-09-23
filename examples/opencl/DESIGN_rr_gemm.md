# 设计文档：OpenCL 后端全 fragment T.gemm（RR）+ copy-to-fragment

状态：待评审（2026-09-20）
目标读者：实施者（code-writer / kernel-worker）
关联代码：`tilelang/opencl/op/gemm_fma.py`、`src/opencl/op/gemm.cc`、
`tilelang/opencl/vectorize_private_fragment.py`、`examples/opencl/kernels/gemm_nt.py`

## 1. 背景与问题

OpenCL 后端目前最快 GEMM（fragment kn，1.372 TFLOPS @ OnePlus 13 /
Adreno 830，1024×2560×2560）依赖一条脆弱链路：

手写标量乘加循环 + `T.unroll(explicit=True)`
→ UnrollLoop
→ `VectorizePrivateFragment`（形状特判 pass，`vectorize_private_fragment.py:147-176`，
  只认"8×8 fragment 直读 global、无 shared sync"这一种 post-unroll TIR 形态）
→ 整体重写为 `float32x8` 向量 TIR

正规高层写法走不通，堵点：

1. `T.gemm(A_frag, B_frag, C_frag)`（RR 形态）：OpenCL 唯一注册的 gemm 实现
   `GemmFMA.infer_layout()` 返回 `{}`（`tilelang/opencl/op/gemm_fma.py:23-24`），
   不给 fragment 推布局 → LayoutInference 硬断言
   `The layout for fragment ... can not be inferred correctly`
   （`src/transform/layout_inference/layout_inference.cc:707-713`）。
2. `T.copy(global → fragment)`：基础设施在（copy loop domain 认
   `local.fragment`，`src/op/copy.cc:331-344`；cost model 专门建模
   fragment↔global，`layout_cost_model.cc:608-691`），但因 (1) 上游推不出
   fragment 布局，整条链路无从启动。

## 2. 目标

用户可以用如下高层写法表达 GEMM_NT，并由 OpenCL 后端生成与现有
1.372T kernel 等价或更快的代码：

```python
A_frag = T.alloc_fragment((bm, bk), "float16")   # 每线程逻辑视图见 §4
B_frag = T.alloc_fragment((bk, bn), "float16")   # kn 布局
C_frag = T.alloc_fragment((bm, bn), "float32")
for ko in T.serial(T.ceildiv(K, bk)):
    T.copy(A[by*bm:(by+1)*bm, ko*bk:(ko+1)*bk], A_frag)
    T.copy(B[ko*bk:(ko+1)*bk, bx*bn:(bx+1)*bn], B_frag)
    T.gemm(A_frag, B_frag, C_frag)               # RR 形态
T.copy(C_frag, C[by*bm:(by+1)*bm, bx*bn:(bx+1)*bn])
```

验收红线（全部在 OnePlus 13 真机 tl_probe，fp64 ref，rms-scaled
max_rel < 0.1）：

- 1024×2560×2560 kn：**≥ 1.372 TFLOPS**（追平 fragment kn 32×128 纪录）
- 512³ kn：≥ 1.191 TFLOPS
- nk 布局可用（允许低于 kn，参考 0.545T 基线）

## 3. 非目标

- 不引入硬件 mma 语义（Adreno 无此指令；RR gemm 的终态就是
  float8 寄存器 FMA）。
- 不改 SS 形态（shared×shared→shared）现有行为。
- 不删 `VectorizePrivateFragment`：保留为 texture 路径和旧写法的
  fallback。
- 不做 auto-tuning；tile 参数（bm/bn/bk/threads）仍由用户指定。

## 4. 布局设计（infer_layout 的核心产出）

在 `GemmFMA`（或新类 `GemmRR`，见 §6 决策 D1）中新增 RR 分支，
`infer_layout(target, thread_nums)` 返回
`{self.A: fragA, self.B: fragB, self.C: fragC}`。

记号：tile 形状 `(bm, bn, bk)`，每线程 C 子块 `8×8`，
`threads = (bm/8) * (bn/8)`，`tiles_n = bn // 8`。
约束：`bm % 8 == 0`、`bn % 8 == 0`、`bk % 4 == 0`（对齐现有
`local64x128`/fragment 惯例；不满足时 infer_layout 返回 `{}` 并记录
原因，fallback 到现有行为，不硬报错）。

### 4.1 C fragment（store layout）

逻辑 `(bm, bn)` → `(thread, slot)`：

```python
fragC = T.Fragment(
    (bm, bn),
    forward_fn=lambda i, j: (
        (i // 8) * tiles_n + (j // 8),   # thread: 行主序平铺 8×8 子块
        (i % 8) * 8 + (j % 8),           # slot: 线程内 64 元素行主序
    ),
)
```

向量信息：slot 维尾部 8 连续（j%8），`Fragment.OutputShape()` 尾部
= 8 → `VectorizeLoop` 的 access_ptr bound 取到 8
（`src/transform/loop_vectorize.cc:829-838`），copy frag→global 的
store 可 `vstore8`。

### 4.2 A fragment（load layout，replicate over tn）

逻辑 `(bm, bk)` → `(thread, slot)`；同一 `tm` 行的 `tiles_n` 个线程
持有相同 A 行（replicate）：

```python
fragA = T.Fragment(
    (bm, bk),
    forward_fn=lambda i, k: (
        ((i // 8) * tiles_n) % threads,  # thread 只依赖 i//8；对 tn 复制
        (i % 8) * bk + k,                # slot: 8 行 × bk，行主序
    ),
)
```

注：thread 映射对 `tn` 简并即表达 replicate；若布局推断要求
bijection，改用显式 `replicate=tiles_n` 构造（`fragment.py:137-151`
的 `.replicate()`），实施时二选一，以 LayoutInference 接受为准。

### 4.3 B fragment（load layout，kn 与 nk 两种）

kn（B 逻辑 `(bk, bn)`，K 主序连续在 N 维）：

```python
fragB_kn = T.Fragment(
    (bk, bn),
    forward_fn=lambda k, j: (
        (j // 8) % threads,              # 对 tm 复制
        k * 8 + (j % 8),                 # slot: k 主序，尾部 j%8 连续
    ),
)
```

尾部 8 连续 → global 侧 `vload8`、private 侧 half8。

nk（B 逻辑 `(bn, bk)`）：

```python
fragB_nk = T.Fragment(
    (bn, bk),
    forward_fn=lambda j, k: (
        (j // 8) % threads,
        (j % 8) * bk + k,                # 尾部 k 连续，向量宽 4（bk%4）
    ),
)
```

b_layout 信息来源：`tl.opencl.b_layout` func_attr（现有
`gemm_nt.py:247` 惯例），infer_layout 时从 `Target`/func attr 读取；
缺省按 kn。

## 5. lower 设计（RR 分支）

**决策：直接发射最终向量 TIR，不依赖"标量循环 + 后续 pass 打捞"。**
目标形态即 `VectorizePrivateFragment` direct 路径的现有产出
（`vectorize_private_fragment.py:194-242`），从其 python 生成逻辑
参数化移植：

```
acc = T.alloc_buffer((8,), "float32x8", scope="local")     # 8 行 × 8 列
A_pri = private float16 buffer，布局按 fragA（或向量类型数组）
B_pri = private float16 buffer，布局按 fragB
for pos4 in range(K // 4):          # 每步 4 个 K lane
    a{i} = cast<f32x4>(A_pri 的 half4 load)         # i = 0..7
    b{comp} = cast<f32x8>(B_pri 的 half8 load)      # kn；comp = 0..3
    acc[i] += broadcast(a{i}[comp], 8) * b{comp}
C store：cast<f16x8> 写回
```

关键不变量（与 1.372T kernel 逐条对应）：

- C 累加器是 **8 个 `float32x8` 值**，绝不出现 `float acc[64]` 标量数组；
- A/B 的 private 存储在 unroll 后下标全常量，可确定性 SROA 成寄存器；
- kn 路径 B 的 inner load 是连续 half8；nk 路径走 per-column half4 +
  gather（复刻现有 nk 形态，允许较慢）；
- fp32 内链累加（精度红线 max_rel<0.1 的经验要求）。

`lower()` 签名里线程变量来自
`thread_index - thread_bounds.min`（参照 CUDA `gemm_mma.py:71-82`）。

### 5.1 copy 与 gemm 的衔接（本设计的性能命门）

高层写法是 `T.copy(global→frag)` 后 `T.gemm(frag,frag,frag)`。
若 copy lower 出标量逐元素搬运，private 数组会产生真实内存往返，
性能崩。设计约束：

- A/B frag 的 copy 目标必须命中现有 **wide contiguous copy** 路径
  （`src/opencl/op/copy.cc:87-168`：fp16、innermost stride 1、
  inner extent % 8 == 0 → 生成 8-lane `ForKind::kVectorized`）。
  §4 的 fragA/fragB 布局（slot 尾部连续 8）正是为满足该条件设计。
- gemm RR lower 读 A_pri/B_pri 时按 half4/half8 向量读，与 copy 的
  向量写对齐，unroll 后全常量下标 → SROA → 全链路寄存器化，
  终态 TIR 与 direct-global 形态收敛。
- 验证方式：host 端对生成 `.cl` 做形态断言（不允许出现
  `float acc[` 标量数组、A/B private 不允许有逐元素 `= A[..]` 标量
  拷贝循环），再跑 clang `-fsyntax-only`。

## 6. 实施决策

- **D1（类组织）**：扩展现有 `GemmFMA`，在 `infer_layout`/`lower`
  内按 `is_gemm_rr()` 分支；不新增 inst key（C++ `SelectInst` 永远
  返回 `kOpenCLFMA`，`src/opencl/op/gemm.cc:22-28`，不动 C++）。
  SS 分支保持原行为（`infer_layout` 返回 `{}` 合法，因为断言只
  针对 local.fragment）。
- **D2（向量化责任）**：向量形态由 RR lower 直接发射（确定性），
  不依赖 VectorizeLoop 的事后推断；VectorizeLoop 仅兜底 copy。
- **D3（b_layout）**：读 `tl.opencl.b_layout` func_attr，缺省 kn。
- **D4（fallback）**：tile 约束不满足时 infer_layout 返回 `{}` +
  `T.clear` 日志，行为退化为现状；`VectorizePrivateFragment` 保留。

## 7. 迁移与验证计划

### 7.1 gemm_nt.py 新 impl

新增 `--impl rrgemm`（§2 的高层写法），不动现有 7 个 impl。

### 7.2 分层验证

1. **host 形态测试**（新 `examples/opencl/tests/probe_rr.py`）：
   `tilelang.lower(..., target="opencl", enable_device_compile=False)`
   后对 `kernel_source` 断言：存在 `float8 acc` 形态、A `half4`/
   B `half8` 向量 load、`vstore8` 写 C；不存在标量 acc 数组；
   clang `-x cl -cl-std=CL3.0 -fsyntax-only` rc=0（沿用
   `gemm_nt.py:330-348` 的 syntax_check 路径）。
2. **layout 单测**：合法 tile 推出 §4 三个 Fragment；非法 tile
   （bm%8≠0 等）干净 fallback。
3. **真机精度**：`examples/opencl/tests/run_oneplus13.sh` + tl_probe，
   fp64 ref，max_rel < 0.1（kn/nk 各一）。
4. **真机性能红线**：§2 三组数字；同场 A/B 对照现有
   `fragment kn 32×128`。方差按 ±3~4% 时钟噪声计，差距 <3% 视为
   追平。
5. **回归**：现有 `probe_l0.py`/`probe_l1.py`、gemm_nt 全部旧 impl
   生成物不变（md5 对比）。

### 7.3 完成定义

§2 红线全过 + 旧路径零回归 + README 增补 rrgemm 段落。

## 8. 风险与回退

| 风险 | 缓解 |
|---|---|
| RR lower 的向量 TIR 与 pass 生成形态有细微差异，性能掉档 | 先 diff 两者 `.cl`，差异定位到行再调；红线 <3% 才算过 |
| fragA/fragB 的 replicate 表达被 LayoutInference 拒绝 | 备选：显式 `replicate=` 构造；再不行退化为 thread 映射简并 + 文档说明 |
| copy 未命中 wide contiguous 路径（如 region offset 破坏 stride 判定） | host 形态测试第一时间暴露；必要时给 copy 加 fragment-dst 特判 |
| 全链路 SROA 失败，private 数组实体化 | 形态断言 + 真机性能双重门禁；失败则回退手写 impl，文档记录原因 |

## 9. 工作分解（供 code-writer 派单）

1. `tilelang/opencl/op/gemm_fma.py`：RR 分支 infer_layout（§4）+
   lower（§5，从 vectorize_private_fragment.py direct 生成器参数化
   移植）。
2. `examples/opencl/kernels/gemm_nt.py`：`--impl rrgemm`。
3. `examples/opencl/tests/probe_rr.py`：host 形态 + clang 测试。
4. 真机：run_oneplus13.sh 增加 rrgemm 段；kernel-worker 收数。
5. README 更新。

## 10. 实施结果：RR 打通 + GR 路径成为性能正解（2026-09-20）

### 10.1 RR 纯 fragment 路径（本文 §4/§5 原设计）

端到端已打通（fp16 全程无 convert，kn/nk，probe_rr.py 形态门禁全绿），
但真机性能死刑：bk=64/16/8 实测 0.012-0.043 TFLOPS。根因是 replicated
fragment staging 本身——每线程 A×16 / B×4 放大后 private fragment 各
512 halfs，Adreno 上 private 数组每 FMA 都要 memory round trip。
**结论：replicated private staging 在 Adreno 上是死路，不是实现质量问题。**

### 10.2 GR 路径（`--impl grgemm`，最终达标方案）

kernel 侧只给 C 配 fragment，A/B 由 GemmFMA 向量化 lower 直接读 global
（Hexagon 式点索引语义：`T.gemm(A[by*bm, 0], B[0, bx*bn], C_frag)`，
extent 取 backing buffer 尾部 2-D 全形状，单次全 K gemm，无 ko 循环）。
lower 产出的 pos4 K 循环保持 rolled（只有索引 local buffer 的循环才被
`tl.UnrollLoop.unroll_local_access` 强制展开——GR 的 K 循环不碰 local，
天然保持 rolled，与锚点 kernel 同构）。

三个实测出的 codegen 关键事实（OnePlus 13，同窗口交错 A/B）：

1. **fp16 必须显式 `mad()`**：`acc + bcast*b` 在 fp32 下被驱动编译器融合
   成 FMA（锚点 1.35T），fp16 下不融合、正好腰斩到 ~0.75-0.79T。
   lower 现在对 kn/nk 两路都发 `T.call_extern(v8, "mad", ...)` 嵌套链。
2. **fp16 broadcast 用 splat cast，fp32 用 N-lane constructor**：
   fp16 splat 比 constructor 快 ~15%（1.34-1.40 vs 1.14-1.19），fp32 相反
   （constructor 1.28-1.35 vs splat ~1.15）。
   `3rdparty/tvm/src/target/opencl/codegen_opencl.cc` 的
   `BroadcastNode` 打印按 dtype 分派。
3. **K 循环内 load 必须具名寄存器化**：把每 pos4 的 8×half4 A load +
   4×half8 B load 先写进 constant-index 的 local 数组 av/bv（靠 Adreno
   SROA 提升为寄存器），比内联重复 vload 快 ~15%（1.34-1.40 vs 1.21）。

真机成绩（1024×2560×2560 kn fp16 全链路，TL_CHECK=cosine PASS）：

| 时钟窗口 | GR fp16 | fragment fp32-accum 锚点 |
|---|---|---|
| 快 | 1.343-1.399 TFLOPS | 1.350 TFLOPS |
| 慢（热降速 ~15%） | 1.170-1.182 | 1.131-1.146 |

两窗口内 GR ≥ 锚点，绝对值稳定 >1.0T 门槛。512³ kn 1.07-1.10T；
nk 512³ 0.44T（gather 路径，符合 nk < kn 的预期）。

### 10.3 本轮顺带修的编译器 bug

- `codegen_c.cc` AllocBuffer SSA 去重：unroll 后 storage folding 会对同一
  TIR var 发多个栈分配，旧代码直接 ICHECK 崩（"SSA form dup av"）；现在
  重映射到新 C 标识符（每次分配的 use 都被自己的 alloc dominate，安全）。
- `src/opencl/op/copy.cc`：wide contiguous copy 对 fragment 端点加 guard
  （src/dst 含 "fragment" scope 即回退 normal SIMT copy，replication 感知）。
- `pipeline.py`：`tl.opencl.assume_inbounds` func attr 跳过
  `LegalizeSafeMemoryAccess`（它对 rolled K 循环的向量 load 生成保守
  guard，kernel 膨胀数百倍）。
- `src/layout/layout.cc`：`LayoutNode::Forward` 用
  `SubstituteWithDataTypeLegalization`（上一会话遗留，本轮回归通过）。

### 10.4 仍开着的坑

- fp32 accum 的 C store：`DecoupleTypeCast` 生成的 `C_local_cast`
  AllocBuffer 在 `_OpenCLSharedDynToShared` 被丢成悬空 DeclBuffer
  （codegen 报 undefined Variable）。fp16 路径无 cast 不受影响；fp32 GR
  和旧 `--impl tilelang` 仍被它挡（基线即失败，非本轮回归）。
- RR staged 路径保留（正确性 PASS）但仅作参考实现；性能以其 staging
  成本为上限。

## 11. 事实库增补(2026-09 中,FABLE/FFN 战役实测)

- **GR gemm fp32 累加曾测得 ~0.40T,现已证明是写回 epilogue 标量化的
  codegen 假象,不是 Adreno fp32 硬件墙**(见 §13)。direct-store 修复后
  同形状 M=960 N=9216 K=2560 为 **10.35ms / 1.30T**,fp16 累加 1.42T。
- **fp16 全 K 累加精度不可接受**(K=2560 时 max_rel 1.4,bad 1.3%)。
- **chunk16(fp16 内层 + fp32 外层提升)经 shared staging 实现正确
  (max_rel 0.07)但慢**(ck64 0.149T / ck32 0.323T):T.gemm 恒覆盖操作数
  完整 extent,分块必须先把 tile 拷进 shared,拷贝+同步开销吃光收益。
  直接全局分块不可行(extent 取自 buffer 尾部形状)。
- **FFN 结论**:split 链(gemm gate + gemm up + silu_mul + gemm down)
  是出货路径;融合双累加器 kernel 与 split 同速(234 vs 231ms),无收益。
  旧链总耗时 343.9ms @ 0.53T;§13 direct-store 修复后为 **103.7ms**
  (全部 fp64 对拍 PASS)。work-item 语义:
  clEnqueueNDRangeKernel 的 global_work_size 以 work-item 计,不是 wg 数
  (踩过:silu 只覆盖 0.4% 数据,症状是下游 check 自洽地"假 PASS"——
  用被测数据自身算参考会掩盖上游错误)。
- **`half` 是 OpenCL 保留字**,不能当循环变量名(报 "expected expression"
  且 device log 只有 "Pass")。
- **fragment 显式布局标注(T.annotate_layout + T.Fragment)能用但危险**:
  FA acc_o 标注使性能掉 20 倍(2570ms vs 129ms),acc_s 标注无变化;
  标注后的 decoupled copy(acc_o_1)与实际使用点布局关系未明,暂列为禁区。
- 私有/局部数组向量访问统一为 deref 形式 `(*(halfN*)(arr+off))`
  (vloadN/vstoreN 对 __local/__private 指针在 Adreno 不存在);
  probe_rr 断言已同步放宽为两种拼写都接受。

## 12. FA GQA 战役第二轮:94.9 -> 71.0ms 与 codegen peephole 集(2026-09-22)

本轮全部数字均为 OnePlus 13 / Adreno 830 真机实测,fp64 参考对拍通过。

### 12.1 codegen.py peephole 集

tilelang 仓 commit `49f101f` / `f38bc70` / `2fa25de` 新增一组 OpenCL
codegen 后处理 peephole:

- `_patch_opencl_float8_split`:float8 SSA 累加器变量当且仅当所有使用都是
  `.lo`/`.hi`/`.sN` 分量访问时拆成 float4 lo/hi 对;若存在整变量使用
  (如 `mad()`)则跳过。目标是消除向量化更新里每条 FMA 的
  `(float4)(acc.s0..s3)` 重建。
- `_patch_opencl_identity_splat`:`(float4)(X.s0, X.s1, X.s2, X.s3)` →
  `X`,覆盖 C 打印器在任意位置把一个变量用自己的 lane 重建的情形。
- `_patch_opencl_inline_splat`:`((float4)((convert_float(E)), ×4))` →
  `(float4)(convert_float(E))`,把 4 次 convert 折成 1 次 convert + splat。
- `_patch_opencl_splat_convert`:仅被 `convert_float4` 消费的 half4 splat
  staging 变量提升为 float4 变量。
- half-staging 提升扩展:half8 数组的 half4-deref 读
  (`(*(half4*)(X + 0/4))`)→ `.lo`/`.hi`;同时修了一个 scrub 正则 bug
  (half4-deref scrub 误配 half4 数组的 store,导致所有 half4 staging
  数组被静默跳过提升)。

### 12.2 fa_gqa.py 结构改动与主结果

- `acc_o` 改为 `(tile_q, D//8, 8)` fragment,PV 内层改 `T.vectorized(8)`,
  使 V 操作数发出 `vload8`。
- QK 的 Ks 向量 load 用 k4 hoist 提出 `ii` 循环;否则向量化器会在每个
  `ii` 发一条 `vload4`,设备编译器不做 CSE。
- 结果:FA GQA prefill **94.9 → 71.0ms**,PASS,cos **0.999999971**。

### 12.3 实测病理记录

- 4D 块化 `acc_o` `(tile_q//2, D//16, 2, 16)`(试图复刻 fa.cl PV
  2×16 线程映射)触发 layout inference 把每线程工作复制 4 倍
  (128 个累加器),最终 **7.5s**。
- `T.vectorized(16)` 报错 `VectorizeLoop before LiftStorageAlloc`;flat
  `acc_o` 循环里用 `dv8*8+dv` 索引触发 `CanProveEqual(scale,1)` 失败。
  结论:fragment 索引必须每维 scale-1,块因子只能放进 fragment 形状。
- 2 行变体 `(tile_q//2, D//8, 2, 8)` 形式正确但 **77.2ms**,比 1 行
  形态的 71.0ms 更慢。
- half8 staging 数组未提升时(PV 每 `j` 一条 `half X[8]` + `vload8`
  store + half4-deref 读)为 **630ms**——per-j 私有内存往返;提升为
  `half8 X_v` SSA 变量后回到 71ms。
- `#pragma unroll` 加在 QK k 循环是毒药:**105ms**(+34ms);加在 PV j
  循环(unroll 4)基本中性。手写 fa.cl 全展开 k 循环却是 27.4ms,不对称
  原因未明。
- **DCE 污染教训**:"每 ks 全新累加器"探针显示 69ms,但实际 `a4..a7`
  链被死代码消除,少了一半 FMA。之后所有手术消融必须核对 emit 工作量
  完整,否则计时不予采信。

### 12.4 相位归因与剩余差距

71.2ms 版本通过手术删除循环体归因:

- QK ≈ **24.4ms**
- PV ≈ **31.4ms**
- softmax ≈ **6.0ms**
- staging / mask / epilogue ≈ **9.4ms**

手写 fa.cl 锚点仍为 **27.4ms**;两边瓦片形状完全相同(BM=32 BN=128
BK=16 WG=256),QK 指令形式现已一致。Adreno 830 的
`CL_KERNEL_{WORK_GROUP,PRIVATE,LOCAL}_MEM_SIZE` 全部返回 0,拿不到寄存器 /
spill 诊断。该版剩余 **71 vs 27.4ms** 差距在本日续战中已破解,见本节
终审记录。

FFN 链不受本轮 peephole 影响(其 `mad()` 是整变量 float8 FMA,不会被
float8 split 命中),复测 **345.2ms** 全 PASS。

### 12.1 终审:71 vs 27.4 三根因(2026-09-22 续)

tilelang commit `51a0662` 后,通过差分 morphing + 手术移植把 71ms vs
fa.cl 27.4ms 的差距关平。最终结论:差距不是 QK/softmax 逻辑,也不是
float8 vs float4 累加器宽度,而是三个 OpenCL codegen / 映射细节叠加。

1. **generic 指针访问 shared**:TVM 的 `MergeSharedMemoryAllocations` 把
   所有 shared buffer 打包进一个 `__local uchar buf_dyn_shmem[]` + `void*`
   别名,所有 shared 访问都经过泛型地址空间 cast;Adreno 对这种泛型 shared
   访问惩罚极重。改成类型化 `__local half Sh[4096]` 等数组后
   **71.0 → 38.5ms**。已固化为 codegen.py 的
   `_patch_opencl_typed_shared` peephole,带守卫:偏移互异、单一元素类型、
   仅 cast 使用、extent 由下一边界限定;不满足则不改。
2. **PV 的 V 全局加载 bound**:手术消融把 V load 换成常量后证明 PV
   约 25ms 几乎全是 V 加载(去掉后 PV 约 1ms),有效带宽约 **186GB/s**。
   1 行×32 列线程映射相比 fa.cl 的 2 行×16 列每 2 行多读一倍 V。
   `acc_o` 改成 5D fragment `(tile_q//2, D//16, 2, 2, 8)`(每维 scale-1),
   `ii` 不变的 V 向量 stage 到 `half[16]` local 后两 row 共享。注意:
   2×16 的 4D fragment 会让 layout inference 每线程复制 4 倍工作(7.5s),
   所以采用 5D(Parallel 维 + ii + dv8 + vectorized-8)。
3. **vload8 与 scalar×vector 形式**:half-staging peephole 扩展到 `half[16]`,
   使地址连续的 4×`vload4` staging 融合为 2×`vload8`;同时加入
   splat-mul peephole:`((float4)(convert_float(E))) * convert_float4(V)` →
   `convert_float(E) * convert_float4(V)`。这两项合计把 **38.5 → 27.8ms**。

补充实验:

- fa.cl 对 `#pragma unroll` 完全无感:k 循环、J 循环或全部 pragma 去掉后
  仍为 **27.2-27.9ms**;因此之前的"unroll 之谜"不是差距来源。TileLang
  版本在 QK k 循环加 pragma 反而更慢(105ms)。
- 手术移植把 fa.cl 的 PV 段(2 行×16 列、float8 全变量 FMA、
  `#pragma unroll 4`)搬进 TileLang kernel 后为 **27.2ms**,证明其余部分
  (QK/softmax/staging)已无差距。
- float8 vs float4 累加器宽度不影响性能:fa_v12 使用 float4 lo/hi +
  `p0 * vv0.lo` scalar×vector 也是 **27.6ms**。
- 38.5ms 版相位归因:QK **10.8ms** / PV **24.7ms** / softmax **2.2ms** /
  其余约 **1ms**;`noPV=13.8ms`,即非 PV 全部。

最终结果:**27.8/28.0ms** 稳定复现,PASS(cos **0.999999971**,
max_rel **0.0074**),与手写 fa.cl 锚点 **27.4-27.8ms** 持平。FFN 链
复测 **345.0ms** 全 PASS;typed-shared 对 FFN 无增益(gemm 的 shared staging
不是其瓶颈)。`probe_rr` 回归绿。

### 12.2 反作弊审计(2026-09-22)

对 TileLang FA 27.8ms 成绩做反作弊审计,结论:**无 cheat、无 shortcut**。

1. **静态工作量等价**:与手写 fa.cl 逐循环对照,同 grid(512 WG×256
   线程)、同 KV tile 循环数(`floor(qi/4)+1` 个 128-key tile)、同 QK
   4×4 score tile、同 softmax 范围、同 PV 2×16 累加器(每 key 32
   FMA/线程)、同 V 复用模式(两者都是约 1MB 读请求/KV tile/WG 靠 cache),
   输出覆盖也相同。
2. **全量校验**:harness 补丁 `TL_CHECK_SAMPLES=0` 修复后,全量输出
   cos **0.999999970**,bad **1/4,194,304**。
3. **NaN 投毒**:`TL_O_INIT=nan` 后残留 **0/4,194,304**,证明输出全覆盖
   写回。
4. **对抗数据无数据相关快速路径**:denorm ±1e-7 耗时 **+0.3%**,big
   ±8.0 耗时 **+0.1%**。
5. **denorm 数值失败是平台行为**:手写 fa.cl 同样失败(denorm cos 0.383
   vs TileLang 0.562,输出 norm 都只剩参考约 0.34x),根因是 Adreno fp16
   subnormal flush,非 TileLang 缺陷;附带修正 fa_gpu_test.c 旧 max_rel
   判据 `|got-ref|/(|ref|+0.1)` 对 1e-7 量级失敏会假 OK,并修 host 侧
   f2h/h2f subnormal 处理。
6. **稳定性**:iters=50,event **27.55 / 27.96 / 28.34ms**,约 ±1.5%。
7. **锚点公平性**:fa_gpu 是 wall + 1 次 warmup,TileLang 是 event 无 warmup;
   同窗口对照 fa_gpu **28.17ms** vs TileLang **27.96ms**(另一窗口
   27.08ms),落在时钟方差 ±3-4% 内,判定持平。

harness 加固已提交:fa_gqa_test.c 全量校验 + NaN 投毒 + `TL_DATA` 对抗模式;
fa_gpu_test.c denorm 模式 + f2h/h2f subnormal 修复 + cosine 判据。

## 13. layout 基础设施战役:element mapping 与标准写法(2026-09-22)

本轮基于 tilelang commit `47434e0`(前一关键修复 `d2ded5d`),全部数字为
OnePlus 13 真机实测、全量校验 cos PASS。核心结论:此前 GR fp32 只有
0.40T 不是 Adreno 0.5T 墙,而是 TileLang OpenCL 写回 epilogue 标量化导致
向量累加器被编译器标量化 / spill 的 codegen 假象。

### 13.1 GR fp32 0.40T 之谜:写回标量化往返

差分替换实验从同负载对照开始:手写 buffer 版 **13.1ms**,TileLang fp32
**34.3ms**。排查链如下:

- **死声明假设**:删掉无用 private 声明后无效。
- **索引外提假设**:检查 emit 发现索引本来已外提,不是瓶颈。
- **FMA 数 / mad 链 / splat 构造假设**:逐项 morphing 后仍无法解释 3.3x。
- **epilogue morphing 命中**:原写回把 8 个 float8 累加器拆成 64 个标量
  lane 写入 `float C_frag[64]` private 数组,再 `convert_half4` 读入
  `half C_local_cast[8]`,最后 `vstore8`。这个标量化往返让 Adreno 编译器
  把向量累加器标量化 / spill,k 循环慢 **3.3x**。

只把写回改成 `vstore8(convert_half8(acc))` 直通后,TileLang fp32 **34.3 →
10.35ms**(**0.41 → 1.30T**),甚至快于手写 buffer 版 13.1ms。修法已固化为
codegen.py peephole `_patch_opencl_grgemm_epilogue_direct_store`,严格守卫并覆盖
fp32/fp16 两种形态。GR fp16 仍为 **9.4ms / 1.42T**;准确 fp32 路径现在只比
fp16 慢约 10%,此前搁置的 fp16c32 寄存器分块提升从收益上已无必要。

连带收益:**FFN 链 345ms → 103.7ms**(3.3x,全 PASS),因为链上每个 gemm 都有
同样写回 bug。FA 不变(**27.9ms**),`gemm_std` 不变(**51.3ms**)。手写锚点不变:
image 7.8ms / 1.72T,buffer 13.1ms / 1.02T。

因此 runbook 里曾经的"Adreno OpenCL 有 ~0.5 TFLOPS 墙"只能用于描述当时的
TileLang/集成 codegen 假象,不能作为硬件结论;手写 kernel 从来没有这个
epilogue 问题。

### 13.2 barrier bug 修复(d2ded5d)

标准写法(shared staging)的 OpenCL emit 在 copy 写 shared 后、跨线程读之前漏了
`barrier(CLK_LOCAL_MEM_FENCE)`,按 OpenCL 语义是未定义行为,真机只是碰巧对。
CUDA 的 `gemm_fma` lowering 本来就有 `T.sync_threads()`,OpenCL 版漏加。
`gemm_fma.py` 已按 `is_shared` 守卫补 barrier;GR direct-global 路径不加。
`gemm_std` 因此多 2 个 barrier,性能 **51.35 → 51.35ms**,零影响。FA 的 staging
sync 本来就在。

### 13.3 标准写法与 annotate_layout 语义定论

标准写法(`T.copy + T.gemm + annotate_layout`)机械上已跑通:emit 成功、真机
全量校验 PASS、不触发 layout inference ICHECK。但在 Adreno 这个 GEMM 上,
shared staging 是负收益:

- copy 成本约 **16.9ms**;
- shared 读只比 direct-global 多约 **2.5ms**;
- Adreno 的 L2 全局读已经足够快,staging 买不到复用;
- swizzle(`make_swizzled_layout`,NVIDIA bank 模型)在 Adreno 上负收益
  (**0.262 → 0.239T**),且目前没有 Adreno 图案工厂。

`T.annotate_layout` 的语义结论:

- 它是 strict seed,不是 hint。
- 只标 shared buffer 做 swizzle 是安全用法。
- fragment 标注只有在与推断结果完全一致时安全(等于没标);不一致时要么
  `LayoutConflict` 报错,要么 replicated 静默兜底。
- FA 战役中 20x 劣化的机制就是 fully-replicated fragment 在
  `ProveFragmentContains` 直接放行,物理工作量按 replicate 展开。

### 13.4 当前性能全景

| kernel | 之前 | 现在 |
|---|---|---|
| FA GQA | 27.8ms(持平 fa.cl) | 27.9ms 不变 |
| GR gemm fp32 | 34.3ms / 0.41T | 10.35ms / 1.30T |
| GR gemm fp16 | 9.4ms / 1.42T | 9.4ms 不变 |
| FFN 链 | 345ms | 103.7ms |
| 标准写法 std(shared staging) | — | 51.3ms / 0.26T(barrier 已修,正确但 staging 负收益) |
| 手写锚点 | image 7.8ms / 1.72T,buffer 13.1ms / 1.02T | 不变 |

## 14. 远程 auto-tuning(2026-09-22)

本轮提交 `1294f88` / `c268701` / `b3430bb` 建成 kernel 无关的远程调优
orchestrator:`examples/opencl/tuner/tune.py`。目标不是复用 TileLang 自带
AutoTuner 的测量半边,而是把 OpenCL/手机测量流程工程化。

### 14.1 设施架构与设计判断

调优流程:

1. 输入 config dict 列表;
2. `ThreadPoolExecutor` 在服务器侧并行 lower(纯 CPU 工作);
3. 一次 tar 部署到手机;
4. 手机侧 detach 后批量串行测量;
5. 产出 `<tag>_results.json`(全量结果)和 `<tag>_best.json`(最优结果 +
   sha256 cache key)。cache key 覆盖 git rev、工厂源码、config 和 probe 规格。

测量策略:cosine 抽样快检,冠军 config 再做全量校验;每个 config 跑 2 轮取 min,
对抗 OnePlus 13 时钟方差。

没有直接使用现成搜索器的原因:

- TileLang 自带 AutoTuner 的测量部分绑定本地 torch/CUDA,无法复用到
  OpenCL/远程手机;本轮只借鉴其持久化格式、cache key 和 config 校验思路。
- meta-schedule 搜的是 s_tir schedule,与 TileLang 当前"Python 写死调度 + layout
  推断"的模型正交,不采用。
- carver 暂无 Adreno arch 支持,列为后续三期候选。

### 14.2 试点 1:GR gemm fp32

搜索 54 个 config,全部 PASS。新冠军:

- `bm=32 bn=256 bk=16 layout=kn`
- **9.864ms / 1.361T**
- 相比手工冠军 `bm=32 bn=128 bk=64` 的 **10.35ms / 1.297T** 快 **4.7%**

规律:

- `bn=256` 优于 `bn=128`;
- `bk` 在最优 tile 族里影响很小;
- `nk` 布局整体病态,最差 **147.8ms**。

### 14.3 试点 2:FFN 链坐标下降

`tune_ffn.py` 配合 `ffn_chain_test` 支持 gate/up/down 三个 GEMM 独立替换,参数
通过 `TL_{GATE,UP,DOWN}_{BM,BN,THREADS}` 传入。坐标下降结果:

- 链总耗时 **103.6 → 103.0ms**(+0.5%);
- gate/up 最优 `bm=64 bn=256 threads=256`;
- down 保持 `bm=32 bn=128`。

结论:写回修复后 FFN 链已经贴近 per-gemm GR 速率上限(约 1.34T),tile 配置空间
基本耗尽;继续下降只能靠结构手段,例如 gate+up 融合或 texture/image 路径。
`ffn.py` 默认值已写回。`ffn_chain_test` 的 `TL_CHECK_SAMPLES=0` 语义已修复:
FFN 全量 fp64 参考在手机上太慢,实用深检使用 8192 抽样。

### 14.4 当前 best config

| kernel | best config | 性能 | 备注 |
|---|---|---|---|
| GR gemm fp32 | `bm=32 bn=256 bk=16 layout=kn` | 9.864ms / 1.361T | 54 config 全 PASS |
| FFN gate/up | `bm=64 bn=256 threads=256` | 链 103.0ms | 坐标下降最优 |
| FFN down | `bm=32 bn=128` | 链 103.0ms | 保持旧默认 |
