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

- **GR gemm fp32 累加只有 ~0.40T,fp16 累加 1.42T**(同形状 M=960 N=9216
  K=2560;512³ fp32 同速 0.37T,与形状无关)。瓶颈不是 mad 嵌套形式
  (改成 `acc+=a*b` 零差别),是 fp32 向量 FMA + convert 的结构性代价。
  手写 gemm.cl 的 1.66T fp32 数字与本路径不同构,未复现。
- **fp16 全 K 累加精度不可接受**(K=2560 时 max_rel 1.4,bad 1.3%)。
- **chunk16(fp16 内层 + fp32 外层提升)经 shared staging 实现正确
  (max_rel 0.07)但慢**(ck64 0.149T / ck32 0.323T):T.gemm 恒覆盖操作数
  完整 extent,分块必须先把 tile 拷进 shared,拷贝+同步开销吃光收益。
  直接全局分块不可行(extent 取自 buffer 尾部形状)。
- **FFN 结论**:split 链(gemm gate + gemm up + silu_mul + gemm down)
  是出货路径;融合双累加器 kernel 与 split 同速(234 vs 231ms),无收益。
  链总耗时 343.9ms @ 0.53T(全部 fp64 对拍 PASS)。work-item 语义:
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
spill 诊断。剩余 **71 vs 27.4ms** 差距在指令形式对齐后仍未解释,需要
SASS 级分析。

FFN 链不受本轮 peephole 影响(其 `mad()` 是整变量 float8 FMA,不会被
float8 split 命中),复测 **345.2ms** 全 PASS。
