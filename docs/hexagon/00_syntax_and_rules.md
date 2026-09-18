# TileLang Hexagon NPU Backend — 语法表与规则表 v0.1

目标生态位:TileLang 的 CUDA 路径生成 CUDA kernel;本 backend 以**相同的前端写法**
生成 **Hexagon HVX/HMX intrinsic C kernel**(AOT,hexagon-clang 编译进 FastRPC skel)。
目标硬件:OnePlus 13 / SM8750,Hexagon v79,VTCM 8MB,HVX 128B 向量,HMX 32×32
fp16 矩阵单元(hexkl micro 接口)。

设计原则:

1. **写法与原生 TileLang 一致**。用户写的是标准 TileLang kernel
   (`T.Kernel` / `T.alloc_shared` / `T.copy` / `T.gemm` / `T.Pipelined`),
   只是 `target="hexagon"`。差异全部收敛在 **scope 体系、layout 注解和
   copy/gemm 的 lowering recipe** 上。
2. **多级存储与 copy 显式暴露**(同 CUDA 路径的 global→shared→fragment)。
   用户控制数据在 DDR / VTCM / 向量寄存器 / HMX 累加器之间的流动;
   编译器负责 tiling、layout 变换、对齐与 dtype 相关细节。
3. **规则即编译器检查**。手写 kernel 踩过的坑(VTCM 标量访问、128B 对齐、
   HMX 线程规则、panel 宽度)全部变成编译期断言或自动满足的 invariant。
4. **逃生舱**:编译器做不出的地方,允许通过 `tilelang.hexagon.language`
   向用户暴露硬件原语手写(§2.2),每个暴露的原语必须在本表有条目、注明
   手写配方出处;暴露即承诺维护。

参照物:手写 kernel 在 `/root/project/backend/npu/attn/skel/src/attnops_*.c`;
硬件事实以 `/root/project/.opencode/skills/oneplus13-npu/SKILL.md` 为准。

---

## 1. 存储层级:scope 映射表

| TileLang scope(CUDA 路径) | Hexagon scope | 物理实体 | 容量 | 访问约束 |
|---|---|---|---|---|
| `global`(`T.Tensor`) | `global`(不变) | DDR rpcmem slab,host 注册 | ~1.6GB static map 预算 | 仅 128B 向量批量读写走 VTCM 中转划算 |
| `shared.dyn`(`T.alloc_shared`) | `vtcm`(保留 `T.alloc_shared` 入口) | VTCM,8MB,`hexkl_micro_hw_init` 取得 | 8MB 编译期静态预算 | **禁止标量访问**(~46ns/次);一切访问 HVX 向量化;地址 128B 对齐 |
| `T.alloc_wscratch` | `wscratch` | per-worker DDR scratch(runtime slot) | 不占 VTCM 预算;随 worker 数复用 | 允许标量递推/随机访问;每个 worker 独享一份;当前 GDN 映射到既有 `hrt_gdn_slot_t` ABI 字段 |
| `local`(`T.alloc_local`) | `vreg`(保留 `T.alloc_local` 入口) | HVX 向量寄存器 v0–v31(128B 宽) | 32×128B | 编译器做寄存器分配;溢出报编译错,不静默 spill 到 VTCM |
| `local.fragment`(`T.alloc_fragment`) | `hmx.acc`(保留 `T.alloc_fragment` 入口) | HMX 累加器,37-bit(fp32 级),双组 | 2×32×32 acc | 只能由 `T.gemm` 写、`T.copy` 读;无随机访问 |
| wmma fragment(CUDA 特有) | `vtcm` + layout `AH` / `WH` | VTCM 内 32×32 fp16 tile(2048B/tile) | 占 VTCM 预算 | 布局见 §3;AH==WH(32×32 时字节级相同,`vshuff` 互换) |

scope 名保持 TileLang 习惯:用户仍写 `T.alloc_shared(...)` / `T.alloc_fragment(...)`,
backend 根据 target 把它们落到 VTCM / HMX acc。需要显式布局时用 TileLang 已有的
layout 机制(`T.alloc_shared(shape, dtype, layout=...)` 或 annotate),新增两个
layout 常量:`"rm"`(row-major,默认)、`"ah"`(HMX 激活 tile 布局)、`"wh"`(HMX 权重 tile 布局)。

## 2. 语法表:原语对照

### 2.1 保持原义的原语(用户写法与 CUDA 路径逐字相同)

| 原语 | Hexagon backend 语义 |
|---|---|
| `T.Kernel(bx, by, ..., threads=N)` | N = QuRT worker pool 线程数(≤6,**6 是甜点**,wave 量化:不足一波按一波算)。block 维映射到任务级并行(head / panel / expert 区间) |
| `T.alloc_shared(shape, dtype)` | VTCM 分配,128B 对齐 bump allocator,编译期静态预算检查(§4 R3) |
| `T.alloc_wscratch(shape, dtype)` | per-worker DDR scratch,不计入 VTCM;当前用于 GDN 显式 scratch,按声明顺序+shape/dtype 映射到既有 runtime slot |
| `T.alloc_fragment(shape, accum_dtype)` | HMX acc 分配;shape 必须 32 的倍数(§4 R5) |
| `T.alloc_local(shape, dtype)` | HVX vreg 数组;每元素一个 128B 向量 |
| `T.copy(src, dst)` | 见 §3 lowering 表;所有 copy 生成 HVX 向量化代码或 pooled copy |
| `T.gemm(A, B, C)` | HMX 链:`acc_clear → K/32 × hexkl_micro_hmx_mm_f16 → (延迟 acc_read)`;A/B 必须是 `ah` layout 的 VTCM tile,C 是 `hmx.acc` |
| `T.Pipelined(n, num_stages=k)` | 与 CUDA `num_stages` 同一作者语义:PipelinePlanning/InjectSoftwarePipeline 先生成 prologue/steady/epilogue 与带 leading version 维的多版本 buffer；Hexagon consume 注入后的纯 stage order,不产生 CUDA async-copy marker。steady-state 循环中第一个 `T.parallel` worker-pool phase 用 `attnops_pool_start_ctx` 异步启动,主线程继续执行后续 HMX/HVX consumer 链；在下一个 pool phase 前(或本轮 loop back-edge 前)自动 `attnops_pool_join`。prologue/epilogue 与其余 pool phase 保持同步 `attnops_pool_run_ctx`。pool 是全局 singleton,所以同一时刻只能有一个 outstanding async phase。 |
| `T.Parallel` / `T.serial` / `T.unroll` / `T.vectorized` | 常规循环构造;`T.serial` 允许动态标量表达式 extent(如三角循环 `T.serial(i + 1)`),由 emitter 用标量表达式打印 bound;`T.vectorized` 仍强制静态 128B 向量化(失败=编译错,见 R2) |
| `T.clear` / `T.fill` | fragment → `acc_clear`;vtcm → HVX 向量填充 |
| `T.reduce_sum(src128, dst_scalar)` | **已支持** 128 维 fp16/fp32 向量→fp32 标量;fp16 先升 fp32,再走双链交错 + `Q6_V_vror_VR` rotate-fold(VLIW in-order 配方)。`dst_scalar` 可是 global/DDR scratch,禁止写 VTCM 标量(R2);src/dst buffer 下标按任意 rank row-major 线性化,不再限 1D/2D |
| dtype: `float16` / `float32` | fp16 存储 + fp32(或 HMX 37-bit)累加是默认;转换由编译器插桩(§4 R7) |
| `T.exp(x)` | 标准写法:fp32 标量走 `expf`,fp32 HVX 向量走 `hrt_exp_fp32_vec` 的 32-lane pair；fp16 HVX 走 `hrt_exp_fp16`。递推链仍应显式保持 fp32 输入类型。 |

### 2.2 Hexagon 扩展原语(对照 CUDA 路径暴露 ldg/sts/wgmma 的方式,
###     以 `tilelang.hexagon.language` 子包提供,不影响通用写法)

| 扩展 | 对应手写配方 | 用途 |
|---|---|---|
| `T.copy(src, dst, layout=("rm","ah"))` | rm→AH zip16:常数掩码 + `vand/vor` + `valign(64)` + `Q6_Vh_vshuff_Vh`(attnops_gemm.c:56-77 同款) | 激活/权重进 HMX 前的布局转换 |
| `T.copy(src, dst, layout=("ah","rm"))` | AH→rm unperm:`Q6_Vh_vdeal_Vh` 奇偶分离 + tile 对处理直写(attnops_gemm.c:82-98) | acc_read 后写回 DDR |
| `T.hexagon.exp_fp16(v)` | 2^frac 三段多项式 fp16 exp(~1e-3 精度) | 非递推链的 exp |
| `T.hexagon.exp_fp32(v)` | qf32 + 8 阶 Taylor(llama hvx-exp.h 移植版) | **递推链上的 exp 强制用这个**(§4 R8) |
| `T.hexagon.silu_fp16(v)` | fp16 exp + rcp16 位魔术 + Newton,末乘 fp32 | FFN 激活 |
| `T.hexagon.h2f(v)` / `T.hexagon.f2h(v)` | `vunpack` 零扩展 + 指数位修正 / `vcvt`+`vdeal` | fp16↔fp32 向量转换 |
| `T.hexagon.dcfetch_hint(addr, dist=8192)` | `Q6_dcfetch_A(src+j+8192)`,每 128B 一次 | 预取;pooled copy 默认带,无需手写 |
| `T.hexagon.load_state128(src,dst,hv)` / `store_state128(src,dst,hv)` | `memcpy` state [128,128] fp32,对应 `attnops_gdn.c:143`/`:256` | 显式 state tile 初末搬运;参数必须是用户代码中的 S0/state/S1 buffer |
| `T.hexagon.load_h2f_rows128(q,k,v,qf,kf,vf,T,hk,hv,t0)` | `gdn_cvt_row`: `Q6_Wuw_vunpack_Vuh` + 指数位修正,对应 `attnops_gdn.c:146-151`/`:77-84` | chunk 内 Q/K/V fp16 行转 fp32 scratch;所有 GDN scratch 由 `T.alloc_wscratch` 显式声明,名称可任意 |
| `T.hexagon.scan_exp32(g,beta_src,eG,eGinv,beta,eGC,T,hv,t0)` | 前缀和 + fp32 `expf`,对应 `attnops_gdn.c:152-162` | **deprecated**:已被用户层 `T.alloc_var` + `T.serial` 标量递推 + `T.hexagon.exp_fp32` 替代;仅保留兼容旧生成物 |
| `T.hexagon.dot128x2_store(kf,qf,eG,eGinv,beta,A,P,i,j)` | 双链交错 128 维 fp32 dot + rotate-fold,对应 `attnops_gdn.c:163-174`/`:95-116` | **deprecated**:已被用户层 `T.vectorized` 乘法 + `T.reduce_sum(src128,dst)` + 标量组合替代;仅保留兼容旧生成物;UT 矩阵不含 decay |
| `T.hexagon.state_x2_matvec128(S,kf,qf,w,o,i)` | 4×HVX fp32 累加器计算 `S^T k` 和 `S^T q`,对应 `attnops_gdn.c:175-198` | 参数化叶片;state/kf/qf/w/o 均为显式 buffer |
| `T.hexagon.affine_rows128(vf,w,beta,eG,i)` | `beta*v - beta*eG*(S^T k)` 的 128-wide fp32 HVX 行变换,对应 `attnops_gdn.c:199-207` | 参数化叶片 |
| `T.hexagon.forward_solve32(A,w,i)` | 32 步下三角前代回消,每步 128-wide fp32 HVX,对应 `attnops_gdn.c:208-217` | 当前暂留参数化复合叶片;回收计划:emitter 支持跨行 fp32 向量数组依赖后拆成 `axpy_row128` |
| `T.hexagon.output_rows128(o_acc,w,P,eG,out,T,hv,t0,i)` | `eG*S0^Tq + tril(P)*w` + fp32→fp16 store,对应 `attnops_gdn.c:218-230`/`:86-93` | 参数化叶片;输出目标 O 显式传入 |
| `T.hexagon.state_decay_rows128(kf,eGC,eGinv)` / `state_update32(S,kf,w,eGC)` | fold `eGC*eGinv` 到 K 行,再 `S=eGC*S+sum k*w^T`,对应 `attnops_gdn.c:231-254` | chunk 末状态更新;decay 只在 v 侧/输出侧/chunk 末三处,UT 不含 decay |

## 3. copy lowering recipe 表(scope 对 → 生成代码)

| src → dst | lowering | 实测带宽/依据 |
|---|---|---|
| global → vtcm | **pooled copy**:4 worker 连续切分 + 128B HVX 拷贝 + 每 128B `dcfetch(+8192)` | ~40GB/s(单线程只有 7–14);dcfetch 距离 2048 起有效,8192 平顶 |
| global → vtcm(小张量/控制块) | 单线程 128B HVX 直拷 `hvx_memcpy` | ~8GB/s,编译器按尺寸阈值选 |
| vtcm → vtcm(纯搬移) | 128B HVX 拷贝循环;不展开不 dcfetch(dense GEMM 消融:两者都是负优化) | 编译器调度最好 |
| vtcm(rm) → vtcm(ah) | zip16(§2.2),tile 对处理 | 手写版 392ms 标量 → 3ms HVX |
| global/slab fp32(rm) → vtcm fp16(AH) | `hexagon.copy_f32_ah`: fp32→fp16 staging + AH zip16,经 `hexagon_rt.h::hrt_tlgdn_stage_f32_to_ah`;`hexagon.copy.trans=1` 表示源按 `[K,M]` row-major 读、逻辑输出为 `[M,K]` | GDN 标准构造中的 fp32 矩阵进 HMX 前 staging |
| vtcm(ah) → vtcm/global(rm) | unperm + `vdeal`,相邻 tile 对拼 128B 整行直写 | 手写版 78ms → ~1ms |
| vtcm → vreg | 128B 对齐 load;**非对齐地址编译错**(硬件静默向下对齐,必须挡住,§4 R1) | — |
| fragment(hmx.acc) → global/vtcm(rm) | `hexagon.gemm_hmx` 必须紧跟 `hexagon.copy_acc_rm`;emitter 延迟 acc_read 后直接 unperm 写回。global 目标走 `hrt_unperm_*_hvx`;VTCM RM fp16 目标走 `hrt_tlgdn_acc_tile_to_vtcm_rm`，供后续 HVX/vectorized 代码继续读取 | 标准 GDN kernel 的 HMX 中间 tile 写回路径 |
| vtcm → global | 128B HVX store;fp32→fp16 转换在此插桩 | — |

未命中特殊 `hexagon.copy_*` recipe 的 `T.copy` 组合会回退到设备无关的
`LowerNormalCopy`，生成普通 loop + `T.vectorized` 的 HVX 向量 load/store；
仍受 R1/R2/R10 等 Hexagon verifier 规则约束，不会放行 VTCM 标量访问。

`T.Pipelined(num_stages≥2)` 时,注入 pass 会给 VTCM buffer 加 leading version 维,
访问下标使用 `floormod(loop_var-min, versions)` 轮转。Hexagon 地址折叠规则把 leading
version 当作完整 tile plane 的批维,因此 version stride = 单版本元素数 × dtype bytes
(AH/WH tile layout 下等价为完整 tile footprint)。StoragePlan 按静态 shape 计入所有
version,所以 VTCM 预算天然反映 2x/多倍 buffer。当前 async 只重叠 steady-state
第一个 `T.parallel` phase 与 caller-thread consumer;如果循环体后面还有 pool phase,
emitter 必须先 join,以维护 pool singleton 不变量。

### 3.0.1 HMX GEMM 发射:legacy GEMM 壳 vs 通用 recipe

Hexagon emitter 目前保留两条 GEMM lowering 路径：

- **legacy GEMM 壳路径**：入口 ABI 为 `slab,w,M,N,K,abl` 的 GEMM_NT kernel 继续使用
  原有 `GM_ACT/GM_WA/GM_OUT/NP/nct/kt` 宏和运行时 panel 选择逻辑。该路径用于
  `examples/hexagon/gemm/gemm_nt.py` / `gemm/gemm_small.py`，要求逐字节回归稳定。
- **通用 recipe**：当 `hexagon.gemm_hmx + hexagon.copy_acc_rm` 出现在非 legacy GEMM
  壳的 kernel 内(例如 GDN/elementwise 壳内有多个异形 `T.gemm`)时，emitter 不再读取
  `GM_*` 运行时量，而是从当前 gemm extern 的静态 `(A_data,B_data,C_data,M,N,K)`
  取每 call 的 M/N/K，并从 `HexagonStoragePlan` 写入的 `hexagon.vtcm_offsets` 查
  A/B/acc/dst 的 VTCM 基址。每个 gemm 对独立生成局部 `mt/nt/kt` 循环：
  `acc_clear → kt × hrt_hmx_mm_f16 → hrt_acc_read_f16 → copy_acc_rm 写回`。

通用 recipe 允许同一 kernel 内顺序出现多个不同形状的 GEMM；所有 tile 数、VTCM
offset、写回目标都绑定在当前 gemm/copy 对上，不使用 emitter 全局单值状态。
HMX 物理 acc 仍是单份资源，顺序段之间通过每段自带的 clear/read 隔离，acc read
scratch 只需一对 32×32 fp16 tile，可在各段复用。

通用 recipe 另有默认关闭的 profiling 模式：设置 pass config
`tl.hexagon_prof=True`（或同名 kernel attr）后，generic mixed pool/HMX shell
会把每个同步 `T.parallel` worker-pool call 包在 `HAP_perf_get_qtimer_count()`
前后，并继续累加主线程 staging / HMX matmul / accumulator readback 聚合槽。
生成文件头会确定性写出 `prof[]` slot 布局；默认关闭时不改变生成文本。

### 3.0.2 通用 mixed pool/HMX 渲染路径

渲染器选择保持向后兼容优先级：先按既有结构事实选择 GDN shell、纯
elementwise/SILU vector shell、legacy/user-tiled GEMM shell；只有这些固定壳都不匹配，
且 kernel 中存在需要交替执行的 `T.parallel`/串行段/`T.gemm` 组合时，才进入通用
`generic mixed pool/HMX shell`。因此 `gemm_nt` / `gemm_small` / `gdn_prefill` /
`silu_mul` 等既有固定例子的生成物必须逐字节不变。

通用路径的 IR 映射约定：

- body 内的 `for x in T.parallel(N)` 被提升为文件级 `static void worker(int job,
  void *opaque)`；`x` 在 worker 开头绑定为 `const int x = job`。入口处用
  `attnops_pool_run_ctx(worker, &ctx, N)` 同步执行并 join；每个 pool 调用就是段间
  barrier，后续串行/HMX 段能看到并行段对 VTCM/global 的写入。
- 普通 `T.serial` 循环、`T.copy` staging、`hexagon.gemm_hmx + hexagon.copy_acc_rm`
  链保持在 FastRPC caller 主线程内联执行。R4 仍禁止 worker 内出现 HMX；通用路径只把
  显式 `T.parallel` 的 body 提升，`T.gemm` 所在串行段不会被提升。
- 若 `T.parallel` 位于外层 `T.serial` 循环内，提升函数的 ctx 结构体会捕获所有 global
  slab 指针、按首次出现顺序推导的 shape 标量、外层循环变量、`abl` 与 `prof` 指针。
  VTCM 指针不放入 ctx，而是在 worker 内重新取 `HRT_VTCM_BASE()`；各 VTCM buffer 地址
  来自 `HexagonStoragePlan` 写入的静态 128B 对齐 offset。
- 通用入口 ABI 为
  `int name(remote_handle64 h, unsigned char *slab, int slabLen, <shape scalars...>, int abl)`。
  `<shape scalars...>` 只来自 global buffer shape 中的符号维度，按 `func.buffer_map`
  中首次出现顺序去重；所有 global tensor 均由 `_emit_slab_layout` 在单个 slab 中按
  128B 对齐顺序切片，尾部预留 int32 profiling 区(`prof`)。

这一路径适合“并行 HVX 段 → 主线程 HMX GEMM 段 → 并行 HVX 段”这类混合 kernel；
如果 kernel 能被上面的固定 renderer 完整识别，应继续走固定 renderer 以保持 ABI 和
生成文本稳定。

### 3.0.3 `tl.hexagon_prof` 可选分阶段计时

通用 mixed pool/HMX shell 支持 PrimFunc attr / pass 配置风格开关
`tl.hexagon_prof`，默认 `False`。关闭时不改变默认生成文本；开启时 emitter 在
generated C 文件头写入 `prof[]` slot 注释，并把 HAP qtimer tick 累加到 slab 尾部
的 int32 `prof[]` 区。slot 0..4 保留既有聚合计数，后续 slot 按源码顺序记录
worker-pool phase；staging、HMX mm、unperm 等主线程子阶段继续使用原有聚合 slot。
该开关只在通用 shell 路径生效，不触碰 legacy leaf renderer。

### 3.1 Hexagon layout 注册框架

Hexagon 专用 layout 不再由 Python 或 lowering 代码散落地判断字符串，而是在
`src/hexagon/layouts.{h,cc}` 中走统一注册表：

- `HexagonLayoutMode`：`NONE/AH/WH/RM/...` 的 C++ mode 枚举，`NONE` 只表示未命中
  注册 layout；普通 row-major/默认路径正式命中 `RM`。
- `HexagonLayoutSpec`：一行注册表项，包含 mode、名字、构造器
  `Layout (*)(Buffer)` 和检测器 `bool (*)(Layout, Buffer)`。
- `MakeHexagonLayout(mode, buffer)`：按 mode 查注册表并调用构造器。
- `DetectHexagonLayoutMode(layout, buffer)`：遍历注册表，用
  `StructuralEqual(layout, MakeXLayout(buffer))` 风格的检测器返回首个命中 mode。

内置 layout 按注册顺序为 AH/WH/RM：

- `AH` / `WH`：二者在 32×32 fp16 HMX tile 上都是同一个 zip16 row-pair
布局：输入行对 `(2*rp, 2*rp+1)` 中第 `col` 列被映射到
`lane = (col % 2) * 32 + (row % 2) + 2 * floor((col % 32) / 2)`，外层输出维是
`row_tile, col_tile, row_pair, lane`。这与 `hexagon_rt.h::hrt_stage_act_hvx` 中
两行拼接后执行 `Q6_Vh_vshuff_Vh` 的 AH 语义一致；WH 当前遵循硬件事实
“32×32 tile 上 WH == AH == vshuff(row-major)”。layout 只描述纯下标置换，
因此只实现 `Forward` 表达式，不实现反向 rewrite 特化。
- `RM`：逐维恒等 layout，`InputShape == OutputShape == buffer.shape`，forward 为
  `InputPlaceholder(0..rank-1)` 原样返回。该模式无 dtype/shape 约束，代表普通
  row-major buffer；`DetectHexagonLayoutMode` 对这类 layout 返回 `"rm"`，而不是
  旧行为的 `"none"`。

### 3.2 Hexagon pipeline 中的 LayoutInference 不变量

Hexagon pipeline 中 **LayoutInference 常驻**：无论 kernel 是否包含 `T.gemm`，都会在
`LowerTileOp` 之前运行。Hexagon 以 pass config
`tl.layout_inference.fill_default_layout=True` 要求 `LowerTileOp` 入口处每个 SBlock 的
`layout_map` 覆盖 block 内所有已知 buffer；没有 AH/WH/用户 strict 标注等特殊布局的
buffer 会补一个逐维恒等 `RM` layout。

同时 Hexagon 设置 `tl.layout_inference.annotate_parallel_loops=False`：`T.Kernel`/
`T.parallel` 在 Hexagon 上是运行时 worker-pool 语义，不是 CUDA 风格 SIMT lane 切分；
因此不把 `parallel_loop_layout` / padding guard 等编译期 SIMT annotations 写到 For 上，
避免触发后续 padding-guard 切分和向量化 planner 降级。

以后添加新 layout 的固定三步：

1. 在 `HexagonLayoutMode` 增加枚举值和字符串转换分支；
2. 编写 `MakeHexagonXLayout(buffer)` 与 `DetectHexagonXLayout(layout, buffer)`；
3. 在 `RegisterBuiltinHexagonLayouts()` 的表中追加一行
   `{HexagonLayoutMode::kX, "x", MakeHexagonXLayout, DetectHexagonXLayout}`。

框架 API、Python `tilelang.hexagon.language.layout.make_layout()` /
`detect_layout_mode()` 和调用方无需再改。

### 3.3 `alloc_shared(layout=...)` 双轨过渡

过渡期 `T.alloc_shared(shape, dtype, layout="ah"/"wh")` 同时走两条通道：

- **fallback 通道**：保留物理 scope 后缀编码，实际分配仍带 `vtcm.ah` / `vtcm.wh`，
  兼容尚未消费 layout annotation 的 Hexagon lowering/工具。
- **正式通道**：分配完成后立即在当前 kernel SBlock 上发
  `T.annotate_layout({buf: T.make_layout(layout, buf)})`，让 AH/WH 进入
  LayoutInference 的 strict annotation 种子；`layout="rm"` 或缺省时不发 annotation，
  由 `fill_default_layout=True` 补 row-major `RM`。

双轨期 verifier 只做轻量一致性检查：若同一 buffer 同时有 `.ah`/`.wh` scope 后缀
和 AH/WH `layout_map` annotation，两者必须同名（`.ah` 对 AH、`.wh` 对 WH）；
缺 annotation、缺后缀，或通用 shell 补上的 `rm`/`none` 恒等 annotation 均合法。B 步
完成后 scope 后缀 fallback 将退役，layout annotation 成为唯一语义通道。
一致性检查会先跨整棵 TIR 收集所有嵌套 SBlock 的 `layout_map`，只在 annotation 明确
识别为 AH/WH 且与 `.ah`/`.wh` 后缀不同名时视为冲突；缺 annotation、`rm`/`none`、以及
其它恒等/未命中自定义 layout 都按 advisory 处理，不覆盖 scope 后缀语义。

`T.gemm` 的 A/B/C 操作数可写成 `buf[row0, col0]` 形式表示“声明 buffer + 动态
基址视图”：矩阵逻辑形状取 backing buffer/输出 tile 的尾部二维，`row0/col0` 只参与
AH/WH tile 基址偏移计算，不把 GEMM 形状收缩成 `1x1`。

向量三角/边界掩码请在 `T.vectorized` 循环中写 `T.if_then_else(j <= i, x, 0)`；
Hexagon emitter 会下降为 HVX word-lane compare + `Q6_V_vmux_QVV` select。

## 4. 规则表(编译期强制;每条都来自真机踩坑)

| # | 规则 | 违反后果 | 编译器行为 |
|---|---|---|---|
| R1 | 一切 VTCM buffer 128B 对齐;HVX load 地址必须 128B 对齐 | 硬件静默向下对齐,读错数据不报错 | 分配器自动对齐;load 地址不可证对齐时报错 |
| R2 | 禁止对 VTCM 的标量 load/store | ~46ns/次,慢 100 倍级 | `T.vectorized` 失败即编译错;所有 vtcm 访问 lowering 必须整 128B |
| R3 | VTCM 静态预算 ≤ 8MB(含 HMX CFG 区) | 运行时越界踩 CFG,杀 DSP PD | 编译期求和断言,报错给出每个 buffer 的占用 |
| R4 | HMX 序列只能出现在 kernel 主线程(RPC/executor 线程);pool worker 不得调 HMX | 跨线程 invoke 杀 PD(`AEE_EBADSTATE`) | 编译期强制:`HexagonVerify`(`tilelang/hexagon/passes.py`) 跟踪 worker-pool/threadIdx/GDN leaf pool 上下文,worker 分支里出现 `T.gemm` 或 `hexagon.gemm_hmx` 即报 R4 |
| R5 | `T.gemm` 的 A/B/C tile 维必须 32 的倍数;K 链长 = K/32 次 mm | HMX 指令粒度即 32×32×32 | 编译期整除检查;tailing 由 padding 或拆分处理 |
| R6 | 已废弃为性能建议:旧手写 GEMM 的 panel 宽 ∈ {1024,512,256};TileLang HMX GEMM 现在接受用户代码里 `T.alloc_shared/T.alloc_fragment` 选出的任意 32 倍数 `block_N` | 小 `block_N` 会增加 panel 数、权重 staging/写回相对开销,性能显著变慢但语义合法 | 编译器不再做 R6 硬校验;只保留 R5(32 整除)和 R3(VTCM 预算),VTCM 偏移来自 `HexagonStoragePlan` |
| R7 | GEMM 类累加:fp16 输入、HMX 37-bit acc(≈fp32);HVX 路径累加一律 fp32 内链 | fp16 内链在 K≥2560 必炸(max_rel 0.15 级) | `T.gemm` 默认 fp32-grade acc;HVX 归约自动 fp32 链 |
| R8 | 递推链上的 exp(逐 token/chunk 复利)必须 fp32 域 | fp16 exp 1e-3 误差 128 步复利成 14% | 当前要求用户显式使用 `T.hexagon.exp_fp32` 或 `T.hexagon.scan_exp32()` 这类 fp32 递推叶片;自动循环进位依赖识别待实现 |
| R9 | mask 用大负数(-32768)不用 -inf | HVX fp16 乘 inf 产 NaN | `T.fill(mask, ...)` 语义即写 -32768 |
| R10 | worker pool 规模 %6 wave 量化;≤6 任务一波 | 不满一波按一波计时 | `T.Kernel(threads=)` 上限 6;代价模型按 wave 计费 |
| R11 | 线程与 buffer 跨调用复用:kernel 内不得 malloc/free 大 buffer、不得每次 spawn 线程 | page fault + spawn 开销吃掉一半性能 | 生成的 skel 入口用全局 VTCM 分配表 + 常驻 pool,一次 open 终身复用 |
| R12 | 精度判据:rms 缩放 `\|got-ref\|/(\|ref\|+0.02·rms)` | K 大时 fp16 中间噪声顶爆 `\|ref\|+0.1` | 测试脚手架按此生成 |

### 4.1 pass 结构

Hexagon 后端在通用 TileLang lowering(`BindTarget`、kernel launch materialize、
negative-index/legalize、assume 注入、`Simplify`、`IfStmtBinding`)之后,先运行
通用 **`LowerTileOp`**。Hexagon 在 C++/Python 两侧注册了 target-specific
TileOp 实现:`src/hexagon/op/copy.cc` 把 `T.copy` 改写成显式
`hexagon.copy_*` intrinsic,`src/hexagon/op/gemm.cc` 选择 `hexagon.hmx`,
`tilelang/hexagon/op/gemm/` 再把 `T.gemm` 改写成 `hexagon.gemm_hmx`。
这些实现不生成 SIMT/threadIdx copy loop,避免与 Hexagon HVX worker-pool/HMX
主线程模型冲突。随后进入专用 pass 链,最后才调用 C emitter。顺序固定如下:

1. **LowerTileOp**(TileOp→Hexagon intrinsic):`T.copy` →
   `hexagon.copy_rm_ah` / `hexagon.copy_ah_rm` / `hexagon.copy_acc_rm` /
   `hexagon.copy_ddr`;`T.gemm` → `hexagon.gemm_hmx`。这是 target-specific
   注册表选择,不是 emitter 里的 recipe 推断；copy recipe 优先读取
   LayoutInference 的 `layout_map`，未命中时才回退到显式 annotation / scope 后缀。
2. **HexagonProductReduceFusion**(TIR→TIR):识别“`T.vectorized` 循环把
   `Mul(x,y)` 写入 128-lane scratch,紧跟 `hexagon.reduce_sum128`”的结构,
   改写为 `hexagon.reduce_prod128`;两个连续同形态 reduce 改写为
   `hexagon.reduce_prod2_128`。匹配只看结构,不依赖 GDN 或 buffer 名。
3. **HexagonWriteSet**(分析):扫描 `BufferStore` 与 Hexagon extern 的输出参数,
   产出全局写入 buffer 集合,挂到 `PrimFunc` attr `hexagon.write_set`。emitter
   只读取该 attr 来判断 slab 参数是否 `const`。
4. **HexagonStoragePlan**(分析):为所有 VTCM buffer 做 128B 对齐静态 bump
    allocation,挂 `hexagon.vtcm_offsets`,并在这里执行 R3 VTCM 8MB 预算检查。
5. **HexagonWScratchPlan**(分析):为 `wscratch` 做 per-worker DDR slot 映射。
   第一版保持真机 ABI 不变:按声明顺序检查 shape/dtype,映射到现有
   `hrt_gdn_slot_t` 字段(S/kf/qf/vf/w/o/A/P/kkprod/qkprod/eG/eGinv/beta/eGC);
   映射结果挂 `hexagon.wscratch_slots`,emitter 只查该 attr,不再看用户 buffer 名。
6. **HexagonVerify**(验证):在 emitter 前集中检查 TIR 层可见规则,当前覆盖 R1
     (VTCM copy 128B 可向量化,含 `hexagon.copy_*`)、R2(VTCM 标量 load/store,
     含 reduce 结果)、R4(pool worker 内禁止 `T.gemm`/`hexagon.gemm_hmx`)、
     R5(`hexagon.gemm_hmx` 静态 MNK/32 整除)、R10(worker≤6),失败错误带规则号。

emitter 只做机械 lowering:看到 `hexagon.copy_*` / `hexagon.gemm_hmx` 就打印
既有 `hrt_copy_*`、HMX acc_clear/mm/acc_read/unperm recipe;不再承担 copy/gemm
recipe 选择、全局写集分析、VTCM 静态 offset 规划或 product-reduce 模式识别。
当前保留的特殊分流也必须来自 TIR/attr 结构事实:weight staging 由
`hexagon.gemm_hmx` 的 B 操作数决定,不是 buffer 名前缀；GDN ABI shell 由
`hexagon.wscratch_slots` 决定,不是 "检测到 GDN 叶片" 开关；显式
`T.hexagon.silu_fp16` 仍机械 lowering,但 emitter
不再自动把普通 SiLU 表达式 peephole 成该 intrinsic。

## 5. v1 验证算子:GEMM(对照手写 attnops_gemm_nt)

用户代码(目标写法,与 TileLang CUDA GEMM 例子结构一致):

```python
@tilelang.jit(target="hexagon")
def gemm_nt(M, N, K, block_M=32, block_N=None, dtype=T.float16):
    # block_N 由用户选择:任意 32 倍数;旧 block_N=1024 路径保持生成物兼容
    A: T.Tensor((M, K), dtype)          # DDR,row-major
    B: T.Tensor((N, K), dtype)          # DDR,row-major,host 侧已转 WH
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), threads=6) as bx:
        A_sh = T.alloc_shared((block_M, K), dtype, layout="ah")   # ACT/LIN
        B_sh = T.alloc_shared((block_N, K), dtype, layout="ah")   # WH panel
        C_fr = T.alloc_fragment((block_M, block_N), T.float32)    # hmx.acc

        T.copy(B[bx * block_N:(bx + 1) * block_N, :], B_sh)       # pooled dcfetch
        T.copy(A, A_sh, layout=("rm", "ah"))                      # zip16
        for m in T.serial(T.ceildiv(M, block_M)):
            T.gemm(A_sh[m], B_sh, C_fr)                           # acc_clear+链式 mm
            T.copy(C_fr, C[m * block_M:(m + 1) * block_M,
                            bx * block_N:(bx + 1) * block_N],
                   layout=("ah", "rm"))                           # acc_read+unperm+写回
    return C
```

生成物:单个 `.c`(hexkl micro + Q6 intrinsics,结构对照 attnops_gemm.c)+
skel idl 方法 + host launcher stub。VTCM map 由用户声明推导:`A_sh`/ACT =
`block_M*K*sizeof(fp16)`,`B_sh`/WA = `block_N*K*sizeof(fp16)`,acc_read 输出 scratch =
`block_M*block_N*sizeof(fp16)`(均 128B 对齐);R3 统一由 `HexagonStoragePlan` 检查。
`block_N=1024,block_M=32` 的 Qwen 锚点仍走兼容 lowering,生成物逐字节不变。
验证:Qwen3.5 形状(M=960,
(N,K) ∈ {(8192,2560),(4096,2560),(2560,4096),(12288,2560),(6144,2560)}),
fp64 对拍 max_rel < 0.1(R12 判据),性能对照手写版 1.6–3.2 TFLOPS,
允许差距 ≤15%,超出则归因到 lowering 差异逐项对齐。

## 6. Qwen3.5-4B F16 形状锚点(语法/规则覆盖检验集)

| 算子 | 形状 | 切分轴 | 手写 kernel 结构 | 涉及规则 |
|---|---|---|---|---|
| GEMM_NT | M=960/1024,N∈{2560,4096,6144,8192,12288},K∈{2560,4096} | N panel | panel→rowblock(32)→K 链 | R5,R6,R7 |
| FFN(SwiGLU) | M=960,FF=9216,K=2560 | FF panel(Wg+Wu 共 stage) | 两 phase gemm + silu 原地 | R5–R8,pipeline |
| GDN prefill | T=1024,Hk=16,Hv=32,D=128,chunk=32 | v-head × chunk | 纯 HVX fp32,无 HMX | R2,R8,R10 |
| GDN std prefill | T=1024,Hk=16,Hv=32,D=128,chunk=32 | v-head × chunk | 标准 TileLang 构造生成，HVX/标量段与主线程 HMX `T.gemm` 混合 | R2,R4,R5,R8,R10 |
| FA D=256 | GQA 16q/4kv,S≤1024 | KV-head group→q-head→Q tile | HMX(S=K^TQ、PV)+ HVX online softmax | R5,R8,R9 |
| GLA decode | T=1,D=128 | head over pool | state 64KB/head 驻 VTCM | R3,R10 |

GDN 显式 scratch 必须使用 `T.alloc_wscratch`，但变量名不参与 ABI 选择：
`HexagonWScratchPlan` 按声明顺序检查 shape/dtype 并映射到 `hrt_gdn_slot_t`
字段，生成 C 仍只出现 `slot->S/kf/.../eGC`，保证真机 ABI 不变；
`examples/hexagon/gdn_prefill_renamed.py` 用任意 scratch 名覆盖此规则。

v0.1 只承诺 GEMM 端到端;GDN/FA/FFN 是语法表必须能表达、但尚未验证的后续目标。


## 6.5 T.Pipelined:软件流水与 pool/HMX 异步交叠(2026-09-13)

标准 `T.Pipelined(extent, num_stages=2)` 在 Hexagon 后端走与 CUDA 相同的
两个 pass(`PipelinePlanning` + `InjectSoftwarePipeline`,插在 IfStmtBinding
之后、LayoutInference 之前),自动完成 prologue/steady/epilogue 拆分和
buffer 多版本化(leading version 维 + floormod 轮换,VTCM 预算按版本数
翻倍计入)。

**stage 标注必须显式给**:自动 stage 分类器(pool-wrapped copy 识别不了)
会把所有语句塞进同一 stage 导致注入空转。写法:

```python
for i in T.Pipelined(N, num_stages=2, order=[0, 1, 2], stage=[0, 1, 1]):
    # stage 0 = producer(T.parallel staging copy,藏在消费相位下面)
    for job in T.parallel(...):
        T.copy(src[i], buf[i % 2 隐含], layout=("rm", "ah"))
    # stage 1 = consumer(主线程 gemm / 其他 pool 相位)
    T.gemm(...)
```

稳态循环 lowering 规则:
- 循环体内**第一个** T.parallel 相位 → `attnops_pool_start_ctx`(异步,
  与后续主线程 gemm 链交叠);
- `attnops_pool_join()` 自动插在**下一个 pool 相位之前**或循环回边
  (pool 是全局单例,同时只能有一个 outstanding 异步相位);
- prologue/epilogue 的所有相位保持同步 `attnops_pool_run_ctx`;
- 安全由注入 pass 保证(依赖分析 + 多版本化防 WAR),emitter 不另做
  依赖检查;带 `num_stages`/`software_pipeline_*` 残留标注到 emitter =
  注入失败,报 HexagonEmitError。

对照测试:/tmp/tl_pipeline_emit_test.py(异步形态)/tl_pipeline_reject_test.py。

**pool 相位融合(num_stages>=3,对齐手写 kernel 的异步 job 数组)**:
稳态体若恰为 `[T.parallel 生产者, gemm 链..., T.parallel 写回]` 且
num_stages>=3(写回相位必须是 stage>=2 的迟到消费者,读的是上一迭代的
buffer 版本),emitter 会把两个 T.parallel 合成**一个**异步 job 数组
(jobs [0,E0) 跑生产者、[E0,E0+E1) 跑写回),一次 pool_start_ctx 盖住,
回边一次 join——即手写 kernel "stage 下一个 + writeback 上一个"的单
异步槽结构。num_stages==2 时写回相位与本轮 gemm 有依赖,**不融合**(
保持 join 后同步跑)。融合要求两个相位 extent 均静态;非 bare For 的
包装(SBlockRealize/SBlock)自动剥壳识别。对照测试:
/tmp/tl_pipeline_fuse_emit_test.py。

## 7. 实现落点(mirror `tilelang/cuda/` + `src/cuda/`)

| CUDA 路径文件 | Hexagon 对应 |
|---|---|
| `tilelang/cuda/language/`(9 files) | `tilelang/hexagon/language/`:scope 常量、§2.2 扩展原语 |
| `tilelang/cuda/backend.py`(注册) | `tilelang/hexagon/backend.py`:`register_backend("hexagon")` |
| `tilelang/cuda/pipeline.py` | `tilelang/hexagon/pipeline.py`:复用通用 pass,替换 copy/gemm lowering + 加 R1–R12 检查 pass |
| `src/cuda/op/copy.cc`、`src/cuda/op/gemm.cc` | copy/gemm 的 Hexagon lowering(§3 recipe) |
| `src/cuda/codegen/codegen_cuda.cc` | C emitter:hexkl micro 调用 + Q6 intrinsics + skel 入口样板 |
| NVCC 编译回调 | hexagon-clang 编译进 skel + FastRPC host stub 生成 |
