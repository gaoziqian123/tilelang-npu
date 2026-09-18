# TileLang OpenCL GPU Backend — 语法表与规则表 v0.1

目标生态位:TileLang 的 CUDA 路径生成 CUDA kernel;本 backend 以**相同的前端写法**
生成 **OpenCL C kernel 文本**。用户仍写 `T.Kernel` / `T.alloc_shared` /
`T.copy` / `T.gemm`,只是 `target="opencl"`。当前主跑法是 AOT 模式取
`kernel_source`,生成物**不在编译机上运行**;它被部署到手机上,由手机侧 OpenCL
驱动完成编译与执行。

目标硬件:OnePlus 13 / SM8750,Adreno 830 GPU,OpenCL 3.0,12 CU,
`cl_khr_fp16` 与 subgroups 可用。硬件事实以
`/root/project/.opencode/skills/oneplus13-npu/SKILL.md` 为准;手写 OpenCL
参照物在 `/root/project/backend/gpu/*/*.cl`。

验证宿主:当前使用 `/root/project/backend/gpu/tl_probe/tl_probe.c` 作为通用
runner。它在手机上 `dlopen /vendor/lib64/libOpenCL.so`,调用
`clCreateProgramWithSource` 编译 TileLang 生成的 `.cl`,再用 fp64 reference harness
对拍。正式接入时走 `/root/project/backend/gpu/` 下各算子工程同款 dlopen 宿主。

设计原则:

1. **写法与原生 TileLang 一致**。用户写标准 TileLang kernel,只是
   `target="opencl"`。差异收敛在 **scope 映射、SIMT lowering recipe 与 legacy
   TVM OpenCL codegen 兼容层**。
2. **尽量复用现有编译链**。OpenCL 后端(已实现于 commit `31d3a80`)复用
   vendored TVM 的 `target.build.opencl` / legacy `codegen_opencl.cc`,而不是照
   Metal 路线自写 C++ codegen。收益是零 C++ codegen 工作量,fp16/vector 类型/
   地址空间发射现成;代价是若干 legacy 缺口必须由 Python pass remap 补齐。
3. **OpenCL 是 SIMT 后端,不是 NPU intrinsic 后端**。`T.copy` / `T.fill` /
   `T.transpose` 下降成 work-item 级 loop;`T.gemm` 下降成 `opencl.fma` 标量 FMA
   循环。没有 Tensor Core/HMX 概念;Adreno 的 `cl_qcom_ml_ops` 私有扩展只是理论
   突破口,当前未做。
4. **规则来自真机踩坑**。`shared.dyn` remap、`DeclBuffer` 物化、legacy
   `PrintStorageScope` 限制、alias 型 dynamic shared memory 等均是已验证问题;
   文档里的红线后续应变成 pass invariant 或 verifier。

---

## 1. 存储层级:scope 映射表

| TileLang scope | OpenCL 物理实体 | 容量 | 约束 |
|---|---|---|---|
| `global`(`T.Tensor`) | `__global` DDR buffer,host 侧 `clCreateBuffer` 传入 | 由 OpenCL buffer 与手机内存决定 | kernel 文本只看到 `__global` 指针;host 负责 buffer 生命周期与拷贝 |
| `shared.dyn`(`T.alloc_shared`) | `__local` local memory(Adreno local memory) | ~32KB/CU 量级,以 SKILL.md 为准;TVM OpenCL target 默认 `max_shared_memory_per_block=16384` | legacy codegen `PrintStorageScope` 只认 `shared`,不认 `shared.dyn`;必须经 remap pass 改写成 `shared` |
| `local.fragment`(`T.alloc_fragment`) | work-item 私有寄存器/私有数组 | 受寄存器与 private memory 限制 | 用作 `T.gemm` accumulator;当前是标量 FMA 私有累加,无 tensor core fragment |
| `local`(`T.alloc_local`) | work-item 私有寄存器/私有数组 | 受寄存器与 private memory 限制 | fresh thread-local `DeclBuffer` 需在 remap pass 中物化成 `AllocBuffer`,否则 legacy codegen 不声明变量 |

OpenCL 后端没有 Hexagon 的 `wscratch`、VTCM、HMX acc 或 AH/WH layout 概念。
scope 名保持 TileLang 习惯:用户仍写 `T.alloc_shared(...)` / `T.alloc_fragment(...)`,
backend 根据 target 把它们落到 OpenCL `__local` / private storage。

关键 remap:TileLang 前端与 WebGPU SIMT 管线会产生 `shared.dyn`,但 vendored TVM
OpenCL legacy codegen 只认 `shared`。`tilelang/opencl/pipeline.py` 在末尾运行
`_OpenCLSharedDynToShared` pass,把 dynamic shared memory 改写到 legacy codegen
能打印的 scope。该 pass 必须放在 `ThreadSync("shared.dyn")` 之后:先让通用 pass
插入 barrier,再改名成 `shared`。

## 2. 语法表:原语对照

### 2.1 保持原义的原语(用户写法与 CUDA 路径逐字相同)

| 原语 | OpenCL backend 语义 |
|---|---|
| `T.Kernel(..., threads=N)` | 映射到 OpenCL workgroup / work-item SIMT 执行;具体 launch 维度由生成 kernel 与 host 调用共同决定 |
| `T.alloc_shared(shape, dtype)` | 生成 OpenCL `__local` storage;前端可出现 `shared.dyn`,最终由 remap pass 改成 legacy codegen 认识的 `shared` |
| `T.alloc_fragment(shape, dtype)` | work-item 私有 accumulator/private array;`T.gemm` 的输出累加落在这里 |
| `T.alloc_local(shape, dtype)` | work-item 私有寄存器/私有数组;fresh `DeclBuffer` 需要显式物化,见 R2 |
| `T.copy(src, dst)` | 走 SIMT `LowerNormalCopy`:每个 work-item 搬一个或多个元素;向量化由 `VectorizeLoop` 与 legacy codegen 的向量类型发射负责 |
| `T.fill(dst, value)` | 走 SIMT lowering,由 work-item 并行写入;向量化同上 |
| `T.transpose(src, dst)` | 走 SIMT lowering,照 WebGPU 样板注册到 OpenCL target |
| `T.gemm(A, B, C)` | lowering 为 `opencl.fma`:标量 FMA 循环,accumulator 为线程私有;实现位于 `tilelang/opencl/op/gemm_fma.py` |
| dtype: `float16` / `float32` | legacy OpenCL codegen 已具备 fp16 pragma、vector 类型与地址空间打印;`cl_khr_fp16` 在目标机可用 |

### 2.2 当前不承诺的原语

| 原语/能力 | 当前状态 | 后续做法 |
|---|---|---|
| `reduce` / `atomic_add` 等其它 TileOp | 未注册;用到会报 `no implementation registered for opencl` | 按 `src/opencl/op/` 现有 `copy.cc` / `fill.cc` / `transpose.cc` 样板补注册 |
| Tensor Core / WMMA 类能力 | OpenCL 标准路径无此概念 | Adreno 私有 `cl_qcom_ml_ops` 是理论突破口,当前未接 |
| Hexagon `wscratch` / HMX / AH/WH layout | OpenCL 后端不存在 | 不在 OpenCL scope 体系内表达 |

## 3. TileOp lowering recipe 表

| TileOp | lowering | 当前实现落点 / 依据 |
|---|---|---|
| `T.copy` | **SIMT `LowerNormalCopy`**:每个 work-item 搬一个或多个元素;由通用 vectorize + legacy OpenCL codegen 打印 OpenCL vector 类型 | `src/opencl/op/copy.cc`,照 WebGPU 样板并 `match opencl` |
| `T.fill` | SIMT 写入,每个 work-item 覆盖一段元素 | `src/opencl/op/fill.cc` |
| `T.transpose` | SIMT 转置搬运 | `src/opencl/op/transpose.cc` |
| `T.gemm` | 指令选择恒返回 `opencl.fma`;Python 侧 `GemmFMA` 生成标量 FMA 循环,accumulator 线程私有 | `src/opencl/op/gemm.cc`, `tilelang/opencl/op/gemm_fma.py` |
| 其它 TileOp | 未注册即报错 | 按需扩展 `src/opencl/op/` |

`T.gemm` 当前是功能优先的朴素 SIMT 路线。L2 GEMM NT 512³ fp16 已真机对拍
PASS,但默认 `T.copy+T.gemm` 生成物性能仍只有 36ms,原因是每 work-item 只算 1 个
输出、global/shared 访问未形成 `halfN/floatN` 向量内链。2026-09-18 的探索性
`examples/opencl/gemm/gemm_nt.py --impl local_tiled` 可生成手写 local-memory 结构
同款的 SIMT OpenCL kernel(64×128 work tile、128 work-items、8×8 thread tile、BK
可调、fp32 FMA 内链);在修正 `tl_probe` kernel-only 多次事件计时后,生产形状
1024×2560×2560 真机 best 为 BK=16 的 17.9ms(0.75 TFLOPS)。`--impl image8x8`
是针对 Adreno 的 escape hatch,生成 B 走 `image2d_t/read_imageh`、无 local/barrier、
8×8 寄存器 tile 的 kernel;需 `gemm_gpu` 同款 host 预转 B 到 RGBA fp16 image,生产形状
真机 7.80ms(1.72 TFLOPS)。这证明标准 buffer/local 路径主要卡在 B 读取与
local/barrier 开销,而纹理+寄存器路径能接近手写上限;该 image 路径目前仍是示例侧
模板,尚未并入通用 `T.gemm` lowering。

## 4. pass / codegen 结构

OpenCL 后端结构(已实现于 commit `31d3a80`):

| 路径 | 职责 |
|---|---|
| `tilelang/opencl/backend.py` | 注册 `BackendModule`, `target_kinds=("opencl",)` |
| `tilelang/opencl/codegen.py` | device codegen 指向 vendored TVM `target.build.opencl`;实际 C++ codegen 为 `3rdparty/tvm/src/target/opencl/codegen_opencl.cc` |
| `tilelang/opencl/pipeline.py` | 克隆 WebGPU SIMT 管线,并在末尾追加 `_OpenCLSharedDynToShared` remap pass |
| `tilelang/opencl/op/` | Python 侧 op lowering,当前含 GEMM FMA 路径 |
| `src/opencl/op/copy.cc` / `fill.cc` / `transpose.cc` | C++ TileOp 注册,SIMT lowering,照 WebGPU 样板并匹配 `opencl` |
| `src/opencl/op/gemm.cc` | GEMM 指令选择,当前恒返回 `opencl.fma` |

选择 legacy TVM OpenCL codegen 而非自写 codegen 的原因:

1. **收益**:零 C++ codegen 工作量;fp16 pragma、OpenCL vector 类型、地址空间打印
   都已在 vendored TVM 内实现,且编进 `libtilelang.so`。
2. **代价**:legacy codegen 对新 TileLang IR 的若干节点不完整,尤其是
   `shared.dyn`、`DeclBuffer` 与 alias dynamic shared memory。当前用 Python pass
   在 codegen 前修正 IR,避免 fork codegen。
3. **长期选择**:若 remap 与 workaround 继续增加,可评估照 `codegen_metal.cc` 路线
   自写 OpenCL codegen,原生支持 `shared.dyn` 并移除兼容层。

## 5. 规则表(编译期强制目标;每条都来自真机踩坑)

| # | 规则 | 违反后果 | 当前/目标编译器行为 |
|---|---|---|---|
| R1 | `shared.dyn` 必须先参与 `ThreadSync("shared.dyn")`,再 remap 成 `shared` | 先改名会导致 barrier 插入 miss 或 storage scope 不一致 | `_OpenCLSharedDynToShared` 放在 pipeline 末尾,注释固定说明顺序 |
| R2 | legacy `codegen_c` 的 `VisitStmt_(DeclBufferNode)` 既不发射声明也不注册变量;fresh thread-local `DeclBuffer` 必须物化成 `AllocBuffer` | 生成 OpenCL 使用未声明变量或 codegen state 缺失 | remap pass 中把 fresh thread-local `DeclBuffer` 转为 `AllocBuffer` |
| R3 | alias 型 `DeclBuffer` 不能转 `AllocBuffer` | data var 已由 `Bind` 绑定;再转会造成 SSA 重复定义 | remap pass 区分 fresh 与 alias;alias 保持原状 |
| R4 | `PyStmtExprMutator` 不能 override `visit_bind_` | 该 tirx fork 会静默丢 `Bind` 节点 | 用 `tirx.stmt_functor.post_order_visit` 预先收集,不要 override `visit_bind_` |
| R5 | legacy `PrintStorageScope` / `PrintStorageSync` 只认 `shared`,不认 `shared.dyn` | OpenCL codegen 打印失败或 barrier scope 不合法 | `shared.dyn` → `shared` remap 是 OpenCL pipeline 必备 pass |
| R6 | `MergeSharedMemoryAllocations(preserve_aliases=true)` 在 OpenCL 路径会生成 `buf_dyn_shmem + handle_add_byte_offset` alias | 生成代码为 `void*` + `(char*)` 强转形式,较丑但 Adreno 编译器实测可编译 | 当前接受该形态,L2 真机 PASS;长期可评估走 WebGPU 的 `preserve_aliases=false` 路径 |
| R7 | TVM OpenCL target 默认 attr 是保守桌面值 | `max_num_threads=256`, `max_shared_memory_per_block=16384`,低于 Adreno 实际潜力 | v0.1 保持默认;Adreno 830 实测校准是路线图项 |
| R8 | L0 生成代码无 bounds guard | launch 尺寸必须整除数据规模,否则越界或漏算 | 当前测试只用整除尺寸;后续补 guard 或要求 host launch 对齐 |
| R9 | L1 生成代码存在多余未使用 `smem_1` 数组 | 无害,但影响可读性与资源观感 | 当前接受;后续清理 lowering 残留 |
| R10 | 精度判据使用 rms 缩放 `\|got-ref\|/(\|ref\|+0.02·rms)` | K 维大时普通相对误差阈值易误判 fp16 中间噪声 | `tl_probe` harness 按该判据对拍 |

## 6. v1 验证算子与性能基线

全部基线来自 OnePlus 13 真机,judge 为 `max_rel < 0.1` 的 rms-scaled 判据。

| 层级 | 算子/形状 | TileLang 特性覆盖 | 真机结果 |
|---|---|---|---|
| L0 | elementwise 128×128 fp32 | 基础 OpenCL kernel 生成、global buffer、SIMT loop | PASS, `max_rel=5.8e-8` |
| L1 | 行归约 128×256 fp32 | `T.alloc_shared` + barrier 树归约 | PASS, `max_rel=3.0e-6` |
| L2 | GEMM NT 512³ fp16 | `T.copy` + `T.gemm` 高层写法,`opencl.fma` lowering | PASS, `max_rel=4.8e-4`,36.1ms |
| L2-opt | GEMM NT 512³ fp16 | `--impl local_tiled --bm 64 --bn 128 --bk 16`:64×128 WG / 8×8 thread tile / fp32 FMA 内链;kernel-only event timing | PASS, `max_rel=4.7e-4`,0.421ms(0.638 TFLOPS) |
| L2-prod-local | GEMM NT 1024×2560×2560 fp16 | `--impl local_tiled --bm 64 --bn 128 --bk 16`,sampled fp64 check | PASS, `max_rel=4.7e-4`,17.9ms(0.750 TFLOPS) |
| L2-prod-image | GEMM NT 1024×2560×2560 fp16 | `--impl image8x8`,B as RGBA fp16 `image2d_t`,no local/barrier,`gemm_gpu` host | PASS, `max_rel=5.0e-4`,7.80ms(1.72 TFLOPS) |

对照性能:同口径手写 local64×128 为 22.4ms(0.598 TFLOPS),手写 image8×8 为
7.84ms(1.71 TFLOPS)。TileLang local_tiled 与手写 local 的差距已在噪声/参数范围内;
剩余 2.3× 差距来自 `image2d_t/read_imageh` 纹理读路径避免 B buffer gather、local memory
填充和每 K tile 双 barrier。TileLang OpenCL L2 的 36ms 是朴素 SIMT 生成物基线;
L2-opt 已使用 `half4` global staging、`half8/float8` local load/FMA 与每 work-item
8×8 输出复用。尚未并入通用 `T.gemm` lowering、未做完整 autotune 或 target attr
校准,不能代表优化后上限。

对拍工具: `/root/project/backend/gpu/tl_probe/tl_probe.c`。用法由 argv 传入
`.cl` 路径、kernel 名与算子类型;host 侧完成 dlopen OpenCL、program build、kernel
launch 与 fp64 reference 对拍。正式 runtime 接入时应迁移到 `backend/gpu/` 各算子
工程同款宿主,但 kernel 文本仍由手机驱动编译。

## 7. 实现落点(mirror WebGPU SIMT + legacy TVM OpenCL codegen)

| 参考路径 | OpenCL 对应 |
|---|---|
| `tilelang/webgpu/pipeline.py` | `tilelang/opencl/pipeline.py`:克隆 SIMT 管线,末尾追加 `_OpenCLSharedDynToShared` |
| `src/webgpu/op/*` | `src/opencl/op/copy.cc` / `fill.cc` / `transpose.cc`:SIMT lowering 样板 |
| target-specific GEMM op | `src/opencl/op/gemm.cc` + `tilelang/opencl/op/gemm_fma.py`:选择 `opencl.fma` 标量 FMA |
| TVM target build | `tilelang/opencl/codegen.py` → `target.build.opencl` → `3rdparty/tvm/src/target/opencl/codegen_opencl.cc` |
| 手机验证宿主 | `backend/gpu/tl_probe/tl_probe.c` 与正式 `backend/gpu/*` dlopen 宿主 |

生成物边界:OpenCL backend 产出的是 kernel source 文本;编译、link、运行发生在
OnePlus 13 手机侧 OpenCL 驱动。不要假设编译机具备 OpenCL runtime,也不要把生成物
设计成必须在编译机上 JIT 执行。

## 8. 路线图

1. **GEMM 向量化与 autotune**:把 `local_tiled` 示例模板的 64×128 work tile、
   `half4/half8` local staging、`float8` accumulator、8×8 thread tile正式并入
   OpenCL `T.gemm` lowering,再搜索 tile 空间。当前通用 lowering 的主要差距是每
   work-item 只产 1 个 C 元素,没有连续向量 load/store 与多输出寄存器复用。
2. **Adreno image GEMM host contract**:`image8x8` 已验证 1.7 TFLOPS,但 ABI 需要 host
   把 B[N,K] 预转为 RGBA fp16 `image2d_t`。若要把该路径产品化,需要在 TileLang
   runtime/AOT metadata 中表达 image 参数与权重预打包,或提供明确的 escape-hatch host API。
3. **Adreno 830 target attr 校准**:实测并设置 `max_num_threads`、local memory
   上限等 target attr,替换 TVM OpenCL 默认保守桌面值。
4. **补齐 TileOp 注册**:按需实现 `reduce`、`atomic_add` 等 OpenCL lowering,
   避免用户碰到 `no implementation registered for opencl`。
5. **清理 legacy 兼容层**:评估让 OpenCL 走 WebGPU `preserve_aliases=false` 路径,
   或自写 OpenCL codegen(照 `codegen_metal.cc`)以原生支持 `shared.dyn`、去掉 remap。
6. **长期突破口**:评估 Adreno 私有 `cl_qcom_ml_ops` 是否可接入矩阵路径;当前未做,
   不在 v0.1 承诺范围内。

## 9. 与 hexagon 后端的差异

| 维度 | Hexagon 后端 | OpenCL 后端 |
|---|---|---|
| 生成语言 | Hexagon HVX/HMX intrinsic C,经 hexagon-clang 编进 FastRPC skel | 标准 OpenCL C kernel source,由手机 OpenCL 驱动编译 |
| 执行模型 | QuRT worker pool + FastRPC executor;HMX 调用线程有严格规则 | OpenCL workgroup/work-item SIMT;host 通过 OpenCL API launch |
| 存储核心 | DDR slab + VTCM(8MB) + HVX vreg + HMX accumulator | `__global` DDR buffer + `__local` local memory + work-item private storage |
| 矩阵单元 | HMX 32×32 fp16 矩阵单元,AH/WH layout 与 VTCM 规则 | 标准 OpenCL FMA loop;无 tensor core/HMX 概念 |
| 主要红线 | VTCM 禁标量访问、128B 对齐、HMX 线程规则、VTCM 预算 | legacy codegen 兼容: `shared.dyn` remap、`DeclBuffer` 物化、alias 保留、target attr 校准 |
| codegen 策略 | 自有 emitter 打印 intrinsic C 与 skel ABI | 复用 vendored TVM legacy OpenCL codegen,Python pass 修补缺口 |
| 验证方式 | FastRPC skel + 手机 NPU/GPU/CPU 对拍 | `tl_probe` dlopen OpenCL + `clCreateProgramWithSource` + fp64 harness |

一句话:Hexagon 后端是“裸机 NPU 规则先行”的 intrinsic C backend,核心难点是
VTCM/HMX/HVX 的硬件不变量;OpenCL 后端是“标准 OpenCL C + SIMT + legacy codegen
兼容层”的 GPU backend,核心难点是把 TileLang 的现代 IR 可靠降到 vendored TVM
OpenCL codegen 能接受、并在 Adreno 驱动上真机通过。
