# Hexagon 后端示例

本目录展示 TileLang 的实验性 `target="hexagon"` 后端：用户仍在 Python 里描述张量、循环、shared/fragment buffer 和硬件叶片，lower 后拿到 **Hexagon intrinsic C 文本**。生成的 C 不是直接在本机运行，而是放进 OnePlus 13 / SM8750 的 FastRPC skel 工程里编译、部署、对拍。语法、布局约定和 Hexagon 红线见 [`docs/hexagon/00_syntax_and_rules.md`](../../docs/hexagon/00_syntax_and_rules.md)。当前脚本使用 `engine.lower(...).kernel_source` 取生成源码，并在本机有 Hexagon SDK 时额外跑 `hexagon-clang -fsyntax-only`。

## 环境与运行方法

从仓库根目录运行，显式设置 `PYTHONPATH` 并使用仓库内虚拟环境：

```bash
PYTHONPATH=/root/project/tilelang /root/project/tilelang/.venv/bin/python examples/hexagon/gemm_nt.py
PYTHONPATH=/root/project/tilelang /root/project/tilelang/.venv/bin/python examples/hexagon/gdn_prefill.py
PYTHONPATH=/root/project/tilelang /root/project/tilelang/.venv/bin/python examples/hexagon/gdn_prefill_renamed.py
PYTHONPATH=/root/project/tilelang /root/project/tilelang/.venv/bin/python examples/hexagon/gdn_split.py
PYTHONPATH=/root/project/tilelang /root/project/tilelang/.venv/bin/python examples/hexagon/silu_mul.py --impl direct
PYTHONPATH=/root/project/tilelang /root/project/tilelang/.venv/bin/python examples/hexagon/silu_mul.py --impl vtcm
```

默认输出在 `examples/hexagon/out/*.c`。所有脚本都有 `--out` 和 `--skip-clang`；形状参数也可在命令行覆盖。成功时关键输出应包含 `EMIT_OK ...` 和 `HEXAGON_CLANG_PASS`（如果本机缺少工具链则会打印 `HEXAGON_CLANG_SKIP`）。

## 可选 per-phase profiling

通用 mixed pool/HMX shell 支持 opt-in pass config / kernel attr `tl.hexagon_prof=True`。开启后生成 C 会在 slab 尾部的 `prof[]` 区写入 int32 qtimer tick 累加计数，并在文件头注释记录 slot 布局；默认关闭，因此 legacy anchors 保持字节稳定。当前仅 `gdn_std.py` 用 `PassContext(config={PassConfigKey.TL_HEXAGON_PROF.value: True})` 开启该模式，用于定位各 worker-pool 阶段和既有 staging/GEMM/unperm 聚合阶段。

## GEMM_NT (`gemm_nt.py`)

用户写法关键代码段如下：A 是 row-major `(M,K)`，B 是 NT 权重 `(N,K)`；脚本显式申请 AH layout scratch 和 fp32 accumulator，再调用 `T.gemm(..., transpose_B=True)`。

```python
@tilelang.jit(out_idx=[2], target="hexagon", execution_backend="aot")
def gemm_nt(M: int, N: int, K: int, block_M: int = 32, block_N: int = 1024, dtype=T.float16):
    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((N, K), dtype), C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), threads=6) as bx:
            A_sh = T.alloc_shared((block_M, K), dtype, layout="ah")
            B_sh = T.alloc_shared((block_N, K), dtype, layout="ah")
            C_fr = T.alloc_fragment((block_M, block_N), T.float32)

            T.copy(B[bx * block_N, 0], B_sh)
            for m in T.serial(T.ceildiv(M, block_M)):
                T.copy(A[m * block_M, 0], A_sh, layout=("rm", "ah"))
                T.gemm(A_sh, B_sh, C_fr, transpose_B=True, clear_accum=True)
                T.copy(C_fr, C[m * block_M, bx * block_N], layout=("ah", "rm"))
```

生成 C 的结构：每个 N panel 一个 worker，B panel 进 AH/VTCM，A 按 `block_M=32` 扫 M，HMX GEMM_NT 使用 fp32 accumulator，最后 AH→row-major 写回 C。

真机验证：五个 GEMM 形状均通过，`max_rel` 约 `4.5e-4`；相同形状下比既有手写路径快约 18–47%。

## GDN prefill (`gdn_prefill.py`)

当前 GDN 示例已经改成 **显式 buffer + 带操作数叶片** 的形态。GDN scratch 使用 `T.alloc_wscratch`，语义是 per-worker DDR scratch，不占 VTCM 预算；lowering 按声明顺序 + shape/dtype 映射到既有 `hrt_gdn_slot_t` ABI 字段，用户变量名可任意（`gdn_prefill_renamed.py` 覆盖此回归）。粒度原则是：数据流必须由 TileLang 层的 tensor/scratch 明确表达，硬件叶片只做一个可命名的动作并接收所有操作数；原语名不绑定上层算子名，避免把整算子语义藏进零参 primitive。少数仍滞留的参数化复合叶片是 `state_x2_matvec128`、`forward_solve32`、`state_update32`，原因是它们分别承载 128×128 state matvec、32 行 forward solve、32 行 state update，内部是较紧的手写 HVX 配方，继续拆细会把示例变成 intrinsics 拼装而不是 TileLang 数据流示例。

用户写法关键代码段如下（摘自当前文件，保留真实操作数形态）：

```python
@tilelang.jit(out_idx=[6, 7], target="hexagon", execution_backend="aot")
def gdn_prefill(TOK: int = 1024, Hk: int = 16, Hv: int = 32, D: int = 128, chunk: int = 32, dtype=T.float16):
    @T.prim_func
    def main(
        Q: T.Tensor((Hk, TOK, 128), T.float16),
        K: T.Tensor((Hk, TOK, 128), T.float16),
        V: T.Tensor((Hv, TOK, 128), T.float16),
        G: T.Tensor((Hv, TOK), T.float32),
        B: T.Tensor((Hv, TOK), T.float32),
        S0: T.Tensor((Hv, 128, 128), T.float32),
        O: T.Tensor((Hv, TOK, 128), T.float16),
        S1: T.Tensor((Hv, 128, 128), T.float32),
    ):
        # Correct-granularity GDN: every scratch buffer is explicit TileLang
        # state. T.hexagon.* calls below are hardware-action leaves with
        # operands; no zero-arg primitive or hidden slot owns dataflow.
        with T.Kernel(Hv, threads=6) as hv:
            state = T.alloc_wscratch((128, 128), T.float32)
            kf = T.alloc_wscratch((chunk, 128), T.float32)
            qf = T.alloc_wscratch((chunk, 128), T.float32)
            vf = T.alloc_wscratch((chunk, 128), T.float32)
            w = T.alloc_wscratch((chunk, 128), T.float32)
            o_acc = T.alloc_wscratch((chunk, 128), T.float32)
            A = T.alloc_wscratch((chunk, chunk), T.float32)
            P = T.alloc_wscratch((chunk, chunk), T.float32)
            eG = T.alloc_wscratch((chunk,), T.float32)
            eGinv = T.alloc_wscratch((chunk,), T.float32)
            beta = T.alloc_wscratch((chunk,), T.float32)
            eGC = T.alloc_wscratch((1,), T.float32)

            T.hexagon.load_state128(S0, state, hv)
            for c in T.serial(TOK // chunk):
                hk = hv % Hk
                t0 = c * chunk

                # a) Q/K/V chunk fp16 -> fp32 scratch rows.
                T.hexagon.load_h2f_rows128(Q, K, V, qf, kf, vf, TOK, hk, hv, t0)

                # b) fp32 serial prefix gate scan and exp factors (R8 explicit).
                T.hexagon.scan_exp32(G, B, eG, eGinv, beta, eGC, TOK, hv, t0)

                # c) A/P triangular matrices from interleaved dot128 reductions.
                for i in T.serial(chunk):
                    for j in T.serial(i + 1):
                        # A[i,j] = beta[i] * eG[i] * eGinv[j] * dot(k_i,k_j)
                        # P[i,j] =           eG[i] * eGinv[j] * dot(q_i,k_j)
                        T.hexagon.dot128x2_store(kf, qf, eG, eGinv, beta, A, P, i, j)

                # d/e) S0^T*k, S0^T*q and beta/eG affine w RHS.
                for i in T.serial(chunk):
                    T.hexagon.state_x2_matvec128(state, kf, qf, w, o_acc, i)
                    T.hexagon.affine_rows128(vf, w, beta, eG, i)

                # f) UT forward substitution over 32 rows of w.
                for i in T.serial(chunk):
                    T.hexagon.forward_solve32(A, w, i)

                # g) output row = eG*S0^T*q + tril(P)*w, stored fp16.
                for i in T.serial(chunk):
                    T.hexagon.output_rows128(o_acc, w, P, eG, O, TOK, hv, t0, i)

                # h) state update. Decay is only in e^{-gamma}/e^{gamma}/chunk end;
                # UT matrix above intentionally carries no decay.
                T.hexagon.state_decay_rows128(kf, eGC, eGinv)
                T.hexagon.state_update32(state, kf, w, eGC)
            T.hexagon.store_state128(state, S1, hv)
```

生成 C 的结构：按 value head 启动 worker，32-token chunk 内依次完成 Q/K/V fp16→fp32、gate scan、三角 dot、state matvec、forward solve、输出、state decay/update；decay 只出现在 v 侧 `e^{-gamma}`、输出侧 `e^{gamma}` 和 chunk 末 state decay，UT 矩阵不含 decay。

真机验证：`max_rel=0.0059`，与手写实现逐位一致；TileLang 生成版 `44.99 ms`，手写版 `45.35 ms`。

## GDN prefill split (`gdn_split.py`)

`gdn_split.py` 把同一 GDN chunk 数学按 FLA/flash-linear-attention 的阶段拆成
五个独立 FastRPC/TileLang 入口。调用方按 chunk 串行执行：

```text
for c in 0..T/32-1:
  tl_gdn_cumsum(c) -> tl_gdn_kkt(c) -> tl_gdn_solve(c)
  -> tl_gdn_fwdo(c) -> tl_gdn_fwdh(c)
```

拆分后的显式 slab 布局（所有 offset 128B 对齐，固定锚点
`T=1024,Hk=16,Hv=32,D=128,C=32`）：

| 名称 | 形状 | dtype | 说明 |
|---|---:|---|---|
| `Q,K` | `[Hk,T,128]` | fp16 | 输入 q/k，k-head = `hv % Hk` |
| `V,O` | `[Hv,T,128]` | fp16 | 输入 v / 输出 o |
| `G,B` | `[Hv,T]` | fp32 | log-decay / beta |
| `S0,S1` | `[Hv,128,128]` | fp32 | 初始/最终 state `[dk,dv]` |
| `state` | `[Hv,128,128]` | fp32 | chunk 间显式 state，`cumsum(c=0)` 从 S0 初始化，`fwdh(last)` 写 S1 |
| `qf,kf,vf` | `[Hv,T/32,32,128]` | fp32 | `tl_gdn_cumsum` 的 fp16→fp32 行缓存 |
| `w,u` | `[Hv,T/32,32,128]` | fp32 | `w` 是 RHS/solve 后结果；`u` 保存 `S^T q` 输出侧 affine 项 |
| `A,P` | `[Hv,T/32,32,32]` | fp32 | FLA 三角矩阵；`A` 严格下三角，`P` 含对角且上三角清零 |
| `eG,eGinv,beta` | `[Hv,T/32,32]` | fp32 | chunk 内前缀 exp 因子和 beta |
| `eGC` | `[Hv,T/32]` | fp32 | chunk-final decay |
| `prof` | `5*int32` | i32 | 五阶段 qtimer tick 累加 |

五个阶段职责：

- `tl_gdn_cumsum`：照单 kernel 用户层 b 段做 inclusive cumsum、clamp(-60)、fp32
  exp，拷贝 beta；同时用 HVX helper 转换 Q/K/V 到 fp32 中间张量。
- `tl_gdn_kkt`：照用户层 c 段用 128-wide fp32 dot 构建 `A/P`；同时计算
  `S^T k`/`S^T q` 两个 state matvec，并做 `beta*v - beta*eG*S^T k` affine。
- `tl_gdn_solve`：逐行串行 forward substitution：`w_i -= A[i,j] w_j`。
- `tl_gdn_fwdo`：矩阵阶段显式走 HMX：stage `P(32x32)` 和 `w^T(128x32)` 到
  AH/VTCM，主线程执行 `T.gemm` 等价链得到 `tril(P)@w`，再加 `eG*(S^Tq)` 并
  fp32→fp16 写 O。HMX 不进入 pool worker。
- `tl_gdn_fwdh`：先把 `eGC*eGinv` fold 到 `kf`，再用 HMX 计算
  `k_hat^T(128x32) @ w(32x128)`，最后 HVX 加到 `eGC*state`；last chunk 写 S1。

这个拆法的目的不是取代已验证单 kernel，而是暴露 FLA 阶段边界，便于以后把
矩阵阶段调度/替换成更标准的 TileLang `T.gemm` lowering。当前 ABI 只追加方法，
不修改 `tl_gdn_prefill`。

## SwiGLU 激活 (`silu_mul.py`)

这个例子覆盖 FFN 中的 `silu(gate) * up`，三个变体的算术表达式一致，差别只在搬运策略。`--impl` 默认是当前最快的 `direct`。

### direct：直读流式冠军

```python
@tilelang.jit(out_idx=[2], target="hexagon", execution_backend="aot")
def silu_mul(M: int, FF: int):
    # direct:DDR 直读 + HVX 流式计算。OnePlus 13 实测 15.5 GB/s
    # (15.53 GB/s),max_rel=0.0027,三变体中最快。
    @T.prim_func
    def main(G: T.Tensor((M * FF,), T.float16), U: T.Tensor((M * FF,), T.float16), O: T.Tensor((M * FF,), T.float16)):
        with T.Kernel(T.ceildiv(M * FF, 1024), threads=6) as bx:
            for p in T.Parallel(16):
                for v in T.vectorized(64):
                    i = bx * 1024 + p * 64 + v
                    # silu(g) * u,纯算术展开;emitter 逐节点 lowering
                    O[i] = G[i] * U[i] / (T.float16(1) + T.exp(-G[i]))
```

### vtcm：staging 对照

```python
@tilelang.jit(out_idx=[2], target="hexagon", execution_backend="aot")
def silu_mul_vtcm(M: int, FF: int):
    # vtcm:把 1024 元素 tile staging 到 VTCM 后计算。OnePlus 13 实测
    # 15.1 GB/s(15.11 GB/s):轻量逐元素算子没有数据复用,copy 无法被
    # 有效重叠,所以比 direct 略亏。
    @T.prim_func
    def main(G: T.Tensor((M * FF,), T.float16), U: T.Tensor((M * FF,), T.float16), O: T.Tensor((M * FF,), T.float16)):
        with T.Kernel(T.ceildiv(M * FF, 1024), threads=6) as bx:
            G_sh = T.alloc_shared((1024,), T.float16)
            U_sh = T.alloc_shared((1024,), T.float16)
            O_sh = T.alloc_shared((1024,), T.float16)
            T.copy(G[bx * 1024:(bx + 1) * 1024], G_sh)
            T.copy(U[bx * 1024:(bx + 1) * 1024], U_sh)
            for p in T.Parallel(16):
                for v in T.vectorized(64):
                    O_sh[p * 64 + v] = G_sh[p * 64 + v] * U_sh[p * 64 + v] / (T.float16(1) + T.exp(-G_sh[p * 64 + v]))
            T.copy(O_sh, O[bx * 1024:(bx + 1) * 1024])
```

### pipe：双缓冲实验（已存档，代码已移除）

该变体曾用函数名特判（`func_name == "attnops_tl_silu_pipe"` 走整 kernel
专用 lowering）实现双槽 ping-pong staging。这种"按名字认算子套模板"的
做法已从 emitter 中删除，pipe 变体随之移除；skel 侧已部署的
`attnops_tl_silu_pipe` 保留不回滚（idl 只增不改），但不再由生成物产生。

真机结论（仍然有效）：direct `15.53 GB/s`（`max_rel=0.0027`）最快；
vtcm `15.11 GB/s`，staging 没有复用且无有效重叠，略亏；pipe
`11.08 GB/s`，双缓冲同步开销反噬。轻量流式算子直读最优，staging 只在
有复用时赚。未来若需要生产者/消费者重叠，应走正式的 `T.Pipelined`
语义（见"已知边界"），而不是函数名特判。

## 已知边界

- `execution_backend="aot"` 的 cache dispatch 尚未接入；当前脚本直接走 `engine.lower(...).kernel_source` 拿 C 文本。
- `T.Pipelined` 还未支持；pipe 实验（函数名特判 lowering）已随模板清理移除，正式 pipeline 语义待做。
- 滞留叶片清单：`state_x2_matvec128`、`forward_solve32`、`state_update32`。它们仍是参数化复合叶片，原因见 GDN 一节。
