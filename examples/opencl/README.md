# OpenCL 后端示例

本目录展示 TileLang 的实验性 `target="opencl"` 后端：用户仍在 Python
里描述张量、循环、`T.alloc_shared` / `T.copy` / `T.gemm`，lower 后拿到
**OpenCL C kernel 文本**。生成的 `.cl` 不是直接在本机运行，而是部署到
OnePlus 13 / SM8750 的 Adreno OpenCL 驱动上编译、执行和对拍。语法、scope
映射与当前限制见 [`docs/opencl/00_syntax_and_rules.md`](../../docs/opencl/00_syntax_and_rules.md)。

## 目录结构（hexagon 标准，2026-09-22 起）

每个标准 kernel 一个目录，三个文件——TileLang kernel 生成脚本、独立
host C 测试（自包含 OpenCL harness，fp64 参考对拍）、生成的 kernel 源码：

```
gemm/     gemm_nt.py   gemm_nt_test.c   out/gemm_nt.cl   # GR fp16 kn, 真机 1.34-1.40 TFLOPS
rmsnorm/  rmsnorm.py   rmsnorm_test.c   out/rmsnorm.cl
silu/     silu.py      silu_test.c      out/silu.cl
experimental/   # 早期 texture/staging 探索脚本（buffer_staging, gemm_nt_tex* 等）
tests/          # probe_l0/l1/rr 形态测试 + run_oneplus13.sh 一键部署
out/            # 历史生成的 golden .cl 归档
```

标准产物再生命令（GEMM 为当前最快配置）：

```bash
python examples/opencl/gemm/gemm_nt.py --impl grgemm --b-layout kn --accum fp16 \
  --m 1024 --n 2560 --k 2560 --bm 32 --bn 128 --bk 64 --threads 64
python examples/opencl/rmsnorm/rmsnorm.py
python examples/opencl/silu/silu.py
```

test.c 编译（Android 交叉，OpenCL-Headers 在 backend/gpu/OpenCL-Headers）：

```bash
$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang \
  -target aarch64-linux-android26 -march=armv8.2-a+dotprod+fp16 -std=c11 -O3 -fPIE -pie \
  -I/root/project/backend/gpu/OpenCL-Headers examples/opencl/gemm/gemm_nt_test.c -ldl -lm -o gemm_nt_test
```

## 历史运行方法（tl_probe 通用 runner）

设备侧验证也可以用主仓库通用 runner
`/root/project/backend/gpu/tl_probe/tl_probe.c`，runner 在手机上 dlopen
OpenCL、编译 `.cl`、运行 kernel，并用 fp64 reference + rms-scaled
`max_rel < 0.1` 判据对拍。

| 文件夹 | kernel 脚本 | 设备对拍 runner | 生成物 | 一键命令 |
|---|---|---|---|---|
| `silu/` | `silu.py` | `tests/run_oneplus13.sh` (`silu`) | `silu/out/silu.cl` | `bash examples/opencl/tests/run_oneplus13.sh` |
| `rmsnorm/` | `rmsnorm.py` | `tests/run_oneplus13.sh` (`rmsnorm`) | `rmsnorm/out/rmsnorm.cl` | `bash examples/opencl/tests/run_oneplus13.sh` |
| `gemm/` | `gemm_nt.py` | `tests/run_oneplus13.sh` (`gemm`) | `gemm/out/gemm_nt.cl` | `bash examples/opencl/tests/run_oneplus13.sh` |
| `experimental/` | `gemm_nt_texstage.py` | `tl_probe` (`gemm_texstage`) | `out/gemm_nt_texstage_*.cl` | 手机:`./tl_probe gemm_nt_texstage_512_bk16.cl gemm_nt_texstage_kernel gemm_texstage` |
| `experimental/` | `gemm_nt_texstaged.py` | `tl_probe` (`gemm_texstage`) | `out/gemm_nt_texstaged_*.cl` | 手机:`./tl_probe gemm_nt_texstaged_512_512_512_64x128_bk16.cl gemm_nt_texstaged_kernel_kernel gemm_texstage` |
| `experimental/` | `texture_copy.py` | `tests/run_oneplus13.sh` (`texcopy`) | `out/texture_copy.cl` | `bash examples/opencl/tests/run_oneplus13.sh` |
| `experimental/` | `texture_staging.py` | `tl_probe` (`texstage`) | `out/texture_staging.cl` | 手机:`./tl_probe texture_staging.cl texture_staging_kernel_kernel texstage` |

从仓库根目录运行，显式设置 `PYTHONPATH` 并使用仓库内虚拟环境：

```bash
PYTHONPATH=/root/project/tilelang /root/project/tilelang/.venv/bin/python examples/opencl/silu/silu.py
PYTHONPATH=/root/project/tilelang /root/project/tilelang/.venv/bin/python examples/opencl/rmsnorm/rmsnorm.py
PYTHONPATH=/root/project/tilelang /root/project/tilelang/.venv/bin/python examples/opencl/gemm/gemm_nt.py
PYTHONPATH=/root/project/tilelang /root/project/tilelang/.venv/bin/python examples/opencl/experimental/texture_copy.py
```

最小管线冒烟回归（不依赖手机，只验证 OpenCL source emission 形态）：

```bash
PYTHONPATH=/root/project/tilelang /root/project/tilelang/.venv/bin/python examples/opencl/tests/probe_l0.py
PYTHONPATH=/root/project/tilelang /root/project/tilelang/.venv/bin/python examples/opencl/tests/probe_l1.py
```

`probe_l0.py` 覆盖 fp32 128×128 elementwise (`B[i]=A[i]+1`，每 work-item 一个元素)；
`probe_l1.py` 覆盖 fp32 128×256→128 行归约（`T.alloc_shared` + barrier 树归约）。

成功时关键输出包含 `EMIT_OK ...`、`SOURCE_PATH ...` 和 `CLANG_SYNTAX_RC 0`
（如果本机无 clang 可加 `--skip-clang`）。形状参数可用命令行覆盖，但当前
OpenCL 生成代码没有边界保护，launch 尺寸必须整除数据规模：例如 SiLU 的
元素数需整除 `--threads`，RMSNorm 的 `--cols` 需整除 `--threads`，GEMM 的
`M/N/K` 需整除对应 tile。

一键生成、部署、md5 校验并在 OnePlus 13 后台运行：

```bash
bash examples/opencl/tests/run_oneplus13.sh
```

手机侧默认目录为 `~/tl_opencl/`，运行环境为
`LD_LIBRARY_PATH=.:/system/lib64:/vendor/lib64`。

## SiLU (`silu/silu.py`)

用户写法关键代码段如下：每个 work-item 处理一个 fp32 元素，签名为
`(__global float* in, __global float* out)`。

```python
with T.Kernel(T.ceildiv(nelem, threads), threads=threads) as bx:
    tx = T.get_thread_binding(0)
    idx = bx * threads + tx
    x = A[idx]
    O[idx] = x / (T.float32(1.0) + T.exp(-x))
```

默认 launch：`local=256`，`global=8192*960`。

## RMSNorm (`rmsnorm/rmsnorm.py`)

一行一个 work-group，`T.alloc_shared((threads,), "float32")` 保存局部平方和，
经 barrier 树归约得到 `scale=rsqrt(mean(x^2)+eps)`，再回写整行。签名为
`(__global float* in, __global float* out, int ncols, float eps)`。

```python
with T.Kernel(rows, threads=threads) as row:
    tx = T.get_thread_binding(0)
    smem = T.alloc_shared((threads,), "float32")
    smem[tx] = T.float32(0.0)
    for i in T.serial(cols // threads):
        col = i * threads + tx
        x = A[row, col]
        smem[tx] = smem[tx] + x * x
    T.sync_threads()
    # tree reduce in smem, then write O[row, col] = A[row, col] * scale
```

默认 launch：`local=256`，`global=960*256`。

## GEMM_NT (`gemm/gemm_nt.py`)

A 是 row-major `(M,K)`，B 是 NT 权重 `(N,K)`；脚本使用 `T.copy` 搬运
shared tile，并调用 `T.gemm(..., transpose_B=True)`。签名为
`(__global half* A, __global half* B, __global half* C)`。

```python
with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), threads=threads) as (bx, by):
    A_shared = T.alloc_shared((bm, bk), "float16")
    B_shared = T.alloc_shared((bn, bk), "float16")
    C_accum = T.alloc_shared((bm, bn), "float32")
    T.clear(C_accum)
    for ko in T.serial(T.ceildiv(K, bk)):
        T.copy(A[by * bm, ko * bk], A_shared)
        T.copy(B[bx * bn, ko * bk], B_shared)
        T.sync_threads()
        T.gemm(A_shared, B_shared, C_accum, transpose_B=True)
        T.sync_threads()
    T.copy(C_accum, C[by * bm, bx * bn])
```

默认 launch：`local=(256,1)`，`global=(32*256,32)`，对应 `M=N=K=512`、
`BM=BN=BK=16`。

### GEMM_NT fragment direct-global variant

`gemm_nt.py --impl fragment` 是纯 TileLang IR 的 direct-global 路线：A/B 直接从
`__global` 读取，不使用 `T.alloc_shared`、`T.copy` 或 barrier；每个 work-item 负责
一个 8×8 输出 tile，累加器为 `T.alloc_fragment((8, 8), "float32")`。OpenCL lowering
在 `UnrollLoop` 之后运行 `VectorizePrivateFragment`，识别所有 fragment 访问均为
编译期常量的形态，并在 codegen 前选择寄存器/向量 lowering：8 个 `float8 acc_*`，
A/B 按 K 维 `vload4`，C 用 `vstore8`，避免 Adreno 编译器把 `float acc[64]` 当作
可寻址私有数组而 spill。

`--b-layout kn` 表示 host 传入预转置的 `Bt[K,N]` buffer。该路径把 B 的 inner load
改为每个 K lane 一次 contiguous `vload8`，内链与手写 `hgemm_buf_8x8` 对齐；默认
`--b-layout nk` 保持旧 ABI(`B[N,K]`)和旧 gather 生成物。

生成命令示例：

```bash
PYTHONPATH=/root/project/tilelang /root/project/tilelang/.venv/bin/python \
  examples/opencl/gemm/gemm_nt.py --impl fragment \
  --m 512 --n 512 --k 512 --bm 64 --bn 128 --bk 64 --threads 128 \
  --out examples/opencl/out/gemm_nt_fragment_512_512_512_64x128_bk64.cl
```

OnePlus 13 / Adreno 830 真机 `tl_probe gemm` kernel-only event 口径，fp64 reference，
rms-scaled `max_rel < 0.1`：

| 形状 | tile | ms | TFLOPS | max_rel |
|---|---:|---:|---:|---:|
| 512³ | 64×64 / 64 threads | 0.608 | 0.441 | 0.000470 |
| 512³ | 64×128 / 128 threads | 0.546 | 0.492 | 0.000470 |
| 512³ | 32×128 / 64 threads | 0.628 | 0.428 | 0.000470 |
| 512³ | 128×64 / 128 threads | **0.529** | **0.507** | 0.000470 |
| 1024×2560×2560 | 64×64 / 64 threads | **24.627** | **0.545** | 0.000473 |
| 1024×2560×2560 | 64×128 / 128 threads | 28.312 | 0.474 | 0.000473 |
| 1024×2560×2560 | 32×128 / 64 threads | 59.630 | 0.225 | 0.000473 |
| 1024×2560×2560 | 128×64 / 128 threads | 27.346 | 0.491 | 0.000473 |
| 1024×2560×2560 (`Bt[K,N]`) | 64×64 / 64 threads | 19.834 | 0.677 | 0.000473 |
| 1024×2560×2560 (`Bt[K,N]`) | 128×64 / 128 threads | 16.432 | 0.817 | 0.000464 |
| 1024×2560×2560 (`Bt[K,N]`) | 64×128 / 128 threads | 12.462 | 1.077 | 0.000464 |
| 1024×2560×2560 (`Bt[K,N]`) | 32×128 / 64 threads | **9.784** | **1.372** | 0.000464 |
| 512³ (`Bt[K,N]`) | 64×64 / 64 threads | 0.272 | 0.985 | 0.000470 |
| 512³ (`Bt[K,N]`) | 64×128 / 128 threads | 0.234 | 1.146 | 0.000470 |
| 512³ (`Bt[K,N]`) | 32×128 / 64 threads | **0.225** | **1.191** | 0.000470 |

结论：旧 `B[N,K]` fragment direct-global 路线正确且寄存器形态达标，但 B gather
导致全局访存重放；切到 `Bt[K,N]` contiguous `half8` load 后，prod 形状达到 1.372T，
高于手写 buffer 8×8 的 1.025T（同为 fp32 accumulate / buffer direct load）。

### GEMM_NT tiled fragment buffer variant

`gemm_nt.py --impl tiled_fragment` 是 buffer 路线的纯 TileLang IR 版本：`T.copy`
把 A/B K tile 搬进 `T.alloc_shared`，`T.sync_threads()` 后用 `T.gemm` 累加到
`T.alloc_fragment((8, 8), "float32")`，每个 work-item 负责一个 8×8 输出 tile。
默认 `--b-layout nk` 保持旧 ABI(`B[N,K]`)并调用 `transpose_B=True`；`--b-layout kn`
表示 host 传入预转置 `Bt[K,N]`，B tile 以 `(BK,BN)` 顺序搬进 shared，调用标准
NN 内链。`VectorizePrivateFragment` 在 TIR 管线中识别该形态，codegen 前选择与
手写 buffer 8×8 同构的向量 lowering：保留 shared staging，inner K loop 用 shared
`half8` 读、`float8 acc[8]` 累加和 `vstore8` 写回。

生成物形态：`nk` 路径的 A/B global→shared staging 为每 work-item 多次 `vload4` +
scalar shared store；`kn` 路径的 B global→shared staging 为每 work-item 连续
`vload8` + `vstore8` 宽 copy。两种路径的 inner K loop 每步各一次 shared `half8`
load（A/B），8 个 `float8` fp32 累加器，每 BK tile 2 个 `barrier(CLK_LOCAL_MEM_FENCE)`。

OnePlus 13 / Adreno 830 真机 `tl_probe gemm` kernel-only event 口径，fp64 reference，
rms-scaled `max_rel < 0.1`；全部 case `bad 0/4096`：

| 形状 | tile/BK | ms | TFLOPS | max_rel |
|---|---:|---:|---:|---:|
| 512³ | 64×64/BK16 | 0.463 | 0.580 | 0.000470 |
| 512³ | 64×64/BK32 | 0.658 | 0.408 | 0.000470 |
| 512³ | 64×64/BK64 | 1.399 | 0.192 | 0.000470 |
| 512³ | 64×128/BK16 | 0.414 | 0.649 | 0.000470 |
| 512³ | 64×128/BK32 | 0.605 | 0.443 | 0.000470 |
| 512³ | 64×128/BK64 | 0.675 | 0.398 | 0.000470 |
| 512³ | 128×64/BK16 | **0.395** | **0.680** | 0.000470 |
| 512³ | 128×64/BK32 | 0.608 | 0.441 | 0.000470 |
| 512³ | 128×64/BK64 | 0.680 | 0.395 | 0.000470 |
| 512³ | 32×128/BK16 | 0.658 | 0.408 | 0.000470 |
| 512³ | 32×128/BK32 | 0.881 | 0.305 | 0.000470 |
| 512³ | 32×128/BK64 | 1.567 | 0.171 | 0.000470 |
| 1024×2560×2560 | 64×64/BK16 | 20.461 | 0.656 | 0.000473 |
| 1024×2560×2560 | 64×64/BK32 | 32.195 | 0.417 | 0.000473 |
| 1024×2560×2560 | 64×64/BK64 | 73.999 | 0.181 | 0.000473 |
| 1024×2560×2560 | 64×128/BK16 | 17.765 | 0.756 | 0.000473 |
| 1024×2560×2560 | 64×128/BK32 | 24.697 | 0.543 | 0.000473 |
| 1024×2560×2560 | 64×128/BK64 | 32.085 | 0.418 | 0.000473 |
| 1024×2560×2560 | 128×64/BK16 | **16.840** | **0.797** | 0.000473 |
| 1024×2560×2560 | 128×64/BK32 | 24.798 | 0.541 | 0.000473 |
| 1024×2560×2560 | 128×64/BK64 | 32.490 | 0.413 | 0.000473 |
| 1024×2560×2560 | 32×128/BK16 | 25.551 | 0.525 | 0.000473 |
| 1024×2560×2560 | 32×128/BK32 | 45.551 | 0.295 | 0.000473 |
| 1024×2560×2560 | 32×128/BK64 | 82.904 | 0.162 | 0.000473 |

`--b-layout kn` 真机扫描（同一 runner；全部 case `bad 0/4096`）：

| 形状 (`Bt[K,N]`) | tile/BK | ms | TFLOPS | max_rel |
|---|---:|---:|---:|---:|
| 512³ | 64×64/BK16 | 0.360 | 0.746 | 0.000468 |
| 512³ | 64×64/BK32 | 0.497 | 0.540 | 0.000468 |
| 512³ | 64×64/BK64 | 0.603 | 0.445 | 0.000468 |
| 512³ | 64×128/BK16 | **0.311** | **0.863** | 0.000468 |
| 512³ | 64×128/BK32 | 0.426 | 0.629 | 0.000468 |
| 512³ | 64×128/BK64 | 0.501 | 0.536 | 0.000468 |
| 512³ | 128×64/BK16 | 0.350 | 0.768 | 0.000468 |
| 512³ | 128×64/BK32 | 0.535 | 0.502 | 0.000468 |
| 512³ | 128×64/BK64 | 0.648 | 0.414 | 0.000468 |
| 512³ | 32×128/BK16 | 0.336 | 0.800 | 0.000468 |
| 512³ | 32×128/BK32 | 0.414 | 0.649 | 0.000468 |
| 512³ | 32×128/BK64 | 0.982 | 0.273 | 0.000468 |
| 1024×2560×2560 | 64×64/BK16 | 15.721 | 0.854 | 0.000466 |
| 1024×2560×2560 | 64×64/BK32 | 21.014 | 0.639 | 0.000466 |
| 1024×2560×2560 | 64×64/BK64 | 30.349 | 0.442 | 0.000466 |
| 1024×2560×2560 | 64×128/BK16 | **13.464** | **0.997** | 0.000466 |
| 1024×2560×2560 | 64×128/BK32 | 16.736 | 0.802 | 0.000466 |
| 1024×2560×2560 | 64×128/BK64 | 24.426 | 0.549 | 0.000466 |
| 1024×2560×2560 | 128×64/BK16 | 15.142 | 0.886 | 0.000466 |
| 1024×2560×2560 | 128×64/BK32 | 20.360 | 0.659 | 0.000466 |
| 1024×2560×2560 | 128×64/BK64 | 30.235 | 0.444 | 0.000466 |
| 1024×2560×2560 | 32×128/BK16 | 14.805 | 0.907 | 0.000466 |
| 1024×2560×2560 | 32×128/BK32 | 21.207 | 0.633 | 0.000466 |
| 1024×2560×2560 | 32×128/BK64 | 52.195 | 0.257 | 0.000466 |

相对锚点：`nk` prod 形状 0.797T，比 fragment direct-global `nk` 0.545T 快 46%，
高于 local_tiled escape 0.750T，但仍低于手写 buffer 8×8 1.025T 约 22%。`kn` +
宽 staging 后 prod 最优 0.997T，比 `nk` tiled 0.797T 快 25%，但仍低于 `kn` direct
fragment 1.372T 约 27%；因此 B 的 staging 在 `kn` 下没有带来净收益，额外的
barrier/shared traffic 抵消了对全局连续读的复用。BK16 仍明显最优；BK32/64 退化来自
local footprint 变大、每 work-group occupancy/issue 下降。

## Texture copy (`experimental/texture_copy.py`)

输入 `B` 使用 `T.Tensor((H, W, 4), "float16", scope="global.texture")`，
OpenCL lowering 会在 `LowerTileOp` 后、`LowerAccessPtr` 前对 texture scope
buffer 运行 TVM Adreno `TextureFlatten`，最终生成 `image2d_array_t` 参数和
`read_imageh` 读取。形状约定为 height、width、channel，其中 fp16 texture
末维必须为 4（RGBA/64bit）。kernel 逐元素执行：

```python
with T.Kernel(width, height, threads=4) as (w, h):
    c = T.get_thread_binding(0)
    O[h, w * 4 + c] = B[h, w, c] * T.float16(2.0)
```

默认 launch：`local=(4,1)`，`global=(128*4,64)`，手机 runner 用
`clCreateImage` 创建 `CL_MEM_OBJECT_IMAGE2D_ARRAY`、`CL_RGBA/CL_HALF_FLOAT`、
`depth=1` 后上传并对拍 `out = B * 2`。

## Texture staging (`experimental/texture_staging.py`)

`T.copy(X[row, :, :], staged)` 把一整行 RGBA texel 从 texture 搬进
`__local`，barrier 后再做行归约。OpenCL copy lowering 对 texture 源有特判
（`src/opencl/op/copy.cc` `LowerTextureCopy`）：一个 work-item 对一个
texel，channel 维标量内循环经 `TextureFlatten` + `VectorizeLoop` 合并成
**每线程一次 `READ_IMAGEH`(half4)+ 一次 `vstore4` 写 `__local`**；通用
SIMT copy 会把线程映射到元素，相邻 4 lane 重复 READ_IMAGEH 同一 texel。
真机 OnePlus 13 PASS：`max_rel=2.1e-07`（64×128 fp16，tl_probe
`texstage` runner）。注意 kernel 脚本不能设 `tirx.disable_vectorize`，
否则 texel 特判发出的 channel 内循环不会被合并。

## GEMM_NT B texture staging (`experimental/gemm_nt_texstage.py` / `gemm_nt_texstaged.py`)

这是 GEMM 的 texture→`__local` staging 探针：A 保持 row-major `__global`
buffer，B 按 `Btex[k, n//4, n%4] = B_nt[n, k]` 存入 RGBA fp16
`image2d_t`，每个 K 分块把 B tile 从 texture 搬进 `__local half Bs` 后，
用 `float8` 寄存器内链计算 8×8 输出。

- `gemm_nt_texstage.py` 是手写参考源码生成器，用来扫 tile/BK 并标定上限。
- `gemm_nt_texstaged.py` 是 TileLang IR 版：B 声明为 `scope="global.texture"`，
  K 分块内用 `T.copy` 触发 texture-source lowering；当前 texture GEMM 路线只要求
  对拍正确，fragment pass 不匹配 texture scope。`gemm_nt_tex.py` 的 direct texture
  探针仍保留 codegen 里的 acc-only fallback（`float acc[64]` → `float8 acc_0..7`）避免
  Adreno 私有数组 spill；它没有替换 GEMM body/staging/load-store 结构。

staging 段形态（生成物可 grep `READ_IMAGEH` / `vstore4`）：

```c
for (int v = lid; v < TL_BN * TL_BK / 4; v += TL_WG) {
    ...
    half4 val = READ_IMAGEH(Bi, smp, (int2)(cc4, kb + kk));
    vstore4(val, 0, &Bs[kk][c4 * 4]);
}
```

OnePlus 13 真机 kernel-only event 口径（`tl_probe gemm_texstage`，fp64
reference，rms-scaled `max_rel<0.1`）：

| 形状 | 最优 tile | ms | TFLOPS | max_rel |
|---|---:|---:|---:|---:|
| 512³ 手写 | 128×64/BK16 | 0.373 | 0.720 | 0.000471 |
| 1024×2560×2560 手写 | 128×64/BK16 | 15.989 | 0.839 | 0.000474 |
| 512³ IR | 64×128/BK16 | 0.868 | 0.309 | 0.000471 |
| 1024×2560×2560 IR | 64×128/BK16 | 38.554 | 0.348 | 0.000474 |

结论：手写 texture staging 比手写 local64×128 的 0.598T 快约 1.4×，但仍只有
image 直读 8×8 的 1.721T 的 49%。IR 版正确触发每线程一次 `READ_IMAGEH` +
`vstore4`，但当前仅 0.35T；瓶颈来自每 K tile 的 `__local` 填充、双 barrier 与
generated shared alias/launch 结构，不如直接在寄存器内从 image 复用 B。该路径
适合作为 texture-copy lowering 的验证和 local 路线对照，不建议作为生产 GEMM 默认路径。

## 已知边界

- `execution_backend="aot"` 的 cache dispatch 尚未作为示例入口；当前脚本直接
  用 `tilelang.lower(...).kernel_source` 取 OpenCL 源码。
- 当前生成代码无 bounds guard，设备验证只覆盖整除形状。
- GEMM 是功能优先的朴素 SIMT FMA 路线，尚未做 Adreno 向量化、autotune 或
  私有矩阵扩展。
- texture 参数只覆盖 fp16 `(H, W, 4)` image2d array 读取示例；写 texture、其它
  channel 数/数据类型暂未作为示例验证。
