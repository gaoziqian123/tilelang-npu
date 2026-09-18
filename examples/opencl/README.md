# OpenCL 后端示例

本目录展示 TileLang 的实验性 `target="opencl"` 后端：用户仍在 Python
里描述张量、循环、`T.alloc_shared` / `T.copy` / `T.gemm`，lower 后拿到
**OpenCL C kernel 文本**。生成的 `.cl` 不是直接在本机运行，而是部署到
OnePlus 13 / SM8750 的 Adreno OpenCL 驱动上编译、执行和对拍。语法、scope
映射与当前限制见 [`docs/opencl/00_syntax_and_rules.md`](../../docs/opencl/00_syntax_and_rules.md)。

## 目录结构、环境与运行方法

本目录现在按职责拆分：`kernels/` 放 TileLang kernel 生成脚本，顶层
`out/` 放生成的 OpenCL C golden 文件，`tests/` 放 OnePlus 13 设备侧
部署与对拍脚本。设备侧验证使用主仓库通用 runner
`/root/project/backend/gpu/tl_probe/tl_probe.c`，runner 在手机上 dlopen
OpenCL、编译 `.cl`、运行 kernel，并用 fp64 reference + rms-scaled
`max_rel < 0.1` 判据对拍。

| 文件夹 | kernel 脚本 | 设备对拍 runner | 生成物 | 一键命令 |
|---|---|---|---|---|
| `kernels/` | `silu.py` | `tests/run_oneplus13.sh` (`silu`) | `out/silu.cl` | `bash examples/opencl/tests/run_oneplus13.sh` |
| `kernels/` | `rmsnorm.py` | `tests/run_oneplus13.sh` (`rmsnorm`) | `out/rmsnorm.cl` | `bash examples/opencl/tests/run_oneplus13.sh` |
| `kernels/` | `gemm_nt.py` | `tests/run_oneplus13.sh` (`gemm`) | `out/gemm_nt.cl` | `bash examples/opencl/tests/run_oneplus13.sh` |

从仓库根目录运行，显式设置 `PYTHONPATH` 并使用仓库内虚拟环境：

```bash
PYTHONPATH=/root/project/tilelang /root/project/tilelang/.venv/bin/python examples/opencl/kernels/silu.py
PYTHONPATH=/root/project/tilelang /root/project/tilelang/.venv/bin/python examples/opencl/kernels/rmsnorm.py
PYTHONPATH=/root/project/tilelang /root/project/tilelang/.venv/bin/python examples/opencl/kernels/gemm_nt.py
```

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

## SiLU (`kernels/silu.py`)

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

## RMSNorm (`kernels/rmsnorm.py`)

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

## GEMM_NT (`kernels/gemm_nt.py`)

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

## 已知边界

- `execution_backend="aot"` 的 cache dispatch 尚未作为示例入口；当前脚本直接
  用 `tilelang.lower(...).kernel_source` 取 OpenCL 源码。
- 当前生成代码无 bounds guard，设备验证只覆盖整除形状。
- GEMM 是功能优先的朴素 SIMT FMA 路线，尚未做 Adreno 向量化、autotune 或
  私有矩阵扩展。
