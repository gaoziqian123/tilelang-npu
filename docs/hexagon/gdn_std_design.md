# 标准 TileLang GDN prefill 设计任务书(gdn_std)

目标:纯标准 TileLang 语言构造写 GDN prefill(T=1024, Hk=16, Hv=32, D=128, chunk=32),
OnePlus 13 真机 fp64 对拍 max_rel<0.1,总耗时 < 45.35ms(手写 attnops_gdn.c 基线)。

## 写法边界

只用 T.copy / T.gemm / T.parallel / T.serial(含动态 extent)/ T.vectorized /
T.alloc_shared / T.alloc_fragment / T.hexagon.exp_fp32 等语言构造。
不在用户层手写 C 字符串块,不直接调 hrt_tlgdn_* 数学叶片(matvec/solve/affine/decay)。
emitter 从标准构造生成的 hrt_* 调用(staging、HMX mm、向量化 HVX)是编译器本职,不算违规。

## 编译器能力(已就绪)

运行环境:
```
cd /root/project/tilelang
export LD_LIBRARY_PATH=/root/project/tilelang/build/lib
export PYTHONPATH=/root/project/tilelang/build/lib:/root/project/tilelang/3rdparty/tvm/python:/root/project/tilelang
# python = .venv/bin/python
```

- 通用渲染路径(_render_generic_c):T.parallel(N) 段 → 提升 worker + attnops_pool_run_ctx
  (同步 join);其余段(串行循环/T.copy/T.gemm)主线程内联;外层可包 T.serial chunk 循环;
  入口 ABI 自动推导(slab + 符号维度标量 + abl + prof 尾区)。结构参照 /tmp/tl_generic_shell_test.py。
- T.gemm(NT,fp16 操作数在 VTCM AH/WH,M/N/K %32)主线程通用 recipe:per-call 静态 M/N/K,
  copy_acc_rm 写回 VTCM RM fp16 或 global。R4:gemm 不许出现在 T.parallel 段(编译期强制)。
- copy recipe(T.copy 按 layout_map 自动路由):global fp16 rm → vtcm ah/wh(copy_rm_ah);
  fp32 rm → vtcm ah fp16(copy_f32_ah,trans=1 表示源按 [K,M] rm 读、逻辑输出 [M,K]);
  acc → rm(copy_acc_rm);rm↔rm(copy_ddr);不命中回退普通向量化循环。
- alloc_shared(layout="ah"/"wh") 双轨(annotation + scope 后缀)。
- 任意 rank row-major 下标;动态 extent 串行循环(三角 j<i+1)合法。
- hexagon_rt.h staging 已 HVX 向量化(逐位等价):(32,32)trans0 ≈ 275ns,
  (128,32)trans1 ≈ 5.8µs,(128,128)trans1 ≈ 23µs。
  acc 单价实测:clear 3.4ns / read 7.2ns / mm 7.6ns。
- 关键结构约束(实测驱动):每 chunk 重 stage 128×128 fp32 state(23µs×1024≈23.5ms)不可行。
  所以 state fp32 主本([Hv,128,128] = 2MB)与 fp16 AH 转置影子(1MB)驻留 VTCM;
  state 更新相位(S = eGC*S + h2f(ΔS) 向量 FMA)顺手产出下一 chunk 的 AH 影子;
  c=0 从 S0 初始化并 stage 一次。

## 数学(oracle = /root/project/backend/npu/attn/skel/src/attnops_gdn.c)

每 chunk(C=32)、每 v-head:
```
G_i = cumsum(g)  (clamp -60);  eG=exp(G), eGinv=exp(-G), eGC=exp(G_31)
A[i,j] = β_i eG_i eGinv_j (k_i·k_j)   j<i
P[i,j] = eG_i eGinv_j (q_i·k_j)       j≤i
w0 = K_c @ S;  u = Q_c @ S            ([32,128] each)
w = β⊙v − β⊙eG⊙w0;  w = (I+tril(A))^{-1} w   (32 行前向代换)
o_i = eG_i u_i + Σ_{j≤i} P[i,j] w_j
S = eGC·S + ((eGC·eGinv)⊙K_c)ᵀ @ w
```
GEMM 化:decay 折进行缩放 Kh=diag(β·eG)K、Kl=diag(eGinv)K、Qh=diag(eG)Q;
[A;P]=[Kh;Qh]@Klᵀ(M=64,N=32,K=128);[w0;u]=[K;Q]@S(M=64,N=128,K=128,B=S 转置由影子布局直接供);
fwdo O'=tril(P)@w(M=32,N=128,K=32);fwdh ΔS=k̂ᵀ@w(M=128,N=128,K=32,k̂=(eGC·eGinv)⊙K 已 fold)。

## 建议 kernel 结构(route b 单 kernel,可调整但要说明理由)

单 RPC 入口,body = for c in T.serial(32),每 chunk 相位:
1. pool(Hv):Q/K/V chunk fp16→fp32;cumsum(串行标量,eG/eGinv/β 写 global——VTCM 标量写禁止);
   行缩放产 Kh/Kl/Qh(fp32 rm VTCM);k̂ fold
2. 主线程 serial hv:staging(KhKlQh→AH;c=0 时 S 影子)+ T.gemm AP/WU + copy_acc_rm 写回 VTCM RM fp16
3. pool(Hv):A/P 上三角清零 + affine + solve(前向代换,串行+谓词+vectorized 128)
4. 主线程 serial hv:P RM→AH(copy_rm_ah)+ w→AH trans(copy_f32_ah)+ T.gemm fwdo + acc 写回
5. pool(Hv):O combine(o = eG·u + h2f(O'),f2h 写 O slab)
6. 主线程 serial hv:k̂→AH trans + T.gemm fwdh + acc 写回 VTCM RM fp16
7. pool(Hv):state 更新(S = eGC·S + h2f(ΔS))顺手产下一 chunk fp16 AH 转置影子;last chunk 写 S1

VTCM 预算(R3):state 2MB + 影子 1MB + 中间张量,总量 <6MB;装不下就把 kf/qf/vf/w 移 global slab。

## 流程

1. 写 examples/hexagon/gdn_std.py(骨架参照 gdn_prefill.py / gdn_split.py)。
   生成 → 审查 C 结构 → hexagon-clang 语法过。
2. skel 集成走 /root/project/backend/npu/attn/collect_tilelang.sh 通道(只追加;
   attnops.idl 加 tl_gdn_std 入口,重建 stub/skel)。注意:build.sh 的 host 阶段在服务器缺
   NDK clang,skel 在服务器编,host 测试二进制按前例在手机 Termux clang 下编
   (参照 gdn_split_test 的构建方式)。
3. 写 gdn_std_test.c(参照 gdn_split_test.c 的 rpcmem/slab/fp64 参考模式;
   fp64 参考照 attnops_gdn.c 数学在 host 侧跑)。判据 max_rel<0.1(rms 缩放)。
4. 手机 detach 运行(nohup + .done 轮询,日志拉回):先 1 iter 对拍,再多 iter 计时,
   prof 尾区分相位计时。
5. 性能迭代到 <45.35ms。杠杆(按预期收益):staging 挪 pool 相位与 HMX 重叠
   (pool_start/pool_join 在 hexagon_rt.h;编译器侧若缺表达就记录缺口);合并 pool 段减少
   相位切换;w 的 AH staging 在 fwdo/fwdh 间复用;kkt 与 WU 的 A 操作数合并 stage。
6. 对照基线:手写 TL_GDN(tl_gdn_test)46.6ms;gdn_split 45.7ms。

## 汇报要求

- M1:kernel 结构定稿(相位划分/数据布局/VTCM 预算表)+ codegen + clang 结果
- M2:skel 集成 + 真机对拍(max_rel_o/max_rel_s1)
- M3:真机计时 + prof 分相位分解 + 迭代记录
- 最终:是否达标;未达标则 prof 数据 + 瓶颈分析 + 下一步建议
- 编译器缺口(标准构造降不下去/生成错误代码)单独列出,带最小复现
- 不要 commit;中间产物保留在工作树
