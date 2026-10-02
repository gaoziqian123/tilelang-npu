# HVX intrinsic 统一层设计草案 v0.1

> 状态：**仅供设计评审，尚未获准实现**。本文不新增可执行 API，不修改代码、
> 测试或精度阈值，不部署、不跑手机、不提交。除明确标注“已有”的接口外，
> 下列名字、签名、模式和版本号都是**拟议名称**，不是当前可调用 API。
> 设计接受之后仍需逐项实现、目标 ASM 审计及独立验证；文档不是完成证明。

## 1. 边界与现状

目标是以通用 map / reduce / cast / permutation 原语表达计算，统一其数值、
布局、所有权和向量化契约；不是把 FA softmax 或整个算子塞进一个 helper。
逻辑层应能表达任意**静态长度**，物理层固定 128B。表达能力不等于当前可
编译能力：未支持的尾部、映射或模式必须拒绝；动态逻辑长度留待后续设计。

| 项目 | 当前事实与边界 | 本草案新增目标 |
|---|---|---|
| `T.transform(src, dst)` | 已有真实入口与 Hexagon 支持，不是空占位；等逻辑形状、copy 非 alias 语义。支持矩阵和限制见下方来源 | 纳入统一 cast/layout 契约，而非重新发明一个同名 API |
| transform 布局/转换 | 当前支持部分静态 rank-2 紧凑布局及 half/float 转换；并非任意 shape/permute | 将可证明映射分解为固定宽度叶 intrinsic |
| HVX 模板 | 已有 `hvx.h`、`exp.h`、`hf_math.h`、reduce/cast 等模板；存在逐 lane 除法及异常值标量处理 | `vector_required` 路径完全禁止数据逐 lane 标量 fallback |
| 编译器 pass | 已有布局、worker、局部 map/reduce 等局部实现 | 本文统一 intrinsic 层及 `VectorLowering` 仍是拟议架构；不能把“lowering1”当成已经落地的统一 pass |
| DMA | 已有同步完成的 `dma_copy_2d_wait` adapter | 异步 submit/ticket/wait 是后续协议，不冒充已有接口 |
| 约 20/21ms FA | 本任务引用的手工模板/手改诊断参照，不是本文新做的测量 | 正式 compiler 发射后才能建立正式基线；不保证达到诊断时间 |

窄范围事实来源（源码/文档阅读，不代表本轮重跑验证）：

- `tilelang/language/copy_op.py`：`T.transform` 及通用 `T.copy` 入口。
- [transform_cast_contract.md](transform_cast_contract.md)：支持矩阵、exact cast、
  O2 审计更正与尚未完成的端到端验证。早期“全部整数实现”的描述已被其中更正取代。
- `src/tl_templates/hexagon/{hvx,exp,hf_math,row_reduce,row_map_reduce,reduce}.h`：
  现有 helper 不能统称为“已经全向量”，特别是 IEEE 异常路径。
- `src/tl_templates/hexagon/dma_copy.h`：真实参数、对齐与范围检查。
- 项目 `.opencode/skills/oneplus13-npu/SKILL.md` 与
  `npu-hetero-mlir/docs/runbook.md`：128B、旋转语义、设备验证纪律及旧 TL 路径退休边界。

## 2. 两层模型与符号

拟议流水线：

```text
logical 静态 map / reduce / cast / layout
  → dtype + 数值模式 + alias/effect 分析
  → layout 与 worker ownership 确定
  → VectorLowering：拆成 128B 物理值、证明地址/掩码/owner
  → 合法的 loop fusion / 调度（不得改变数值依赖）
  → 叶 intrinsic codegen → 实际 -O2 object + ASM 审计
```

这是依赖关系而非已确定的 pass 注册顺序；fusion 可有前后两阶段，但都要重新
校验上述证明。不能在 ownership 未定时猜测哪些逻辑 lane 属于同一寄存器。

| 记号 | 拟议定义 |
|---|---|
| `L<T,N>` | N 为静态整数的逻辑值/区域；N 不必是硬件 lane 数的倍数 |
| `h64` | 一只 128B 寄存器，64 个 fp16 lane |
| `f32x32` | 一只 128B 寄存器，32 个 fp32 lane；禁止用 `f32` 简写以免误认为标量 |
| `i32x32`, `u32x32`, `u8x128` | 同宽整数/原始位视图；跨 lane 数的 bitcast 必须显式 |
| `predLane<T>` | 与 T 物理 lane 对应的布尔谓词；到 HVX byte predicate 的复制规则必须证明 |
| `scalar<T>` | 显式标量，例如地址控制、常量或最终一行的归约结果 |
| `V<T>` | 上述受支持的单个 128B 物理向量，不表示任意宽度 |

HVX 硬件有 64B 模式；**本项目本设计固定 128B**，不得按硬件可选模式自动
生成 64B 数据通路。pair 是两只寄存器，不是偷偷改变单向量宽度。

## 3. intrinsic 详细表（全部新名称均为拟议）

### 3.1 内存、构造、比较与重排

| 类别 / 逻辑 frontend 意图 | 固定物理签名草案 | 必须携带的语义 / 拒绝条件 |
|---|---|---|
| `load(L<T,N>)` | `load_aligned128<T>(ptr) -> V<T>` | 地址 128B 对齐；完整 128B 可读，所属地址空间合法；无隐含 cast/越界读取 |
| `store(dst, x)` | `store_aligned128<T>(ptr, V<T>) -> void` | 完整 128B 可写，唯一写 owner，证明无危险 alias |
| 带尾部读取 | `masked_load<T>(ptr, predLane<T>, fill) -> V<T>` | 显式 `read_only_safe`、`no_overread` 契约：仅有效地址可读、无内存副作用；“先读整向量再 select”不是合法实现。安全尾部实现尚未承诺 |
| 带尾部写入 | `masked_store<T>(ptr, predLane<T>, V<T>) -> void` | 无越界读写、无对无效 lane 的 RMW；不借读邻区实现尾写。未支持就拒绝 |
| 广播 | `splat<T>(scalar<T>) -> V<T>` | 所有 lane 位/数值规则一致；不构造逐 lane 标量数组 |
| 条件选择 | `select(predLane<T>, V<T> a, V<T> b) -> V<T>` | lane 位选择；不代表惰性求值，不能靠它掩盖上游非法 load 或异常计算 |
| 比较 | `compare<T>(a, b, relation, nan_policy) -> predLane<T>` | eq/ne/lt/le/gt/ge；浮点 ordered/unordered 显式，NaN 规则不可继承未知硬件默认 |
| 位重解释 | `bitcast<U>(V<T>) -> V<U>` | 总位数同为 1024，不做数值转换；定义字节序/lane 映射；不可代替 dtype cast |
| 旋转 | `rotate_bytes<T>(V<T>, amount) -> V<T>` | amount 单位为**字节**，0..127；常量或可证明合法的 scalar amount，且为 sizeof(T) 倍数。语义 `out_byte[j]=in_byte[(j+amount)%128]` |
| 静态单向量重排 | `permute<T>(V<T>, static_indices) -> V<T>` | 完整 index 映射；只接受已支持网络并有 lane 证明的模式，否则 reject |
| 静态双向量重排 | `permute2<T>(V<T> a, V<T> b, static_indices) -> V<T>` | index 指向 `[a,b]` 的合法 lane；最多两个输入，不承诺通用 gather/任意 permutation |

`Q6_V_vror_VR(v,64)` 的 64 是 64 **bytes**：对 h64 是旋转 32 lane，
对 f32x32 是 16 lane。不能把 lane offset 原样传给该 intrinsic。
动态地址须在指令发射前通过对齐、范围、乘加溢出、能力上限检查；合法运行时
guard 可拒绝调用，不能未经证明发 aligned load 再指望硬件处理。动态 offset
不等于本期已支持动态逻辑长度。

### 3.2 dtype 转换及精确策略

| 逻辑意图 | 固定物理签名草案 | 契约 |
|---|---|---|
| fp16 → fp32 | `widen_f16_f32(h64 x, order, mode) -> (f32x32 a, f32x32 b)` | `order=linear`：a[i]=x[i], b[i]=x[32+i]；`evenodd`：a[i]=x[2i], b[i]=x[2i+1]；0≤i<32 |
| 两只 fp32 → fp16 | `narrow_pair_f32_f16(f32x32 a, f32x32 b, order, rounding=RNE, special_policy) -> h64` | 与 widen 的 order 互逆；禁止丢掉中间 fp16 舍入；native conversion 如产生交错，必须显式重排 |

拟议模式/策略使用精确名字，而不是含糊的“IEEE-like”：

| 名称草案 | 语义及状态 |
|---|---|
| `strict_exact_v1` widening | finite half 精确扩展，含 subnormal、signed zero、infinity；NaN quiet、保留 sign 及可保留的 payload。以当前 transform 契约为提案起点，不宣称统一层已实现 |
| `rne_gradual_quiet_payload_v1` narrowing | ties-to-even、渐进下溢、保留零符号、overflow→同号 infinity；NaN quiet、保留 sign 和高 payload 位。应冻结独立 bit oracle 后才接受为正式 API |
| `native_target_v1` | 目标/工具链/模式限定的硬件数值行为，不与 strict 自动等价；FTZ、NaN payload、rounding 等未证明项必须标为 unknown，不能靠 device flag 名称推断 |

候选 bit oracle 要写清：half NaN fraction 左移 13 并置 fp32 quiet bit；
fp32 NaN fraction 取高 10 位并置 half quiet bit，sign 保留；有限 narrowing
用独立整数 RNE 判定。这里是**待评审提案**，尤其 sNaN quiet、payload 选择及
是否承诺异常标志尚未决定；不能把 payload 未决定伪装成 strict 已完成。
native 使用域若限制 finite/normal，需静态证明或显式 guard→报错分支，
不得在失败分支偷偷运行标量转换。不得把旧 helper 的逐 lane 异常修补引入
`vector_required` strict 路径。

### 3.3 算术、归约及超越函数

| 逻辑 frontend 意图 | 固定物理签名草案 | 契约 / 首期边界 |
|---|---|---|
| `map(add/sub/mul, ...)` | `add/sub/mul<T>(V<T>, V<T>, mode) -> V<T>` | T 为明确支持的 h64/f32x32 或整数类型；结果 dtype 舍入点固定；整数溢出策略独立声明 |
| 显式乘加 | `fma<T>(V<T> a, V<T> b, V<T> c, mode) -> V<T>` | 只有显式 fma 才允许单次舍入；`mul` 后 `add` 禁止隐式 contraction |
| max/min | `max/min<T>(a,b,nan_policy,zero_policy) -> V<T>` | 明确 NaN propagate 或 number-preferred；signed-zero tie 策略明确。候选 strict max(-0,+0)=+0、min=-0；NaN payload tie 尚待裁定 |
| 寄存器横向 max | `hreduce_max(V<T>, tree_version, nan_policy, zero_policy, seed_spec) -> V<T>` | 输出为全 lane 相同的 **splat vector**；不是 scalar；seed dtype 和插入点显式 |
| 寄存器横向 sum | `hreduce_sum(V<T>, acc_dtype, tree_version, seed_spec) -> V<acc_dtype>` | 首期优先 f32x32→f32x32；half 累加前显式 widen。不同 acc_dtype 的组合需单独定义，不能暗改树 |
| 取单 lane | `extractlane<T>(V<T>, constexpr lane) -> scalar<T>` | lane 固定且在范围内；允许最终行结果/控制用途，不允许重复 extract 形成 elementwise fallback |
| fp32 exp | `exp_f32(f32x32, approx_version, domain) -> f32x32` | 近似算法、有效输入域、underflow/overflow/NaN/inf 行为必须版本化；误差界 **TBD**，不编造数字 |
| 非正 fp16 exp2 | `exp2_f16(h64, approx_version, domain=nonpositive) -> h64` | 有限非正域是候选，准确下界与 -inf 行为待定；需证明/guard，不能把历史经验误差当新 API 保证；误差界 **TBD** |
| 倒数 | `rcp_f32(f32x32, mode, version, domain=positive_finite) -> f32x32` | native/approx 分版本；安全下界、subnormal 和倒数溢出处理待定。不得宣称存在直接的硬件向量 rcp 指令；可能是向量算法组合 |
| 通用除法 | `div(L<T,N>, L<T,N>)` | 不在首期向量 intrinsic 承诺中；不得自动降为逐 lane scalar div，也不得未经语义许可改写 rcp×mul |

归约树是数值 ABI 的一部分。拟议正式 fp32 树 `asc_bytes_v1` 的旋转顺序
为 **4,8,16,32,64 bytes**；现有 legacy/helper 中可见 **64,32,16,8,4**，
命名为 `legacy_desc_bytes_v1` 对照。二者浮点 sum 不保证 bit-identical，
不得悄悄切换。配对方向 `op(acc, rotate(acc, shift))` 也必须固定。
seed 候选规范为先用单位元执行树，再只合并一次 seed，最后 splat；若源程序
指定 seed-first 顺序，必须保持或拒绝，不能把 seed splat 到每个输入 lane
导致重复累加。正式 seed 策略需与逻辑 reduce 契约共同批准。

### 3.4 memory reduction 与 DMA 不是一条 HVX 指令

逻辑 `reduce(memory_region, axis, op, seed, order)` 拆为：

1. 按 owner 和逻辑顺序 load 多只向量，做 lane accumulator；固定 chunk 合并顺序。
2. 显式 horizontal tree，必要时 extract 一个行结果。
3. 仅当轴跨 worker 时产生 partial 存储、发布 barrier、固定跨 worker 合并顺序、
   消费 barrier 与生命周期结束；单 owner 不凭空插 collective。

这三个阶段不能统称一个 `hreduce`。barrier、pipeline、DMA completion 是独立
effect，不藏在纯寄存器 intrinsic 内；跨 worker 计划尚未证明则 fail closed。

已有真实 DMA 签名（不是拟议 HVX API）：

```cpp
int tl::dma_copy_2d_wait(void *dst, uint64_t dst_offset,
                       const void *src, uint64_t src_offset,
                       uint64_t width, uint64_t rows,
                       uint64_t src_stride, uint64_t dst_stride);
extern "C" int tl_hex_dma_copy_2d_wait(void *dst, const void *src,
                                    uint32_t width, uint32_t rows,
                                    uint32_t src_stride, uint32_t dst_stride);
```

offset/width/stride 单位为 byte。当前 wrapper 检查非空、正 width/rows、
width/stride ≤0xFFFFFF、rows≤65535、stride≥width、128B 对齐与地址溢出；
返回 0 成功、负数失败，adapter 返回时写入已完成。它是平台 adapter，不是
直接 QuRT 动态符号，也不是 HVX 算术 intrinsic；地址不溢出还不等于有 allocation
权限，allocation bounds 与 owner 仍需上层证明。

未来 `dma_submit_2d(...) -> ticket` / `dma_wait(ticket)` **仅为设计占位**，
需另定 ticket 生命周期、错误传播、slot reuse 与 completion visibility。
本设计中的 physical-compatible `T.copy` 路径只搬数据、不得暗中 cast；
`T.transform` 才表达 logical dtype/layout 转换，绝非 mere bitcast。
这不是声称所有 backend 的现有 `T.copy` 都无 cast：通用 frontend 的 scalar
fast path 目前可插 Cast；需区分 Hexagon physical-copy 契约与通用 API。

## 4. 硬规则：vector_required、尾部与 effect

拟议 opt-in `vector_required` 是严格契约，不是性能提示：

- 真实目标 `-O2` 编译到 object 并审计 ASM；源码里写 vector 或做 syntax-only
  都不构成证据。追踪 vmem、算术、shuffle、spill 与 scalar load/store 的数据来源。
- 禁止逐元素 scalar 数据运算/访存 fallback，含 IEEE NaN/inf/subnormal 的
  scalar 异常修补循环。不能以“罕见分支”为由豁免。
- 允许 scalar 地址计算、loop/control、状态检查、固定 lane 最终行结果提取；
  用这些操作拼逐元素算法仍然违规。
- strict 无全向量实现、unsupported permutation、tail、dtype 或模式：编译期拒绝。
  运行时必要条件失败：显式报错，不切 legacy scalar helper，不静默降级。
- native/finite-domain 的“静态宣称”必须有证明；运行时 guard 自身也不得逐 lane
  scalar 扫描数据，可向量检测后 scalar 分支拒绝。未知硬件 flag 不等于已知契约。

tail 的 sum 无效 lane 用 0，max 用 -inf（限约定支持该语义的 max 路径）；
不能先越界 load 再填单位元。exp 前将无效 lane 置安全输入，再把输出置 0；
不能计算 `-inf - -inf` 后才 select，避免 inactive lane 的 NaN。
全 masked 行需显式结果/分母处理，不允许 rcp(0) 偷过 positive-finite 域。

每条 IR 至少携带：read/write region、alignment、allocation bounds、alias set、
owner、有效 lane mask、dtype/rounding、异常/近似版本、reduction order、
barrier/completion effect。fusion 或 CSE 不能跨越未知 alias、owner 变化或
未完成 DMA；overlap 不能仅凭变量名不同就判定安全。

## 5. 融合边界与 FA 通用原语 recipe

允许融合的是证明等价的 load→map→cast→store 等通用链，而非 kernel-name
guard 或 `fa_softmax` / whole-op helper。必须保留 dtype 舍入与逻辑依赖：

- `P_half = narrow(exp(...))` 后再 `widen(P_half)` 求 sum，不能优化成直接
  sum 尚未 round 的 fp32 exp；P 的 fp16 舍入是数值算法的一部分。
- scale/mask/max 的第一遍先确定行 max；第二遍才能做依赖它的 subtract/exp/sum。
  不能把两遍简单合成同一 map，也不能越过 causal mask 或跨 K-block recurrence。
- 约简重排、FMA contraction、half round 消除需要单独授权的数值模式，默认禁止。

FA 示例为说明组合能力，不新增专用 intrinsic：

1. HMX QK 结果经明确 layout/cast 映射到 HVX 行；load、scale、causal mask、
   lane max、hreduce_max；在线算法还需与上一 K-block 的 max 按既定顺序合并。
2. 第二遍 load、scale/mask、subtract row max、exp、narrow 得 P_half；再 widen
   P_half、sum、hreduce_sum。旧 state 的 rescale、sum 更新也保留原始顺序。
3. P 按已有 HMX 输入布局 transform，PV 更新输出 accumulator；normalization
   用显式 rcp/mul/cast/store。任何算法变化另行评审，不能只因形式相似而替代。
4. 两个 HVX worker 做行 partition 时，各阶段显式证明 row→worker、vector→row
   lane map、完整写集合、无重叠、所有有效行均覆盖。跨阶段复用 partition 必须
   是同一映射，不能默认 worker id 不变就代表 owner 不变。
5. HMX/DMA producer 与两个 HVX consumer 之间有发布/消费同步；最后一个**有效**
   K-block 才执行归一化 epilogue。causal skip/tail 下 last-active-K 不必等于循环
   最后一项，需证明判定唯一、可达且不漏行。

runtime polling 的 1024 参数与 core/MAX 配置是独立变量，不能混为一个调优
开关或从一个推导另一个。worker 数、轮询、MAX、DMA、tile、缓存等必须在 matched
A/B 中分别固定/记录；调整 runtime 不能冒充 compiler intrinsic 带来的收益。

## 6. 待实施验证计划（本轮不执行测试）

| 层级 | 必须收集的证据 |
|---|---|
| host 位语义 | 全部 65536 half bit patterns 的 widen、half bitcast 位保持；独立 oracle 而非被测 helper 自证。narrow 覆盖 RNE tie、subnormal 边界、overflow、±0、±inf、NaN sign/payload；完整 lane 映射而非抽几个值 |
| 布局与负测 | linear/evenodd 及受支持 1/2-vector permutation 全映射；所有余数尾部、负/越界 index、非法 shift、未对齐地址、allocation 边界、alias、owner crossing、未知模式必须正确拒绝或有安全实现 |
| reduction | 每种 tree/seed/acc_dtype 与独立顺序 oracle；跨 chunk、跨 worker、全 masked 行；不能以实数结合律替代浮点顺序验证 |
| 编译/ASM | 实际目标工具链 `-O2 -mv79 -mhvx -mhvx-length=128B` 编 object 与 ASM（需要 HMX 时加 `-mhmx`）；覆盖正常与异常分支，禁止数据标量回退；保留编译命令、版本、源码/object/ASM hash |
| 设备正确性 | 授权之后运行正式 compiler 生成的完整 standard FA，标准形状 B=1,HQ=16,Q=1024,D=256，全量 **4,194,304** 输出；global 与每个 head cosine 均 ≥0.999 |
| 精度防作弊 | cosine 门限按用户批准口径；max_rel 作为诊断同时报告，不以此静默删改已有测试断言。保留 NaN poisoning、全量 finite 扫描、guard/canary、FP64 独立参考及 pack/unpack 验证，不使用被测中间结果构造参考 |
| 性能 | hash 对应正式生成物、实际部署二进制与 launch；包含 pack/转换、kernel、同步及端到端 wall 分列；同窗口同配置 matched A/B、多轮分布，不只挑最快一次 |

20/21ms 手工模板只能作诊断参考；将来正式 compiler 的可复现输出经上述验证后
才能成为正式参照。没有性能保证，也不能把旧 legacy 路径或局部叶测试成绩
升级为完整 FA 验证。测试如与新设计矛盾须停下报告，不为通过而改 fixtures、
阈值或评分逻辑。

## 7. 未决事项与优先级

待明确决策：strict NaN payload/tie 与异常标志承诺；native 各目标的 FTZ/rounding
事实；exp/exp2/rcp 的算法版本、域和误差界；tail 的安全向量实现；支持的 permute
集合；seed/tree 与现有逻辑 reduction 的兼容规则；动态长度；跨 worker tree；
DMA ticket 协议。上述 **TBD 不是许可隐藏 fallback**。

实施门顺序（每一阶段需单独证据与评审）：

1. **设计接受**：批准范围、名字、数值模式与未决项处理；当前仅到此之前。
2. **leaf intrinsics**：独立位语义与实际 ASM；strict 不得留 scalar exception 洞。
3. **VectorLowering**：layout/ownership 后的固定宽度拆分、tail 与 rejection。
4. **fusion**：通用 effect/依赖证明，保留 half 舍入与 reduction 顺序。
5. **ownership / FA emit**：两 HVX 行 partition、同步和 last-active-K epilogue 证明；
   正式生成物可重现，不引入算子级模板捷径。
6. **matched device verification**：用户授权后才构建/部署/真机全量验证及计时。

本次交付仅为此 README；没有授权实施上述步骤。独立 verifier 审阅与所有
未来实现/设备验证仍待完成。
