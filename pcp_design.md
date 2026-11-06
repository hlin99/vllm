# DeepSeek V4 XPU PCP: KV 通信与 Attention 流水线设计

## 1. 目标与范围

目标是让本卡的 Q 在远端 KV 分块到达时就开始计算，而不是等待完整
all-gather。Kernel 应直接理解 KV 的物理布局，不为满足连续输入的要求增加
KV 拼接、重排或临时反量化副本。

P0 的 paged KV 融合读取已有可切换实现；P1 到 P3 仍是待实现的设计，
不表示下述通信/计算流水线已经完成：

- 先完成 V4 的 SWA、C128，再处理 C4 的 indexer 依赖。
- 保持当前历史 KV cache 的复制语义，不同时改成分布式历史缓存。
- 保留原路径和小形状回退。
- 压缩比例和窗口通过元数据表达，不把 C4 的常数写进通信协议。
- 不在这一步设计通用 V4.1 indexer K，也不宣称完整 V4.1 XPU 支持。
- MoE 的 AG/RS 融合作为独立阶段，不与 KV 通信混为一谈。

## 2. 当前路径与目标路径

### 2.1 当前 SWA KV 路径

```text
本卡 KV 投影、RoPE
    |
完整 KV all-gather
    |
量化、写入复制的 paged cache
    |
反量化并收集为连续 BF16 workspace
    |
Sparse attention
    |
本卡 token 的 O projection
```

当前压缩 KV 有两条可切换路径：

- `VLLM_XPU_PCP_BOUNDARY_EXCHANGE=0`：收齐全部压缩输入，各卡重复压缩。
- `VLLM_XPU_PCP_BOUNDARY_EXCHANGE=1`：交换 segment 尾部，各卡只压缩本地
  token，再发布量化后的主 KV 和 indexer K。默认仍为 `0`。

这两条路径都还不是 KV 通信与 attention 的分块流水线。

`VLLM_XPU_PAGED_SPARSE_ATTN=1` 可以跳过 BF16 KV workspace，由 attention
直接读取主 KV/SWA 两个 paged cache，并在 kernel 内反量化。实际 page stride
可以包含 padding 或间隔。默认 `0` 保留原路径；这个开关本身不改变 AG 调度。

### 2.2 目标路径

```text
通信：   收 chunk0 ---- 收 chunk1 ---- 收 chunk2
计算：                 算 chunk0 ---- 算 chunk1 ---- 算 chunk2 ---- 合并
```

图中每个计算块只等待自己需要的数据。本卡已就绪的 KV、历史缓存可以在依赖
允许时先算；不要求按 token 的逻辑顺序等待所有块。

需要区分三个概念：

| 概念 | 含义 |
| --- | --- |
| 逻辑 segment | 同一请求中逻辑位置连续的一段 token |
| 物理连续区间/page | 内存中可以按固定 stride 访问或传输的范围 |
| 通信 chunk | 为流水线划分的一次传输和就绪发布单位 |

三者不必等长，也不必一一对应。

## 3. KV 不能只用“一个连续张量”描述

### 3.1 PCP 的逻辑顺序与物理顺序不同

例如，两卡处理一个请求的 16 个 token，采用首尾配对切分：

```text
rank0 本地排列：[0..3][12..15]
rank1 本地排列：[4..7][8..11]
```

rank-major 收到的顺序不是 `[0..15]`。多请求、尾块和 padding 会进一步增加
不规则性。不能为恢复逻辑顺序再执行一次 `cat`、重排或 `contiguous`。

现有 sparse attention 已有 stride 和稀疏 indices，并非完全不支持非连续访问。
但其 KV 地址仍基于单一 base 和统一行 stride；它不能直接表达多个接收区域、
paged 历史缓存以及不同的量化数据区域。

### 3.2 所需布局信息

“几段、每段多少 token 连续”可描述规则情况，但完整接口至少需要：

| 信息 | 作用 |
| --- | --- |
| base/段偏移、有效长度、行 stride | 找到物理数据，排除 padding |
| 请求标识、逻辑起始位置或已有索引映射 | 维持请求隔离、因果与 SWA 语义 |
| page table、page 大小、实际 page stride | 直接访问历史 paged cache |
| 数据格式及 data/scale 偏移 | 正确读取量化主 KV 和 indexer K |
| chunk-ready 依赖与调用代次 | 保证读到本次调用已经发布的数据 |

优先复用现有 metadata 和 page table。规则 rank/chunk 布局用公式计算地址，
仅对不规则 segment 增加描述，不创建不必要的逐-token 指针表。

主 KV cache 的一个 page 内，token 数据区与 scale 区分开。不能将它误认为
每个 token 都是连续的 584 字节记录，也不能忽略 page padding。

### 3.3 “不增加 copy”的边界

应消除或禁止新增：

- 为满足 kernel 输入而做的 KV 拼接、重排、`contiguous`。
- paged FP8 cache 到大块 BF16 workspace，再由 attention 读取的往返。
- 通信接收 staging 到另一个 attention 输入张量的纯搬运。

跨卡数据传输、必要的 cache 写入和量化本身仍然存在。分块 softmax 的中间状态
也可能需要额外读写；它不是 KV 重排，但必须计入性能成本。

优先沿用既有必要的 cache 写入，让 attention 直接读取写好的 page。若希望通信
直接写最终 paged cache，还需验证该缓存的对称内存注册、目标地址和 data/scale
分区传输能力，不能假设普通 all-gather 可以直接完成。

## 4. 从 TPSP 借什么，不直接照搬什么

参考项目：`/workspace/TPSP`。

| 可借用的机制 | 注意事项 |
| --- | --- |
| 按 chunk 发布就绪事件和 generation | 发布必须在 payload 写入完成之后 |
| 独立计算/通信 queue 与设备侧依赖 | 不用 CPU 轮询决定何时启动下一块 |
| 对称内存、输出池与异步生命周期管理 | consumer 结束前不能覆盖或回收 buffer |
| 本地块优先、远端块到达即消费 | 不在 consumer 前增加完整 AG 的 join |

TPSP 的现成 GEMM 接口不是 attention 接口。需要接入或抽出分块通信的就绪协议，
并让 attention 成为 consumer；不是调用返回完整结果的 AG/GEMM 后再启动 attention。

它也不是所有路径都零额外 copy：

- `MatmulReduceScatterNormAllGather.cpp` 的 `p2p` 路径包含 staging 到 output
  的搬运。
- 独立 chunk ring 路径可以直接写输出，但其设备侧等待和 forward progress
  仍需在目标硬件、驱动及代表性形状上验证。
- Python AG/consumer 路径也有发布与接收复制，不能直接当作零副本实现。

第一版优先使用可验证的 queue/event 依赖组织多个 kernel。暂不强求一个常驻
attention kernel 在内部等待所有远端块，避免等待 kernel 占满计算资源，
阻塞它所等待的 producer。

已验证的另一个接入点是 PyTorch XPU symmetric memory：直接在对称 arena
生成本地 KV，再将远端分块复制到本卡接收段。它不必依赖 TPSP 的 GEMM 扩展，
也不必在通信前额外复制一份本地 KV。接收段仍通过必要的量化写入进入最终
paged cache，attention 不需要重新拼接接收 buffer。

两个 rank 的独立原型已验证 IPC、连续代次的 barrier、分块事件和读取非连续段的
attention。大传输控制实验的 timeline 能看到 peer copy 与 attention 重叠；
小传输并没有稳定的重叠。这不是模型流水线已经完成或小 budget 能获益的证据。

## 5. Attention 的正确性约束

### 5.1 分块计算必须合并 softmax 状态

不能对每段独立完成 softmax，再简单相加输出。每个 query/head 需要保存或
可正确合并的状态：

```text
最大 score
指数和
未归一化的加权 V 累积
```

新块到达时按照 online softmax 更新，或者通过 partial output 与 LSE 正确合并。
保留必要的 FP32 中间精度，最后统一归一化。

同一条应参与计算的 KV 不能因为收到多个副本而重复计数。必须保持已有的位置、
mask、缩放等语义，正确处理空块和全部被 mask 的块。

### 5.2 SWA、C128 与 C4 分开推进

| 路径 | 流水线依赖 |
| --- | --- |
| SWA | 根据请求、位置和窗口确定条目；对应数据就绪即可算部分贡献 |
| C128 | 无 C4 indexer；在压缩窗口完整且产物就绪后消费对应压缩 KV |
| C4 | 还需确定该 query 的全局 top-k，不能把先到的候选当成最终选中项 |

C4 的 indexer 打分依赖候选 K。候选未看齐前，局部 top-k 不等于最终 top-k。
可先计算不依赖 top-k 的 SWA 贡献，并让主 KV 通信与 indexer 重叠；
压缩 KV 贡献要在选择结果确定后接入。

后续可以研究分块 indexer 打分与 top-k 合并，但它是额外工作，不是第一版
“换一个 KV 地址参数”就能自然获得的能力。

### 5.3 协议与生命周期

- 各 rank 使用一致的通信调用计划，空 rank 不能自行跳过必要 collective。
- 压缩边界、indexer K 和主 KV 的信号不能互相混用。
- generation 必须区分调用，防止上一轮 ready 被误认为本轮完成。
- Triton consumer 与 SYCL 通信 queue 间的可见性和依赖需要实际验证。
- chunk、page、segment 边界交叉时，不通过额外 KV 重排来修正布局。
- cache 在下一步、decode 和压缩继续执行时仍保持原有有效状态。

## 6. AG 和 RS 为什么产生

### 6.1 两种切分不要混淆

| 切分对象 | 一张卡持有什么 |
| --- | --- |
| PCP 的 token 切分 | 一部分 token 的 hidden states/Q |
| MoE 内部的专家 TP | 每个专家的一部分中间维度权重 |

当前支持范围是 attention TP=1、PCP>1、EP 关闭。但 FusedMoE 会将 PCP
维度并入专家 TP，因此不能推导出“每卡都有完整 MoE，可以各算各的”。

### 6.2 两卡、四个 token 的例子

假设某个专家的中间维度被切成 A、B 两份：

```text
rank0：token0、token1 的输入 + 专家权重分片 A
rank1：token2、token3 的输入 + 专家权重分片 B
```

只算本地输入会漏掉：

```text
token0、token1 需要分片 B 的贡献
token2、token3 需要分片 A 的贡献
```

所以先 AG，让两卡都拿到这四个 token 的输入：

```text
rank0 用权重 A 计算：[A0, A1, A2, A3]
rank1 用权重 B 计算：[B0, B1, B2, B3]
```

这里 `Ai`、`Bi` 是同一个 token 的不同中间维度分片经 down projection
产生的输出贡献，不是两份独立的完整答案。

完整结果要相加，而每卡最后只需要自己负责的 token：

```text
RS 后：
rank0 得到：[A0+B0, A1+B1]
rank1 得到：[A2+B2, A3+B3]
```

这就是 reduce-scatter：

- reduce：同一个 token 的贡献求和，不是取平均，也不是把不同 token 相加。
- scatter：每卡只保留自己负责的 token 输出。

可以理解成 all-reduce 后只取本卡那一段的结果，但不需要真的先物化全部结果。
MoE 的 top-k 加权归并、量化和 shared expert 分支还需按原实现处理，不能简单
将 shared expert 的完整结果在所有 PCP rank 间重复求和。

### 6.3 一层中的通信位置

```text
本卡 token 的 Q/KV 准备
    |
KV AG / 压缩边界交换与产物发布
    |   attention 按依赖消费已经到达的块，与通信流水重叠
本卡 token 的 attention 输出
    |
O projection                 当前 attention TP=1，不需要跨 PCP RS
    |
本卡 mHC、norm、gate
    |
AG hidden states 和 router logits
    |
MoE 各权重分片计算
    |
RS routed output
    |
本卡 token 的输出，进入下一层
```

KV AG 只是在收集不同 token 的 KV，不是在收集同一 token 的输出部分和，
因此它本身不要求再做一次 RS。Attention 的 Q 和输出仍按 token 留在本卡。

MoE 则既切了输入 token，又切了专家权重，因此有上述 AG 和 RS。它们当前就
存在，不是 KV 流水线新增的通信。

当前代码先在本地计算 gate，再 AG hidden states 和 router logits。优化时可
研究融合或改变这两种数据的传输方式，但不能直接删除输入覆盖或输出求和语义。
此外，模型 forward 有一次供 hash MoE 使用的 input IDs AG，输出恢复也有
独立的 gather；它们不等于每层 O projection 后新增一个 RS。

### 6.4 MoE 的后续融合

参考 TPSP `moe_tp`，可独立推进：

```text
AG + expert permute
    |
专家计算
    |
unpermute + top-k 加权归并 + RS
```

TPSP 的 dispatch 已支持 FP8 数据和 scale 随 permute 传递，并不只是 BF16
数据搬运。但接入时仍需要匹配 V4 当前量化、路由、shared expert 和 padding 语义。

P3 尚未实现。当前 `XPUExperts.apply` 调用的 `XpuFusedMoe.apply` 内部完成
`remap_hidden_states → 两次 grouped GEMM → moe_gather`，没有公开“直接消费
已 permute 输入”和“返回未 unpermute 专家输出”的接入点。只替换外层 AG/RS
会重复 permute、归并或加权，不是真融合。需要先拆出这两个 backend 边界，
shared expert 仍单独处理，不能重复参加 routed output 的 RS。

另外，TPSP 的 Python `SymmBuffer` 可以导入，但实际加载 `deep_symm_cpp`
时因缺少 SYCL `submit_kernel_direct_with_event_impl` 符号失败；
这份预编译插件与当前 SYCL 运行库不匹配。需要匹配工具链重新构建或提供兼容
插件，不能隐藏加载错误或强行替换共享环境的运行库。

## 7. 实施顺序与验收

| 阶段 | 工作 | 验收 |
| --- | --- | --- |
| P0 | 分段/paged KV 地址协议；attention 内融合 load/反量化 | 暂不异步，先验证数值与地址映射；去掉大块 BF16 KV workspace |
| P1 | 分块 AG、就绪事件、SWA/C128 partial attention 与合并 | timeline 显示块间通信/计算重叠，无新增 KV 重排 |
| P2 | C4 indexer 依赖与主 KV/SWA 调度 | 全局 top-k、压缩窗口、attention 结果保持一致 |
| P3 | MoE AG/permute 与 unpermute/RS 融合 | 路由、权重分片求和和本地 token 输出布局不变 |

P0 当前验证：

- sparse attention kernel 测试 7 项通过，包括非连续 page、offset view、尾部和空行。
- PCP2/4 的真实 `LLM.generate` 入口测试 6 项通过，覆盖 SWA、C4/C128、
  两种压缩路径、分块 prefill 和 decode；这里使用的是小配置 dummy 权重。
- 真实 V4 Flash、PCP8 的三个短输入任务与原路径 token、top-5 logprob 完全一致。
  长输入仍有原路径自身也存在的变化，不能宣称长上下文逐 token 等价。
- 完整适用 pre-commit 检查通过，没有绕过 hook。

P0 的收益与代价必须分开看：消除 BF16 KV workspace 不等于一定更快，
attention 内重复反量化可能抵消搬运收益。默认路径保持不变。

P1a 的 kernel 已能跨调用保存 FP32 numerator、指数和与最大值，并在最后一块
统一归一化，不是相加各块独立 softmax 的输出。新增测试按原有 16-key block
边界拆分，包含空块、全 mask 行和状态复用，结果与单次调用逐元素完全相同。
仅 P1a 完成时模型尚未接入分块传输调度，不能把这个内核能力当作完整 P1。

P1b 当前原型用 PyTorch XPU symmetric memory 建立共享 IPC arena，Qnorm/RoPE
直接写本卡发送段；通信 stream 拉取远端 chunk，按 event 就绪后量化入 cache，
attention 按物理 slot 的 chunk 归属累积 SWA。历史 KV 单独先算，不额外拼接
或重排 KV。P1b 只在 SWA/C128 启用，C128 压缩贡献最后合并。

P2 将同一调度接入 C4：先累积 SWA 的 FP32 状态，等现有 indexer 的全局
top-k 完成后，再读取所选 compressed KV，合并贡献并最终归一化。
Indexer K 的通信与全局选择逻辑不变，没有宣称流水化 indexer 自身，
也没有引入 V4.1 indexer K 的通用设计。PCP2/4、两种 compressor 路径的
4 项 `LLM.generate` 回归通过，覆盖分块 prefill、decode 和空 peer。
真实 PCP8、chunk=8 的长任务仍得到 56088，但只与原路径共享前 5 个输出
token，结尾标签不同；首 token logprob 差 0.00405，共同前缀最大差 0.23097。
尚不能宣称长上下文等价。

补充将 chunk 降为 1，确保三个短任务的 prefill 和 decode 都激活流水线，
各检查前 32 个生成 token：算术任务的 token 和完整 top-5 logprob 一致；
KV cache 任务的 32 个 token 一致，但 logprob 最大差 0.22082；
Python 任务只共享前 26 个 token，之后发生分歧。这里检查的是前缀，
不是完整回复，不能把先前走回退的精确结果代替这组激活路径的结果。

进一步在真实权重的第一个 128-token prefill 上，对 8 个 rank、43 层逐层
使用完全相同的 Q 和 cache 比较流水线、原 attention 和 FP32 softmax 参考。
344 组均满足既有 BF16 的 `rtol=atol=1e-2`；C4 的流水线与原实现最大
relative RMS 为 0.1285%，C128 为 0.0281%，SWA 完全相同。
两种实现相对 FP32 参考的最大 relative RMS 都约 0.18%。
这验证了该输入上的局部数值误差，没有证明完整模型生成等价，也不能把
所有生成分歧都归因于原路径自身波动。

P2 相同 serving 配置三轮全部完成 16 个请求、失败 0 个：

| 路径 | 三轮整批耗时中位数 (s) | 相对原路径 |
| --- | --- | --- |
| 原路径 | 287.950 | 基线 |
| P0 paged | 340.339 | 慢 18.19% |
| P2 SWA/C4/C128 流水线，chunk=8 | 323.864 | 慢 12.47% |

P2 三轮分别为 323.878、323.864、323.856 秒，比 P0 paged 的耗时减少
4.84%，但没有超过原路径。三组均为真实 V4 Flash、8 张 B70、PCP8/TP1、
16 个 8192-token 输入、各输出 1 token、budget=128，首轮前预热 8 次，
计时期间没有开启 profiler；原路径和 P0 对照在本 session 较早时测量。
默认继续关闭。

整机 timeline 还捕获了超过 2048-token 上下文、实际执行 indexer logits 与
全局 top-k 的一个迭代。8 个 rank、21 个 C4 层均确认同一 SYCL queue 上
先提交 3 次 SWA partial attention，再执行 indexer logits/top-k，最后提交
compressed contribution 的 attention。部分 profiler 设备时间戳在同一 queue
上有约 1 微秒重叠，因此不把时间戳不等式当作独立的同步正确性证明。

该迭代每卡有 172 次 attention 和 688 次 cache quantize launch。
Attention 的 kernel 时间和约 232–242 毫秒，占该卡所有 kernel 时间和的
72–75%；KV peer copy 时间和约 1.36 毫秒，profiler 记录与 attention 重叠
约 55–61 微秒。这里不是整次 serving 的墙钟耗时，也不是可直接相减的
加速收益。下一步应先优化 attention 本身，而不是假定把通信再重叠一点
或只减少约 3 毫秒的 quantize kernel 就能解决这组回归。

分布式 cache 与 sparse kernel 的 10 项测试通过；原路径、paged 对照及新增
PCP2/4 pipeline 模式共 8 项 `LLM.generate` 测试通过。真实 V4 Flash、PCP8、
chunk=8 的三个短任务也与原路径 token 和完整 top-5 logprob 一致。5122-token
长任务生成的 8 个 token 相同，但首 token logprob 差 0.02452、共同输出前缀的
最大 logprob 差 0.39433；原路径自身复跑也有变化，不能据此宣称长上下文等价。
注意这三个短输入在 PCP8 下每卡仅有 2–6 个 padded query rows，chunk=8 时
实际走的是原路径回退，只能证明回退行为不变，不能当作流水线激活时的精度证据。
相同 serving 配置的一轮测试完成 16 个请求、失败 0 个，耗时 290.85 秒；
相对新测原路径三轮中位数 287.95 秒慢约 1.0%，比 P0 paged 的
340.34 秒短。但 P1b 的 C4 仍走原路径，不能把这个差值归因于流水线收益。
流水线目前只有一轮，不能称为三轮中位数或证明加速；
新开关默认关闭。

原型的两卡 B70 算子对照已覆盖每卡 16、128、1024、4096 token，128 heads、
128-token SWA，BF16；每个形状扫描 2/4/8 个 chunk，交替测量各方案五次，
每次取最慢 rank，最后取中位数。基线也是 paged attention，不是原 serving 路径。
计时包含 Qnorm/RoPE、通信、必要量化写 cache、metadata 和 partial attention；
状态已预留，输入准备和编译不计时。Q 与 cache 逐元素一致，attention 使用
现有 BF16 的 `rtol=atol=1e-2` 验证。

| 每卡 token | 最佳 chunk | paged 串行基线 (ms) | 流水线 (ms) |
| --- | --- | --- | --- |
| 16 | 8 | 2.084 | 2.865 |
| 128 | 64 | 8.806 | 12.909 |
| 1024 | 512 | 66.272 | 90.273 |
| 4096 | 2048 | 264.262 | 350.928 |

4096-token、四 chunk 的生产 wrapper timeline 中，约 132 微秒远端拷贝
与 attention 重叠约 46–48 微秒，证明调度可以重叠，但收益远小于新增开销。
扫描后的各形状仍然更慢；不能把它当作 serving 加速结果。下一步需要检查
partial-state 流量、空 chunk 的处理及量化 launch 数，而不是默认启用。

模型验证还暴露了独立的 XPU FP8 GEMM 单 K-block scale 问题：底层将 compact
block scales 当成 per-channel scales，读取了本不属于该 scale 的内存，可能
返回错误值或 NaN。已用有限输入和 CPU 参考复现，单独修复 scale 展开；
单/多 K-block 四项精确测试通过，没有修改容差或绕开失败。

各阶段需要覆盖：

- 多请求、两段 PCP、尾块、padding、空 peer、非连续 page。
- 分块 prefill、多步续算与 decode，不能只比较第一次 forward。
- 延迟 rank、连续调用、buffer 复用和交错就绪顺序。
- 量化 cache 字节与有效状态，以及 attention 输出的数值误差。
- PCP2/4/8、不同 chunk 和 batch 形状；同时测小形状回退。
- 单算子和端到端延迟，并检查 timeline、内存占用与 KV 搬运。

已有实测基线：真实 V4 Flash、8 张 B70、PCP8/TP1、16 个 8K 输入请求、
各输出 1 token、budget=128、预热 8 次、每模式三轮。
原压缩路径整批耗时中位数 287.40 秒，边界交换路径 291.31 秒。
这不是通信/attention 流水线的性能结果，也不能外推更大 chunk。
该配置每卡 token 很少，融合仍必须评估调度开销，不能直接照搬 TPSP 的阈值。

P0 paged attention 用相同配置重新测了三轮，并重新跑原路径作对照：
原路径中位数 287.95 秒，paged 为 340.34 秒，整批耗时增加 18.2%。
所以不能因为 decode wrapper 的局部 microbenchmark 更快就打开全模型默认值；
本次融合确实消除了 BF16 KV workspace，但这个 prefill workload 上没有性能收益。
默认仍保持原路径。

三组短输入的真实权重检查输出与 top-5 logprob 完全一致；5122-token 长输入
存在输出/logprob 差异，原路径自己复跑也会变化。后续精度验收需单独厘清这条
基线，不能以“算术答案正确”替代长上下文等价性验证。

## 8. 已明确与仍需验证的事项

已明确的是数据切分、现有通信位置、地址协议需要表达的内容、softmax 合并方法、
C4 依赖，以及分阶段实现路线。

尚不能提前保证的是：

- Triton/SYCL 跨 queue 的具体接入和可见性实现。
- 目标硬件上设备侧等待的 forward progress。
- 当前 paged cache 能否直接作为通信落点。
- 融合后重复反量化、累积状态访存与调度开销的真实成本。
- 哪些形状能获得端到端收益，以及长上下文精度验收。

这些需要原型、数值检查和设备 timeline，不应靠模型推理代替验证。

## 9. 代码入口

vLLM：

- `vllm/models/deepseek_v4/xpu/xpu_qnorm_rope_kv_fp8_insert.py`：
  SWA KV AG 与量化插入。
- `vllm/models/deepseek_v4/xpu/xpu_sparse.py`：
  压缩 metadata、边界路径、prefill cache gather 与 attention 调用。
- `vllm/models/deepseek_v4/xpu/pcp_compressor.py`：
  当前压缩 cache 的打包与发布。
- `vllm/v1/attention/ops/xpu_mla_sparse.py`：
  现有 BF16 sparse attention 和地址计算。
- `vllm/v1/worker/gpu/pcp_manager.py`：
  请求 segment、token 切分、padding 与恢复。
- `vllm/model_executor/layers/fused_moe/config.py`：
  PCP 维度并入专家 TP 的配置。
- `vllm/model_executor/layers/fused_moe/runner/moe_runner.py`：
  `_maybe_dispatch` 中 AG、`_maybe_combine` 中 RS。

TPSP：

- `deep_symm/async_tp/__init__.py`：分块 AG 与 consumer 的 Python 参考。
- `csrc/modules/async_tp/AllgatherGemmOps.cpp`：原生 AG/GEMM 调度。
- `csrc/modules/async_tp/MatmulReduceScatterNormAllGather.cpp`：
  chunk-ready 协议、queue 依赖与输出生命周期。
- `deep_symm/moe_tp/symm_buffer.py`：AG/permute 与 unpermute/RS 接口。
