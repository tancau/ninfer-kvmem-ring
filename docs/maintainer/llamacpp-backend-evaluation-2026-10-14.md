# 第二条赛道评估：KVMem 的 llama.cpp 后端 + Qwen3.8-27B（Phase 0）

起因：B 站有人发了一条「Saluki 27B + KVMem + actKV，3060Ti 8G 跑 128K/256K/1M」的视频。
顺着查下去发现一件比 Saluki 本身更重要的事——**KVMem 有两个后端，我们只用了其中一个。**

本文只记录能证实的东西。所有数字有出处；没有出处的标注为未知。

## 一、两个后端，同一套 KV 架构

KVMem 官方 README（包内 `licenses/KVMem-README.md`）的下载表：

```
Windows RTX 30/40/50   NInfer      .ninfer    v0.18.0-ninfer-rc1   ← 我们在用的
Windows/Linux NVIDIA   llama.cpp   GGUF       v0.16.0-rc3          ← 另一条
Windows Bonsai 2       Bonsai 专用 PTQ1 GGUF  prism.3              ← Bonsai 专用实验版
自行编译               llama.cpp   GGUF       v0.17.0 仅源码
```

README 原话：**"The optional LRFU and GQA max switches are shared by the llama.cpp
CLI/server and ninfer server."** 两条线共用同一套 KVMem 检索与排名。

llama.cpp 那支的自述：**"KVMem adds a bounded GPU KV working set, host-memory storage
and query-based retrieval to llama.cpp."** —— 和我们 NInfer 上跑的是同一套东西。

## 二、llama.cpp 那支的官方配方（和我们结构完全一样）

```
-m Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
--mmproj mmproj-Qwen3.8-27B-Q5_K-MIX.gguf --no-mmproj-offload --image-max-tokens 512
-c 262144 -n 16384
--kvmem-budget 36864 --kvmem-gen-reserve 16384
--kv-dtype q8_0
--spec-type draft-mtp
--enable-thinking --reasoning-budget 4096
```

**`--kvmem-budget 36864` + `--kvmem-gen-reserve 16384` = 53,248** —— 和我们 NInfer 生产的
数值一字不差。也就是说我们量出来的那条"预填充台阶 = B+R"（见 `prefill-speed-2026-10-14.md`）
**在这条线上同样适用**，而且官方推荐值也停在这个位置。

## 三、显存账：官方实测把它钉死了

release notes 里 16 GiB 卡上的 256K 压测（IQ3_S MTP、Q8_0 主 KV、上下文 262144、
budget 36864、reserve 16384、**GPU 上的 Q8_0 视觉头**、关闭 thinking）：

```
峰值整卡显存    15.41 GiB
最低剩余        约 0.4 GiB
压测规模        261,573 输入 tokens，33 轮工具累积，3/3 标记检索
prefill（加权） 316.95 tok/s      decode（加权） 43.72 tok/s
```

拆开：IQ3_S-mtp 权重 12.1 GB = 11.27 GiB → **非权重部分 = 15.41 − 11.27 = 4.14 GiB**
（KV q8_0 在竞技场 53,248 上 + MTP 草稿 KV + GDN 状态 + 缓冲 + GPU 视觉头）。

换到我们的 12 GiB 卡，只换权重（非权重先按同样 4.14 GiB 算）：

```
IQ3_S-mtp      11.27 GiB → 15.41 GiB   装不下（官方推荐，直接排除）
IQ2_XS-mtp      8.16 GiB → 12.30 GiB   略超
IQ2_XS          7.83 GiB → 11.97 GiB   极限，几乎零余量
Saluki IQ2-mix  7.35 GiB → 11.49 GiB   约 0.5 GiB 余量
```

**但 4.14 GiB 这个数有两个不确定性，都朝好的方向：**

1. 那次压测把视觉头放在 **GPU** 上；rc3 起 IQ3 脚本**默认改成 CPU 视觉头**
   （`--no-mmproj-offload`），视觉头（Q8_0 约 0.6 GB）就不再占显存。
   扣掉后非权重约 3.1-3.5 GiB。
2. 那次是 **16 GiB 卡且只剩 0.4 GiB** —— 说明引擎是**把卡吃满**，不是"需要这么多"。
   换到 12 GiB，它应当自己缩（缩竞技场、缩缓冲）。

按 CPU 视觉头重算：

```
IQ2_XS-mtp      8.16 + 3.3 = 11.46 GiB   约 0.8 GiB 余量
Saluki IQ2-mix  7.35 + 3.3 = 10.65 GiB   约 1.6 GiB 余量
```

**结论：算不出定论，只能真加载。** 这正是要做的加载测试。

## 四、为什么先测 IQ2_XS-mtp，不先测 Saluki

```
Saluki IQ2-mix（7.9 GB）
  优点  更小、厂商声称工具调用反超原版
  疑点  文件名没有 -mtp 标记。我们的解码速度靠 MTP（实测 2.1 倍）；
        没有 MTP 头就没有投机解码，速度不可比。是否带 MTP 头未证实。

Qwen3.8-27B-GSQ-RCO-IQ2_XS-mtp（8.77 GB，ISTA-DASLab）
  优点  官方配方指定的同一系列；明确带 -mtp 头；配套 mmproj 齐
  缺点  比 Saluki 大 0.87 GB

先测 IQ2_XS-mtp：唯一变量是"更小的量化"，且 MTP 有保障。
```

顺带纠正一条：新闻说 Saluki"仅保留文字 I/O"已过时——HF 仓库 6 天前加了视觉
（`mmproj-Underdog-Saluki-27B-1.0-Q8_0.gguf` 629 MB / F16 928 MB）。

### 下载后实测的 GGUF 头（2026-10-14，本地解析，未占卡）

```
valid GGUF  version=3  tensors=866  kv=53
general.architecture              qwen35        ← 和我们三元 Bonsai 同一个架构家族
qwen35.block_count                65            （64 主层 + 1 nextn 层）
qwen35.context_length             262144        ← 原生 262K，与我们一致
qwen35.attention.head_count       24
qwen35.attention.head_count_kv    4             ← GQA 4 个 KV 头
qwen35.embedding_length           5120
qwen35.nextn_predict_layers       1             ← MTP 头【在】，投机解码可用
qwen35.rope.dimension_count       64
qwen35.rope.freq_base             10000000.0
```

**三条重要含义：**

1. **架构相同（qwen35）**。也就是说 Ternary Bonsai 2 27B 和 Qwen3.8-27B 是**同一个
   模型家族的不同量化**——不是"换一个模型"，是"同一个模型的更新版本 + 换一种量化"。
   这让对照干净得多：KV 几何（4 个 KV 头、65 层）一致，KV 占用可直接比较，
   变量只剩（a）底座版本、（b）量化方案（1.58-bit 三元 vs 2-bit IQ2）。
2. **MTP 头确认存在**（`nextn_predict_layers=1`）。投机解码可用，速度可比。
3. **原生 262144**，和我们一致——不需要 RoPE 扩窗，也就不涉及"扩窗掉质量"的问题。

（本地下载的文件是 8.17 GB，HF 页面标 8.77 GB；差异未解释，但 GGUF 头完整可解析、
tensors=866，文件本身有效。）

## 五、这条线多出来的能力（rc3 实测 --help 核实，2026-10-14）

**先纠正本文早先的两条猜测**（以 rc3 预编译包实测 `--help` 为准）：

```
✅ NVMe 在 rc3 里就有，只是名字不同
   --kvmem-nvme-gb GB     NVMe file in GiB (0 = off)
   --kvmem-nvme-dir PATH  NVMe directory
   （README 里那套 --kvmem-session-nvme-gb / --kvmem-session-cache-dir 是源码版 v0.17.0 的新名）
   语义待测：是"会话跨重启持久化"，还是仅仅是溢出文件

❌ 多会话不在 rc3
   --kvmem-conversations 在 rc3 的 --help 里【不存在】
   -np, --parallel N      明确写着 "only 1 is currently supported"
   → 换会话即丢、不能并行，这两条在 rc3 上拿不到
```

**rc3 实际提供的分层与上限开关：**

```
--kvmem-gpu-ratio R     GPU 槽池上限 = VRAM 的这个比例（默认 0.50）
--kvmem-cpu-gb GB       CPU 溢出竞技场（0 = 关）
--kvmem-nvme-gb GB      NVMe 溢出文件（0 = 关）
--kvmem-budget N        GPU 工作集 tokens（0 = n_ctx）
--kvmem-block-tokens N  块大小（默认 128）
--kvmem-gen-reserve N   解码余量（rc3 默认 256；IQ3 配方设 16384）
--kvmem-sink-tokens N   常驻前缀（默认 0 = 一块）
--kvmem-recent-tokens N 常驻最新后缀（默认 0）
--kvmem-method          recency | retrieval（默认 retrieval）
--kvmem-query-policy    legacy | user（配方用 user）
--kvmem-mtp-state       snapshots | auto | replay（有 MTP 时默认 replay）
--kvmem-harvest-v       预填充时把 V 以 raw-K 形式搬下主机
--kvmem-raw-k-nvme      把 raw-K 和 V 存到 NVMe（需 --kvmem-nvme-gb）
```

**`--kvmem-gpu-ratio` 是个新杠杆**：默认把 GPU 槽池限制在 **VRAM 的 50%**。也就是说
引擎是**主动自我限制**的，不是"能吃多少吃多少"。12 GiB 上默认上限是 6 GiB 槽池——
而 53,248 的竞技场在 q8_0 下约 1.8 GiB，远没顶到上限。**所以真正的约束是权重，不是 KV。**

另外注意：rc3 的 `--help` **没有列出 LRFU 相关开关**（`--kvmem-lrfu` 等），而 README 说
LRFU 是两个后端共享的。也就是说 LRFU 可能在 NInfer 侧有、rc3 的 llama.cpp 构建里没有。
我们本来就实测 LRFU 贡献约 0，所以影响不大。

## 六、闸门与成本

```
闸门（一次加载就能判死）
  1. 12 GiB 上能不能起
  2. 起来之后竞技场（budget+reserve）能开多大
  3. 能不能到 262K 上下文
  → 若竞技场只能开 45K 以下：我们真实负载 57.7-62K 会落在台阶上面，
    预填充掉进慢车道 → 对我们没意义，到此为止
  → 若能开 55K+：值得往下走（召回门 + greedy 门 + 曲线对照）

成本
  运行包    684 MB
  模型      8.77 GB
  视觉头    约 1 GB
  合计      约 10.5 GB 下载（不占卡）
  加载测试  占卡约 10 分钟
```

## 七、Phase 1 实测结果（2026-10-14，12 GiB，同卡同形状族）

配方 = 包自带 `start-iq3.ps1` 原样，**只把模型换成 IQ2_XS-mtp**，视觉头留 CPU。

### 闸门结果

```
闸门 1  能否在 12 GiB 上加载    ✅  11392 / 12288 MiB（剩 896 MiB）
                                事前估算 11.46 GiB，实测 11.39 GiB
闸门 2  上下文                   ✅  262144（与原生一致，无需扩窗）
闸门 3  召回                     ✅  PARROT-5/15/25 逐字全对 3/3
闸门 4  前缀复用                 ✅  258K 热探针 6.5 秒；日志 cache = 58971
闸门 5  竞技场                   ⚠️  卡在配方原值 53,248（显存不够抬）
```

### 对照（同一台机、同一张卡、同一形状族）

```
                        NInfer（我们）      llama.cpp + Qwen3.8 IQ2_XS-mtp
60K 预填充              711.5 tok/s         290.8 tok/s        NInfer 2.45x
60K 解码（散文 256 字）  35.2 tok/s          14.1 tok/s         NInfer 2.5x
258K 新算速率           332.4 tok/s         283.5 tok/s        NInfer 1.17x
258K 冷读墙钟           772 s               682.8 s（含 58,971 缓存）
258K 热探针             3.0 s               6.5 s
显存                    约 11.0 GiB         11.39 GiB
```

### 三条结论

1. **能力上通过。** 12 GiB 装得下、262K 通、召回逐字 3/3、前缀复用在两个后端都工作
   （`cache = 58971` 是引擎自己报的）。事前算不出的显存账，实测落在我预估的 1% 以内。

2. **预填充的差距在长上下文处收窄。** 60K 上差 2.45 倍，但那一半以上是**竞技场**造成的
   （NInfer 已调到 73,728，llama.cpp 卡在 53,248，而请求 59,534 在它之上）。
   258K 上两者都在各自竞技场之上、都进入计算受限区，差距缩到 **1.17 倍**。
   → 说明后端本身不慢，"小显存装不下大竞技场"才是主因。

3. **解码是真实短板，2.5 倍。** 而且**不能用 ngram 解释**：实测把 NInfer 的
   ngram 起草关掉，散文解码 35.196 vs 开着 35.153，**完全相同**（ngram 只在输出
   重复输入时才命中，散文上从不触发）。所以 14.1 对 35.2 是模型/MTP 路径的真实差距。

### 顺带纠正我们自己的一个数

生产文档里写的"解码 ~103 tok/s"是在**可预测输出**上量的（工具调用 JSON、重复标记行）。
散文上 NInfer 只有 **35.2 tok/s**（约 14 个汉字/秒）。两个后端都表现出这个规律：

```
              可预测输出     散文输出     倍数
NInfer        101-130        35.2        约 3-3.7x
llama.cpp      27.8-30.0     14.1        约 2x
```

机制是 MTP 草稿在接受率高的文本上命中率高。**所以"解码 103"这个对外数字需要加限定：
那是可预测输出；散文是 35。**

## 八、判断与后续

**结论：这条赛道在 12 GiB 上是可运行的，但性能上不划算。**

```
它的优势    ~~底座更新~~【已纠正，见第十二节：三元 Bonsai 2 27B 本身就是
            Qwen3.8-27B 的量化，两者同底座，"更新"不成立】
            原生 262K、MTP 头齐、视觉齐（含第三方视觉头）
它的代价    解码慢 2.5 倍（14.1 tok/s ≈ 5.6 汉字/秒，交互体感明显卡）
            竞技场被显存卡在 53,248，真实负载 59.5K 落在慢车道
            我们量过的 NInfer 优化（竞技场调优、device-profile）在这条线上不通用
```

### 质量对照（2026-10-14，与 NInfer 主线同一形状族）

**召回门 ✅ 通过。** llama.cpp 后端 + IQ2_XS-mtp，258K 形状 + PARROT-5/15/25 三个标记：

```
probe-258K OK wall=6.5s prompt_n=631 prefill=179.9 decode=30.0 finish=stop
ANSWER="Output PARROT-5 was written to the log at 5.
Output PARROT-15 was written to the log at 15.
Output PARROT-25 was written to the log at 25."
```

三行逐字全对。**KVMem 的检索在这条后端上同样工作。**

**greedy 门（与 NInfer 主线同 8 题、温度 0、seed 20261013）：** `greedy-qwen38.json`
vs 已存的 `greedy-int8arm.json`，逐字对比 + 人工逐组判定。注意口径：这里换的是
底座+量化+引擎，分叉是预期的——门要回答的是"分叉的方向"（新底座答得更好还是更差），
不是"有没有分叉"。

```
组          结果              判定
code        不同（命名不同）  平：同一算法都对（reverse_linked_list vs reverseList）
chinese     不同（措辞不同）  平：都对；qwen38 点名 B+ 树，略具体
english     不同（详略不同）  平：都对；int8 多一句人眼对蓝敏感，略完整
toolshape   不同（措辞不同）  平：都正确拒绝 + 请贴内容
reason      不同（排版不同）  平：同一条算路（120/2.5），都对
longctx     一字不差          平：精确引用在 IQ2 上完好
format      不同              int8 胜：Qwen/unknownB（族对、数诚实）vs
                              qwen38 自称 "Claude"、"180B"（身份错）
negation    不同              int8 胜：三条实质正确的否定；
                              qwen38 三条浅（伦敦/木石/300米），第三条边界错
                              （塔高 330 米，"只" 300 米约成立，不是否定）
```

**计：6 平、int8 胜 2、qwen38 胜 0。** 新底座在这 8 题上**没有任何质量优势**；
输的两组都是"知识/身份"类（自称、事实选择），不是推理类。IQ2 本身健康：
无乱码、无召回失败，longctx 逐字一致。

**综合判断（性能 + 质量）：不切换。** 解码慢 2.5 倍是硬伤，质量上新底座没有
给出任何补偿。llama.cpp 那台已停，生产恢复 NInfer（竞技场 57344）。

这条赛道的剩余价值只剩一条：**混合精度 `-ctv q4_0`**（官方已验证）可能把竞技场
抬到 73K，让日常档回快车道。但那要先过 greedy 门（4-bit V 会动开放式输出，
我们实测 rk4v4-e8 是 3/8）。在门通过之前，这条线封存。

**三个能把它问清楚的后手：**

```
① 混合精度 -ctk q8_0 -ctv q4_0（官方列为已验证）
   → V 砍半，省约 0.9 GiB → 竞技场可到约 73K → 真实负载回快车道
   代价：4-bit V 会移动开放式输出（我们实测 rk4v4-e8 greedy 等价 3/8）
   → 必须先过 greedy 门

② 质量对照（召回门 + greedy 门）—— 这才是"要不要换"的真正理由。
   速度快慢已有数；模型好不好还没量。

③ Saluki IQ2-mix（7.9 GB，比 IQ2_XS-mtp 小 0.27 GB）
   省下的显存只能多开约 4K 竞技场，不足以解决慢车道问题
   → 优先级低；除非它的工具调用质量明显更好
```

**当前状态：生产已恢复 NInfer（竞技场 57344，ngram 31）。llama.cpp 那台已停。**

## 十、其它模型候选的状态（2026-10-14）

### Saluki 27B：结构性出局（速度维度）

远程读 HF 上的 GGUF 头（Range 请求，未下载）：

```
valid GGUF  version=3  tensors=851  kv=52
general.architecture   qwen35
qwen35.block_count     64              ← 对比：带 MTP 的 IQ2_XS-mtp 是 65（64+1）
qwen35.context_length  262144
MTP keys: NONE                         ← 无 MTP 头
```

851 vs 866 个张量，差的 15 个正是 nextn 层。**Saluki 没有 MTP 头，不能投机解码。**
而我们的解码速度靠 MTP（NInfer 实测 2.1 倍；prism.3 官方在 5060Ti 上 MTP1 +34%，
接受率 97%）。

Saluki 的定位（agent 工具调用）与我们的生产用途重合，但用户的抱怨是**性能**，
而它在结构上拿不到投机解码——速度维度出局。质量维度（工具调用是否真比原版好）
没有量，但那不是当前要解决的问题。封存。

（另注：新闻说它"仅保留文字 I/O"已过时，HF 6 天前加了视觉投影器 629 MB。）

### Bonsai PTQ1 + prism.3：在测（同一模型，换跑道）

KVMem 有个 Bonsai 2 专用构建（prism.3，RTX 20/30/40/50 已编译），吃我们盘上已有的
`Ternary-Bonsai-2-27B-PTQ1_0.gguf`（5.54 GB）。这组对照最干净：**同一模型，
变量只剩后端+量化**。

```
NInfer .ninfer（8.87 GB）   生产：解码 35 tok/s 散文
llama PTQ1（5.54 GB）       未测：权重小 3.3 GB → 竞技场可开更大？
MTP 头                      社区 r3（849 MB BF16）→ graft 为 Q4_0（+228 MiB）
                            官方 5060Ti 实测 MTP1 +34%，接受率 97%
```

风险（官方自己写的）：长上下文召回有已知问题；30 系有编译架构但未经实机验证——
正好由我们验证。默认上下文 128K（可调），视觉默认 CPU。

下载中：运行包 742 MB + MTP 头 849 MB + 源码 38 MB（graft 脚本在源码包里）。
下一步：SHA 校验 → graft Q4_0 头 → 加载测试（峰值显存 + 262K + 召回门）。

## 十一、Bonsai PTQ1 + prism.3 实测 verdict（2026-10-14，12 GiB）

同一模型（Ternary Bonsai 2 27B），只换后端+量化。配方 = 包自带启动器原样，
MTP 开（graft 的 Q4_0 头），CPU 视觉。

```
                        Bonsai PTQ1+MTP1   NInfer 生产       Qwen3.8 IQ2
16K 预填充              243.1 tok/s        795.8            —
60K 预填充（小竞技场）  222.2              —                359.8
60K 预填充（大竞技场）  217.0              734.4            —
60K 解码                36.9 / 28.6        35.2（散文）      14.1
16K 解码                43.6               —                —
显存                    8186→9824 MiB      ~11000           11760
```

**三条结论：**

1. **解码打平（36.9 vs 35.2）。** MTP 头接上了，graft 有效。Qwen3.8 那条 14.1
   相形见绌——底座新不等于解码快。

2. **预填充慢 1.5-3.3 倍，而且与竞技场无关。** 16K（任何竞技场之下）只有 243，
   而 NInfer 同档 796。把竞技场从 34,816 抬到 73,728：217.0 vs 222.2，
   **纹丝不动**——NInfer 上那条 B+R 阈值规律在这条后端上不存在。
   瓶颈在预填充计算路径本身（很可能是三元解包/核未优化），不是检索。

3. **显存省约 1.2 GB 是真的**（9824 vs ~11000），但买不到速度。

**Verdict：封存。** 解码打平不够——冷读占体验大头，而冷读慢 3 倍。
附带的一个认识：`.ninfer` 制品 8.87 GB vs PTQ1 GGUF 5.54 GB，多出来的 3.3 GB
很可能就是预排好的快速核布局。**格式本身是性能的一部分**，不是单纯的体积。

召回门没跑：预填充差距已能定罪，为一个改不了 verdict 的数字再花 20 分钟
（258K 冷读）不值得。官方自己也写了长上下文召回有已知问题。如将来重审这条线，
召回是第一个要补的数。

## 九、尚未证实 / 待办

```
未证实  Saluki IQ2-mix 是否带 MTP 头
未证实  rc3 预编译包里 --kvmem-conversations 是否存在（源码版确定有）
未证实  llama.cpp 后端在 12 GiB 上实际占用（本文件的估算是估算）
待办    加载测试 → 记录峰值显存与可用竞技场
待办    若过闸：262K 召回 3/3 + 真实负载档的预填充速度
```

## 八、这件事和主线的关系

不是替换，是**第二条赛道**。主线（三元 Bonsai 2 27B + NInfer）的一切实测——竞技场、
召回门、量化格式判定、思考对照——都建立在 NInfer 上。llama.cpp 支是另一条引擎线
（v0.16.0-rc3 vs 我们 v0.18.0-ninfer-rc3），配方、开关、我们量过的优化都不通用。

它的价值在两点：
1. ~~**底座更新**~~【已纠正，见第十二节】三元 Bonsai 2 27B 的底座就是
   Qwen/Qwen3.8-27B（HF 官方元数据 `base_model`），两者同底座、同架构
   （`qwen35`），差别只在量化（1.58-bit 三元 vs 2-bit IQ2）。
2. **多会话与持久化**：NInfer 那支明确拒绝的能力，这条线上是设计内的。

---

## 十二、底座身份纠正 + 快引擎能挂别的模型（2026-10-14 收口）

本节结清第八、九节留下的全部悬念，并纠正其中一处错误判断。

### 12.1 纠正：三元 Bonsai 2 27B 就是 Qwen3.8-27B

第八、九节把"底座更新"列为 llama.cpp 这条线的优势，**这是错的**。

证据（HF 官方元数据）：

```
prism-ml/Ternary-Bonsai-2-27B-gguf   base_model: Qwen/Qwen3.8-27B
（我们生产的 Ternary-Bonsai-2-27B-ninfer-v3.ninfer 即该三元的 NInfer 打包）
```

也就是说：**我们早就在跑 Qwen3.8-27B**，只是 1.58-bit 三元量化。
`Qwen3.8-27B-GSQ-RCO-IQ2_XS-mtp` 是同一个底座的另一种（更高）精度量化。

因此"换到 Qwen3.8"不是换更新的底模，而是**同一套权重换量化**——第八节那条
"底座优势"不成立。第四节的猜测（"同一个模型家族的不同量化"）才是对的。

### 12.2 快引擎（NInfer）能读社区的 gguf_blocks_v1 制品

社区已把 Qwen3.8-27B 打包成 `.ninfer`（例：`fyb1214/Swift-1.5-Qwen3.8-27B-GSQ-RCO-NInfer`）。
其 README 写明：需要"能读 `gguf_blocks_v1` 容器布局的引擎构建；它与三元
`PQ2_0`/`PTQ1_0` 是两种方言，互不通用"。

实测（扫我们生产用的 `ninfer-kvmem-server.exe`，661 MB，未占卡）：

```
二进制内同时含有：
  gguf_blocks_v1
  gguf_iq1_s / gguf_iq1_m / gguf_iq2_xxs / gguf_iq2_xs / gguf_iq2_s
  gguf_iq3_xxs / gguf_iq3_s / gguf_iq4_xs
  gguf_q2_k / gguf_q4_k / gguf_q6_k / gguf_q8_0 / gguf_q4_0 / gguf_q5_0
  qwen3_5 / ternary
```

→ **同一支引擎，两种方言都支持**。所谓"不同方言"是**制品不同**，不是引擎不同。

### 12.3 实测：Swift-1.5 Qwen3.8 IQ2_XS 挂到快引擎（12 GiB）

制品 `swift15_iq2xs_mtp.ninfer`（8.77 GiB；SHA256 与官方 `SHA256SUMS` 一致）。

```
加载   权重 8.15 GiB 完整载入
       竞技场 73,728 装不下（运行时预留需 3.48 GB，仅剩 3.05 GB，差约 230 MiB）
       降到 61,440 成功（空闲显存仅 196 MiB）
```

实测（生产同款配方：int8、MTP adaptive-4、fast-prefill、device-profile auto）：

```
上下文        预填 tok/s    解码 tok/s    MTP 接受
16K           444.5         73.2          73/104
60K（散文）   437.5         27.8          —
258K          423.3         59.7          37/38
258K 召回     3/3 逐字正确（竞技场 61,440，超配 4.2 倍，靠 KVMem 检索兜住）
```

与生产并排（同口径，散文解码）：

```
                        预填 60K    解码 60K
生产 三元（1.58 bit）   712         35.2
Swift IQ2_XS（2.06 bit） 437.5       27.8     ← 预填 −39%、解码 −21%
```

**结论：同一支快引擎上，IQ2_XS 比三元慢。** 解码降幅基本等于数据量比（1.30 倍）；
预填降得更多（−39% > 1.30 倍），说明 IQ2 的解包核效率也不如三元。

### 12.4 质量对照（gsm8k 前 50 题，配对检验）

```
                 总分            思考开    思考关
生产 三元        94/100  94.0%   47/50     47/50
Swift IQ2_XS     97/100  97.0%   48/50     49/50
```

配对拆解（100 次请求）：两边都对 92、都错 1、**生产错/Swift 对 5**、
**生产对/Swift 错 2**。不一致对数 7。
McNemar 精确检验双尾 **p = 0.453 → 不显著**。

方向略微偏向 IQ2_XS（5:2），但样本太小，证明不了质量优势。

**必须标注的偏差**：Swift 是**微调版**（efficient-thinking），不是 base。
所以这个对比是"精度 + 微调"的混合，**不能单独归因给精度**。

### 12.5 llama.cpp 混合精度实测：崩塌

第八节留的后手①（`-ctk q8_0 -ctv q4_0` 把竞技场抬到 73K）已实测：

```
budget 49152 + reserve 16384 = 竞技场 65,536（≥ 请求 59,534）
加载         成功（peak 11,458 MiB / steady 11,439）
60K 预填     156.3 tok/s   （比 q8_0 基线的 217 更慢）
60K 解码     0.84 tok/s    （比基线 28.6 慢 34 倍；每 token 1185 ms）
显存         11,873 / 12,288（几乎撑满）
```

现场观测与数字一致：GPU 占用满、功耗仅约 60 W —— SM 在**忙等访存**，不是算力打满。

**未分开的变量**：本次同时动了（a）V 精度 q4_0、（b）竞技场大小。到底是
q4_0 把注意力路径搞坏，还是显存顶满导致 KV 反复倒腾，**没有分开**。
但无论哪个原因，这条线对"提速"已无出路（最好情况也只是解码打平）。

### 12.6 综合 verdict

```
速度   生产三元的快引擎是已知最优；IQ2_XS 更慢；llama.cpp 更慢且混合精度崩塌
质量   IQ2_XS 无显著优势（p = 0.45）；且制品是微调版，无法归因
底座   本来就是 Qwen3.8-27B，不存在"换更新底座"这回事
```

**结论：不切换，生产保持不动。**

这轮真正拿到的不是"更快的模型"，而是一个**能力**：
**快引擎能挂任意 `gguf_blocks_v1` 的 `.ninfer` 制品**（含 MTP、含 258K 检索召回）。
以后有更好的模型发布成 `.ninfer`，可直接挂上，**不必再碰 llama.cpp 那条慢线**。

### 12.7 结清第八/九节的待办

```
未证实  Saluki IQ2-mix 是否带 MTP 头           → 已结：无 MTP 头，结构性出局（第十节）
未证实  rc3 里 --kvmem-conversations 是否存在   → 仍在：源码版有，预编译包无（第五节）
未证实  llama.cpp 后端在 12 GiB 的实际占用      → 已结：IQ2_XS-mtp 加载 11.39 GiB（第七节）
待办    加载测试                               → 已做（第七节 + 12.3 + 12.5）
待办    262K 召回 3/3 + 真实负载档预填          → 已做：三条线都过了召回门（第七节、12.3）
```

---

## 十三、引擎优化三条路（2026-10-15，全部闭环）

问题："能在引擎上优化吗？" 结论先行：**三条路全走完了，没有低成本的提速空间。**

### 13.1 换引擎（wei231 sm_86 构建）：双死

```
现象  加载生产 .ninfer，30 秒退出：
      FATAL | artifact magic is not NInfer v2
含义  它们的制品格式是 v2，我们的是 v3，互不认。
      另：该引擎无 --kvmem-* 参数（上游 NInfer，无检索），
      12 GiB 上撑死 60-70K 固定 KV，262K 不可能。
结论  格式 + 架构双死。出局。
```

附带收获：`--help` 对比确认两条架构路线（KVMem 检索 vs 上游固定 KV），
以后选引擎先看有没有检索，没有直接毙掉，不用下载。

### 13.2 官方 v0.18.0 stable A/B：打平，留作备胎

```
对象  kvmem-v0.18.0-ninfer-windows-rtx30-sm86.zip（10-13 stable，SHA 对上）
      vs 生产 RC3（10-11）。同模型（只读引用）、同配方（竞技场 73728、
      int8、MTP4、ngram31、vision resident），只换 exe。端口 18220。

60K 预填   693 / 719      vs 712 / 734     → −2~3%（噪声）
60K 解码   33.5           vs 35.2          → −5%（边界，大概率噪声）
258K 召回  3/3 逐字正确                    → 长上下文门槛过
显存       ~11.0 GB       vs ~11.0 GB      → 一样
```

**结论：不换。** stable 没更快。包留盘（`H:\kvmem-v0180`，1GB）当备胎。
CLI 与 RC3 几乎一字不差（差 2 字符），**无 ActKV**（`--kvmem-action-lrfu`
不在两个构建里，要更新的版本才有）。

### 13.3 源码 + kernel A/B：默认路由是对的，不用写代码

```
源码  github.com/kvmem/kvmem（Apache-2.0，已克隆到 H:\kvmem-src，3841 文件）。
      三元 kernel 在 backends/ninfer/src/ops/linear/t2/（v2/MMA/SIMT 多条路），
      路由在 t2_dispatch.cpp 按形状选，注释的数在 RTX 3090 上量。
      另有 calibration 系统（device_calibration.cu，51KB）+ 环境变量开关
      NINFER_T2_DECODE / NINFER_T2_SMALLT（RC3 二进制里都在，无需重编即可 A/B）。

A/B   同模型同配方，只差解码 kernel（端口 18220/18221，无 vision 以省显存）：
      A 默认 v2 tensor-core   预填 760.4   解码 37.71
      B 强制 SIMT GEMV        预填 735.3   解码 36.73
      → v2 赢（+2.7%）。3090 上 v2 赢 50%+，3060 上只赢 3%，
        说明小卡 tensor-core 吃不饱，但还是比 SIMT 快。
```

**结论：默认路由在 sm_86 上正确，不换 kernel，更不用写 kernel。**
物理天花板：解码带宽 roof 约 60 tok/s，MTP 打折后 40 出头；
当前 37.7 已贴近，改 kernel 上限 +10%、大概率个位数，要几周专家工作。
**不值得。** 源码留着当文档看，不装构建工具链。

附带两组微调对照（同模型同配方，60K 散文解码；基线 A = 默认 v2、无 vision）：

```
D  NINFER_T2_SMALLT=v1（MTP 验证步换第一版 kernel）  解码 35.42  vs 37.71 → −6%，别碰
E  --draft-tokens 6（默认 4）                        解码 31.31  vs 37.71 → −17%！
   草稿越多散文越亏（接受率太低，白算）；adaptive-mtp 已动态下调，
   手动加码只加浪费。往小调不用动手（adaptive 在干）。
```

至此引擎旋钮穷尽：kernel v2 对、drafts 4 对、ngram/cublas/chunk 之前扫过、
KV int8 对、arena/device 已最优。**以后"调调参数"直接拿这张表挡。**

### 13.4 vision 成本：约 3%，留着

```
A 无 vision        预填 760.4   解码 37.71  （同一天）
C vision resident  预填 740.1   解码 36.45  （同一天，只差 vision）
→ vision 成本约 3% + 约 0.5GB 显存。
  （注：跨天 variance 约 3-4%，单次测量噪声带；±5% 才算数。）
```

用户决策：**vision 留着**（能力优先，3% 接受）。生产不动。

### 13.5 未来引擎/模型的四道闸（死线）

```
1. 有 KVMem 检索吗？没有 → 出局（262K 是硬门槛）
2. 读 v3 .ninfer 吗？不读 → 出局（v2/v3 互不认）
3. 12 GiB 装得下吗？（权重 + 竞技场 73728 + vision ≈ 11GB）→ 装不下 → 出局
4. 速度超生产 ±5% 吗？没有 → 不换（单次测量噪声带约 3-4%）
```

四道全过才值得动手。在此之前，生产就是答案。
