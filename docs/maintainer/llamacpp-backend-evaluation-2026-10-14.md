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
它的优势    底座更新（Qwen3.8-27B 的智能体基准远高于 Qwen3.5 系）
            原生 262K、MTP 头齐、视觉齐（含第三方视觉头）
它的代价    解码慢 2.5 倍（14.1 tok/s ≈ 5.6 汉字/秒，交互体感明显卡）
            竞技场被显存卡在 53,248，真实负载 59.5K 落在慢车道
            我们量过的 NInfer 优化（竞技场调优、device-profile）在这条线上不通用
```

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
1. **底座更新**：Qwen3.8-27B 的智能体基准（SWE-bench Pro 61.7、OSWorld 84.3、
   LiveCodeBench 90.3）远高于我们 Qwen3.5 系的三元底座。
2. **多会话与持久化**：NInfer 那支明确拒绝的能力，这条线上是设计内的。
