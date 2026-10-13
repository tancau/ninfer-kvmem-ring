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

## 五、这条线多出来的能力（NInfer 那支给不了的）

```
--kvmem-conversations N        同时保留多个会话（默认 1 = 换会话即丢）
--kvmem-conversations-gb GB    活跃+非活跃会话的内存软上限
--kvmem-session-ram-gb GB      会话内存软上限别名
--kvmem-session-nvme-gb GB     不活跃会话落盘
--kvmem-session-cache-dir DIR  落盘目录
--parallel N                   1..4 条推理车道共享权重
```

前四条正是我在 NInfer 上试过、**全被拒**的"会话持久化"（见
`prefix-persistence-2026-10-13.md`：`--slot-save-path` / `--auto-save-evicted` /
`--disk-kv-*` 四种组合全被 KVMem 运行时拒绝）。

**但要注意**：rc3 预编译包的 release notes 明确写 "This release does not add ...
NVMe offload"，README 也说 NVMe 快照是 source 版（v0.17.0）才有、rc3 预编译包早于该特性。
所以 `--kvmem-conversations` 可能在 rc3 里（主机内存多会话），**NVMe 落盘要源码自编**。
这条待加载测试时用 `--help` 核实。

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

## 七、尚未证实 / 待办

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
