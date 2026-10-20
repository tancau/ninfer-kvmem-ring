# 执行单 · 2026-10-20 生产栈提速（换抓手，不换栈）

> 决定：**换"提速的入手点"，不换生产栈。** 生产继续跑厂商 KVMem RC3（vision + 262K 逻辑上下文 + 宿主层）；
> 多环嵌套 / 异步搬运 / 换模型 / 换并发 / int4 KV **全部不做**（理由见"不做清单"）。
> 依据：2026-10-20 的实测（本文末"今天的数"）+ 24 天生产日志的字段级统计。

## 排序（按量化收益）

| # | 事项 | 层 | 预计收益 | 验收口径 | 状态 |
|---|---|---|---|---|---|
| 0 | **DFlash2 快档**（第二服务，端口 8099，短提示专用） | 引擎启动配方 + 客户端路由 | 3.4K 提示 **200.5 t/s** vs 同引擎 MTP K=4 118.8（+68.8%）；短问答体感直接 | 三条短任务冒烟（算术/代码/事实）+ 引擎请求日志的接受率 | **已就绪**：`C:\Bonsai-App\Start-RC3-DFlash2-Fast.bat`；ZCode 已加 provider `bonsai2-fast`（contextWindow 32768、不支持图像）；冒烟测试已排队（等质量门腾卡） |
| 1 | **思考预算**（代理改写 `thinking.budget_tokens` + `output_config.effort`） | 代理/客户端 | 规划题 28.45s → 18.68s（-34%）；关思考 8.83s（-69%）。生产 24 天思考 token 1,320,638（占生成 70%，其中 79% 集中在 top 10% 请求） | MMLU-Pro 280 题，effort=low vs 基线 none 76.4%，判定阈值：降幅 ≤ 2 个点即上生产 | **通过**：effort=low **217/280 = 77.5%**（基线 214/280 = 76.4%）；配对 净 **+3 题（+1.1 点）**，McNemar 精确检验 **p = 0.701**，截断 0/0 → 质量中性，**可上生产** |
| 2 | **共享前缀钉住**（用树里已有的 `--auto-prefix-grid` / shared prefixes / long anchors / ring 模式） | 引擎参数 | root 678 条 0% 命中 = 2–4.2 小时冷算；长锚点 54.2% | 复跑 KV 召回统计：分路径命中率、分长度段、root 尺寸分布 | 待做（不需要改引擎） |
| 3 | **DFlash2 长上下文**（一处引擎 bug 修复，见下） | 引擎源码 | 53K 上 **64.5 t/s** vs 同引擎 MTP K=4 53.1 t/s（**+21.5%**）；短提示 K=7 **200.5 t/s** vs MTP K=4 118.8（**+68.8%**） | 同引擎同 flag 的 MTP 回归臂必须一致（今天 53.1 vs 53.1 ✓） | **修复已落地并验证**：`prefill.cpp` 一行；待提交 |
| 4 | **重启热启动**（前缀索引与热度跨重启保留） | 引擎 | 117 个引擎实例/24 天，每次重启后首个会话冷付 | 重启后首个请求的 prefill 秒数 | 待做 |

## 不做清单（含理由）

- **多环嵌套 + 异步读取**：优化的是我们已经做得最好的部分（召回 66.7%，多轮路径 99%）。剩余未命中 9–11 小时/24 天，且其中大部分是"新 token 本来就得算"，不是浪费。天花板 ≈ 每天 23 分钟。
- **换模型（iq3xxs）**：MMLU-Pro 80.7% vs 三元 76.4%（+4.3），但短上下文解码 52–64 vs ~85 tok/s、设备窗口 12,288 vs ~57K–74K，净亏。
- **int4 KV / 提高并发 / 换 24 GB 卡**：都不买命中率。
- **把自建树当生产**：它是 ring 原型（无 vision、缓存语义不同），生产继续用厂商 RC3。

## 今天的数（RTX 3060 12G，干净窗口，greedy，读引擎自身计时，3 轮中位）

| 场景 | decode | prefill | 备注 |
|---|---:|---:|---|
| 生产配方（KVMem RC3，MTP4+ngram31，thinking on），3.4K 提示 | 73.6 t/s | 905 t/s | 早先报的 63.8/54.4 是 ComfyUI 污染值 |
| 生产配方，53K 提示 | **65.6 t/s** | 770 t/s | 真实流量形状（57.7–62K/轮） |
| 生产配方 + ngram 63，53K | 60.2 t/s | — | 无增益 |
| 生产配方 + MTP draft 10，3.4K | 51.1 t/s | 962 t/s | 更慢且方差极大 |
| 生产配方 + MTP draft 12/15 | 起不来 | — | 运行时预留 4.28/4.75 GB > 可用 4.16 GB |
| 自建树引擎，MTP K=4，3.4K，thinking off | 118.8 t/s | — | DFlash2 的同引擎对照 |
| 自建树引擎，**DFlash2 K=5**，3.4K | **156.5 t/s** | — | +31.7% |
| 自建树引擎，**DFlash2 K=7**，3.4K | **200.5 t/s** | ~700 t/s | **+68.8%**；接受率 96.2 / 99.7 / 98.5% |
| 自建树引擎（修复后），**DFlash2 K=5 @53,324 token** | **64.5 t/s** | 511.6 t/s | 环窗口 32,768 + chunk 4096；接受率 42.3%、3.12 tok/轮、TTFT 103 s |
| 自建树引擎（修复后），MTP K=4 @53,320 token | 53.1 t/s | 551.3 t/s | 回归对照 → 补丁无害 |
| 快档冒烟（DFlash2 K=7, 32K, thinking off） | 161.8 t/s | 347 t/s | 3/3 答对，TTFT 112–190 ms |
| 对照 · 对方归档（同口径，另一台部署） | 108.7 t/s | 640 t/s | 接受率 96.2% · 7.77 tok/轮 |

**DFlash2 的边界：根因是我们自己的一行代码**

原边界写成"提示必须在一个 `--prefill-chunk` 内，否则 HTTP 500"。**这不是厂商策略，也不是 dflash2 的
固有限制**：`src/models/qwen3_5/program/prefill.cpp`（ring 移植的 LOCAL FIX 里）首块映射写的是

```cpp
ensure_sequence_kv_mapped(sequence, initial_map,
                          initial_map < prompt_tokens ? initial_map : backend_materialized);
```

`backend_materialized` 是**按后端分型**的：MTP 要 `prompt+extent-1`、DFlash 要 `prompt`、
**DFlash2 要 0**（dflash2 没有 full draft pool，草稿 KV 在环里）。ring 分支却把 backend 侧无条件写成
`initial_map` → 提示一旦跨 chunk，就去要一个规划器从没给过的 backend 空间 →
`backend KV materialization requested without an allocation` → HTTP 500，引擎随后 latch 住失败、
后续请求全 503。实测复现：3,408 token 提示 + `--prefill-chunk 256` 必炸；把 chunk 提到覆盖提示就正常
——所以这个边界此前被误读成"dflash2 的限制"。

修复（`min(initial_map, backend_materialized)`）对 MTP/DFlash **可证明无影响**（它们的
`backend_materialized` 从不低于 chunk，min 是恒等）；回归臂 53K 上 53.1 t/s 与修复前一致。

**修复后 DFlash2 可以小 chunk + 长提示**，这正是 12 GB 卡需要的：chunk 64K 的目标模型 prefill 暂存
要 17.18 GB（64 层 × 8 头 × 128 维 × chunk 的 BF16 K/V），chunk 4096 只需约 1 GB。
实测可用配置：设备窗口 32,768 + chunk 4096 + 环模式（`NINFER_KV_RING=1`）→ 53,324 token 提示跑通。

> 更正：本执行单早前把它写成"预留按整块 chunk 的 BF16 KV 算，改成 live window"——不准确。
> dflash2 的 context materializer 本来就按 ring live window 算（`startup.cpp:583-591`）；
> 爆卡的是**目标模型**的 prefill 暂存。

**另需产品方支持**：KVMem worker 本身拒绝 `--spec dflash2`（"KVMem requires ordinary/fixed/adaptive
MTP1..15, ngram1..63"），所以 dflash2 要装进**生产栈**（带 vision + 宿主层）需要 worker 侧支持。
需求文档见 `docs/maintainer/kvmem-dflash2-request-2026-10-20.md`（含复现步骤与数据）。

## 思考档位阶梯（生产配方 RC3，greedy，15 次全部答对）

| 任务 | xhigh（模板默认） | high | medium | low | off |
|---|---:|---:|---:|---:|---:|
| 算术 | 4.85s（思考 189） | 4.39s | 6.05s | **4.19s** | 3.70s |
| 规划 | 28.45s（1500 截断） | 24.47s | 24.64s | **18.68s** | 8.83s |
| 代码 | 5.92s | 5.12s | 4.05s | **4.24s** | 1.60s |

结论：`medium` 无收益；`low` 既快又答对（此前"low 不可靠"是把 max_tokens 设成 2000 造成的截断假象）；
`off` 最快但改变模型行为，只做按需开关，不做全局默认。

## 第 1 项的实现与回滚

- 实现：`keepalive_proxy.py` 增加请求体改写，两个环境变量控制，**默认全关**（不设即行为不变）：
  - `NINFER_THINK_BUDGET_CAP`：把 `thinking.budget_tokens` 收到上限（Anthropic 路径的字段）；
  - `NINFER_REASONING_EFFORT`：写 `output_config.effort`（Anthropic）与 `reasoning_effort`（OpenAI 路径）。
- 端到端已验证：代理日志 `request rewrite: thinking.budget_tokens 24576->4096, output_config.effort ->low`，
  引擎日志 `requested_reasoning_effort=low, thinking_budget=4096, effective_thinking_budget=4096`。
- 回滚：把两个环境变量从配方里去掉即可（代理本身仍然只做 keepalive）；原文件备份 `keepalive_proxy.py.bak-20261020`。
- 上生产的前置：MMLU-Pro（effort=low）不低于基线 76.4% 太多。

## 待拍板

- 第 3 项（DFlash2 预留口径）动源码：做 / 先搁置。
- 第 1 项的最终档位：`low`（推荐）还是 `off`（按任务类型分流）。
- 第 2 项是否并入本执行单（需要先复跑 KV 召回统计拿分路径基线）。
