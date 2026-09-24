# 给 NInfer 移植 KVMem 能力 —— 技术方案（**修订版 v2**）

> v1 有三处实质性错误，本版已按源码查实修正。修正记录见 §0。
> 目标：NInfer 单引擎在 12GB 卡上把上下文从 160K 推到 **256K**，且 decode **不随深度衰减**。
> 基线（实测）：nvfp4 160K = decode 33.8 t/s @60K；rk8v4 96K = 51.2 t/s @60K。

---

## §0 修正记录（v1 错在哪）

| # | v1 写的 | 实际（源码证据） | 影响 |
|---|---|---|---|
| 1 | "每步取回（per-step fetch）" | ❌ 错。KVMem **每轮只检索一次**，在 prefill→decode 边界；之后 pinned。<br>证据：`docs/architecture.md` "Retrieval... After retrieval the selected blocks are **pinned**"；`docs/retrieval-stagein-optimization.md` "copies sit at the **prefill→decode boundary**" | 性能模型完全变了：**decode 是平坦的**（只读驻留窗口），代价转移到 **TTFT** |
| 2 | 没提**有损性** | ❌ 漏。KVMem 自称 **"block-sparse KV working-set manager"**，模型**只注意选中的子集**，不是全历史 | **最大遗漏**。"256K 上下文" ≠ "注意全部 256K" |
| 3 | "必须改内核做稀疏位置" | ✅ 结论对，但理由不完整。根因是 **NInfer 与 llama.cpp 的掩码语义不同**：<br>NInfer `small_t_*.cuh`: `const int window = positions[tokens-1] + 1;`（**位置区间**，每个位置 0..p 都必须有页）<br>llama.cpp FA：掩码是 `pos_j ≤ pos_i`（**比较存储的 pos**，天然支持稀疏子集） | 决定了改动形态：**要么加"按块 -inf 掩码"（改动小但计算量仍 O(p)），要么改成"驻留单元数 + 位置比较"（改动中等但换来平坦 decode）** |
| 4 | 没提**单轮生成长度上限** | ❌ 漏。KVMem："One generation therefore cannot exceed `--kvmem-gen-reserve`" | NInfer 侧需要等价的"生成预留"策略 |
| 5 | 性能预测 35–40 t/s | ❌ 偏低。按修正后的模型应 **~50 t/s 且平坦** | 见 §4 |

---

## §1 PoC 硬数据（本机实测，仍然有效）

`C:\ninfer-build\poc-vmm-host-kv\poc.cu`（RTX 3060 12G，256MB × 20）

| 读取来源 | 带宽 |
|---|---|
| 设备内存 | **308.0 GB/s** |
| 主机 pinned（跨 PCIe） | **24.5 GB/s** |
| 比值 | **12.6×** |

**用途**：估算"取回/驱逐"的 PCIe 代价。**不再**用来否定方案——因为取回是**每轮一次**，不是每步。

---

## §2 KVMem 的真实机制（源码实证）

```
GPU 池 = budget（选中的工作集）+ gen_reserve（解码预留）
  · 每轮在 prefill→decode 边界做一次检索，选中块 pinned
  · 检索打分 = mean-K（F32、pre-RoPE、首次写入时捕获）与查询的 dot
  · 检索 query = 最后一段 role=user 的 span
  · 选中块按 **原始 pos** 压紧到连续 slot；RoPE 坐标 = 原始 pos
  · 冷块 stage-in = copy_k_gpu/copy_v_gpu + slab H2D（32 MiB slab）
  · 未选中块：不驻留（在 host 存储里），本轮不可见
  · GDN/循环状态每 token 更新，不占这些注意力 slot
```

**关键**：llama.cpp 的 FA 掩码比较存储的 `pos`，所以"压紧到连续 slot + 原始 pos"就能让 FA 只注意选中子集，**FA 一行不用改**。

---

## §3 NInfer 侧的真实情况（源码实证）

| 层 | 文件 | 结论 |
|---|---|---|
| 注意力掩码 | `ops/softmax_attention/dense/causal_cache/small_t_*.cuh`<br>`const int window = positions[tokens-1] + 1;` | ⛔ **位置区间寻址**，与 llama.cpp 语义不同 → **必须改** |
| 页表 | `causal_softmax_attention(..., kv_table_rows, cache)` | ✅ 走页表间接寻址，取回路径在物化层 |
| KV 量化 | `ops/kv_cache/`（rk8v4/k8v4/nvfp4/int8/fp8/bf16） | ✅ 现成 |
| 主机存储 | `core/host_kv_arena.*` + `program/storage/host_kv_store.h`<br>`DeviceKVPagePool::copy_to_host/from_host` | ✅ 现成 |
| 页 residency | 页同时跟踪 `device_resident` / `host_resident` | ✅ 现成 |
| 压力规划 | `program/planning/pressure.cpp`（147KB） | ✅ 现成，但**只驱逐非活跃状态** |
| VMM 池 | `core/evictable_kv_pool.*` | ⚠️ 只借**空闲** granule，**无主机镜像** |
| 前缀物化 | `ops/context_kv_materialize/` | ✅ 可复用为"把选中块物化进窗口"的搬运器 |

---

## §4 性能模型（修正后）

设驻留窗口 W、rk8v4 26,112 B/token、设备 308 GB/s。
实测锚点：rk8v4 96K 全注意 = 51.2 t/s，步长 43 ms（MTP3 ≈ 2.2 tok/步）。
→ 权重+固定开销 ≈ 43 − (96K×26KB/308GB/s = 8.1 ms) = **~35 ms**

| 方案 | 驻留窗口 | 每步 KV 读 | 步长 | **预估 decode** |
|---|---|---|---|---|
| 现状 rk8v4 96K | 96K（全注意） | 8.1 ms | 43 ms | **51.2 t/s**（实测） |
| 现状 nvfp4 160K | 160K（全注意） | 6.8 ms | 65 ms | **33.8 t/s**（实测） |
| **KVMem 式 256K** | **64K** | **5.4 ms** | **40 ms** | **~54 t/s，且平坦** |
| KVMem 式 256K | 32K | 2.7 ms | 38 ms | ~58 t/s，且平坦 |

**TTFT = 现有 prefill + 一次检索**
- 检索原始字节：选 64K 窗口 = 1.67 GB ÷ 24.5 GB/s = **68 ms**（纯带宽）
- 实测参照（KVMem 在 5090 上）：16K 检索 1.31 s；128K 早期 229 s → 优化后 ~20 s
- 我们的实现若做到 KVMem PR2/PR3 的水平，**256K 轮次 TTFT 增加约 1–5 s**（主要不确定项）

⚠️ **最大不确定项**：检索实现质量。KVMem 从 229 s 优化到秒级靠的是"批量 get/set + D2D 布局 + 预取 V"，**这些优化必须一起做**，否则会出现"每 token 同步"的性能灾难。

---

## §5 内核改动形态（**已按源码定准**）

**源码事实**（`dense/causal_cache/small_t_nvfp4.cuh`）：内核里 `key` **就是位置**，直接用于
```c
const int window = last_pos + 1;                                    // 循环上界 = 位置
physical_page = physical_pages_s[(position >> kPagedKVPageShift) - first_page];
const int page_offset = position & kPagedKVPageMask;                // 页内偏移 = 位置低位
... key0 <= qabs0                                                    // 掩码 = 位置比较
```
⇒ 位置 → (物理页, 页内偏移) 是**直接映射**。所以**不需要**把 slot 与 position 分离的大改。

### 形态 A′（推荐）：dummy 页 + 块位图 + tile 跳过

1. **页表**：未选中的位置块，全部指向**同一个 dummy 页**（1 页共享，不占显存）。
2. **新增一张"每块是否选中"的位图**（`selected_blocks`，与 block table 同粒度）。
3. **tile 循环里加一次跳过**：`if (!selected[block]) continue;`
4. 页表本身仍按位置索引 → **`window`/`positions`/RoPE 语义全部不变**。

| | 效果 |
|---|---|
| 显存流量 | ∝ 选中集（未选中块只读 dummy 页，L2 命中） |
| 计算量 | ∝ 选中集（跳过的 tile 不做 MMA） |
| **decode** | **平坦**（只与驻留窗口有关，与上下文无关） |
| RoPE / 位置语义 | **不变**（仍是原始位置） |
| 改动面 | 加位图 + 循环跳过 + 页表构建；**不重写注意力数学** |

### 形态 A（仅掩码，最省事但无效）
只做"dummy 页 + -inf 掩码"，不跳过 tile。显存流量降了，但**计算量仍 O(p)** → decode 仍随上下文衰减。**不推荐**，仅可作过渡。

### 形态 B（完全压紧，改动大，不必要）
把 slot 与 position 分离、按 slot 压紧。既然 A′ 已能达到同样效果，**不必做**。

### 为什么 A′ 可行而 KVMem 不需要它
llama.cpp 的 FA 掩码比较**存储的 pos**，所以 KVMem 只需"压紧 slot + 原始 pos"就能稀疏。NInfer 的掩码是**位置区间**，所以必须显式告诉内核"哪些位置块被选中"——这正是位图要解决的问题。


---

## §6 新增组件清单

| # | 组件 | 复用 | 新写 |
|---|---|---|---|
| ① | 活跃序列的主机 KV 存储与 residency 生命周期 | `HostKVArena`/`HostKVExtentStore`/`copy_*` | 生命周期管理（现在只用于 checkpoint/前缀） |
| ② | 驻留窗口 + 检索策略 | `candidate_selector`/`linear_topk`（块打分） | mean-K 捕获、每轮检索、pinned 语义 |
| ③ | 稀疏/压紧注意力（§5 形态 A 或 B） | 现有 MMA 路线 | 位置/掩码解析（+ 形态 B 的循环上界） |
| ④ | 调度与物化接线 | `context_kv_materialize` 搬运器 | 每轮 prefill→decode 边界的检索/驱逐/建表 |
| ⑤ | 生成预留策略（§0-4） | — | 单轮生成长度上限 + 溢出处理 |

---

## §7 验证判据（照 KVMem/NInfer 方法学）

| 级 | 验什么 | 判据 |
|---|---|---|
| L0 | residency 不变量 | 每页 device/host residency 与引用计数自洽，无泄漏 |
| L1 | 取回数值等价 | 同一页 host 往返后注意力输出**逐 bit 相同** |
| L2 | **T>1 引擎侧** | 形态 B 必须 T>1 验证（T=1 掩盖布局错误） |
| L3 | PPL | `ninfer-perplexity --quick` 对比全注意基线（rk8v4 4.346413） |
| L4 | 长上下文召回 | 三针检索（1/3、2/3、9/10 埋码），32K/131K/250K 全中 |
| L5 | 负控 | 假位置/错窗口尺寸必须被拒；**"对照组全零"是方法失效信号** |
| L6 | 速度/显存 | 引擎自报 timings；decode 应**平坦**（这是形态 B 的核心验收） |
| L7 | **有损性边界** | 明确测出"窗口 < 上下文"时丢什么（needle 全中但细节可能丢） |

---

## §8 工作量与风险

| 项 | 量级 |
|---|---|
| ① 主机 KV residency 管理 | 中 |
| ② 驻留窗口 + 检索策略 | 中 |
| ③ **内核（形态 B）** | **大**（最难，T>1 数值验证） |
| ④ 调度接线 | 中 |
| ⑤ 生成预留 | 小 |
| L0–L7 验证 | 中 |
| **合计** | **引擎级项目，量级"周"** |

**风险排序**
1. **检索实现质量**（可能重现 KVMem 229 s 的坑）→ 必须先做批量搬运 + D2D 布局
2. **形态 B 内核的静默错误**（T=1 全绿、T>1 全乱）→ 必须引擎侧 T>1 dump
3. **有损性带来的质量下降**（agent 任务可能丢关键细节）→ 必须 L7 实测
4. GDN 循环状态与稀疏注意力的层间交互 → 需验证

---

## §9 内核原型验证结果（**已实测通过**）

### 改动（3 个文件 + 2 个共享头，全部带 `.orig` 备份）

| 文件 | 改动 |
|---|---|
| `ops/kernel/paged_kv_address.cuh` | 加 `paged_kv_block_selected()` 判据；`PagedKVDirectMetadata` / `PagedKVBatchMetadata` 各加 `selected_blocks` 字段 |
| `ops/softmax_attention/dense/causal_cache/launch.h` | `CausalSmallTInvocation` 加 `selected_blocks`；原型位图 helper（`NINFER_PROTO_SELECT`） |
| `.../causal_cache/small_t_nvfp4.cuh` + `.cu` | 内核加参数；4 处掩码加选中判据（解码路径） |
| `.../causal_cache/prompt_nvfp4.cuh` + `.cu` | 4 处掩码加选中判据，**并让 `full_score_tile` 快速路径不能绕过掩码**（prefill 路径） |

**位图粒度 = KV 页 = 64 token**（`kPagedKVPageSize`）；`nullptr` = 全选中。

### 验证判据与结果（PPL，261,167 token，同语料同协议）

| 判据 | 预期 | 实测 | 结论 |
|---|---|---|---|
| **A 回归**：`selected_blocks == nullptr` | 必须与原版**逐 bit 一致** | `mean_nll 1.729799 / PPL 5.639521`，与原版**完全相同**；29 个逐窗口采样点 diff 全为 0 | ✅ **PASS** |
| **B 功能**：`NINFER_PROTO_SELECT=even`（丢一半页） | 必须**显著变差**（否则是死代码） | `mean_nll 1.965951 / PPL 7.141698` = **1.266×** | ✅ **PASS** |

### 关键副产物（影响后续设计）

**丢掉 50% 的全注意力 KV，PPL 只涨 26.6%。**
原因：Qwen3.8-27B 是**混合注意力**，**GDN 层是循环状态、不受掩码影响、仍看到全部历史**，大幅补偿了全注意力层的 KV 损失。

⇒ 而且这是**最笨的策略（隔一个丢一个）**；换成按查询相关性检索，损失会显著更小。
⇒ **驻留窗口可以比原计划更小**（质量损失可控），进一步扩大上下文。

### 踩到的坑（记下来）

**第一次测试 B 失败**：我最初只改了 `small_t_nvfp4`（解码路径），但 **PPL 跑的是 prefill，走 `prompt_nvfp4`** → 掩码根本没被执行到，B 的 PPL 与 A 完全相同（死代码）。
**教训**：① 测试必须能证伪；② **多轮对话里第 N 轮的 prefill 也要注意被检索的历史，所以 prompt 路径同样需要掩码**，不是只有解码路径。

### 下一步（M1 余下部分）

**④ 位图接线（已完成）**
- `PagedKVLayerView` / `PagedKVBatchLayerView` 加 `selected_blocks` 字段
- `PagedKVCache` 加成员 + `set_selected_blocks()` / `selected_blocks()`
- `decoder_state.cpp` 两个 view 构造点传入位图
- `small_t_nvfp4.cu` / `prompt_nvfp4.cu` 的 invocation/metadata 从 `cache.selected_blocks` 取值
  （环境变量 `NINFER_PROTO_SELECT` 仅作测试回退）
- ✅ **验证**：接线后回归仍**逐 bit 一致**（PPL 5.639521，62 采样点 0 差异）→ 接线是惰性的

**⑤ 策略机制（已完成）**
- `PagedKVCache` 加设备位图缓冲（懒分配 + 析构释放）
- `install_selection(span<const uint32_t>)` —— 安装显式位图（供检索打分器使用）
- `install_sink_recency_selection(committed_pages, sink_pages, window_pages)` —— sink + 最近窗口的占位策略
- `clear_selection()` —— 回到 dense

**仍待做（按依赖顺序）**

| # | 待做 | 说明 |
|---|---|---|
| 1 | **策略调用方** | 在"每轮 prefill 后、decode 前"用 `kv_store` 的 `committed_columns` 调用 `install_sink_recency_selection()`。**做完只能演示 decode 变平坦，上下文不会变长**（显存未省） |
| 2 | **mean-K 检索打分** | 替换 sink/recency；需给 KV 写入路径挂钩子捕获每块 mean-K（KVMem 的做法） |
| 3 | **主机 KV 存储 + dummy 页表** | 未选中块指向同一个 dummy 页 → 真正省显存 → **真正扩上下文** |
| 4 | 铺开到 `rk8v4` 与 prompt 各 dtype | 机械重复 |
| 5 | 完整验证 | T>1 oracle、三针检索、速度/显存 |

**⑥ 环形回收（ring / recycle，已实现，功能验证进行中）**

让**设备池小于逻辑上下文**成为合法配置：地址空间仍是逻辑大小（`max_context`），但物理页池可以更小；
序列增长超过池时，分配器**复用最老的逻辑页**（`recycle_cursor`），检索掩码把被复用的旧位置挡在注意力之外。

### 约束层清单（原版引擎是"层层设防"，共 8 层）

| # | 层 | 位置 | 处理 |
|---|---|---|---|
| 1 | 启动检查 | `src/serve/serve_options.cpp` | 去掉 `kv_capacity >= max_context` |
| 2 | 规划器曲线 | `.../planning/startup.cpp`（**两处**） | `minimum_pages` 由 `max(logical, concurrency)` 改为 `concurrency` |
| 3 | plan_cache | `.../state/decoder_state.cpp` | 去掉 `physical_page_groups < logical_pages` |
| 4 | 准入可行性 | `.../planning/pressure.cpp` | `isolated_request_feasible` / `persistent_backfill_safe`：逻辑页需求**钳到池大小** |
| 5 | 峰值拟合 | `.../storage/context.cpp` | `physical_peak_fits` + 两个 deficit：钳位（**主 KV 与后端 KV 都要**） |
| 6 | 物化预检 | `pressure.cpp`（`revalidate_materialization`） | 靠 #5 通过；**曾用插桩定位**（见下） |
| 7 | 页预留 | `.../storage/kv_store.h`（4 处 `resize_reservation`） | 预留改为**逻辑承诺**（见下） |
| 8 | 整段提示一次性映射 | `.../program/prefill.cpp` | 改为**按 prefill 块增量映射** |

**关键教训**：同一约束出现在**多个不同层**（启动 / 规划 / 准入 / 会计），只改一处不够；
定位靠**插桩打印**（在 `revalidate_materialization` 的每个 `StalePolicyState` 返回点、`fits` 失败点、entitlement 断言点打 `fprintf`），
否则会一轮一轮盲撞。

### 会计模型（最重要的一处认知）

引擎的不变式是：**一个 owner 独占的设备 KV 页数 = `entitlement`（常驻 + 预留）**，且必须与请求计划一致
（`commit.cpp` 里有 `actual != expected` 的**精确断言**；`owner_exclusive_resources` 用
`常驻页 + (entitlement - 已映射页)` 计算）。

⇒ 所以**不能把预留钳到池大小**（那样断言必失败）。
正确做法是**分离逻辑与物理**：

| 概念 | 语义 | 上限 |
|---|---|---|
| `reservation` / `entitlement` | 该 owner 的**逻辑**页承诺 | 逻辑页容量（`page_capacity_`） |
| `allocated_pages_` | 真正占用的**物理**页 | 设备池（`usable_pages()`） |

实现：`DeviceKVPagePoolSpec` 加 `allow_logical_reservation`（仅当 `池 < 逻辑容量` 时为真）；
`can_resize_reservation` 在该模式下不再用 `usable_pages()` 约束预留；`available_pages()` 钳到 0 防下溢。
dense 模式（池 ≥ 上下文）该标志为假，行为完全不变。

**而环形回收在会计上自洽**：`DeviceKVPagePool::dematerialize_one` 会 `++reservation.pages_`（把页**还回**预留），
`materialize` 再 `--reservation.pages_` —— 一进一出净零，所以预留值恒定 = 逻辑 entitlement。

### 掩码时序（第二处关键认知）

**回收发生在映射时**（`ensure_mapped_to_tokens`），而注意力在映射之后、提交之前。
所以掩码必须**在映射之后、注意力之前**安装——原先把安装放在 `commit_sequence_kv`（提交后）**来不及**，
会让 prefill 读到刚被回收（内容已被新数据覆盖）的旧页。

实现：把安装移到 `ensure_sequence_kv_mapped()`（映射后立即安装），`commit_sequence_kv` 保留一次（无害）。
掩码的 `committed_pages` 用**本次映射的前沿**，于是 prefill 第 N 块看到的是
"以第 N 块末尾为基准的最近窗口"，正好与被回收的位置对齐。

### 其他必须修的细节

| 问题 | 症状 | 修法 |
|---|---|---|
| 位图只按 `committed_pages` 分配 | prefill 当前块的键（逻辑页 > committed）→ **越界读**位图 | 位图跨度改为**整个逻辑页空间**（`page_count(max_context_)`） |
| sink 与回收冲突 | 被回收页上的"sink"读到的是新数据 | 池 < 逻辑容量时**自动禁用 sink** |
| 后端（MTP）KV 未装掩码 | 草稿模型读到被回收的旧页 | 后端 KV 装**同一掩码**（两者都是 `PagedKVCache`） |
| 掩码指针随 CUDA Graph 固化 | 启动时捕获 Graph，`selected_blocks_ == nullptr` 被烧进 kernel 参数 → **decode 掩码失效（静默）** | 环模式下**构造时就预分配全 1 位图**，指针稳定且非空 |

### 验证

| 判据 | 预期 | 实测 | 结论 |
|---|---|---|---|
| **A 回归**：池 == 上下文（回收循环惰性） | 与原版**逐 bit 一致** | `mean_nll 1.729799 / PPL 5.639521`，62 采样点 diff 全为 0 | ✅ **PASS** |
| **A′ 全改动后回归**（8 层约束 + 会计模型 + prefill 按块映射 + 掩码四处修正之后重跑） | 仍与原版**逐 bit 一致** | `62/62 采样点 diff 全为 0`；overall `261167 / 1.729799 / 5.639521`，**delta = 0**；四领域 PPL 亦逐位相同 | ✅ **PASS（生产配置零影响）** |
| **B 启动**：池 32K + 上下文 128K | 引擎能启动（原版硬性拒绝） | `capacity \| KV 32,768 tokens \| pages 512/2,048`，VRAM 9233 MiB（对比 160K dense 11955 MiB，**省 2.7 GB**） | ✅ **PASS** |
| **C 功能**：喂 > 池的长提示（~79K token），针在开头 | 回收触发；被回收位置**不可检索** → 答不出 `BLUEBIRD-42` | **机制验证通过**：60K token 提示（池仅 32K）+ 主机降级 → **无任何错误完成**（`prompt_tokens 60322`，wall 250s） | ✅ **机制 PASS** |
| **C′ 检索判读**：同 60K 提示，`max_tokens=256`，在**完整响应（含思考）**里搜针 | 模型**看不到**开头的针，且输出**连贯** | 模型明确回答：**"The text doesn't seem to contain any specific 'secret access code' - it's just a long stream of generated words and phrases"**；全响应无 `BLUEBIRD`；输出连贯（准确描述文本性质） | ✅ **PASS** |
| **C″ dense 对照**：同 60K 提示、池 131072（全设备） | 模型**应能**回忆出针 | **`content: "BLUEBIRD-42"`**，思考里复述了原句；`RECALL: YES` | ✅ **PASS** |

### ✅ A/B 决定性结论

| 组 | 提示 | 设备池 | 驻留窗口 | 结果 |
|---|---|---|---|---|
| dense 对照 | 60K token | 131K（全在设备） | 全上下文 | **答出 `BLUEBIRD-42`** |
| 环模式 | **同一条** 60K | **32K** | ~28.7K（其余 D2H 到主机） | **"文本里没有这个码"** |

⇒ ① 针可回忆（不是提示设计问题）② 被降级到主机的页**真的不可检索**（掩码与降级位置严格对齐）
③ 输出连贯（无陈旧槽位污染）④ **32K 设备池承载 60K 序列（1.9×）**

**KVMem 式机制（设备池 = 注意力窗口；主机 = 历史）在 NInfer 内端到端成立。**

**C′ 为什么算强证据**：模型(a)在**整个响应**里都没出现 `BLUEBIRD`，(b)**明确声明文本中不存在这个码**，
(c) 同时**准确描述了其余文本**（"generated words and phrases"）→ 说明注意力**只覆盖了驻留窗口**，
被降级到主机的开头部分**既没被读到、也没污染注意力**（否则输出会乱或会幻觉出一个码）。

**C″ 是必需的对照**：若 dense 也答不出，则本测试不具判别力（提示本身的针太弱/GDN 也不记得），
需要换更醒目的针或更短的距离重做。

**测试 C 第一阶段的结论**：60K token 提示在 32K 设备池上跑通（序列 60K、驻留窗口 ~28.7K、其余 D2H 到主机），
**四个缺陷全部修复**。但 `max_tokens=64` 被思考通道吃光（`reasoning_tokens: 64`）→ `content` 为空，
**无法判读是否回忆出针**；需 `max_tokens=256` 重跑 + **dense 对照**才能定论。

**另一个真实问题**：同一提示第二次请求（走**前缀复用**）报
`isolated-feasible request is blocked in an idle Engine`；重启引擎（清缓存前缀）后正常
→ **reuse 路径与环模式不兼容**，需修（否则多轮对话不可用）。

### 测试 C 的根因与修复：回收改为"降级到主机"（已实现，待复测）

**第一版的错误**：回收循环用 `LogicalKVPageStore::dematerialize()`，它会 `release_descriptor()` **销毁逻辑页描述符**
并把 membership 置空。但引擎的模型是：**一个已映射（block table 有槽位）的逻辑页必须持有 device 或 host 副本**。
销毁后逻辑页仍被 block table 引用 → 遍历 `mapped_pages` 的路径读到失效句柄 → `logical KV page handle is stale`。

**修复（已实现）**：在 `ProgramImpl` 层新增 `demote_sequence_kv_to_host()`，五步复用引擎已有 API：

| 步 | 调用 | 作用 |
|---|---|---|
| 1 | `host_kv_extents->prepare(*pages, batch)` | 在主机 KV 区预留 extent |
| 2 | `host_kv_extents->device_sources(reserved)` | 取 device 源页 |
| 3 | `pages.physical_pool().copy_to_host(sources, writable_view(reserved), device.stream)` | **D2H**（用计算流以保证写后读顺序） |
| 4 | `host_kv_extents->publish(std::move(reserved))` | 挂 host 副本（**逻辑页句柄保留**） |
| 5 | `set_writer(false)` + `release_active_reference()` + `drop_device_replica()` | 释放 device 副本 → 物理页回池 |

配套改动：
- 移除 store 内的销毁式回收循环与 `Address::recycle_cursor`（回收上移到 `ProgramImpl`）
- `release_active_reference()` 容忍 0（降级页不再持有 active 引用）
- `KVAddressSpaceStore::logical_page_capacity()` 访问器（判定环模式）
- `ensure_sequence_kv_mapped()` 在映射前先降级，且仅当 `池 usable < 逻辑页容量` 且窗口非 0 时启用（**dense 路径完全不进入**）

### 两处必须严格对齐的量（第一次实现踩的坑）

**坑 1：保留量不能等于池大小。**
第一版用 `保留量 = NINFER_KV_WINDOW`，而测试里窗口(512 页) = 池(512 页) → 降级后设备页**零空闲** →
当前 prefill 块无处可写 → `Paged KV reservation invariant was violated`。
**修正**：`保留量 = min(请求窗口, 池 − 本块需要)`，保证下一块总有空页。

**坑 2：掩码窗口必须等于实际驻留集合。**
若掩码窗口取自环境变量（512）而实际驻留只有 448，则掩码会"选中"被降级的 64 页 →
注意力读到它们**已被新数据覆盖的陈旧设备槽位** → 静默错误。
**修正**：由降级边界推导窗口 —— 新增 `lowest_resident_kv_page()` 扫描最低驻留页，
掩码窗口 = `总页数 − 最低驻留页`。两处掩码安装（映射后 / 提交后）都用同一个量，不可能不一致。

**另外**：sink 支持被移除——环形降级从**最老页**开始，sink 恰好是最先被降级的页，
"保留 sink"必然读到被复用的槽位。已在代码注释中说明。

### 坑 3：MTP 后端 KV 是**独立的池**，必须单独回收

设备上有**两个**分页 KV 池：

| 池 | 页数 | 逻辑容量 | 用途 |
|---|---|---|---|
| 文本（target） | `kv_capacity / 64` = 512 | 2,048 | 目标模型全注意力层 |
| **后端（MTP/draft）** | **512 + 1 = 513** | 2,048 | 草稿模型（1 层） |

只给文本池做降级的话，**后端池无人回收** → 涨到 513 页就爆
（诊断里 `usable=513` 暴露了它，因为文本池是 512）。

**修正**：把降级泛化为 `demote_kv_pages_to_host(addresses, pages, address, keep, need)`，
对**任意 KV 地址空间**生效；`ensure_sequence_kv_mapped` / `commit_sequence_kv` 里
**文本与 MTP 各降级一次**，各自用**自己的池大小**算保留量。
掩码也改为**每个 cache 一个独立窗口**（MTP 草稿读后端池，所以后端掩码必须与后端驻留集合一致）。

**关键复用**：主机区支持**按几何多布局**（`HostKVExtentStore::page_layout` →
`arena_->layout_for(geometry)`），所以同一个主机区能同时服务文本池（多层）和后端池（1 层）。
后端只有 1 层，所以主机占用很小。

### 坑 4（Windows 特有，影响容量账）：主机 KV 会**计入 VRAM**

`program_impl.cpp` 的注释与 `host_kv_clamp.h` 写得很明确：

> WDDM 会把 pinned 主机分配映射进 GPU 地址空间，并**按同一预算计费**。
> 启动时按空闲显存钳制：`budget = (free_device − 1 GiB) / 2`。

实测（本机）：

| 配置 | 请求 | 实得 |
|---|---|---|
| ring（池 32K） | 8.00 GiB | **783.8 MiB** ← `(2.5 − 1)/2` |
| dense（池 160K） | 8.00 GiB | 8.00 GiB |

⇒ **主机溢出量不是免费的**，它与设备池**争同一块显存**。
本机 12 GB：权重 7.28 GB，留给「设备池 + 主机溢出」的约 4.2 GB。

| 方案 | 设备池 | 主机溢出 | 合计上下文 | 显存 |
|---|---|---|---|---|
| dense（现状） | 181K | — | **181K** | ~11.1 GB |
| 环：池 96K + 主机 128K | 1.77 GB | 2.36 GB | **~224K** | ~11.4 GB |
| 环：池 64K + 主机 192K | 1.18 GB | 3.54 GB | 超预算 | ✗ |

所以**在 Windows 上总上下文 ≈ 220–230K**（不是 256K），除非放宽这个钳制。
放宽 `host_kv_clamp.h` 的 `(free−1 GiB)/2` 是启动时的一个可用旋钮（prototype 可试），
但要留出显存余量否则设备 OOM。

**更重要的推论**：这条约束解释了为什么"主机放多少历史"在 Windows 上**不等于**"内存有多少"——
它受显存会计限制。若换 Linux（非 WDDM），主机溢出只吃系统内存，那时 256K/512K 才真正由 32 GB 内存决定。

**为什么现在能无损**：被降级的页内容保留在主机 KV 区（`--host-kv-mib`，默认 8 GiB），
检索策略日后可以把页 **H2D 取回** device 并重新选中——不像第一版是丢弃。

**注意 C 的判读陷阱**：GDN（线性注意力）层是**循环状态、不受掩码影响**，
可能仍"记得"开头的针——这与 §9 的副产物（丢 50% 全注意力 KV 只 +26.6% PPL）同源。
所以 C 必须**与 dense 对照**（同提示、池 ≥ 提示 → 应能答出）才有说服力，不能只看环模式答不出。

### 顺手修掉的一个**生产安全隐患**

回收循环的守卫原本是 `available_pages() < count`。但在 **dense 模式**下，一个接近上下文上限的长请求
也会把 `available_pages()` 压到 0 → **回收循环会误触发**，进而销毁生产配置的页！
（测试 A 没暴露它，因为 PPL 每个窗口只调用一次映射，`begin == 0` 使循环体不执行。）

已修：守卫加上 `usable_pages() < page_capacity_`（严格环模式才回收），dense 路径完全不走回收。

### 环模式已**默认关闭**

`--kv-capacity < --max-context` 现在**默认被拒绝**（与原版一致），
只有显式设 `NINFER_KV_RING=1` 才放开——因为存储层还没实现，放开会得到 500 而不是可用的长上下文。
**因此生产配置（`--kv-capacity == --max-context`）保证不受本次改动影响。**

**注意 C 的判读陷阱**：GDN（线性注意力）层是**循环状态、不受掩码影响**，
可能仍"记得"开头的针——这与 §9 的副产物（丢 50% 全注意力 KV 只 +26.6% PPL）同源。
所以 C 必须**与 dense 对照**（同提示、池 ≥ 提示 → 应能答出）才有说服力，不能只看环模式答不出。



## §9⑦ 检索策略（词法重叠 + IDF 加权，**已实现并验证**）

§9⑥ 的环形回收把窗口外的页搬到主机，但**只是隐藏**，找不回来——那还是滑窗。
本节加上"取回"：给即将降级的页打分，把相关的页 **H2D 拉回设备**并纳入掩码。

### 为什么先做词法而不是 KVMem 原版的 mean-K

| | mean-K（KVMem 原版） | 词法重叠（本实现） |
|---|---|---|
| 新 CUDA 内核 | 需要（写路径捕获每页 K 均值） | **不需要** |
| query 钩子 | 需要 | **不需要**（token id 已在主机侧 `sequence.ledger`） |
| 擅长的场景 | 语义相似 | **词法重复**：变量名、错误信息、代码片段 —— 与 `--lookup-ngram` 同一现象 |
| 实现风险 | 高 | 低（纯主机侧） |

**IDF 加权是必需的**：测试用的噪声提示词汇表只有 17 个词，不加权时**每页得分几乎相同**；
加权后稀有词（`secret`/`access`/`code` 只出现在针句与问题里）**决定性主导**排序。

### 每次映射前执行的对称流程

```
1. 打分 : 对即将降级的页 [sink, recent_begin) 按 IDF 词法重叠打分 → top-K
          预算 = min(NINFER_KV_RETRIEVE, 池/4)
2. 降级 : 未选中的页 D2H → 主机（sink 与检索页保留在设备）
3. 映射 : 本次需要的页
4. 恢复 : 选中的页 H2D → 设备 + 重发布 block table 槽位
5. 掩码 : 只选中 [0,sink) ∪ [实际驻留窗口) ∪ 成功恢复的页
```

### 关键安全设计

**掩码始终由"实际驻留集合"推导，而不是"意图"**：
降级或恢复失败只会**多隐藏**，绝不会暴露被降级页的陈旧设备槽位（静默错误最容易出在这里）。

### 验证（同一条 60K 提示、同一个 32K 设备池，只改检索开关）

| 组 | 设备池 | 检索 | 结果 |
|---|---|---|---|
| dense（全设备） | 131K | — | **`BLUEBIRD-42`** |
| 环模式 | **32K** | **关** | "文本里没有这个码" |
| 环模式 | **32K** | **开** | **`BLUEBIRD-42`** ✅ |

**这是检索能力的决定性证明**：同样的池、同样的窗口，打开检索后窗口外的相关页被从主机取回，
模型答对，与全设备 dense 一致。

### 池稳定 + 性能（实测）

```
mapped=944 pool=512 sink=4 budget=64 keep_recent=436 retrieved=64  allocated=504 runs=1
                                            ↑ 436+4+64+8(slack)=512 ✓ 精确符合预算
```

| 指标 | 环+检索 | dense（参考） | 说明 |
|---|---|---|---|
| prefill | 239 t/s | 365 t/s | −35%：D2H 用计算流，未做传输重叠 |
| decode | 41.9 t/s | 85 t/s | −51%：逐步的降级/恢复抖动 + 打分 + 位图上传 |
| MTP 接受率 | 73.3% | 76.9% | 基本不变 |

### 调试链（检索功能的 4 个真实缺陷）

| # | 症状 | 根因 | 修复 |
|---|---|---|---|
| 1 | `materialize_one` 无空闲页 | 预算零余量（`空闲 = budget − 已常驻检索页` 恰好等于所需） | 8 页 slack + 恢复前检查 `free_runs()`（拿不到就跳过 → **只隐藏，安全**） |
| 2 | `materialize` `count 16 > free 15` | **host+device 双副本的页被降级准入排除**（`!host_resident`）→ 取回过但不再被选中的页**永久常驻** → **真实泄漏** | 双副本页**直接释放 device 副本**（内容已在主机，无需拷贝） |
| 3 | 掩码漏选"本来就常驻"的检索页 | `restore_*` 只返回"恢复成功"的 | 语义改为返回"检索后**实际设备常驻**的检索页" |
| 4 | 诊断在失败路径不可见 | 打印位于映射之后 | 移到映射之前 |
| 5 | `logical KV batch materialization exceeds descriptor capacity`（`--no-prefix-reuse` 路径） | **`materialization_scratch_` 只按物理池大小（512 页）预留，而逻辑容量是 2048 页** → 一次映射跨越 >512 逻辑页时溢出（潜伏缺陷，被环模式的"逻辑映射跳得很远"暴露） | `reserve(logical_capacity)` |
| 6 | reuse 请求被拒（`isolated-feasible ... idle Engine`） | **主机区容量**：环形的降级页 + checkpoint 副本超出主机区（实测需 1.71 GB、可用 1.53 GiB） | 配置 A（`--no-prefix-reuse`）或配置 B（`NINFER_KV_HOST_OVERCOMMIT_MIB` 放大主机区）——详见下文"主机区容量需求" |

### 后续优化方向

| 项 | 状态 |
|---|---|
| **滞回**（`SequenceState::ring_keep`：上次检索集保持常驻）+ **打分节奏**（mapped 页数不变不重算） | ✅ 已实现并验证（40K×2 通过，decode ~50 t/s）；越界防护（lane 复用钳制） |
| **传输流重叠** | ❌ 不做（已证伪）：单缓冲池下新页复用必须等旧内容存完，串行化不可避免；且传输只占约 0.1%（17MB/chunk），不是瓶颈 |
| **mean-K 打分**替换词法 | ⏸ 暂缓：实测瓶颈在 kernel 内掩码求值（检索开关对速度几乎无影响：49.6 vs ~50），换打分不提速；词法 A/B 已通过 |

### 性能定论（40K 同深度实测）

| 配置 | prefill | decode | 瓶颈 |
|---|---|---|---|
| dense | 365 t/s | 85 t/s | — |
| 环，检索关 | 307 t/s | **49.6 t/s** | kernel 掩码（slow path + 逐块判断） |
| 环，检索开 | ~240–310 t/s | ~47–53 t/s | 同上（检索抖动影响 <5%） |

### ⚠️ 主机区容量需求（环模式多轮对话的硬约束）

实测定位（诊断命中 `pressure.cpp:2350` → `physical_peak_fits` 的 `host.kv_bytes` 维度）：

```
环形降级页（历史）              ≈ (上下文 − 驻留窗口) × 18432 B
已发布的 checkpoint 副本        ≈ 上下文 × 18432 B（一整份）
新请求自己的 checkpoint 副本    ≈ 上下文 × 18432 B（一整份）
───────────────────────────────────────────────────────────
主机区需求 ≈ (3 × 上下文 − 驻留窗口) × 18432 B
本机实测：3×60K − 27K ≈ 153K token ≈ 2.8 GB   而主机区上限只有约 2.1 GB → 第二个请求被拒
```

**已排除的其它原因**（逐一实测）：主机区放大到 2.02 GiB、设备状态槽 3 个、事务未清、无空闲 lane。

**试过但失败的修法**：环模式下禁用 `publish_continuation` → 破坏前缀簿记（`PrefixShortlistDigests::at` 越界），
连第一个请求都失败 → **已回退**（`request_plan.cpp` 恢复原状）。

**⇒ 结论（Windows）**：环模式的**多轮对话**需要 3 份上下文的主机空间，而主机区受 `free_device − 余量` 限制。
本机 2.0 GiB 主机区对应的多轮上限约 **36K token**；60K 以上只能单请求。

**真正的修法**（未做，属引擎深度改动）：让 checkpoint **直接引用环形已在主机的那些页**，而不是复制一份
—— 需要状态/检查点子系统理解环形的 host extents。

**另一条已知不兼容**：`--no-prefix-reuse` 会把地址空间按 `kv_capacity` 而非 `max_context` 构建
→ 环模式判据失效（`logical_capacity` 变成 512）→ 退化为 dense 映射并溢出。**不要与环模式同用**。

---

## §9⑧ 突破：可分页主机内存（解除 Windows 的显存耦合）

### 问题回顾

引擎的主机 KV 区用 `cudaMallocHost`（**pinned**）。而引擎作者自己的注释就写明了后果：

> 在 Windows/WDDM 上，pinned 主机分配被映射进 GPU 地址空间并**按卡容量计费**——
> 在 24 GiB 的 3090 上实测：**常驻显存 + pinned 主机总会逼近卡容量**

于是启动时要把主机区钳到 `(free_device − 1 GiB) / 2`（本机 ≈ 1.5–2.1 GB）。
**这把"用内存换显存"的杠杆几乎抵消了**：主机区上限 ≈ 2.1 GB，而多轮对话需要 3 份上下文（≈2.8 GB）。

### 关键洞察

**计入显存的是"pinned 映射"，不是"主机内存"本身。**
改用**可分页内存**（`std::malloc`，不 pin、不映射）→ 只吃系统内存 ✓

### 实现（约 40 行，环境变量门控，生产不变）

| 文件 | 改动 |
|---|---|
| `core/arena.h` / `arena.cu` | `PinnedHostBuffer` 加 `pageable_` 标志；`NINFER_HOST_PAGEABLE=1` 时走 `std::malloc`/`std::free`（移动语义同步处理） |
| `program_impl.cpp` | pageable 时**跳过 WDDM 钳制**（不再按空闲显存打折） |

**代价**：`cudaMemcpy` 对可分页内存是 staged（经 bounce buffer），不是真异步 → D2H/H2D 变慢（可测）。

### 实测结果（本机 12 GB）

| 指标 | pinned（原） | **pageable（新）** |
|---|---|---|
| 主机区实得 | 784 MiB ~ 2.1 GB（被钳） | **8.00 GiB**（= 请求值） |
| 分配耗时 | 55–135 ms | **1.86 ms**（无 pinning） |
| **显存占用** | 被主机区挤占 | **9259 MiB（不变）** ✓ |
| 设备空闲显存 | — | **2.51 GiB（不变）** ✓ |

⇒ **可分页内存确实不计入显存**，Windows 上主机区上限从 ~2.1 GB 变为**系统内存可用量（20+ GB）**。

---

## §9⑨ 准入钳位 bug：第二个请求必失败（已定位并修复，待验证）

### 现象

环模式下第一个 60K 请求成功，**第二个相同请求必被拒**
（`isolated-feasible request is blocked in an idle Engine`），与主机区大小无关
（8 GiB 可分页主机区下依然失败）。

### 根因（诊断钉死）

```
[diag] peak_fits FAIL ... main 504+512/512 backend 505+513/513 hoststate 0+0/8 hostkv 559M+0/8G
```

引擎判据是 `used + added <= capacity`。我之前的修法是把逻辑页需求**钳到池大小**
（`added = min(peak, 512) = 512`），于是 `capacity − added = 0`，要求 `used == 0`：

| 请求 | 池中已有 `used` | 结果 |
|---|---|---|
| 第一个 | 0 | `0 <= 0` ✓ 通过 |
| 第二个 | > 0 | `used <= 0` ✗ **必失败** |

hostkv 维度（`559M+0/8G`）当时是**通过**的 —— 之前"主机区容量不足"的结论是误判。

### 修复

环模式下设备 KV 池由环形自己管理，准入**不该**拿逻辑 entitlement 和池比较。
三个位置把主/后端 KV 维度**清零（跳过）而非钳位**：

| 位置 | 函数 |
|---|---|
| `storage/context.cpp` | `physical_peak_fits` |
| `planning/pressure.cpp` | `isolated_request_feasible` |
| `planning/pressure.cpp` | `persistent_backfill_safe` |

全部以 `usable_pages() < logical_page_capacity()` 为门控，dense 路径不受影响。

### 后续三层（多轮对话真正的链条）

准入通过后，第二个请求又倒了三次，一次比一次深：

| # | 症状 | 根因 | 修复 |
|---|---|---|---|
| 1 | `Paged KV reservation invariant`（50ms） | 复用候选要把 443 个主机页**一次性全搬回**（`prepare_kv_restores`），池子只有 8 页空位 | **拒绝主机页复用**（`inspect_admission` 门控：源含主机页 → nullopt → 回退全新 root；全在设备的源照样快速复用） |
| 2 | 同上（全新 root 第一块 prefill，`count=16 reserved=625`） | 新地址是空的，504 页被**上一个请求的旧地址**占着；ring 降级只管当前地址 | **跨地址降级**（`demote_other_addresses_to_host` + `for_each_address`）：先搬冷数据（非活跃地址），不够再动自己窗口 |
| 3 | 跨地址降级 `freed=0`（`prot=504`） | 完成发布 checkpoint 时 `commit_active_snapshot` 给**每一页**加 `protect_coverage` | 去掉该检查：保护是为 fork 复用保持常驻（复用已拒绝），且降级不改列数/不销毁描述符，与保护不冲突 |
| 4 | `active KV snapshot full page is not stable`（终点快照） | 完成时发布 continuation 要求**全页在设备**，环模式结构性做不到 | **完成时跳过发布**（`inspect_capture` 门控：含主机页 → 不发布；规划期不动，避开 digest 坑；短请求照样发布复用） |

### ✅ 多轮验证通过（40K×2，60K×2）

```
40K: attempt 1/2 均答出 BLUEBIRD-42
60K: attempt 1 wall 245.5s / attempt 2 wall 239.9s，均答出 BLUEBIRD-42
```

代价：环模式长上下文失去前缀缓存（每轮重 prefill）；短上下文多轮不受影响。

---

## §9⑩ 256K 缩放验证（**通过**）

配置：池 96K token（1500 页）/ 上下文 262144（4096 页）/ 窗口 96K / 检索 12K token / 主机 8 GiB 可分页。

| 判据 | 实测 |
|---|---|
| 引擎启动 | `pages 1,500/4,096`，`runtime 2.15 GiB` |
| 152K 提示 prefill | ✅ 完成（`prompt_tokens 152,481`），123.5 t/s |
| 针（开头） widh 检索 | ✅ **`BLUEBIRD-42`**（思考过程明确引用"第一行的码"） |
| decode @150K | 22.8 t/s（随总页数缓慢衰减：掩码逐块检查成本 ∝ 总页数；40K 时 ~50 t/s） |
| 显存 | **10413 MiB**（10.2 GB，12 GB 内 ✓） |
| MTP 接受率 | 68.4%（正常） |

**结论**：96K 设备池承载 152K 序列（1.6×），**"256K on 12GB" 成立**。
decode 随长度衰减是 kernel 掩码逐块检查的固有成本（与 dense 的 OOM/崩溃不同，是缓慢线性衰减）。

---

## §10 结论与建议

**修正后，方案比 v1 更有吸引力**：

| | NInfer nvfp4 160K（现状） | NInfer + KVMem 式 256K（移植后） |
|---|---|---|
| 上下文 | 160K | **256K** |
| decode @60K | 33.8 t/s | **~54 t/s（平坦）** |
| 全历史注意力 | ✅ 是 | ⛔ **否（block-sparse）** |
| TTFT | 现有 prefill | prefill + 1–5 s 检索 |
| 工作量 | 0 | 数周 |

**建议路径**：
- ✅ **已完成**：内核原型（掩码）+ 两项验证（回归逐 bit 一致 / 功能 PPL ×1.266）——见 §9
- ✅ **已完成**：**环形回收 + 主机降级**（池 < 上下文合法）+ 8 层约束放开 + 掩码时序/位图跨度/后端/Graph/MTP 五处修正 —— 见 §9⑥
- ✅ **已完成**：**A/B 决定性验证**（60K 提示、池 32K）：dense 答出针 / 环模式答不出且输出连贯 —— 见 §9⑥
- ✅ **已完成**：**检索策略**（词法重叠 + IDF 加权 + H2D 回取）：同池同窗口，检索开 → **答出 `BLUEBIRD-42`** —— 见 §9⑦
- ✅ **已完成**：dense 生产配置回归**逐 bit 一致**（62/62 采样点，delta = 0；build52 最终版复验通过）
- ✅ **已完成**：**多轮对话**（40K×2、60K×2 均答出针）——准入清零 + 拒绝主机页复用 + 跨地址降级 + 跳过终点发布 —— 见 §9⑨
- ✅ **已完成**：**256K 缩放**（池 96K / 上下文 262144：152K 提示完成并答出针，显存 10.2 GB）—— 见 §9⑩
- ✅ **已完成**：性能优化取舍（实测）—— 滞回 + 打分节奏已实现；传输流重叠已证伪（串行化不可避免 + 传输仅占 ~0.1%）；mean-K 暂缓（瓶颈在 kernel 掩码）—— 见 §9⑦

**关键增益预期**：由 §9 的副产物（丢 50% KV 仅 +26.6% PPL）可知，**驻留窗口可以比原计划更小**；
且 GDN 层是全历史的循环状态，进一步补偿窗口外信息 → 检索窗口 64K–96K 即可获得很长的有效上下文。

---

## 附：PoC 复现

```
C:\ninfer-build\poc-vmm-host-kv\poc.cu
编译: nvcc -arch=sm_86 -O2 -o poc.exe poc.cu   (CUDA 12.8 + MSVC)
```
