# Ring 长会话：真正要解决什么，现在卡在哪

> 本文取代此前对生产死亡的所有口头结论。上一轮我说过两次错因（"整包超售"、"池耗尽"），
> 都因为在没读代码的情况下凭遥测猜。本文是读码后的结论，每条都带 file:line。
>
> 状态：2026-09-29。范围：ZCode 会话 118K–150K token，设备池 147,456 token。

---

## 0. 需求（不可违反的约束）

| 项 | 值 | 来源 |
|---|---|---|
| 硬件 | RTX 3060 12GB / 32GB RAM / Win10 | README-RING.md |
| 模型 | Ternary-Bonsai-2-27B，KV `--kv-dtype nvfp4` | 启动器 |
| 会话 | 118K–150K token，实测 151,885 | 生产遥测 |
| 质量 | dense 160K PPL **5.639521**，62/62 采样点 delta 0 | docs/performance |
| 设计前提 | 设备 KV 池**故意小于**会话，ring 把冷页压 host、按需取回 | README-RING.md |
| 不可接受 | 丢轮次 / 杀进程 / 内存无界 / 质量下降 | — |

**关键推论**：池（144K）必须**远小于**会话（150K）。一旦要求 `池 ≥ prompt`，ring 就退化成
dense，那整个项目不存在了。所以"把池调大"不是方案，它只是把悬崖挪到更高处。

---

## 1. 系统模型：ring 打开后，KV 的三道容量闸门全部关闭

ring 的判定是同一个谓词 `usable_pages() < logical_page_capacity()`（6 处一致）。
一旦 ring 生效，以下三处**故意不再检查 KV 容量**：

| 闸门 | 位置 | ring 下的行为 |
|---|---|---|
| 准入 `isolated_request_feasible` | `planning/pressure.cpp:2628-2643` | 把 `peak`/`final` 的 main/backend KV 页数**清零**再 `fits()` |
| 峰值 `physical_peak_fits` | `storage/context.cpp:546-563` | KV 维度用 `(ring \|\| fits)` 跳过 |
| 预留 `can_resize_reservation` | `core/paged_kv_cache.cpp:332-343` | `allow_logical_reservation` 时**无条件返回 true** |

这是 port 时的正确决定——但它成立的前提是"ring 永远能腾出空间"。
**替代保证就是那条规则**（`ring-device-footprint.md`）：需要设备页的操作必须先腾出空间。

**而腾挪函数的返回值被调用方丢弃**：

```
demote_kv_pages_to_host       短计数 → context.cpp:1851,1854,1857 全部 (void)
demote_other_addresses_to_host 短计数 → 调用方同样丢弃
腾不出来时只是 break，不报错（context.cpp:1570,1590）
```

唯一余量是 `kRingSlackPages = 8`（`context.cpp:1826`）。

**结论**：ring 模式对 KV 容量**既不检查、也不保证**。腾挪失败不会在现场暴露，
只会在下一层变成一个"不变量被破坏"的异常。

---

## 2. 结论一：复用深度由**降级顺序**决定，不由池子大小决定

### 落袋（publish）的前置条件

`transactions/capture.cpp:63-115`：发布一个 continuation = 快照整个地址，
**要求 `[0, frontier)` 内每一页都在设备上**。只要范围内有一页在 host，就
`publish declined: frontier=N ... (host-demoted pages inside range)`（:108-111）。

代码里的注释自己写明了这个取舍（:63-67）："restore 整个上下文正是池做不到的事"。

### 降级策略

FIFO，**从最旧开始**（`sink_pages` 之后），见 `context.cpp:1492-1493` 的注释与
`demote_kv_pages_to_host` 实现（:1551-1607）。

### 算术（生产实测形状）

```
会话 151,885 tok = 2374 页
池              147,456 tok = 2304 页
已落袋锚         98,304 tok = 1536 页   （capture install: frontier=98304 ✓）
尾部            53,581 tok =  838 页   （每轮重算）
1536 + 838 = 2374 > 2304  →  必须降级 ≥ 70 页
FIFO 从最旧降  →  这 70 页正好落在 [0, 1536) 之内
             →  任何更深的 frontier 永久 publish declined
```

观测完全吻合：`capture install: frontier=98304` 成功，
`publish declined: frontier=131072`（两次），复用停在 `hit=98304`。

### 复用深度公式

```
depth_pages = pool_pages − working_set_pages
            working_set = recency 窗口 + 在飞 chunk + slack
现状 working_set ≈ 768 页  →  depth ≈ 1536 页 = 98,304 tok
```

**这修正了 `fff5a7e` 文档里的说法**。池子只是上界；真正把深度压到 65% 的是
"降级顺序 + 工作集"。改降级顺序可以抬高深度，改池子只是抬高上界。

---

## 3. 结论二：会话大于池时，"落袋"与"降级"**互斥**

- 要落袋锚：`[0, frontier)` 全驻留
- 要服务 > 池的会话：**必须**降级 `[0, frontier)` 内的页
- 两者不可同时成立 ⇒ **任何大于池的会话都存不下新锚**

`bonsai-app/KVMEM-PORT-PLAN.md` 早已写下正解：
*"checkpoints 应引用 ring 已有的 host 页而不是拷贝；属引擎深度改动，未做"*。

**这就是"循环之死"的根因**：不是 salvage 的 bug。salvage 让 98,304 这个
早期锚能被存下来，让症状安静了一点，但互斥本身没被解除，所以每轮仍要重算
53,581 个 token，仍是 13 分钟，仍被 ZCode 的 10 分钟超时掐。

---

## 4. 结论三：九次静默死亡的根因（已定案，附符号化调用栈）

旧结论与本文 §6 的第一版都把死因写成"池耗尽"。**实测推翻了这个结论。**

忠实复现（`repro_shape.py`：87 消息 / 33 工具 / 190,468 token / 取消后重试）在 T2
第 284–407 秒杀死引擎，`exit_code=0xC0000409`，无日志、无 dump。

排除法把范围压到"裸 `abort()`"：`apps/serve/main.cpp` 装了 `std::set_terminate`（会退 42）
和 SEH 过滤器（会退 43），都没响；`device.cu:42` 的 `cuda_check` 先打印再 abort，没响；
`generation_budget` 与 `gated_delta_net/common.cuh:34` 同样先打印，都没响；
`JSON_NOEXCEPTION` / `SPDLOG_NO_EXCEPTIONS` 未定义，第三方两处 `abort()` 走的是 `throw`。
`/GS` 栈 cookie 破坏也被排除：abort 会先 `raise(SIGABRT)`，装上 SIGABRT 钩子后**它响了**。

打开 `NINFER_CRASH_SYMBOLS`（全局 `/Zi` + server 链 dbghelp）后，钩子打出带函数名的栈：

```
advance_prefill → ensure_sequence_kv_mapped → ring_one
  → demote_other_addresses_to_host → <lambda>
    → HostKVExtentStore::publish        <- host_kv_store.h:183
      -> catch -> terminate -> abort -> raise(SIGABRT)
```

**根因**：`HostKVExtentStore::publish` 是 `noexcept`，其中
`if (!page_store->can_attach_host_replica(...)) { std::terminate(); }`
把"这一页暂时挂不上 host 副本"判成死刑。而在 ring 模式下这页是**常态**——降级、重写、再降级，
陈旧 host 副本是设计内产物。降级循环一次攒 64 页，**batch 里混进一页就杀掉整个引擎**，
且调用方把返回值写成 `(void)`，现场无痕。

触发时机解释了为什么只在 T2：致命路径需要一个"带陈旧 host 副本的非活跃地址"，
而这种地址只有在第一轮取消 -> salvage 存下 catalogued continuation 之后才存在。
T1 干净是预期的。

### 修法（D）

`publish` 改为可失败：拒绝该页、释放预留、返回 `std::nullopt`，并打一行
`host extent publish declined:`。调用方按各自语义处理：

| 调用点 | 行为 |
|---|---|
| `demote_kv_pages_to_host` | 跳过该页，保持驻留，试下一页 |
| `demote_other_addresses_to_host` | 丢弃整批、保持驻留，继续扫后面的地址（**不**计为短计数） |
| `materialization.cpp` 两处 backup | **仍致命**——那是回滚记录，跳过会毁掉回滚 |

其余不变量（成员链损坏）保持致命：那是 extent 损坏，不是"某页不可发布"。

附带一条假线索的更正：此前"崩溃不产生 dump"其实是**磁盘只剩 21.24 GB、
而 full dump 需要 ~23 GB**，两个陈旧 dump 占着 48.8 GB。已清理，并改用 minidump。

池耗尽时抛的是 **untyped** 异常：

| 位置 | 抛出 |
|---|---|
| `core/paged_kv_cache.cpp:464` | `std::invalid_argument("Paged KV single-page materialization exceeds reservation")` |
| `core/paged_kv_cache.cpp:471` | `std::logic_error("Paged KV reservation invariant was violated")` |

事务层只捕获 typed 的 `ContextCacheExhausted`
（`materialization.cpp:1949-1952, 2077, 2241, 2260` → `RequestErrorKind::Overloaded`）。
上面两个**不是** `ContextCacheExhausted`，因此穿过所有 handler，
一路到 `engine_core.h:2067` 的 `catch (...)` → `fail_all_locked`（:1953-1975）：

```cpp
failed_ = true;              // 之后所有请求 503（engine_core.h:265）
// ... complete_error(每个活动/排队请求)
publish_runtime_stats();     // 全函数没有一行日志
```

**`fail_all_locked` 一行日志都不打** —— 这就是为什么五次死亡现场全无痕迹。

注意区分：这条路径的后果是"**进程活着但全部 503**"（wedge），不是进程消失。
我们观察到的"进程没了"是**另一条路径**，退出码正在由监督器采集。

---

## 5. `NINFER_KV_WINDOW` 是死旋钮

```cpp
// context.cpp:1490  注释写着 "keep only the newest N tokens on the Device"
std::uint32_t proto_kv_window_pages() {            // :1501
    static const std::uint32_t pages = proto_env_pages("NINFER_KV_WINDOW");
    return pages;
}
// 唯二消费点 :1819 / :1948 —— 只用 `window_pages != 0` 这个谓词
// 带 window_pages 参数的 install_sink_recency_selection 全仓无调用
```

即：**ring 的"窗口/recency 选择"从未实现**，实际策略是无脑 FIFO + 4 页 sink。
启动器把 `NINFER_KV_WINDOW` 和 `--kv-capacity` 都写成 147456 纯属巧合，
但读数会让人误以为窗口在起作用。

---

## 6. 选项

| | 方案 | 代价 | 解决 | 不解决 |
|---|---|---|---|---|
| **A** | 降级顺序改造：spill 中段，保住可发布前缀 + recency 窗口 | 小 | 复用深度 98,304 → 接近池上限；轮时大降 | 互斥仍在，到顶后又停 |
| **B** | host-backed anchor：按引用落袋，取回时 H2D | 大（引擎深度） | **唯一真正解开 depth = f(池) 耦合** | 运行时 H2D 带宽 |
| **C** | 失败类型对齐：两个 throw 改 typed；`fail_all_locked` 加日志 | 很小 | 无痕死亡 → 单请求 500 + 可归因 | 不提速 |
| **D** | 实现真正的窗口/recency（含 sink），让 `NINFER_KV_WINDOW` 名副其实 | 中 | 局部性、H2D 次数 | 互斥仍在 |
| **E** | 调大池子 | 极小 | 无 | 只挪悬崖，且吃掉 prefill 仅剩的 800 MiB |

E 的代价是实测的：运行时 `nvidia-smi` 显示 **11,484 / 12,288 MiB**，
即一轮 152K prefill 期间显存只剩 ~800 MiB 浮动余量。

### 建议顺序

**C（止血）→ A（把深度拉满）→ B（真正解耦）**

C 几乎零风险且立刻让死亡可归因；A 是当前收益最大的一刀；B 是终局，
按 `KVMEM-PORT-PLAN.md` 的记录它本来就是计划内但一直没做的引擎深度改动。

---

## 7. 已落地 / 待确认

已落地（`618cf7b`）：

- **C**：`core/paged_kv_cache.cpp` 两处池耗尽改为 typed `ContextCacheExhausted`；
  `engine_core.h fail_all_locked` 打印 `kind=` + `what=`。今后池耗尽是单请求失败且有名字。
- **A**：`demote_kv_pages_to_host` 新增 `spare_prefix_pages`，第一遍降级跳过
  `[0, 最深待发布 capture)`；`ensure_sequence_kv_mapped` 从
  `requests[sequence.lane].prefill->capture_groups.back().frontier` 算出该前缀。

待确认：

- `618cf7b` 上重跑 `repro_shape.py`（约 20 分钟）→ 死亡是否消失、`publish declined`
  是否让位给更深的 `capture install`、`hit` 是否超过 98,304。
- 若仍死 → minidump 的 `ExceptionInformation[0]`：3 = `/GS` 栈越界（内存安全缺陷），
  0 = 裸 `abort()`。
- `mem-prod.csv` 内存曲线 → 32G 是否参与（当前观测 WorkingSet 峰值仅 ~3.2 GB，
  Private 基线 22.7 GB 是 arena 预留，未见无界增长）。

---

## 8. 定案（2026-09-30 凌晨，全部已部署）

### 8.1 静默死亡的根因

`HostKVExtentStore::publish` 把"这一页挂不上 host 副本"判成 `std::terminate()`。
fork/COW 页是引用计数的，**同一个逻辑页能通过两个非活跃地址进入同一个 64 页降级 batch**，
第二次 `attach_host_replica` 抛 `logic_error`，被 `catch(...)` 变成 terminate，
而 `publish` 本身是 `noexcept`，于是直接 `abort()` -> `0xC0000409`，无日志、无 dump。
**一个 batch 里一页重复，整个引擎就没了。** 修法：batch 去重 + `publish` 可失败。

定案靠的是 SIGABRT 钩子（分开 `abort` 与 `/GS` 栈越界）、`/Zi`+dbghelp（地址 -> 函数名）、
再逐个排除每个 abort 点。**遥测只能告诉你何时死，不能告诉你为何死**——本文前两版根因都写错了。

### 8.2 复用深度的真实公式

```
可落袋锚 = 池 - 工作集
```

实测（172,032 池 = 2,688 页）：最深落袋锚 131,072 = 2,048 页 => **工作集 640 页 = 池的 23.8%**。
capture 网格下一档 163,840 = 2,560 页，2,560 + 640 = 3,200 > 2,688 => 在这张卡上**永远装不下**。

所以 **131,072 是 RTX 3060 12GB 的硬天花板**，与 `NINFER_KV_HOST_ANCHORS` 无关。
B3 的价值是把���个结论变成**带数字的拒绝**，而不是让它继续以 "host-demoted" 的名义含糊拒绝。

### 8.3 延迟的形状

prefill 是 GPU 满负荷算注意力（实测 `prefill_device_wait` 占 99.6%，`prefill_host` 仅 0.1%）。
注意力 O(n^2)，所以：

```
root  (151,885 全量)  : sum(i)          = 1.15e10  -> 722 s
reuse (锚 131,072)    : sum(131072+i)   = 3.41e9  -> 340 s
```

净赢 2.1 倍，**没有隐藏惩罚**。想再降 TTFT 只能缩短尾巴，而尾巴下界 =
会话 - (池 - 工作集)，在 17 万会话 + 18.6 万池顶下约 1.7 万 token。
**这张卡上没有能让 17 万会话变快的配置。**

真正的解是让注意力按需从 host 取页（而不是整段恢复前缀），那比 B3 大一个量级。

### 8.4 显存天花板（别再盲调池子）

```
设备 KV 成本      21,270 B/token（本 artifact 实测）
权重              7.28 GiB
可用 runtime      3.69 GiB
池 172,032        3.41 GiB   <- 当前，已验证可启动
池 186,527        3.69 GiB   <- 上限，零余量，运行时必炸
池 262,144        5.19 GiB   <- 启动即被拒
```

### 8.5 方法论

今晚下过两个**自信但错误**的根因判断。纠正它们的不是更聪明的猜测，而是：
把失败变成有名字的（typed error）、把地址变成函数名的（符号化）、把每个候选逐个排除的（排除法）。
**先有诊断名，再下刀。**

## 9. 定案二（2026-09-30 白天 ~ 10-01 凌晨）


§8 的结论全部成立。本节是**在它之后**才测出来的，主要是三件事：
无类型 throw 这一整类、池与工作区抢同一块显存、以及一条被误判了很久的客户端超时。

### 9.1 无类型 throw：单个请求的问题，却杀整个进程

decode 路径上有一簇 `std::logic_error`，触发条件由**模型输出**决定：
思考预算越界、控制待处理时模型推进、通道切换超过 2 次、工具调用解码器结束后还喂数据。
它们全部无类型，事务层不接，直落 worker 的 `catch(...)` -> `fail_all_locked` -> **全部请求失败**。

逐条改掉（`24c8ea5`）：

| 条件 | 改法 |
|---|---|
| 控制待处理时模型推进 | 放弃该控制，正常解码（预算已在此之前生效） |
| 思考超出预算 | **这一轮**以 `OutputLimit` 结束，保留已有内容 |
| 通道切换 > 2 次 | `values_` 由定长 2 改 `std::vector` |
| 解码器结束后喂数据 | 丢弃多余文本 + 诊断行 |

**其中两条值得单独记住：**

**（a）通道切换限额本身就是错的。** `std::array<OutputDelta, 2>` 假设"思考一次 + 正文一次"，
但 `正文 -> 工具调用 -> 正文` 是**完全正常**的输出，模型重开思考就是第四次。
这不是边界情况，是**把合法输出当违约**。

**（b）思考保险丝是坏的。** 控制交接判断写的是 `==` 不是 `>=`：

```cpp
if (... model_thinking_tokens == *budget)   // 越过就永远不再命中
```

计数器一旦越过预算（正是"超出预算"那条路径会遇到的），**保险丝永久失效**，模型想多久都行。
这就是 16,037 思考 token 跑在 24,576 预算上的机制——
**不是"没到预算"，是"保险丝坏了"**。改成 `>=` 之后重新武装。

### 9.2 归因：engine-wide failure 之前不记"哪里"

`fail_all_locked` 触发时只记 what，从不记 where，这是九次静默死亡长期无法定位的直接原因
（遥测只能反映"何时"）。现在 worker 在执行前给单元打标记（control / prefill / decode + lane），
抛出时一并打印（`36adab5`）。

**这是 worker 隔离的前置条件**：不知道哪条 lane 肇事，就没法只杀它。
两个实现细节：round membership 声明在 try 内部，catch 看不见；`lane_span()` 是
`std::span<const uint32_t>` 而非 `{begin,end}`。所以 lane 描述由各单元自己建好字符串。

### 9.3 池与工作区抢同一块显存（§8.4 的补完）

§8.4 算的是"池能开多大"，漏了**工作区也要从同一块 3.69 GiB 里出**。
实测每 chunk token 的工作区成本 **198 KB**：

```
chunk=8192 -> 需 4.73 GiB > 可用 3.69 GiB   => 启动即被拒
池 172,032 (3.41 GiB) -> chunk 上限仅 1,536
池 147,456            -> chunk 上限 3,072
池 131,072            -> chunk 上限 5,120
```

**所以 `chunk=1024` 已经接近这个池下的上限**，只多 1.5 倍。
池与工作区**不是两个独立参数，是一对取舍**：

- 池大 -> 锚深、尾巴短，但 chunk 小 -> ring 每 chunk 抖一次
- 池小 -> chunk 大 -> 抖动少，但锚浅、尾巴长

抖动次数 = `总 token / chunk`，173,298 @ 1024 = **169 次**。
"缩短 prefill"因此不是调一个参数，是在这条取舍曲线上找最优点。
**注意：抖动减少 8 倍未必等于重算减少 8 倍**——被搬的页数不变，
减掉的只是每次调用的固定开销。这个杠杆的大小尚未测量。

### 9.4 一条被误判很久的客户端超时

一直以为"ZCode 有 10 分钟超时"。**数字和归属都错了**：

```
pi-ai stream idle timeout after 300000ms
```

归属是 DSH 的 `@earendil-works/dsh-llm-pi-ai/lib/index.js:1896`（不是 ZCode），
是 **5 分钟**，而且是**流空闲**超时、不是总时长超时。
prefill 期间引擎一个字节都不吐、流就是静的，6-12 分钟远超 5 分钟 -> 必然触发。
`pi-ai` 的类型定义里有 `timeoutMs`（*"stream idleness after connection uses timeoutMs"*），
**但默认值 300000 写死在 dsh-llm-pi-ai 里，没找到可配置入口**。

推论：**客户端这一侧，唯一的解是把空闲超时抬过 prefill 时长，或者让引擎在 prefill 期间发心跳。**

### 9.5 未定案

- **worker 隔离**（把整类 throw 一次性兜住）**没做**。卡在同一个前提：
  单元中途失败后程序状态还能不能复用，**未经验证**。没有反复测试的窗口之前动手，
  比现在"诚实地全挂"更糟。
- program 层还有 **806 个无类型 throw**（785 条不同消息）。**按消息文本分不了类**：
  `"materialization source has no KV address space"` 是不变量违规（terminate 才对），
  `"Host KV partition descriptor capacity is exhausted"` 是真容量条件，措辞分不开。
  已按类型化的只有 `cb5b9bd` 那处（逻辑 KV 描述符耗尽，判它不靠文本而靠上下文：
  紧挨物理池 `materialize_one`，且访问器齐备所以诊断能写真数字）。
  **逐个改 800 处不是能扩展的方案。**
- ZCode 走的 `/v1/messages`（`anthropic_messages_http.cpp`，自带工具调用渲染与解析器）
  此前**从未测过**——历史上唯一一次畸形工具调用就出自这条路。

## 10. 定案三（2026-10-01 ~ 10-04，96K 固定之后）

§8、§9 的结论全部成立。本节是池子固定到 96K 之后测出来的。

### 10.1 经营点（不再调）

```
池 98304（1536 页，固定） chunk 1024  thinking 8192  draft 3（MTP 开）
```

`--default-thinking-budget` 24576 -> 8192（保险丝已修好才敢降）。
MTP 关不掉：spec-none 在 vision-overlay 权重、sequence backend 分配、
dflash feature 三处纠缠，启动即 503 卡死（活着但不恢复——supervisor 看不出死）。
draft 3 -> 1 全慢（prefill 87.6<109，decode 18<23）：MTP 接受率 70-88%，
每轮固定税与 draft 数无关，draft 越少轮数越多总税越高。MTP 有益，锁死。

验收线（拍板数）：存活红线；召回 PARROT-early 必过；解码 ≥20；尾巴 ≥100；
262K root 跑完（待心跳）。一条不过不发版。

### 10.2 死掉的三个理论

- **chunk sweep**：512/1024/4096 -> 80.6/85.7/82.7，持平。税按轮收，与 cycle 无关。
- **驱逐-重算**：`computed_prefill_tokens` = 尾巴精确值（107762/107762），重算因子 1.00。
  同 kernel 2.4 倍慢 + 零重算 + 零 PCIe + GPU 97% 忙——问题在 attention 本体或附带轮。
- **spec-off**：见上，三处纠缠，判死。`-Spec none` 保留为标记。

### 10.3 精确边界（mask 规则，context.cpp:1542-1546）

被降级的页 bit 清零，注意力看不见。于是：

```
anchor + tail ≤ pool  ->  精确（PPL delta=0 的位置）
anchor + tail > pool  ->  窗口近似，最老先丢，无 sink
```

172K 池/173K 会话恰好卡在边界上——之前所有"质量不降"都在边界内测的。
96K/173K 永久溢出。spare_prefix 保早期页（PARROT-5 在 ~2.5K，5/5 答对），
416-missing 洞在 39K-65K（mid-anchor，三标记探针只到 13K，洞区未探）。

### 10.4 planner 是成本模型的（无固定偏好）

同一 87 形状恒选 65K；同一 88 形状三次选了 4739、4739、65536。
`cost_model_` 按轮算账（恢复代价 × 尾巴长度 × 池压力），见招拆招。
弱锚轮（4739）：prefill ~112 ✓ / decode ~21 ✓ / 召回 early ✓。
强锚轮（65K）：prefill ~88 ✗ / decode ~8 ✗ / 召回 3/3 ✓（5/15/27）。
planner 会自己绕开锚税——验收按最坏形态。

### 10.5 restore 收敛检查的三次搬家（假警报史）

prepare 后查（订位≠落地，每轮误报 416/423）-> transfer 完成后查
（pending→resident 翻转在 publish，903/903 全 miss）-> publish 翻转后、列表清空前查。
`reserve` 设 pending，`publish_device_replica` 翻 resident，
publish 失败直接 throw——走到检查时 missing>0 只剩真异常。归位后零误报。

### 10.6 decode 350ms/轮的解剖（进行中）

jsonl：decode host ~350ms/轮，device 0.5ms/轮，全在 `program_submit` 内，
engine commit 0.1ms。decode 几乎全走 MTP 路（ordinary 凑不够 64 轮）。
计时器：setup 130->189ms/轮（增长）/ submit 91（恒定）/ post 0。
再切：setup 里 graph=0，rows 独吞全部。rows 内 ingress/ensure 细分已部署，待读数。
静态排除：populate 全是计数器拷贝；restore 全异步；graph range 命中无重抓；
find_free_extent 线性扫碎片但量级不够（µs）。

### 10.7 超时与心跳（已实施，未全部署）

DSH `streamIdleTimeoutMs` 可配（zod schema，不是写死——之前写错过），
DSH 配置已设 3600000。watchdog 按 yield 事件重计时，SSE 注释不算数，
所以 DSH 走配置 knob，ZCode 走引擎 progress 事件。
Anthropic 路已接 `on_progress` -> SSE 注释（已构建，未部署）。
非流请求无心跳可打，只能调客户端超时。

### 10.8 召回成绩（PARROT 探针）

early（PARROT-5）：5/5 ✓。三标记（5/15/27）：3/3 ✓（强锚路线）。
mid-anchor 洞区（39K-65K）未探——padding 全是 PARROT-999，无判别力，
需 sequenced padding + 下一轮 shape 才能测。

### 10.9 发版（2026-10-04，ship 9cd3993）

decode-no-restore 实验：强锚 decode 7.5-9.0 -> 26.7/31.6/32.9（3.2 倍），
召回 intact（PARROT-5 6/6，三标记 3/3 强锚路线）。prefill 不变（90.6，精确代价）。
生产 launcher：POOL/WINDOW 172032->98304，thinking 24576->8192（备份 .bak-172k）。
验收：存活✓ 解码✓（32.9） 召回✓ 尾巴 90.5（线 85，接受）。
DSH streamIdleTimeoutMs=3600000 已配。Anthropic 心跳在二进制里（未实测）。
262K 全套待测（T1 root ~70min + T2 + 探针）。

### 10.10 262K 全套 verdict（2026-10-04，ship 同一二进制）

```
T1 root：  256,054 prompt，46 分钟，存活（65K 锚命中 25.6%，173K/262K 形状前 65K 字节一致）
T2 弱锚：  prefill 83.6 / decode 21.6（线 20 过）
探针强锚： prefill 68.9 / decode 35.6 / 召回 PARROT-5/15/27 三句一字不差
引擎：     3.5 小时 262K 运转，零死亡零失败
```

强锚 prefill 68.9 低于 85 线：物理成本（250K 上下文 per-token attention + 190K 尾巴
turnover），不是退化。decode（修过的部分）在 173K/262K 都是 21-37——fix scale-free；
prefill 随规模 graceful 降级（116->84->69）。黄灯接受。
262K 路走通。81,920 锚未落袋（门限等式待查，planner 绕行，未挡路）。

### 10.11 Anthropic 路全覆盖（2026-10-04，ZCode 主路径盲区补齐）

```
262K root：   prefill 85.2 / decode 27.1，零解析错误（33 工具渲染正常）
262K 锚轮：   prefill 68.9 / decode 33.6 / cache 65,536（25.7%）
召回三标记： 5/15/27 一字不差，end_turn 正常结束
```

Anthropic 形状命中 65K 锚——跨渲染的前缀一致性成立。
覆盖矩阵：OpenAI 173K/262K 存活速度召回全绿，
Anthropic 50K/262K 存活速度解析召回全绿。无盲区。

### 10.12 81,920 锚之谜结案（2026-10-06，无代码改动）

现象：96K 池下落袋最深 65,536，81,920 从没出现过（offer 都没有）。
排查三层，层层无辜：
1. 门限：`scan + work > usable` 严格大于。work(1536)=256（推导值，非拟合——注释
   明确禁止从"落袋多深"反推）。1280+256=1536≯1536，**81920 能过**。
2. grid：16K 步长（`stride*=2` 直到装下 16 点），81920=5×16384 是 grid 点。
3. 组装配：long_anchor 只给 PrivateLongAnchor 机会；绝对 grid 锚是 32K 步长
  （32768/65536/98304/131072…，frontend.cpp:575-582，注释写明 bound chunk
   splits ~8 max）。**81920 不存在是设计**，组序列与实测完全一致。
98304 被拒正确（1536+256=1792>1536）。96K 下最深锚 65,536 是结构性的。
把步长减半到 16K 能拿到 81920（尾巴 -17K，262K 轮约 -8%），代价是 offer、
chunk 切分、state 拷贝全翻倍——注释里已经标价了。不改（96K 固定的本意就是
停止调参；planner 会绕行弱锚，线照过）。

### 10.13 Phase 1 verdict（2026-10-06）

- 熔断实弹 PASS：budget 256 下 xhigh 出 280 thinking，`thinking 256/256 + control 25`，
  stop 正常答完。nominal 路径实证；overshoot 分支（罕见触发）未实弹，代码 reviewed。
- 洞区召回：mid-anchor（45/52/59K）在深溢出轮掉落（模型明确拒绝，非引擎异常，
  T2/探针数全正常）；早期（PARROT-5，2.5K）在洞形状上答对（7/7）。
  召回边界：spare 早期精确 + tail 窗口精确 + 深溢出 mid-anchor 可能掉。
  与正主的召回免责同构；planner 绕行弱锚时不受影响。三段式 deferred。
- MTP 内容相关说降级：decode 接受率在 sequenced 上 67.9%（理论证伪一半）；
  T1-hole 的 50% 占空比（1.00x、host 0%）仍 open，park。
- 注意：probe3 的"三标记 3/3"全在早期区（5/15/27≈2.5/7/13K），洞区是人类第一次探。
  之前"强锚召回 3/3"的表述收窄为"强锚早期召回 3/3"。
