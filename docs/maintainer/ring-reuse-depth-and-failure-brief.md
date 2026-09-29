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

## 4. 结论三：进程死因已从"池耗尽"改判为 **fail-fast**

`docs/maintainer` 旧结论与本文 §6 的第一版都把死因写成"池耗尽"。**实测推翻了这个结论。**

忠实复现（`repro_shape.py`：87 消息 / 33 工具 / 151,908 token / 取消后重试）在 T2
第 353 秒杀死引擎，监督器读到：

```
DEATH pid=9372 exit_code=0xC0000409 uptime=1338s
VERDICT: NTSTATUS exception 0xC0000409
```

排除法：

| 候选 | 判定 | 依据 |
|---|---|---|
| 外部杀 / wedge | **排除** | `is_available()` 只影响 503，进程不会消失 |
| `std::terminate` | **排除** | `apps/serve/main.cpp:45-56` 装了 handler，会打印 `NINFER-CRASH` 并 `exit 42`；实测无该行、码不是 42 |
| 未处理结构化异常 | **排除** | `main.cpp:65-72` 的 SEH 过滤器会打印并退 43 |
| `device.cu:54` `cuda_check` abort | **排除** | `device.cu:42` 先打印 file:line + CUDA 错误名才 abort，日志里没有 |
| `generation_budget.h:16` | **排除** | `request_plan.cpp:246-248` 的 `effective_limit_reason` 恒为 OutputLimit/ContextCapacity |
| `generation_budget.h:25` | **排除** | `commit()` 只在解码提交路径（`engine_core.h:1252,1942`），死亡发生在 prefill |
| **裸 `abort()` / `/GS` 栈 cookie 破坏** | **剩余候选** | 两者都产生 0xC0000409 且绕过全部 handler |

`0xC0000409` 是 `__fastfail`。剩余两种来源：
`FAST_FAIL_FATAL_APP_EXIT`（`abort()`，值 0）与
`FAST_FAIL_STACK_COOKIE_CHECK_FAILURE`（`/GS` 检出栈越界，值 3）。
**dump 的 `ExceptionInformation[0]` 直接分辨这两个**。

顺带修正一个假线索：此前"没有 dump"被当成崩溃不产生 dump。真相是
**磁盘只剩 21.24 GB，而 full dump 需要 ~23 GB**，两个陈旧 dump 占着 48.8 GB。
已清理，并把 `LocalDumps\ninfer-serve.exe` 的 `DumpType` 从 2（full）改为 1（minidump），
现在剩 66.65 GB。

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
