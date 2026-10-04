# 12G 显存跑满 262K 上下文：四天实录

## Ring KV on RTX 3060: from nine silent deaths to a shippable 262K

**结论先行**：Bonsai 2 27B，RTX 3060 12G，256K 会话——存活 100%，解码 27–37 tok/s，
召回三标记一字不差，引擎零死亡。生产已发版。

---

## 1. 起点：死得悄无声息

九次。引擎进程直接消失，`0xC0000409`，无日志、无 dump、无遗言。
每次都是长会话跑到深夜，人睡了，卡空转，早上起来发现进程没了。

遥测只能告诉你**何时**死。`prefill_device_wait` 99.6%——GPU 在干活，
然后没了。两版根因判断，自信，写进文档，然后被推翻。

转折点是两件工具：SIGABRT 钩子（分开 `abort` 和栈破坏）+ `/Zi` 符号化
（地址变函数名）。第三次，栈帧第一次说出了真凶：

```
advance_prefill → ensure_sequence_kv_mapped → ring_one
  → demote_other_addresses_to_host → HostKVExtentStore::publish
  → catch(...) → terminate → abort
```

`publish` 是 `noexcept`。一个引用计数的 fork/COW 页通过两个非活跃地址
进入同一个 64 页降级 batch，第二次 `attach_host_replica` 抛 `logic_error`，
被 `catch(...)` 接住变成 terminate。**一个 batch 里一页重复，整个引擎就没了。**
修法：batch 去重 + `publish` 可失败。九次死亡，一次结案。

**教训**：把失败变成有名字的（typed error），把地址变成函数名的（符号化），
把每个候选逐个排除。先有诊断名，再下刀。

## 2. thinking 失控：一字之差烧掉一整轮

模型一轮能想 16,037 个 token，预算 24,576，保险丝没触发。
读代码：

```cpp
if (... model_thinking_tokens == *budget)   // 越过就永远不再命中
```

`==` 不是 `>=`。计数器一旦越过预算，保险丝永久失效。
**不是"没到预算"，是"保险丝坏了"。** 改成 `>=` 之后重武装。

顺带把 decode 路径上同类的五个无类型 throw 全兜住
（预算越界、控制待处理、通道切换、工具解码器），之前每一个都能拖垮全进程。
program 层还有 800 个——按消息文本分不开类，逐个改不是方案，
worker 隔离才是，但前提（中途失败后状态可复用）未经验证，先不动。

## 3. 池子：从 172K 追到 96K，然后固定

池子 147456 → 172032（测出显存天花板）→ 用户拍板：**96K 固定，不再调**。
这是全周最正确的决定。随后测出池和工作区抢同一块 3.69 GiB
（每 chunk token 198 KB 工作区），`chunk=8192` 启动即被拒——
之前把它们当独立参数是错的，是一对取舍。

## 4. 正主出现了：kvmem/kvmem-llama.cpp

`KVMem-style` 注释是照它仿的（MTP3、ReplaySSM、F16 draft，连 reserve
都是同一个 256）。prism.3 直接给了 Bonsai 2 打包：128K 上限、思考预算 4096、
MTP 默认关——我们的 24576 预算和 MTP3 全开显得很激进，预算随后降到 8192。

但正主自己写：128K 召回"存在已知问题"，thinking 失控靠文档约束。
**有些墙是共有的，不是我们的 bug。**

## 5. 超时乌龙：5 分钟，不是 10 分钟；DSH 的，不是 ZCode 的

```
pi-ai stream idle timeout after 300000ms
```

流空闲超时。prefill 期间引擎一个字节不吐，6–20 分钟静默，必触发。
修法：DSH 配 `streamIdleTimeoutMs`（可配，之前误判写死）+ 引擎 prefill 心跳
（Anthropic 路 SSE 注释，已构建）。

## 6. 锚税：350ms/轮的三级解剖

强锚轮解码 7–9 tok/s，GPU 每轮只干 0.5ms，host 烧 350ms。
三级计时器（写完即删）点名：

```
setup 176ms = graph 0 + rows 176 = ingress 0 + ensure 176
ensure 186ms = restore+install（且随轮增长），score/demote/map ≈ 0
```

机制：demote_own（decode 无 spare 保护）为 8 页 slack 每轮吃锚，
下轮 restore 花 186ms 复活——自噬循环。修法：decode 跳过 mid-ensure H2D
restore（mask 照跟，turn-start restore 不动，新页本来就不需要 restore）。

结果：**decode 7.5 → 32.9（3.2 倍），召回一字不差**。
churn 不是 load-bearing。prefill 不动（近似写是毒化下游，只能付 90 tok/s 的精确代价）。

路上死掉的理论：chunk sweep 持平（80–86），重算因子 1.00
（`computed_prefill_tokens` = 尾巴精确值），spec-off 三处纠缠启动即 503。

## 7. 验收（自己定的线，自己守）

```
存活：  不死无静默（红线）
召回：  PARROT-5 6/6，三标记 3/3（强锚路线）
解码：  ≥20（实测 27–37）
尾巴：  ≥85（实测 90–116；262K 强锚 68.9，物理成本，黄灯接受）
262K：  T1 46 分钟不死 / T2 21.6 / 探针 35.6 + 三标记全对
```

planner 是成本模型的（87 消息恒选 65K，88 消息三次选了 4739/4739/65536），
弱路线（4739）常年飘绿：prefill 110+，decode 20+。

## 8. 发版

- 二进制 `9cd3993`（franken，SHA 对过）
- 生产 launcher：POOL/WINDOW 172032→98304，thinking 24576→8192
- DSH：`streamIdleTimeoutMs: 3600000`
- 文档：`docs/maintainer/ring-reuse-depth-and-failure-brief.md` §8–§10

## 9. 没做的（诚实清单）

- worker 隔离（结构性兜底，缺验证窗口）
- mid-anchor 洞区（39K–65K）召回直测（padding 无判别力，需 sequenced padding）
- 262K 的 81,920 锚（门限等式待查，planner 绕行中）
- submit+sync 81ms/轮（下一税，已不挡线）
- 心跳实测（ZCode 真实流量）
