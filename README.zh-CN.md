# ninfer-ring —— 让 NInfer 在 RTX 3060 12G 上跑 256K 上下文

> English: [`README.md`](README.md)（上游文档，3090 环境）·
> Ring 设计与用法：[`README-RING.md`](README-RING.md) ·
> 全部实测数据：[`bonsai-app/BENCH-RESULTS.md`](bonsai-app/BENCH-RESULTS.md)

把 KVMem 式的环形 KV 机制移植进 NInfer 引擎：**显存里只放 96K 的 KV 池**（注意力窗口），
其余页无损放到内存，需要时按需召回。实测环境 **RTX 3060 12GB / sm_86 / Windows 10 / 32GB 内存**。

模型：三元 Bonsai 2 27B（`bonsai2-27b`，9.52GB 制品，Apache-2.0）。

## 快速开始

| 启动器 | 模式 | 上下文 | 说明 |
|---|---|---|---|
| `bonsai-app/Start-Bonsai-WebUI.bat` | dense（生产） | 160K | 默认，WebUI 在 `:8080` |
| `bonsai-app/Start-Bonsai-WebUI-256K.bat` | **ring（实验）** | **256K** | 设备池 96K + 内存 8GiB |

```bat
REM 生产（dense 160K）
C:\Bonsai-App\Start-Bonsai-WebUI.bat

REM 长上下文（ring 256K）
C:\Bonsai-App\Start-Bonsai-WebUI-256K.bat
```

API：`http://127.0.0.1:8080/v1`（经 WebUI shim 代理），模型 id `bonsai2-27b`。
请求体必须带 `"model": "bonsai2-27b"`。

Ring 开关（环境变量，不设就是原版 dense 行为）：

```bat
set NINFER_KV_RING=1
set NINFER_KV_WINDOW=96000
set NINFER_KV_RETRIEVE=12288
set NINFER_HOST_PAGEABLE=1
```

## 实测数据（本机，RTX 3060 12G）

| 项目 | 数值 |
|---|---|
| prefill（池内） | 658.6 / 506.4 / 252.1 t/s @ 4K / 17K / 65.5K |
| prefill（环通路 129K） | 147.8–150.6 t/s |
| decode（短问答） | 63.6–77.8 t/s |
| decode（129K 深度） | 18.3–21.5 t/s |
| PPL（261,167 token） | **5.639521，与 dense 版小数点后 6 位全同** |
| HumanEval+ | **152/164 = 92.68%**（官方三值 95.12，同测试集同协议，统计上不可区分） |
| NIAH 64K | **0.9591**（英文 11/11，中文 10/11） |
| 能效 | **0.521 mWh/token**（比 RTX 5090 的 0.582 还省） |
| 256K 运行时显存 | 10.2 / 12 GB，主机 KV 8 GiB（不占显存） |

完整表格、跨卡对照、官方分数三级链见 [`bonsai-app/BENCH-RESULTS.md`](bonsai-app/BENCH-RESULTS.md)。

## 质量有没有损失

没有可测出的损失：ring 路径与 dense **逐 bit 一致**（62/62 采样点，PPL delta=0）；
唯一的代价是三值化本身（官方数据：保留 98.2%），那在量化时已经付完。
nvfp4 KV 相对 rk8v4 是 +0.17% PPL，噪声级。

## 开关建议（实测结论）

| 开关 | 结论 |
|---|---|
| `--prefill-cublas` | ✅ 开（prefill +3%，几乎零代价） |
| `--spec mtp` | ✅ 保持开（3060 上约 2.1x，不开掉到 37 t/s） |
| `--embedding-q4` / `--lm-head-q6` | ❌ 本制品不兼容（要求 Q8 存储，制品是三值存储，引擎拒绝启动） |
| `--mlp-a8-decode` | ⚠️ 不开（decode 抖动 ±25%） |

## 出处与致谢

- 引擎：`iamwavecut/ninfer-3090` 分支 `franken/v0.11`（上游 `ashalliants/ninfer-3090` ← `Neroued/ninfer`）
- 模型：PrismML 三元 Bonsai 2 27B（基座 Qwen3.8-27B），制品来自 WaveCut
- 环形 KV 思想来源：KVMem；B 站 UP 主沈三殊（https://space.bilibili.com/85280961）的先例工作亦有启发（未复用其代码）；NInfer 上游与 prefill 数据来自公开资料
- 评测协议对齐 PrismML 白皮书附录 B（EvalScope，温度 1.0 / top-p 0.95 / top-k 20，xhigh）

## 仓库内容

- `src/`：引擎源码（含 ring 移植与崩溃诊断插桩）
- `bonsai-app/`：一键启动器、WebUI shim、全部实测记录
- `eval/`：评测 harness 配置（HumanEval+ / NIAH）、复现脚本、9 条评测避坑清单
- `eval/repro/`：长 prefill 复现器、中断复现器

已知问题与边界见 `BENCH-RESULTS.md` §8.4（引擎偶发静默退出，干净长请求已 42 次零故障；插桩已保留，崩溃会自动留证）。
