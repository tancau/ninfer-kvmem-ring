# 需求 · KVMem worker 支持 DFlash2 投机后端

> 2026-10-20 · 面向 KVMem worker 的产品方（`ninfer-kvmem-server` 形态，例：`kvmem-ninfer-windows-rtx30-sm86-938c2f1`）
> 状态：待提交 · 复现材料在本机可跑

## 一句话

KVMem worker 目前只实现 MTP + ngram 两种投机后端，**拒绝 `--spec dflash2`**；同一份
`Ternary-Bonsai-2-27B-ninfer-v3.ninfer` 制品在**同一张 RTX 3060 12G** 上，用普通 ninfer 引擎
（`ninfer-serve`）跑 DFlash2 可以拿到 **+68.8% 的 decode**。希望 worker 也支持该后端。

## 拒绝点（现状）

```
$ ninfer-kvmem-server.exe <model.ninfer> --spec dflash2 --draft-tokens 7 ...
ninfer-serve: KVMem requires ordinary/fixed/adaptive MTP1..15, ngram1..63
              (automatically capped at 15 with 2..8 lanes), single-GPU or layer-pipeline
              BF16/INT8/FP8/NVFP4/K8V4/RK8V4/RK4V4/RK4V4-E8/RK2V4-E8 generation,
              1..8 text or resident/CPU-vision lanes,
              INT8/RK8V4/RK4V4/RK4V4-E8/RK2V4-E8 fast prefill, and exclusive
              ownership of history caching
```

同一引擎的 `--help` 里写着 `--spec mtp|dflash|dflash2`，但 KVMem 的能力检查只放行 MTP 与 ngram。

## 为什么值得做（实测）

**环境**：RTX 3060 12G（sm_86）· Windows 10 · CUDA 12.8 · MSVC 14.44 · 制品
`Ternary-Bonsai-2-27B-ninfer-v3.ninfer`（9,520,051,456 B，SHA256
`cdc4810b0ff17c40d0f62cf214b6e0bcd08346e9eb05ca53371507037793c14a`，其中已含
`ProCreations/Ternary-Bonsai-2-27B-DFlash2`（Q4）适配器与 proposal head）。

**协议**：随机 nonce 前缀 + 长提示 + 700 输出 + 丢弃预热 + 3 轮取中位 + 读引擎自身计时 + greedy
（与对方归档的 `bench.py` 同口径）。

| 引擎 | 投机 | 提示 | decode |
|---|---|---:|---:|
| ninfer-serve（普通） | `--spec mtp --draft-tokens 4` | 3,409 | 118.8 t/s |
| ninfer-serve（普通） | `--spec dflash2 --draft-tokens 5` | 3,409 | **156.5 t/s**（+31.7%） |
| ninfer-serve（普通） | `--spec dflash2 --draft-tokens 7` | 3,409 | **200.5 t/s**（**+68.8%**） |
| KVMem worker | `--spec mtp --draft-tokens 4` + ngram 31 | 3,409 | 73.6 t/s（thinking on） |

机理（`docs/maintainer/dflash.md`）：多列验证轮几乎免费——decode 受内存带宽约束，一轮 16 列和一轮
单列都是把权重扫一遍；**把列变成 token 的唯一东西是接受率**。DFlash2 的 K=7 在本机把 tok/轮从
MTP 的 3.51 提到 5.57（3090 上的既有记录：同一比率 +38.6%）。

## 要点：两个已知边界（不是缺陷，是配置/规划边界）

1. **提示必须在一个 `--prefill-chunk` 内**。过了第一个 chunk，规划器不给 dflash2 的 draft 形状发
   backend 权益 → `backend KV materialization requested without an allocation` → HTTP 500
   （`docs/serving.md` "Draft-KV operational boundaries"）。小池下限可用配置：
   `--max-concurrency 1 --media-cache-mib 0 --response-store-max-mib 16
   --max-private-continuations 1 --max-shared-prefixes 1 --max-long-anchors-per-continuation 0`。
2. **长提示需要两处规划器改动才能进 worker**：
   - 目标模型的 prefill 暂存按 chunk 宽度分配整块 chunk 的 BF16 K/V
     （64 层 × 8 头 × 128 维 × 2 × 2 B × 65,536 token ≈ 17.18 GB），12 GB 卡放不下 → chunk 必须保持
     4K–8K；
   - 与第 1 点配套，dflash2 的 backend 权益要能跨 chunk。
   两者齐备后，53K 级别的提示才可能跑 dflash2（当前 KVMem 配方下 53K 是 65.6 t/s）。

## 复现步骤（本机）

```
set EXE=C:\ninfer-submit\ninfer-ring\build-ninja\apps\ninfer-serve.exe
set MODEL=C:\Bonsai-App\models\Ternary-Bonsai-2-27B-ninfer-v3.ninfer
%EXE% %MODEL% --host 127.0.0.1 --port 8099 --model-id bonsai2-27b-fast ^
  --max-context 32768 --kv-capacity 32768 --prefill-chunk 4096 ^
  --max-concurrency 1 --kv-dtype int8 --device 0 --greedy --no-thinking ^
  --spec dflash2 --draft-tokens 7 ^
  --media-cache-mib 0 --response-store-max-mib 16 ^
  --max-private-continuations 1 --max-shared-prefixes 1 ^
  --max-long-anchors-per-continuation 0
```

对同一请求把 `--spec dflash2 --draft-tokens 7` 换成 `--spec mtp --draft-tokens 4` 即可得对照。

## 附：K=15 不该用

我们自己的 2026-09 记录（`TODO.md` 第 1 条，3090、kv int8、thinking off、decode 时间读服务端请求日志）：
合成语料下 K=15 更快（326.2 vs 249.3），**但真实生成上 K=15 输给 K=7**（156.3 vs 172.3 tok/s），
因为多出来的列不再被接受却照样付费。K=7 是甜点。
