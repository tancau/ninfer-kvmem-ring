# Bonsai 2 27B 本地部署实测记录

RTX 3060 12GB (sm_86) / 32GB RAM / Win10 LTSC 19044
模型: Ternary-Bonsai-2-27B-PTQ1_0.gguf (5.93 GB, sha256 53107F53...3EE3)
测试提示: 19.8K token 与 59.5K token 两档，各生成 400 token

## 一、三方对拉（同提示、同模型）

| 深度 | 引擎 | prefill | decode | VRAM |
|---|---|---|---|---|
| 20K | llama.cpp @96K  | **433 t/s** | **26.4 t/s** | ~9.5G |
| 20K | llama.cpp @128K | **441 t/s** | **26.6 t/s** | 9.9G |
| 20K | llama.cpp @256K | 132 t/s | 17.2 t/s | 11.8G |
| 20K | KVMem @256K | 234 t/s | 19.3 t/s | 7.6G |
| 60K | llama.cpp @96K  | **334 t/s** | 17.0 t/s | ~9.5G |
| 60K | llama.cpp @128K | **336 t/s** | 17.1 t/s | 9.9G |
| 60K | llama.cpp @256K | 175 t/s | 12.2 t/s | 11.8G |
| 60K | KVMem @256K | 198 t/s | **19.2 t/s** | 7.6G |

结论:
- **llama.cpp 128K 是甜点**: 相比 96K 零速度损失，多 33% 上下文。
- **llama.cpp 拉到 256K 是负优化**: 显存只剩 468MiB，计算缓冲外溢，prefill 腰斩、decode 掉到 12。
- **KVMem decode 是平坦的**: 20K→60K 几乎不变 (19.3→19.2)；llama.cpp 会衰减 (26.6→17.1)。
- **交叉点约 45K**: 浅上下文 llama.cpp 全面领先；深上下文 KVMem 的 decode 反超。
- KVMem 的 prefill 上限约 230 t/s（提高 -ub 到 512 也只从 194→234），是它最大短板。

## 二、KVMem 正确参数（关键！）

`--kvmem-budget` = **检索允许常驻显存的 token 数**，也就是 decode 时的注意力窗口。
不是"显存越大越好"——把它设成 131072 会让每步 decode 都注意 13 万 token，速度崩到 3.65 t/s。

官方 Bonsai 分支默认（适配 8/12G 卡）:
    -c 262144 --kvmem-budget 24576 --kvmem-gen-reserve 10240
    --kv-dtype q8_0 --kvmem-block-tokens 128 --kvmem-cpu-gb 9

另外 `--kvmem-sink-tokens` 必须 ≥4096 且严格小于 budget，否则 HTTP 500。

## 三、过度思考问题（比速度更重要）

B站评论区独立用户实测反馈（森角初楓, 4070TiS 16G）:
> "ninfer的实际速度感觉很一般，而bonsai2这个过度思考有点太离谱了"
> 让模型做自我介绍，它调用一堆工具去查资料、翻后台日志，花了近 10 分钟；
> 读一个日志写简要报告的任务花了 12 分半，各种调与任务无关的工具。
> "总结，目前 ninfer+bonsai2 这个方案从实用角度不值得花时间折腾部署"

原因定位:
1. chat 模板 `reasoning_effort` **默认就是 xhigh**，会往系统提示注入
   "仔细思考、验证假设、考虑各种替代方案" —— 过度思考的助燃剂。
2. llama.cpp / KVMem 默认 `--reasoning-budget -1` = **无限思考预算**。
3. ninfer 默认开启 thinking；KVMem 默认 `think=0`（关闭）——这就是两者体验差异的主因。

对策（已写入启动器）:
    --reasoning-effort medium     覆盖 xhigh 默认值
    --reasoning-budget 2048       思考超过 2048 token 强制 </think>

验证: 简单问题（17*23）2.4 秒答完，思考仅 115 字符，说明模型不是无脑话痨，
      过度思考只在复杂/agent 任务上发作，所以预算是"保险丝"。

## 四、NInfer 三值路线 —— **最终成功（当前最强方案）**

> 早期结论"ninfer 不可用"是**错的**：那是只看发放包里的预编译引擎得出的。
> 真正可行的路线在社区 fork 里，已实测跑通。

### 4.1 正确的组件组合

| 组件 | 来源 | 说明 |
|---|---|---|
| 引擎 | `iamwavecut/ninfer-3090` 分支 **`franken/v0.11`** | 原生支持 **sm_86**（CMakeLists 默认 `ARCH=86`），已内置三值格式 `t2_g128_fp16` + Hadamard 旋转 |
| 模型制品 | `WaveCut/Ternary-Bonsai-2-27B-NInfer-v3` | **9,520,051,456 字节**，含 MTP 头 + DFlash2 + 提案头 + 视觉 |
| CUDA | 12.8.93（旁装到 `...\CUDA\v12.8`） | 从官方安装包提取合并，**未碰 616.92 驱动** |
| 工具链 | CMake 3.31.6 / Ninja 1.13.2 / MSVC 14.44 | VS2022 BuildTools 自带 |

**不需要**：Ada 移植补丁、`pack.py`、20GB 模板、55GB safetensors。

### 4.2 实测性能（RTX 3060 12G，96K 上下文，rk8v4 KV，MTP3 + 提案头）

| 深度 | prefill | decode | MTP 接受率 | TTFT |
|---|---|---|---|---|
| 短（67 tok） | 141 t/s | **97.2 t/s** | 79.6% | 476 ms |
| 20K | **662 t/s** | **61.2 t/s** | 43.9% | 30.0s → **118 ms**（缓存后） |
| 60K | **516 t/s** | **51.2 t/s** | 44.5% | 1m55s → **170 ms**（缓存后） |

**前缀缓存 100% 命中**（20K/60K 复测均 100.0%）。

### 4.3 三方终局对比（同模型、同提示）

| 引擎 | 20K decode | 60K decode | 20K prefill | 60K prefill | 最大上下文 | 显存 |
|---|---|---|---|---|---|---|
| llama.cpp 128K | 26.4 | 17.0 | 433 | 334 | 128K | 9.9G |
| KVMem 256K | 19.3 | 19.2 | 234 | 198 | 256K | 7.6G+9G RAM |
| **NInfer 三值** | **61.2** | **51.2** | **662** | **516** | 96K | 11.1G |

**NInfer 在 60K 深度是 llama.cpp 的 3.0 倍 decode、1.5 倍 prefill。**

### 4.4 必须打的两个补丁（本机 fork 的 bug）

1. **FFmpeg/curl 依赖**：官方 Windows 构建靠 vcpkg 提供。ZCode 用不到视觉，故把
   `src/media/decode/decode.cpp`（FFmpeg，4 个函数）与
   `src/product/media_acquire/acquire.cpp`（curl，1 个函数）替换为 stub，
   并在 `cmake/Dependencies.cmake` 用空接口库顶替依赖。原文件均留 `.orig`。

2. **`kMinSupportedSmCount = 82` 假设过窄**（`bf16_gdn_gating_proj_plan.cpp`）：
   GDN gating 的 MMA 路由用**协作启动**，驻留预算 = 每 SM CTA 数 × **设备 SM 数**。
   路由表的 cols 区间按 82 SM（RTX 3090）标定；**RTX 3060 只有 28 SM**，预算只剩 1/3，
   于是启动即报 `BF16 GDN gating: candidate is not legal for exact problem`。
   修复 = 首选调度不合法时逐级回退 `Split8 → Split4 → Split2 → MmaUnsplit`
   （`MmaUnsplit` 非协作启动、无驻留约束，一定能兜住）。
   ⚠️ **这个 bug 影响所有 SM 数 < 82 的 sm_86 卡**（3060/3070/3080）。

### 4.5 起服

```
C:\Bonsai-App\Start-Bonsai-Ninfer.bat [上下文]     默认 98304
API : http://127.0.0.1:8080/v1     model id: bonsai2-27b
```

⚠️ 请求**必须带** `"model": "bonsai2-27b"`，否则 400。

### 4.6 KV dtype 与投机的质量验证（实测，非假设）

**A. KV dtype 质量代价 —— PPL（同制品、同语料、同协议 4096/2048，261,167 token）**

| 领域 | rk8v4（作者文档） | **nvfp4（本机实测）** | 差异 |
|---|---|---|---|
| WikiText | 7.820 | 7.8409 | +0.27% |
| PG-19 | 8.663 | 8.6657 | +0.03% |
| 中文 | 7.903 | 7.9146 | +0.15% |
| 代码 | 1.861 | 1.8642 | +0.16% |
| **总体** | **5.630** | **5.6395** | **+0.17%** |

⇒ **`nvfp4` 用 +0.17% PPL 换来 +67% 上下文（160K vs 96K），是明确的净收益。**
⇒ 单次 PPL 耗时约 13 分钟（加载 1.5 min + 评分 11 min，343 tok/s）。

**B. 投机配置 —— 逐字质量比对（8 道题：数学/逻辑/代码/中文/指令遵循）**

| 配置 | 与基线逐字一致 | 结论 |
|---|---|---|
| `--lookup-ngram 16` | **8/8 完全一致** | 零质量变化 |
| `--spec dflash2` d7 | 5/8 一致，3 处**仅措辞/格式差异**（`11,13,17` vs `11, 13, 17`；`$0.05 (5 cents)` vs `$0.05`；中文换说法） | 语义无变化 |

⇒ 与理论一致：**投机是"验证式"的，按构造不改变输出分布。**

**C. 投机配置速度（rk8v4 @ 64K，同提示）**

| 配置 | 短问答 | 20K decode | 60K decode | VRAM@64K | 接受率@20K/60K |
|---|---|---|---|---|---|
| **mtp3 + 提案头**（默认） | 94.2 | 61.4 | 53.5 | 8157 MiB | 43.9% / 44.5% |
| `--lookup-ngram 16` | 94.2 | 56.7 ⬇️ | 56.9 | ~8157 MiB | 40.4% / 50.1% |
| `--spec dflash2` d7 | **139.1** ⬆️ | **74.9** ⬆️ | 53.9 ≈ | **11033 MiB** ⬆️ | 28.5% / 22.3% |

⇒ **`--lookup-ngram` 无收益（20K 反降 7.7%）→ 弃用。**
⇒ **DFlash2 短/中上下文 +22~48%，但 60K 收益归零**（接受率从 57.6% 崩到 22.3%），且多吃 **2.9 GB 显存** → 在 12G 卡上**与 160K 上下文无法共存**。
⇒ **默认保持 MTP3 + 提案头**；DFlash2 仅用于短上下文极速场景（`Start-Bonsai-Ninfer-Fast.bat`）。

### 4.7 视觉支持（已恢复，端到端验证通过）

**问题**：官方 Windows 构建靠 vcpkg 提供 FFmpeg + libcurl 开发库；我们最初为省时间把
`media/decode`（FFmpeg）和 `media_acquire`（curl）换成了 stub ⇒ **视觉不可用**。

**解法（比装 vcpkg 便宜得多）**：

| 依赖 | 原方案 | 采用方案 |
|---|---|---|
| 图片解码 | FFmpeg（vcpkg，构建 30–60 分钟） | **stb_image 单头文件**（283 KB，已内置 `third_party/stb/`） |
| 媒体获取 | libcurl | **Data(base64) / Path / Bytes** 原生实现；远程 URL 明确报错 |

**实测（160K 上下文 + MTP3 全开）**：

| 测试图 | 问题 | 回答 | 耗时 |
|---|---|---|---|
| PNG（左红右蓝 + 白横条） | 从左到右两个颜色？ | **"Red and blue"** ✅ | 2.3s |
| JPEG（左绿右黄） | 同上 | **"green and yellow"** ✅ | 2.5s |
| **真实截图（中文复古书封）** | 这是什么 | **准确读出「陕甘宁边区政府《惩治贪污暂行条例》」**，并给出颁布主体/性质/年代/历史意义，**连图片的仿古做旧风格都看出来了** ✅ | 1,117 tok / 19s / **56.74 t/s** |

> 第三项是**通过 WebUI 粘贴截图**完成的端到端验证（浏览器 → shim → NInfer → 视觉塔），
> 覆盖了中文 OCR、图像风格判断与历史推理——这是最有说服力的一项。

启动参数：`--vision --vision-residency overlay --vision-max-merged 12288`
（overlay 让视觉塔常驻主机内存，几乎不占常驻显存；实测 160K 上下文下 free 仍有 ~318 MiB）

**已知边界**：
- **视频不支持**（`decode_video`/`inspect_video` 明确报错）——ZCode 不发视频，不影响
- **远程 URL 图片不支持**（无 libcurl）——base64 内联与本地文件都可用，覆盖截图场景
- JPEG 的 **EXIF 方向**未处理（stb_image 不解 EXIF）；截图通常无 EXIF 方向，影响很小

**改动文件**（均有备份）：`src/media/decode/decode.cpp`、`src/product/media_acquire/acquire.cpp`、
`src/media/CMakeLists.txt`（加 stb include 路径）、`third_party/stb/stb_image.h`（新增）

### 4.8 WebUI（llama.cpp 的界面跑在 NInfer 上，已实测可用）

**问题**：NInfer 没有内置 WebUI（源码里无 `/` 路由、构建产物无 UI 资产）。

**解法**：KVMem 包（llama.cpp 分支）**自带 llama.cpp 的 UI 资源在磁盘上**
（`C:\Bonsai-KVMem\share\kvmem\ui\`，51 个资源），写一个 shim 把两者接起来：

```
浏览器 ──► http://localhost:8080/     ← llama.cpp 的 UI（静态托管）
              ├─ /props   ← shim 合成（llama.cpp 形状；n_ctx/vision/模板）
              ├─ /tools, /v1/streams/lookup ← 空响应（UI 的可选面板）
              └─ /v1/*    ← 代理到 NInfer（含 SSE 流式透传）
                    http://127.0.0.1:8081/v1   ← NInfer 引擎
```

**关键的三处适配**（都是 llama.cpp 与 NInfer 的差异，不是 bug）：

| # | 差异 | shim 的处置 |
|---|---|---|
| 1 | llama.cpp 的 `/props` 有 `endpoint_slots`/`endpoint_metrics`/`cors_proxy_enabled` 开关 | 设为 **false** → UI 不再调 NInfer 没有的端点 |
| 2 | **llama.cpp UI 不带 `model` 字段**（单模型时可省略），**NInfer 强制要求** | 代理层**自动补 `model: "bonsai2-27b"`** |
| 3 | UI 会探测 `/tools`、`/v1/streams/lookup` | 返回空结构（200），不报错 |

**实测结果**：UI 完整渲染（思考折叠区、流式输出、速度统计），
对话 **74.98 t/s**，模型选择器显示 `bonsai2` / `27B`。

**一键启动**：`C:\Bonsai-App\Start-Bonsai-WebUI.bat`
（起 NInfer@8081 → 等就绪 → 起 shim@8080 → 自动打开浏览器）

**文件**：`C:\Bonsai-App\webui.py`（纯 Python 标准库，无依赖）

## 五、推荐配置

**主力（带 WebUI，推荐）**: `C:\Bonsai-App\Start-Bonsai-WebUI.bat`
    引擎 @8081 + WebUI shim @8080 → 浏览器打开 `http://localhost:8080/`
    ZCode 仍指向 `http://localhost:8080/v1`（shim 会代理）｜模型 id `bonsai2-27b`

**主力（无 UI，只有 API）**: `C:\Bonsai-App\Start-Bonsai-Ninfer.bat`
    端口 8080 | **160K 上下文** | nvfp4 KV | prefill 662/516 t/s | decode 61.2/51.2 t/s | VRAM 11.1G

> 模型文件已搬到 **C 盘**（`C:\Bonsai-App\models\`），启动从 **1分45秒 → 4.7 秒**（快 22 倍）。H 盘那份保留作回退。

**短上下文极速**: `C:\Bonsai-App\Start-Bonsai-Ninfer-Fast.bat`
    端口 8080 | 64K 上下文 | DFlash2 | decode 139/75 t/s | VRAM 11.0G

**回退 A（显存更宽松）**: `C:\Bonsai-App\Start-Bonsai.bat`
    端口 8080 | 128K 上下文 | prefill 441/336 t/s | decode 26.6/17.1 t/s | VRAM 9.9G

**回退 B（超长上下文 256K）**: `C:\Bonsai-KVMem\Start-KVMem-256K.bat`
    端口 18200 | prefill 234/198 t/s | decode 19.3/19.2 t/s (平坦) | VRAM 7.6G + RAM 9G

**停止**: `C:\Bonsai-App\Stop-Bonsai.bat`（四个引擎都能停）

四者不能同时跑（显存合计超 12G）。原 32K 配置 `Start-Bonsai-32K.bat` 保留未动。

## 六、官方与社区资源

- **NInfer 引擎（本机所用）**: https://github.com/iamwavecut/ninfer-3090 分支 `franken/v0.11`
  （父仓库 `ashalliants/ninfer-3090`，上游 `Neroued/ninfer`）
- **三值制品**: https://huggingface.co/WaveCut/Ternary-Bonsai-2-27B-NInfer-v3
- **Ada 三值移植材料（参考）**: ModelScope `shensanshu/ninfer-ada-ternary`
- KVMem: https://github.com/kvmem/kvmem-llama.cpp （Bonsai 分支 `/tree/kvmem-bonsai-llama.cpp`）
- B站作者教程: BV1r2ey6EEh5 / BV1ZKhH6iEWr；三值实测: BV1hjhh6DEbn

## 七、本机路径

```
C:\ninfer-build\franken\ninfer-3090-franken-v0.11\  源码 + build-ninja\apps\
H:\ninfer-models\Ternary-Bonsai-2-27B-ninfer-v3.ninfer   9.52 GB 制品
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8  旁装 CUDA（回退=删目录）
```

## 八、Ring 256K 深度基准（RTX 3060 12G，实测完成）

> 2026-09-25 干净环境实测（ZCode / dsh 均已退出，单 slot 无争用；`queue` 全 < 65 ms 可证）。
> 数据取自**引擎自报日志**（`req#N done` 行），非客户端估算。脏数据（排队污染）不进表。

### 8.1 测试矩阵

| # | 内容 | prompt tokens | 是否超 96K 池 | 结果 |
|---|---|---|---|---|
| T1 | 短问答（自我介绍） | 58 / 65 | 否 | 质量正常，身份正确 |
| T2-4K | prefill 扫频 | 4,422 | 否 | ✅ |
| T2-16K | prefill 扫频 | 17,028 | 否 | ✅ |
| T2-64K | prefill 扫频 | 65,513 | 否 | ✅ |
| T3 | **藏针召回**（STARFRUIT-88） | 129,376 | **是（1.35×）** | ✅ **PASS** |
| T4 | 双轮记忆（PARROT-31） | 65 → 88 | 否 | ✅ **PASS**（cache 68.2% turn closure） |

### 8.2 prefill / decode 曲线（服务端权威数）

| prompt tokens | prefill t/s | TTFT | decode t/s | MTP 接受率 |
|---:|---:|---:|---:|---:|
| 4,422 | **658.6** | 6.7 s | 67.3 | 48.6% |
| 17,028 | **506.4** | 33.7 s | 52.5 | 42.1% |
| 65,513 | **252.1** | 4 m 20 s | 37.2 | 60.0% |
| **129,376（环通路）** | **147.8 / 150.6** | **14 m 20–36 s** | **18.3 / 19.5** | 35.9 / 38.9% |
| 65（短问答） | 345.8–368.7 | **208–511 ms** | 63.6–71.9 | 41.8–50.0% |

注：短 prompt 的 prefill 数被模板/预热固定开销主导，不作吞吐参考；有效曲线是 4K→129K 四档。
两行长请求各测一次（两台引擎进程），数值一致性 ±2%。

### 8.3 结论

1. **池内零开销**：4K 658.6 t/s ≈ dense-160K 生产的 662 t/s（§4.2），门控旁路没拖慢。
2. **深度衰减与官方同趋势、更陡**：4K→64K 掉 62%（官方 3090：1794→1234，掉 31%）——
   3060 带宽 + 全页 mask 检查的代价。
3. **环通路可用**：129,376 tokens（1.35× 池）走降级+IDF 检索，召回 PASS，prefill 147.8 t/s
   （官方 3090 在 250K 是 613 t/s，约 1/4，符合卡差）。
4. **decode 天花板随深度**：129K 处 18.3 t/s（官方 3090 在 250K 是 71 t/s）。
5. **MTP 接受率随深度下降**：短问答 ~42–50% → 129K 处 35.9–38.9%，与官方 3090 的 22–57% 区间一致。
6. **多轮正常**：第二轮 cache 68.2%（turn closure），decode 69.4 t/s，短轮 TTFT 194 ms。

### 8.4 稳定性观察（待续，非分数）

| 时间 | 现象 | 环境 | 定性 |
|---|---|---|---|
| 02:52:57 | 引擎 fail-fast `0xc0000409`（ucrtbase.dll） | T2-64K prefill + dsh 并发排队 | 并发诱发，疑栈检查/abort |
| 03:19:47 | 引擎**静默退出**（无 WER 事件、无 stderr 报错） | 长请求结束、空闲后 | 原因不明，需更多数据 |
| 03:57–04:12 | 129K 长请求 + 双轮全程无异常，空闲后仍存活 | 干净环境 | 未复现 |

⇒ 两次异常都发生在**并发或空闲边界**，干净长请求路径稳定。已挂独立退出码监视进程
（`engine2.exitcode`）继续观察；复现后另案排查。

### 跨卡吞吐与能效对照（白皮书 Table 5 / Table 6 全表，tg128 / pp512 标准）

协议：`tg128` = depth 0 生成 128 token（带宽受限），`pp512` = 512-token prompt（算力受限）；
batch 1、排除视觉塔、预热后测 3 次取均值。后端 llama.cpp + GGUF 包。

| 硬件 | PQ2_0 TG128 | PQ2_0 PP512 | PQ2_0 mWh/tok | PTQ1_0 TG128 | PTQ1_0 PP512 | PTQ1_0 mWh/tok |
|---|---:|---:|---:|---:|---:|---:|
| RTX 5090 (32 GB) | 142.5 | 4121 | 0.582 | 134.4 | 1901 | 0.609 |
| RTX PRO 6000 Blackwell | 140.6 | 4520 | 0.637 | 136.8 | 2290 | 0.642 |
| H200 SXM (141 GB) | 118.0 | 2818 | 0.708 | 89.5 | 1216 | 0.814 |
| B200 (180 GB) | 117.1 | 3112 | 0.830 | 91.9 | 1406 | 1.019 |
| H100 NVL (94 GB) | 106.4 | 2484 | 0.541 | 81.6 | 1098 | 0.647 |
| H100 SXM (80 GB) | 103.2 | 2467 | 0.584 | 77.3 | 1097 | 0.706 |
| RTX 4090 (24 GB) | 90.9 | 3134 | 0.714 | 96.7 | 1634 | 0.682 |
| RTX 6000 Ada (48 GB) | 84.8 | 2430 | 0.731 | 92.0 | 1627 | 0.701 |
| L40S (48 GB) | 74.6 | 2827 | 0.812 | 82.8 | 1601 | 0.743 |
| A100 SXM (80 GB) | 74.0 | 1328 | 0.776 | 54.6 | 703 | 0.932 |
| L4 (24 GB, 72 W) | 29.7 | 778 | 0.629 | 32.1 | 468 | 0.585 |
| M5 Pro（Metal） | 27.7 | 397 | — | 27.1 | 369 | — |
| **RTX 3060 12G（本机 NInfer ring+MTP，实测）** | **≈62–72** | **658.6（4K prompt）** | **0.521** | — | — | — |

Apple Silicon（Table 6，PQ2_0 + Metal）：M5 Max 46.8 / 765；M5 Pro 27.7 / 397；M4 Pro 18.0 / 125。

**三条读数**

1. **速度站位**：本机短问答 decode ≈62–72 t/s，落在 M5 Max（46.8）与 A100（74.0）之间；
   4K prefill 658.6 t/s 是 5090 pp512（4121）的约 1/6，但 prompt 长度不同，只看量级。
   3060 显存带宽 360 GB/s vs 5090 的 1792 GB/s，decode 比值 66/142.5 ≈ 46%，
   超出带宽比值（20%）——MTP 投机（3 draft）补回了一块。
2. **能效反超**：本机 **0.521 mWh/tok**，比 5090（0.582）和 B200（0.830）都低。
   同口径（`nvidia-smi` 板功耗含显存、不扣 idle）：生成 128 token 期间均值 116.7 W、
   峰值 168.2 W（170 W TDP 打满）。**低功耗卡做 7×24 常驻是划算的**——
   这正是"12G 老卡跑 256K"这套方案的实际价值。
3. **架构规律（白皮书 §D）**：PTQ1_0 在 **Ada** 与 L4 上更快（4090 96.7 > 90.9），
   PQ2_0 在 **Ampere / Hopper / Blackwell / Apple** 上更快（A100 74.0 > 54.6，H100 103.2 > 77.3）。
   ⚠️ **RTX 3060 是 Ampere（GA106，sm_86），不是 Ada**。
   **好消息：本机制品本来就在 PQ2_0 路线上**——转换配方用的是
   `--source ternary=Ternary-Bonsai-2-27B-PQ2_0.gguf`，容器格式 `t2_g128_fp16` 是
   "2-bit codes + 每 128 列一个 fp16 scale" = 2 + 16/128 = **2.125 bpw**，
   正是白皮书 PQ2_0 的块格式（34 字节 / 128 权重 = 2.125 bpw）。
   ⇒ 在 Ampere 上更快的那个打包，我们**已经在用**，无需切换（此前的"应换 PQ2_0"判断有误，已更正）。

口径声明：白皮书是 llama.cpp 后端、纯自回归；本机是 NInfer 引擎 + MTP 投机 + ring/nvfp4，
runtime 与协议都不同，数字只对量级不对点；本机行严格对齐附录 D（depth 0 / batch 1 / 预热 3 次）
后待补。

## 九、官方分数对照（计划内，未跑）

三级分数链（每一级都有发布方，其责权分明）：

**L1 基座**（PrismML 官方发布，thinking 模式）：Qwen3.8-27B（85.4）→ 三值 Bonsai 2 27B（83.9，
**保留 98.2%**，1.76 bit/权重，5.9GB，262K 上下文，Apache-2.0）：

| 能力域 | 三值 Bonsai 2 27B | Qwen3.8-27B（全精度） |
|---|---|---|
| Agent/工具调用（τ²、BFCLv3） | 77.57 | 79.74 |
| 代码（HumanEval+、LCBv6、MBPP+、BigCodeBench） | 81.58 | 82.17 |
| 指令遵循（IFBench、IFEval） | 82.66 | 81.25 |
| 知识推理（MMLU-Redux、GPQA-D、AA-LCR） | 83.95 | 86.66 |
| 数学（AIME26/25、GSM8K、MATH-500） | 96.57 | 97.06 |
| 视觉 | 78.59 | 81.64 |
| 总分 | 83.9 | 85.4 |

**L1b 白皮书逐项分**（`bonsai-2-27b-whitepaper.pdf` 附录 C **Table 10 全表**，xhigh thinking，
EvalScope + vLLM / H100×（TP2·DP4），温度 1.0 单样本 pass@1（非 greedy）；
AIME 平均 8 采、GPQA 平均 5 采。**粗体 = ZCode 自用最相关**）：

> ⚠️ **可信度声明**：下表是**厂商发布数字**，本机只独立复现过其中一项（HumanEval+，
> 见 §十：92.68 vs 95.12，落在同一区间，弱佐证而非证实）。其余 19 项在本机上重跑一遍
> 需要 H100 级硬件与数周时间，不可行。引用时注意以下四点：
>
> 1. **98.2% 是聚合数，藏着弱项**：TerminalBench 52.8 vs 69.7（76%）、SWE-bench 60.8 vs 80.6
>    （75%）——白皮书自己公布了，这是加分项，但"保留 98.2%"这个标题数会让人忽略长程 agent 的 1/4 损失。
> 2. **协议偏慷慨**：AIME 取 8 采平均、GPQA 取 5 采平均（不是单发）；温度 1.0 单样本的方差本身就大，
>    且**未公布误差棒与重复次数**。
> 3. **对照组可能偏弱**：IQ2_XXS 的具体配置（提示模板、预算、是否开思考）未逐项披露；
>    拿它当"常规低比特代表"时要打折。
> 4. **吞吐与我们无关**：Table 5 是 llama.cpp 后端 + 高端卡，本机 NInfer 线的数字以 §八 为准，
>    跨表只看量级。

| benchmark | 三值 Bonsai 2 27B | Qwen3.8 全精度 | Qwen3.6 全精度 | IQ2_XXS（7.3G） |
|---|---:|---:|---:|---:|
| **知识推理** | | | | |
| MMLU-Redux | 89.09 | 91.46 | 93.26 | 85.79 |
| GPQA Diamond | 85.76 | 90.51 | 86.87 | 65.45 |
| AA-LCR（长上下文推理） | 77.00 | 78.00 | 74.00 | 64.00 |
| **数学** | | | | |
| GSM8K | 96.66 | 97.19 | 95.60 | 95.38 |
| MATH-500 | 98.80 | 99.80 | 99.20 | 94.8 |
| AIME25 | 95.00 | 96.67 | 90.42 | 82.5 |
| AIME26 | 95.83 | 94.58 | 93.33 | 78.6 |
| **代码** | | | | |
| **HumanEval+** | **95.12** | 93.29 | 95.73 | 87.95 |
| **MBPP+** | **83.07** | 83.86 | 83.07 | 78.31 |
| **LiveCodeBench v6** | **90.07** | 90.05 | 89.10 | 70.05 |
| **BigCodeBench** | **58.07** | 61.49 | 62.37 | 49.81 |
| **指令遵循** | | | | |
| IFEval | 91.31 | 91.50 | 88.72 | 81.52 |
| **IFBench** | **74.00** | 71.00 | 60.33 | 52.33 |
| **Agent / 工具调用** | | | | |
| **τ²-Bench** | **80.22** | 82.73 | 82.90 | 69.43 |
| **BFCL v3** | **74.92** | 76.74 | 77.19 | 66.1 |
| **视觉** | | | | |
| CharXiv | 80.00 | 82.02 | 81.60 | 78.76 |
| A-OKVQA | 86.81 | 89.34 | 89.87 | 86.2 |
| OmniDocBench v1.6 | 89.13 | 92.46 | 84.17 | 82.87 |
| RealWorldQA | 80.13 | 83.40 | 81.83 | 78.56 |
| OCR Bench v2 | 56.88 | 60.99 | 61.64 | 55.37 |
| **平均（20 项）** | **83.9** | **85.4** | **83.6** | **75.2** |
| 体积 / bpw | **5.93 GB / 1.76** | 53.8 GB / 16.0 | 53.8 GB / 16.0 | 7.3 GB / 2.2 |

长程 agent（白皮书 §4，Harbor 框架驱动，同服务配置）：

| benchmark | 三值 Bonsai 2 27B | Qwen3.8 全精度 | 保留 |
|---|---:|---:|---:|
| Terminal-Bench 2.1（89 任务，Terminus-2） | 52.8 | 69.7 | ~76% |
| SWE-bench Verified（500 实例，mini-swe-agent） | 60.8 | 80.6 | ~75% |

**reasoning effort 对照**（Table 11，medium）：总分 79.3 vs xhigh 83.9（全精度 82.6）；
AIME25 74.58 vs 95.00、GPQA 75.56 vs 85.76、τ² 70.86 vs 80.22 —— **effort 一降，硬项雪崩**。

**智能密度**（Table 9，ID = −log₂(1−avg/100)/GB）：三值 5.93GB → **0.444 /GB**；
IQ2_XXS 7.3GB → 0.276；全精度 53.8GB → 0.051。

要点：① 三值在 agent 长程任务上保留约 3/4，常规低比特量化（IQ2_XXS 在 LiveCodeBench 70.05、
AIME26 78.6）直接崩；② **代码三件套几乎无损甚至反超**（HumanEval+ 95.12 > 93.29，
LCBv6 90.07 ≈ 90.05）——ZCode 场景押对了；③ reasoning effort 必须保持 xhigh；
④ **L2 的 greedy-slice 与白皮书 temp-1 是两种协议，分数不可混比**；本机 L3 对照跟白皮书协议
（EvalScope，温度 1.0 / top-p 0.95 / top-k 20，xhigh，宽松 output budget），与本仓库 harness 同族。

**L2 制品**（WaveCut HF 卡，同文件，RTX 3090 常驻 rk8v4），1,179 题固定 slice、greedy：

| suite | 题数 | 官方 MTP | 官方 DFlash2 |
|---|---|---|---|
| GSM8K | 200 | 95.5 | 96.5 |
| MMLU-Pro | 280 | 73.2 | 73.6 |
| HumanEval | 164 | 90.9 | 90.9 |
| MGSM（俄） | 250 | 90.4 | 90.0 |
| Global-MMLU（俄） | 285 | 78.6 | 79.3 |
| pooled | 1179 | 84.4 | 84.7 |

本机对照方案（本仓库 `eval/` harness 打 `127.0.0.1:8081`，production 配置不动）：

| 优先级 | 内容 | 耗时估计 | 说明 |
|---|---|---|---|
| P0 | NIAH 长上下文 64K/128K/200K（thinking 关） | 几小时 | 直接给 ring 定级，对标官方 250K 三码召回 |
| P1 | HumanEval-164 全集 | ~1 小时 | 对标官方 90.9，ZCode 自用最相关 |
| P2 | IFBench-300 | 3–5 小时（过夜） | 指令遵循，harness 有 Qwen3.8 参考分 |
| P3 | GPQA-Diamond / AIME | 数十小时 | 可选，3060 上太慢 |

前置条件：机器安静（ZCode、dsh 等让出单 slot），脏数据不进表。
预期：**L3 落在 L2 值 ±1% 内**（符号检验噪声水平，官方卡内 p≈1.0）；
速度不对标（3060 vs 3090/5090 + ring mask + host 溢出，必然更慢）。
参考：官方称三值版在 5090 上 143 t/s；本机 ring 短问答实测 decode ≈43 t/s。

## 十、L3 本机实测：HumanEval+（RTX 3060 ring）

**结果：152 / 164 = 92.68%**（pass@1；温度 1.0 单样本、xhigh 思考、预算 49,152，
与白皮书附录 B 同协议）。同一测试集（EvalPlus 扩展，164 题）对照：

| 模型 | HumanEval+ | 来源 |
|---|---:|---|
| Qwen3.6-27B FP16 | 95.73 | 白皮书 Table 10 |
| 三值 Bonsai 2 27B（官方，H100） | **95.12** | 白皮书 Table 10 |
| Qwen3.8-27B FP16 | 93.29 | 白皮书 Table 10 |
| **本机 ring（RTX 3060 12G）** | **92.68** | 本节 |
| Qwen3.8-27B IQ2_XXS | 87.95 | 白皮书 Table 10 |

**差值 4 题（−2.44 分）**。n=164 时 95% 置信区间约 ±3.98 分（SE=2.03%），
官方 95.12 落在本机区间内 → **统计上不可区分**；点估计略低，归因见下。

失败明细（12 题，**全部为真实代码缺陷**，无环境假阴性、无超时）：

| 类型 | 题数 | 例 |
|---|---:|---|
| AssertionError（边界条件不满足） | 9 | HumanEval/103、116、124、132、141、145、151、91、99 |
| NameError | 1 | HumanEval/154 `name 'i' is not defined` |
| IndexError | 1 | HumanEval/160 `pop from empty list` |
| TypeError | 1 | HumanEval/32 `Value after * must be an iterable, not float` |

同一次运行的性能（引擎自报）：

| 指标 | 值 |
|---|---:|
| 平均延迟 | 46.5 s/题 |
| 平均输出 | 2,745 token（含 reasoning） |
| 平均输入 | 220 token |
| 吞吐 | 59.05 t/s |
| 收尾方式 | 全部 `stop token` 自然收尾，无 output-limit 截断 |

**口径与偏差**：

1. 本机 EvalScope 1.10.0，协议同白皮书（temp 1.0 / top-p 0.95 / top-k 20 / 单样本 / xhigh）。
2. **沙箱差异**：官方用 Docker `python3.11-numpy`；本机无 Docker，改为等价子进程执行
   （同样的 `prompt + completion + test + check(entry_point)` 拼接），Python 3.12 + numpy 1.26。
3. 单样本 temp 1.0 的方差：官方自身两次修订间也有 ±1% 抖动（1,179 题中 27 项翻转）。
4. 未复现的偏差源：nvfp4 KV（+0.17% PPL）、int8 lm-head、MTP 投机（验证式，理论不改分布）；
   ring 在这些短上下文（<1K token）**完全不激活**。
5. 因此本机这一分应主要反映**三值基座本身**，而非 ring 移植引入的退化。

**⚠️ 避坑（踩过的坑，供复现者参考）**：EvalScope 的 `humaneval_plus` 适配器走
`CodeExecutionSandboxMixin`；sandbox 未启用时 `execute_code_in_sandbox` 返回
`{'error': 'Sandbox is not initialized.'}` 且**不抛异常**，适配器取不到 `status`
→ **每题判 False，总分 0**。两种解法：启用 Docker 沙箱，或像本机一样用等价子进程执行重打分
（预测已落盘，不需要重跑 GPU）。

复现：`eval/configs/bonsai2_27b_ring.yaml`（suite `humaneval_plus`）；
本次运行目录 `eval/runs/20260924T202723Z-89f9e0aa`。

## 十一、评测避坑清单（本机实测踩过的）

跑一遍仓库自带的 `eval/` harness，踩到 9 个坑。前 3 个会导致**分数完全错误**，必须知道。

### 会算错分的（严重）

| # | 坑 | 现象 | 修法 |
|---|---|---|---|
| 1 | `humaneval_plus` 沙箱未启用 | `execute_code_in_sandbox` 在 `use_sandbox=False` 时返回 `{'error': ...}` 且**不抛异常**，适配器取不到 `status` → **每题判 False，总分 0** | 启用 Docker 沙箱，或用等价子进程本地执行重打分（预测已落盘，不必重跑 GPU）。建议给 harness 加 fail-fast |
| 2 | NIAH 的 `judge_strategy: rule` | 走 `exact_match`（整串归一化逐字比对）。模型答对但省了框架前缀（如漏掉 "The best thing to do in San Francisco is"）→ **判 0** | 用 LLM judge（EvalScope 对 NIAH 的默认，`llm_judge_default = True`） |
| 3 | NIAH 的 LLM 裁判 prompt 含全文 | 上游 `question = task_state.input_text` 是**整个渲染后的 64K prompt** → 每个样本要跑两条 65K 请求（主请求 + 裁判），耗时翻倍 | 补丁：`question = self.retrieval_question`（短问题）。见 `eval/patches/needle_haystack_judge_prompt.patch` |

### 会拖慢或误导的

| # | 坑 | 修法 |
|---|---|---|
| 4 | LLM 裁判默认走模板 xhigh 思考，`max_tokens 4096` | 裁判 `generation_config` 加 `extra_body.enable_thinking: false` |
| 5 | `eval/README.md` 的命令是 Linux 路径（`eval/.venv/bin/python`） | Windows 用 `eval\.venv\Scripts\python.exe`；README 应补双平台 |
| 6 | `eval/configs/qwen3_6_35b_needle_haystack.yaml` 硬编码原作者 Linux 路径（`/home/neroued/...`） | 参数化/环境变量，或像本仓库新增的 `bonsai2_27b_ring_niah.yaml` 一样写本机路径 |
| 7 | README 提到的 `--check-runtime` 在 1.10.0 里不存在 | 改文档（`validate` 无此 flag） |
| 8 | Python 3.14 装不上 EvalScope 栈 | 用 Python 3.12（`uv venv --python 3.12`）；本机 3.14 是默认解释器，容易踩 |
| 9 | 拼长 prompt 前不换算 token | 我按字符估 token 错了 4 倍，130K 请求撞 `context_length_exceeded`（引擎行为正确，是我的估算错）。**先小样本实测 `prompt_tokens` 再放大** |

### 环境噪音（无害）

- `matplotlib` 在无头 Windows 上报 `Starting a Matplotlib GUI outside of the main thread`，图仍正常生成。
- 退出时 `atexit` 里 tkinter 报 `main thread is not in main loop`，不影响结果。
- `read` 工具不支持 PDF；白皮书需用 `pypdf` 提取文本（本机已装）。

## 十二、NIAH 长上下文（64K，环通路实测）

**结果：22 样本，accuracy 0.9591**（英文 11/11 = 100%，中文 10/11 = 90.9%）。

配置：`eval/configs/bonsai2_27b_ring_niah.yaml`，64K 上下文（`Context#65536`），
中英双语 × 11 个深度（0%–100%），thinking 关闭，LLM 裁判（指向本机引擎），
rule 评分 → 见 §十一 坑 2/3。

| 子集 | 通过 | 热力图 |
|---|---|---|
| English | **11 / 11 = 100%** | `bench-figures/niah-64k-english.png`（全绿） |
| Chinese | **10 / 11 = 90.9%** | `bench-figures/niah-64k-chinese.png`（仅 Depth#100 红） |
| 合计 | 21 / 22 满分 + 1 个 0.1 | EvalScope 报 0.9591 |

**唯一的失败：中文子集 Depth#100**（针埋在文档最末端）。核实过：

- 针**确实在 prompt 里**（11/11 样本的 prompt 都含 `San Francisco`）→ 不是数据集/构造问题
- 模型答 `The provided text does not contain information about San Francisco.`（直接否认）
- LLM 裁判给最低档 `[[1]]`（1/10 → 0.1），与"完全无关"的档位一致
- 英文子集同深度（Depth#100）**通过**，所以是"针在最末端 + 中文语料"这一档的边缘失败

**结论**：环通路在 64K 下，**针埋在 0%–90% 任何位置都能稳定捞出**（英文全深度、中文 0–90），
只有"最末端 + 中文"这一档漏了 1 次。考虑到 64K < 96K 窗口（全程未触发降级），
这一档反映的是**模型本身**的边缘行为，而非环通路退化。

热力图由 EvalScope 自动生成（`reports/bonsai2-27b/needle_haystack_heatmap_{english,chinese}.png`），
已复制到 `bonsai-app/bench-figures/` 便于引用。

## 十三、长 prefill 崩溃：复现与插桩（进行中）

**现象**：`ninfer-serve.exe` 观察到 4 次非正常退出。

| # | 时间 | 场景 | 签名 |
|---|---|---|---|
| 1 | 02:52:57 | 64K prefill + 并发排队 | fail-fast `0xc0000409` @ `ucrtbase.dll+0x7286e` |
| 2 | ~03:19:47 | 129K 请求结束、空闲 | 无 WER 事件、无 stderr |
| 3 | ~07:44:02 | 65K prefill 中请求被 `cancel` 后 | 无 WER 事件 |
| 4 | 13:26:39 | 64K prefill 中途（单客户端） | fail-fast `0xc0000409` @ `ucrtbase.dll+0x7286e`（与 #1 逐字相同） |

`0xc0000409` = `STATUS_STACK_BUFFER_OVERRUN`；模块与偏移在 #1/#4 完全一致 → 指向同一条确定性代码路径，
最可能是 `std::terminate` → `abort()`。

**插桩**（`apps/serve/main.cpp`，仅崩溃时输出，不影响正常运行）：

| handler | 触发 | 输出 |
|---|---|---|
| `std::set_terminate` | 未捕获 C++ 异常 | `NINFER-CRASH kind=std::terminate detail=<what()>`，退出码 **42** |
| `_set_invalid_parameter_handler` | CRT 参数校验失败 | `kind=invalid_parameter`，退出码 **43** |
| `SetUnhandledExceptionFilter` | SEH 异常 | `kind=unhandled_exception code=... addr=...` |

**复现结果（重要）**：

| 触发条件 | 样本 | 崩溃 |
|---|---:|---:|
| 干净串行长 prefill（唯一前缀，`eval/repro/long_prefill_crash.py`） | 6 | **0** |
| 真实 NIAH 64K 全程（22 样本，含 22 次 64K prefill） | 22 | **0** |
| **prefill 中途 FIN 断连**（`abort_prefill_crash.py close`） | 8 | **0** |
| **两条并发 64K 请求 + 中途断连**（`... pair`） | 6 | **0** |
| 合计 | **42** | **0** |

⇒ 三种最可疑的条件**全部排除**：干净串行、中途取消、并发排队都无法触发。
四次死亡都发生在并发/取消/空闲边界，但**构造同样的边界也不崩**，说明触发条件更隐蔽
（可能是时序窗口、内存压力或运行时长相关的罕见路径）。

**当前处置**：

1. **插桩保留进生产**——三个 handler 只在崩溃路径执行，零开销；下次崩溃会自动留下
   `NINFER-CRASH kind=... detail=...` 与退出码 42/43，可就地取证。
2. WER 全量 dump 已配置（`HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\ninfer-serve.exe`）。
3. Release 构建无 PDB，dump 难符号化；若崩溃再现，插桩输出是主要线索。
4. 罕见崩溃（观测 ~4 次 / 数小时运行）不阻塞生产：干净长请求路径已 42 次零故障验证。

**已知的非崩溃行为**（正常，勿误判）：

- 同一 prompt 重复请求会 100% 命中前缀缓存（0.3 s 返回，不走 prefill）——这是设计行为。
  复现长 prefill 必须每次用唯一前缀。

## 十四、速度开关矩阵（实测，含三处更正）

方法：ring 256K 同配置，逐配置重启引擎，测 4K/16K/64K prefill + decode。
**decode 一律取引擎自报值**（`req#N done` 行的 `decode X tok/s`），每配置 5 次取中位。

| 配置 | 4K prefill | 16K prefill | 64K prefill | decode 中位（范围） | 显存 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| baseline | 587.5 | 469.1 | 233.7 | 74.1（71.9–78.2） | 10305 | — |
| **`--prefill-cublas`** | 607.9<br>**+3.5%** | 486.1<br>**+3.6%** | 240.4<br>**+2.9%** | 71.3（70.1–77.4） | 10310 | ✅ **建议开**（prefill 稳定小赚，decode 噪声内，显存 +5 MiB） |
| `--embedding-q4` | — | — | — | — | — | ❌ **制品不兼容** |
| `--lm-head-q6` | — | — | — | — | — | ❌ **制品不兼容** |
| `--mlp-a8-decode` | 621.8<br>+5.8% | 489.9<br>+4.4% | 236.2<br>+1.1% | 65.5（**55.3–88.7**） | 10293 | ⚠️ **不建议**（prefill 小赚，但 decode 抖动 ±25%） |

### 三处更正（相对此前的记录）

1. **`--prefill-cublas` 不是 1.63–1.83x，实测 +2.9–3.6%。** 此前那条数字应是在 **dense 160K**
   （显存宽裕、cublas tile 更大）下测的；ring 模式下 prefill 走另一条实现，收益大幅缩小。
   仍建议开：稳定小赚、几乎零代价。

2. **`--embedding-q4` / `--lm-head-q6` 对本制品不可用**（不是"没开"，是"开不了"）：

   ```
   FATAL server failed during startup |
     --embedding-q4 requires text/token_embedding to be stored as row-split Q8_G32
   FATAL server failed during startup |
     --lm-head-q6   requires text/output_head   to be stored as row-split Q8_G32
   ```

   本制品的 embedding / output_head 是 **三值 `t2_g128_fp16`**，两个开关要求 **Q8_G32** 存储。
   所以此前记录的"embedding-q4 +24.3K ctx、lm-head-q6 +12.9K ctx"**对本制品无效**——
   要它们生效必须重新转换制品（把 embedding/head 改成 Q8）。

3. **decode 不能用客户端计时。** 同一配置单次测量在 62–78 t/s 之间跳（±20%），
   据此得出的"mlp-a8-decode 掉 20%"是**噪声**，不是结论。改用引擎自报值后：
   baseline 74.1（±4%，很稳），mlp-a8-decode 中位 65.5 但范围 **55.3–88.7**——
   真实结论是该开关**引入 decode 抖动**，而非单纯变慢。

## 十六、MTP 开关 A/B（3060 实测：MTP 是正收益）

背景：沈三殊的第三方三档包称 MTP 在小显存是**负收益**（llama-KVMem 线实测 decode −43%），
且刻意不发 MTP。本机 launchers 全开 `--spec mtp`，但**从未测过关 MTP 的基线**——补上。

方法：文本 ring 256K（**去掉 `--vision`**，见下），同负载 5 次 decode（128 token），
取引擎自报中位。两组各一次引擎重启。

| 配置 | decode 中位（范围） | MTP 接受率 | sanity 17×23 |
|---|---:|---|---|
| `--spec mtp --draft-tokens 3 --lm-head-draft` | **77.8**（72.0–79.5） | 58.5% | 正常 |
| 无 `--spec` | **37.2**（35–38，极稳） | — | 391 ✅ |

**结论：3060 上 MTP ≈ 2.1x，launchers 保持现状。** 与第三方线的 −43% 相反——
差异来源：不同引擎线（franken/v0.11 vs dev fork v1.0.8）、不同投机实现
（MTP3+提案头 vs 他们的配置）、不同 KV（nvfp4 vs fp8/q4）。**投机是正是负与卡无关，
与实现有关，不可跨引擎搬运结论。**

附带发现（启动约束）：

```
FATAL ... --vision-residency overlay needs 816 MiB of evict-ranked weights ...,
but ... provide 656 MiB
```

去掉 `--spec` 后，vision-overlay 借不到足够的 evict 权重，引擎**拒绝启动**。
所以"关 MTP + 开视觉"在此构建上是不可能的组合——A/B 双方统一去掉 `--vision`，
测的是纯文本 decode，不影响结论（视觉与 decode 速度无关）。

日期注：本节及之后时间为 2026-09-26（过零点）。

## 十八、流式是真的（逐字节验证，之前的"假流式"是测量 bug）

方法：socket 层 `read(1)` 逐字节计时（`eval` 之外的独立脚本），短 prompt，
`max_tokens=100`，`stream:true`。

| 指标 | 值 |
|---|---|
| 首字节 | 0.00 s（响应头 + role 行立即到） |
| 首个内容 token | 0.43 s |
| 数据行 | 33 行，散布在 **1.55 s / 全程 1.56 s** |
| 结论 | **真流式** |

之前"93 个 chunk 同一毫秒到达"的结论，是测试脚本用 `read(65536)` 阻塞攒满导致的假象
（小响应第一次 `read` 直接拿到全文），不是服务端缓冲。特此更正——
**测量工具的 bug 被当成被测对象的 bug，这是本轮第三次**（前两次：mlp-a8 的 20%、
MTP 测试打错端口）。

附带观察：`extra_body.enable_thinking:false` 在此请求里**仍有少量 reasoning 输出**
（首个 reasoning 块 0.43s）；NIAH 那次 15 token 纯答案。thinking 开关的行为不完全一致，
待单独验证（不影响流式结论）。

## 十七、thinking-budget 上生产（P0 迭代）

动机：生产 6 个启动器**全没设**思考预算，走模板默认 xhigh + 无限思考；
HumanEval+ 实测单题吃过 **31,360 token / 9 分 44 秒**。白皮书证明**降 effort 会雪崩**
（AIME25 95.00 → 74.58），所以正确做法不是降档，而是设**上限当保险丝**。

改动（仅 ring 256K 两个启动器先试点，dense 生产不动）：
`Start-Bonsai-Ninfer-256K.bat` / `Start-Bonsai-WebUI-256K.bat` 加
`--default-thinking-budget 8192`。

选 8192 的依据：正常问答实测思考 21–146 token；8192 只砍失控个例，
不碰正常输出。预算耗尽后按 `output limit` 收尾（有答案，比无限转圈强）。

验证：新 flag 引擎一次点亮；短请求 0.8s 答对（17×23=391，思考 21 token），
正常请求不受影响。已同步到 `C:\Bonsai-App`。

## 十五、kernel mask 快路径（实验性，未部署）

**状态：已实现、已验证正确、已证伪收益，保留在分支里，不进生产。**

动机：ring decode 的天花板是 attention kernel 内逐元素 mask 检查（decode 85→50）。
`small_t_nvfp4.cuh`（decode 路由）对每个 key 做 `block_selected(key)` 位图检查，
把未选中页的 score 置 -inf——**算完再扔**。

改法（仅 `small_t_nvfp4.cuh`，6 处；prefetch 链 / `cp_wait` 配对 / 所有
`__syncthreads` / grid 配置 / kernel 参数全部不动）：

| # | 位置 | 改法 |
|---|---|---|
| 1 | kb 循环头 | 加 CTA-uniform 的 `tile_active` 判断（tile 内无选中页则跳过） |
| 2 | K 反量化 | 整段包进 `if (tile_active)` |
| 3 | QK 条件 | `warp < ProducerWarps` → `tile_active && ...` |
| 4–5 | V 展开（worker / compact） | 包谓词 |
| 6 | PV | 整段包谓词 |

等价性：全灭 tile 原来产出 alpha 0/1 + 零 block sum，acc/m/l 不变；跳过与之一致；
`alpha_s` 写读同 tile；dense（bitmap=null）恒走 active 路径。

验证：

| 验证 | 结果 |
|---|---|
| PPL 回归（dense） | **5.639521，小数点后 6 位全对** |
| 藏针 129K（稀疏路径） | **PASS**（STARFRUIT-88） |
| 短 prompt decode | 64.9 t/s（基线 63.6–71.9，无退化） |

A/B（同负载 129K 藏针，新旧二进制对照）：

| | 旧（无 skip） | 新（tile skip） |
|---|---|---|
| prefill | 138.2 t/s | 136.6 t/s（prefill 路由未动，属运行方差） |
| **decode** | **21.4 t/s** | **21.5 t/s**（+0.5%，噪声级） |
| MTP 接受 | 56.2% | 42.6% |
| 召回 | PASS | PASS |

**结论：129K 处无可测量加速，不部署。** 原因经 `install_resident_selection`
（`context.cpp:1516`，"检索掩码 = 设备常驻集"）确认：129K 上 1500 页池装下 74%，
只能跳过 ~26% tile；而 decode 主导成本是权重加载 + MTP 多遍验证，attention 只是其中一块。
收益随深度递增（262K 处可跳 ~63%），但 200K+ 验证留待以后。

### v2：条件预取 + 同步点谓词（已部署生产）

v1 复盘发现两个漏项：① HBM 预取（`issue_kv_tile`）无条件执行，被跳 tile 的 K/V 照样上载；
② 循环内 3 个 `__syncthreads` 全在谓词之外——而按 129K 的 tile 成本结构，
同步+调度才是大头，v1 几乎没碰。

v2 改动（同文件，增量）：`tile_has_selected_keys(k0+Bc)` 求下 tile 活跃度；
prefetch 与 `cp_wait` 改走 `next_active`（配对不断）；三个循环内同步点移入谓词。
谓词 CTA-uniform，收敛性不变；dense 恒真，行为与原版一致。

验证（greedy 定温，temperature 0 + seed 42，两次输出逐 token 确定，零干扰）：

| | 旧（无 skip） | v2 |
|---|---|---|
| prompt / 输出 / 思考 | 129376 / 66 / 58 | **完全一致**（129376 / 66 / 58） |
| MTP 接受 | 43/66（65.2%） | 43/66（65.2%） |
| prefill | 138.0 t/s | 136.8 t/s（−0.9%，路由未动，属方差） |
| **decode** | **21.8 t/s** | **23.8 t/s（+9.2%）** |
| 召回 | PASS | PASS |

**结论：+9.2% 稳定可测，已部署生产**（franken 目录二进制替换，原版留 `.pre-maskv2` 备份；
部署版冒烟通过）。129K 处选择 74% 密，v2 的收益主要来自跳过的同步点；
262K 处（63% 可跳）预期更高，待测。

教训：先读选择集的产生逻辑（"掩码=常驻集"而非"稀疏检索子集"），再动手——
本优化的理论上限在动手前就能算出来；以及 A/B 必须定温，否则 MTP 接受率差异
（上次 56% vs 43%）会淹没真实信号。

