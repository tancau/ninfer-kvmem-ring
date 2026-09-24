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

