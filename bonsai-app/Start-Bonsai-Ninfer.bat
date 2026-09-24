@echo off
REM ============================================================
REM  Ternary Bonsai 2 27B  -  NInfer engine  (DEFAULT / best balance)
REM  franken/v0.11 built locally for sm_86 | 160K context | nvfp4 KV
REM  MTP3 speculation + proposal head
REM
REM  Why this config (all measured on RTX 3060 12G, same model+prompts):
REM    prefill  662 / 516 t/s @20K / @60K      (llama.cpp 128K: 433 / 334)
REM    decode   61.2 / 51.2 t/s @20K / @60K    (llama.cpp 128K: 26.4 / 17.0)
REM    prefix cache 100%  (TTFT 30s -> 118 ms on a repeat)
REM    VRAM ~11.1 GB / 12 GB
REM
REM  Quality (verified, not assumed):
REM    nvfp4 KV vs rk8v4 KV : PPL 5.6395 vs 5.630  = +0.17%  (261,167 tokens)
REM    MTP3 vs no-spec      : 8/8 prompts bit-identical output
REM    -> nvfp4 buys +67% context for ~0.2% perplexity. Use it.
REM
REM  API : http://127.0.0.1:8080/v1   model id: bonsai2-27b
REM  (requests MUST include "model":"bonsai2-27b")
REM
REM  VISION: enabled. Images are decoded with the vendored stb_image (PNG/JPEG/BMP/
REM  GIF/WEBP) instead of FFmpeg, and arrive as base64 data URIs or local paths
REM  (remote URLs are not supported in this build - no libcurl). The tower runs in
REM  overlay residency so it costs ~0 resident VRAM. Verified end-to-end:
REM  a left-red/right-blue test image was described correctly in 2.3 s.
REM
REM  Usage: Start-Bonsai-Ninfer.bat [context_tokens]   default 163840
REM  Max context on 12 GB is ~181K (192K is refused at startup).
REM  For 256K long context (ring KV): use Start-Bonsai-Ninfer-256K.bat instead.
REM ============================================================
setlocal
set CTX=%~1
if "%CTX%"=="" set CTX=163840

set MODEL=C:\Bonsai-App\models\Ternary-Bonsai-2-27B-ninfer-v3.ninfer
if not exist "%MODEL%" set MODEL=H:\ninfer-models\Ternary-Bonsai-2-27B-ninfer-v3.ninfer
set SRV=C:\ninfer-build\franken\ninfer-3090-franken-v0.11\build-ninja\apps\ninfer-serve.exe

if not exist "%SRV%"   ( echo [ERR] server not built: %SRV% & pause & exit /b 1 )
if not exist "%MODEL%" ( echo [ERR] artifact missing: %MODEL% & pause & exit /b 1 )

echo Starting NInfer ternary Bonsai 2 27B ...
echo   artifact : %MODEL%
echo   context  : %CTX% tokens (nvfp4 KV, MTP3 + proposal head)
echo   api      : http://127.0.0.1:8080/v1  (model id: bonsai2-27b)
echo.

"%SRV%" "%MODEL%" --model-id bonsai2-27b --host 127.0.0.1 --port 8080 --max-context %CTX% --kv-capacity %CTX% --kv-dtype nvfp4 --gdn-state-fp16 --spec mtp --draft-tokens 3 --lm-head-draft --vision --vision-residency overlay --vision-max-merged 12288

echo.
echo [ninfer-serve exited]
pause
