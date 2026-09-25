@echo off
REM ============================================================
REM  Bonsai 2 27B  +  llama.cpp WebUI  (RING / 256K long context)
REM
REM  Starts two things:
REM    NInfer engine  on 127.0.0.1:8081   (ring KV; 256K ctx; vision)
REM    WebUI shim     on 127.0.0.1:8080   (llama.cpp's chat UI + /v1 proxy)
REM
REM  Ring layout (measured on RTX 3060 12G):
REM    device KV pool  96K tokens  (attention window, resident)
REM    logical context 256K tokens (rest lives on host RAM, lossless)
REM    retrieval pulls relevant host pages back per chunk (IDF lexical)
REM    VRAM ~10.2 GB / 12 GB  |  host KV 8 GiB pageable (not charged to VRAM)
REM
REM  Trade-offs vs Start-Bonsai-WebUI.bat (dense 160K):
REM    + 256K context, multi-turn OK (each turn re-prefills, no prefix cache)
REM    - prefill 120-240 t/s (dense 365), decode ~23-50 t/s (dense 85)
REM    - TTFT on a 150K prompt is ~20 min; short prompts are unaffected
REM
REM  ZCode can keep pointing at http://localhost:8080/v1  (the shim proxies it).
REM  model id: bonsai2-27b
REM
REM  Usage: Start-Bonsai-WebUI-256K.bat [context_tokens] [pool_tokens]
REM         defaults: 262144 96000
REM ============================================================
setlocal
set CTX=%~1
if "%CTX%"=="" set CTX=262144
set POOL=%~2
if "%POOL%"=="" set POOL=96000

set MODEL=C:\Bonsai-App\models\Ternary-Bonsai-2-27B-ninfer-v3.ninfer
if not exist "%MODEL%" set MODEL=H:\ninfer-models\Ternary-Bonsai-2-27B-ninfer-v3.ninfer
set SRV=C:\ninfer-build\franken\ninfer-3090-franken-v0.11\build-ninja\apps\ninfer-serve.exe

if not exist "%SRV%"   ( echo [ERR] server not built: %SRV% & pause & exit /b 1 )
if not exist "%MODEL%" ( echo [ERR] artifact missing: %MODEL% & pause & exit /b 1 )

REM --- KVMem-style ring (all gated; unset = stock dense behaviour) ---
set NINFER_KV_RING=1
set NINFER_KV_WINDOW=96000
set NINFER_KV_RETRIEVE=12288
set NINFER_HOST_PAGEABLE=1

echo Starting NInfer engine on 8081 (RING ctx %CTX% / pool %POOL%) ...
REM --default-thinking-budget 8192: fuse against runaway xhigh thinking
REM (measured 31,360 tokens / 9m44s on one item); normal questions use <200.
start "Bonsai-2 Engine RING (8081)" "%SRV%" "%MODEL%" --model-id bonsai2-27b --host 127.0.0.1 --port 8081 --max-context %CTX% --kv-capacity %POOL% --kv-dtype nvfp4 --gdn-state-fp16 --spec mtp --draft-tokens 3 --lm-head-draft --prefill-cublas --default-thinking-budget 8192 --vision --vision-residency overlay --vision-max-merged 12288

echo Waiting for the engine to become ready ...
:wait
timeout /t 3 /nobreak >nul
curl.exe -s --max-time 3 -o nul http://127.0.0.1:8081/v1/models || goto wait

echo Engine ready. Starting the WebUI shim on 8080 ...
start "" http://localhost:8080/
py C:\Bonsai-App\webui.py --port 8080 --upstream 127.0.0.1:8081

echo.
echo WebUI shim exited - stopping the engine.
taskkill /F /IM ninfer-serve.exe 2>nul
pause
