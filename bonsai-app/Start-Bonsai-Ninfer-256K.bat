@echo off
REM ============================================================
REM  Ternary Bonsai 2 27B  -  NInfer engine  (RING / 256K long context)
REM  franken/v0.11 + ring port | pool 96K | context 256K | nvfp4 KV
REM  MTP3 speculation + proposal head + IDF retrieval + pageable host KV
REM
REM  Ring layout (measured on RTX 3060 12G):
REM    device KV pool  96K tokens  (attention window, resident)
REM    logical context 256K tokens (rest lives on host RAM, lossless)
REM    VRAM ~10.2 GB / 12 GB  |  host KV 8 GiB pageable (not charged to VRAM)
REM
REM  Trade-offs vs Start-Bonsai-Ninfer.bat (dense 160K):
REM    + 256K context, multi-turn OK (each turn re-prefills, no prefix cache)
REM    - prefill 120-240 t/s (dense 365), decode ~23-50 t/s (dense 85)
REM
REM  API : http://127.0.0.1:8080/v1   model id: bonsai2-27b
REM  (requests MUST include "model":"bonsai2-27b")
REM
REM  VISION: same flags as dense (overlay). Vision-in-ring is not yet
REM  explicitly validated - text long-context is the verified path.
REM
REM  Usage: Start-Bonsai-Ninfer-256K.bat [context_tokens] [pool_tokens]
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

REM --- thinking budget fuse: caps runaway xhigh thinking at 24576 tokens.
REM Normal questions use far less (sanity: 23-146 tokens); only pathological
REM cases (measured: 31,360 tokens / 9m44s on one HumanEval item) hit the cap.
REM v1 value 8192 was TOO LOW: a legitimate creative-coding task
REM ("3D pelican on a bicycle") burned through 8192 thinking and got cut,
REM showing "Reasoning Cancelled" with a rushed answer. Raised to 24576.
REM Raises output-limit stop instead of thinking forever. Tune per machine.

echo Starting NInfer ternary Bonsai 2 27B (RING) ...
echo   artifact : %MODEL%
echo   context  : %CTX% tokens logical / %POOL% tokens device pool (nvfp4 KV)
echo   api      : http://127.0.0.1:8080/v1  (model id: bonsai2-27b)
echo.
REM --request-log-jsonl: post-mortem forensics for failed requests (takes effect next start).
if not exist C:\Bonsai-App\logs mkdir C:\Bonsai-App\logs

"%SRV%" "%MODEL%" --model-id bonsai2-27b --host 127.0.0.1 --port 8080 --max-context %CTX% --kv-capacity %POOL% --kv-dtype nvfp4 --gdn-state-fp16 --spec mtp --draft-tokens 3 --lm-head-draft --prefill-cublas --default-thinking-budget 24576 --request-log-jsonl C:\Bonsai-App\logs\requests-8080.jsonl --vision --vision-residency overlay --vision-max-merged 12288

echo.
echo [ninfer-serve exited]
pause
