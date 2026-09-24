@echo off
REM ============================================================
REM  Ternary Bonsai 2 27B  -  NInfer SHORT-CONTEXT MAX-SPEED mode
REM  DFlash2 speculation (7 drafts) | 64K context | rk8v4 KV
REM
REM  Measured vs the default MTP3 profile (RTX 3060 12G):
REM    short Q&A decode : 139.1 / 119.6 t/s   (MTP3: 94.2 / 93.4)  +48%
REM    @20K decode      : 74.9 / 75.2 t/s     (MTP3: 61.4 / 60.3)  +22%
REM    @60K decode      : 53.9 / 54.0 t/s     (MTP3: 53.5 / 54.6)  ~0%
REM    VRAM @64K        : 11033 MiB           (MTP3: 8157 MiB)    +2.9 GB
REM
REM  Quality: identical answers; only cosmetic wording differences.
REM  DFlash2 acceptance falls with depth (57.6% short -> 22.3% @60K),
REM  so the gain vanishes on long contexts -- use this only for
REM  short-context work where the 2.9 GB VRAM cost is acceptable.
REM
REM  API : http://127.0.0.1:8080/v1   model id: bonsai2-27b
REM ============================================================
setlocal
set CTX=%~1
if "%CTX%"=="" set CTX=65536

set MODEL=C:\Bonsai-App\models\Ternary-Bonsai-2-27B-ninfer-v3.ninfer
if not exist "%MODEL%" set MODEL=H:\ninfer-models\Ternary-Bonsai-2-27B-ninfer-v3.ninfer
set SRV=C:\ninfer-build\franken\ninfer-3090-franken-v0.11\build-ninja\apps\ninfer-serve.exe

if not exist "%SRV%"   ( echo [ERR] server not built: %SRV% & pause & exit /b 1 )
if not exist "%MODEL%" ( echo [ERR] artifact missing: %MODEL% & pause & exit /b 1 )

echo Starting NInfer ternary Bonsai 2 27B (DFlash2 max-speed) ...
echo   context : %CTX% tokens (rk8v4 KV, DFlash2 7 drafts)
echo   api     : http://127.0.0.1:8080/v1  (model id: bonsai2-27b)
echo.

"%SRV%" "%MODEL%" --model-id bonsai2-27b --host 127.0.0.1 --port 8080 --max-context %CTX% --kv-capacity %CTX% --kv-dtype rk8v4 --gdn-state-fp16 --spec dflash2 --draft-tokens 7

echo.
echo [ninfer-serve exited]
pause
