@echo off
REM ============================================================
REM  Bonsai 2 27B  +  llama.cpp WebUI  (one click)
REM
REM  Starts two things:
REM    NInfer engine  on 127.0.0.1:8081   (fastest; 160K ctx; vision)
REM    WebUI shim     on 127.0.0.1:8080   (llama.cpp's chat UI + /v1 proxy)
REM
REM  Then it opens http://localhost:8080/ in your browser.
REM
REM  Why the shim: NInfer has no web UI. llama.cpp's UI ships with the KVMem
REM  package on disk; webui.py serves it and translates the few llama.cpp
REM  endpoints it needs (/props) while proxying /v1 to NInfer (streaming included).
REM
REM  ZCode can keep pointing at http://localhost:8080/v1  (the shim proxies it).
REM  model id: bonsai2-27b
REM
REM  Usage: Start-Bonsai-WebUI.bat [context_tokens]   default 163840
REM ============================================================
setlocal
set CTX=%~1
if "%CTX%"=="" set CTX=163840

set MODEL=C:\Bonsai-App\models\Ternary-Bonsai-2-27B-ninfer-v3.ninfer
if not exist "%MODEL%" set MODEL=H:\ninfer-models\Ternary-Bonsai-2-27B-ninfer-v3.ninfer
set SRV=C:\ninfer-build\franken\ninfer-3090-franken-v0.11\build-ninja\apps\ninfer-serve.exe

if not exist "%SRV%"   ( echo [ERR] server not built: %SRV% & pause & exit /b 1 )
if not exist "%MODEL%" ( echo [ERR] artifact missing: %MODEL% & pause & exit /b 1 )

echo Starting NInfer engine on 8081 (context %CTX%) ...
start "Bonsai-2 Engine (8081)" "%SRV%" "%MODEL%" --model-id bonsai2-27b --host 127.0.0.1 --port 8081 --max-context %CTX% --kv-capacity %CTX% --kv-dtype nvfp4 --gdn-state-fp16 --spec mtp --draft-tokens 3 --lm-head-draft --vision --vision-residency overlay --vision-max-merged 12288

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
