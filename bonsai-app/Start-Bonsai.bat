@echo off
REM ============================================================
REM  Bonsai 2 27B  -  DEFAULT launcher  (128K context)
REM  llama.cpp 3-PR build | q4_0 KV | reasoning capped
REM  Measured on RTX 3060 12G: prefill 441/336 t/s, decode 26.6/17.1 t/s @20K/60K
REM  VRAM ~9.9 GB / 12 GB
REM  Fallback: Start-Bonsai-32K.bat   Long context: C:\Bonsai-KVMem\Start-KVMem-256K.bat
REM ============================================================
cd /d C:\Bonsai-App
if not exist C:\Bonsai-App\bin-218\llama-server.exe (
  echo [ERR] custom build not found at C:\Bonsai-App\bin-218\llama-server.exe
  pause
  exit /b 1
)
if not exist models\Ternary-Bonsai-2-27B-PTQ1_0.gguf (
  echo [ERR] model file not found - model still downloading?
  pause
  exit /b 1
)
echo Starting Bonsai 2 27B @ 128K ... chat http://localhost:8080
start "Bonsai-2 Server" C:\Bonsai-App\bin-218\llama-server.exe -m models\Ternary-Bonsai-2-27B-PTQ1_0.gguf --chat-template-file C:\Bonsai-App\chat-template-bonsai2.jinja --alias bonsai-2-27b -ngl 99 -fa on -c 131072 -ctk q4_0 -ctv q4_0 --reasoning-effort medium --reasoning-budget 2048 --host 127.0.0.1 --port 8080
