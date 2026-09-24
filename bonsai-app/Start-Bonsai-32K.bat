@echo off
REM Bonsai 2 27B - one-click launcher (custom #218 build, PTQ1_0)
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
start "Bonsai-2 Server" C:\Bonsai-App\bin-218\llama-server.exe -m models\Ternary-Bonsai-2-27B-PTQ1_0.gguf --mmproj models\Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf --chat-template-file C:\Bonsai-App\chat-template-bonsai2.jinja --alias bonsai-2-27b -ngl 99 -fa on -c 32768 -ctk q4_0 -ctv q4_0 --temp 1.0 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port 8080
echo Starting server, opening chat in browser...
timeout /t 15 /nobreak >nul
start http://localhost:8080
echo Done. Chat at http://localhost:8080  API at http://localhost:8080/v1
