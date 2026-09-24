@echo off
REM Stop any local Bonsai engine (llama.cpp, KVMem, NInfer)
taskkill /F /IM llama-server.exe 2>nul
taskkill /F /IM llama-kvmem-server.exe 2>nul
taskkill /F /IM ninfer-serve.exe 2>nul
echo Server(s) stopped.
