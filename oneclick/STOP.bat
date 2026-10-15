@echo off
chcp 65001 >nul
title Bonsai 2 27B - stop
echo Stopping the local model server ...
taskkill /F /IM ninfer-kvmem-server.exe 2>nul
echo Done. (If it said "not found", it was already stopped.)
echo.
pause
