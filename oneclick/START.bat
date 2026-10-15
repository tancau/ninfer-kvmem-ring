@echo off
chcp 65001 >nul
title Bonsai 2 27B - one click
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "setup.ps1" %*
echo.
echo (window will stay open; close it any time)
pause
