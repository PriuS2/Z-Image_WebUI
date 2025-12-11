@echo off
chcp 65001 > nul
title Z-Image WebUI

echo ========================================
echo        Z-Image WebUI 시작 중...
echo ========================================
echo.

cd /d "%~dp0"

if exist "venv\Scripts\activate.bat" (
    echo [*] 가상환경 활성화 중...
    call venv\Scripts\activate.bat
) else (
    echo [!] 가상환경을 찾을 수 없습니다. 시스템 Python을 사용합니다.
)

echo [*] WebUI 실행 중...
echo.
python app.py

echo.
echo [*] WebUI가 종료되었습니다.
pause


