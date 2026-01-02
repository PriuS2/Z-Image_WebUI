@echo off
chcp 65001 > nul
title Z-Image WebUI

echo ========================================
echo        Z-Image WebUI 시작 중...
echo ========================================
echo.

cd /d "%~dp0"

REM Git 업데이트 확인
if exist ".git" (
    echo [*] Git 저장소 업데이트 확인 중...
    git fetch origin
    git pull origin
    
    REM 서브모듈 업데이트
    if exist ".gitmodules" (
        echo [*] 서브모듈 업데이트 중...
        git submodule update --init --recursive
    )
    echo.
) else (
    echo [!] Git 저장소가 아닙니다. 업데이트를 건너뜁니다.
    echo.
)

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


