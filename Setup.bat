@echo off
chcp 65001 > nul
title Z-Image WebUI 환경 설정

echo ========================================
echo    Z-Image WebUI 환경 설정
echo ========================================
echo.

cd /d "%~dp0"

:: 가상환경 확인 및 생성
if exist "venv\Scripts\activate.bat" (
    echo [✓] 가상환경이 이미 존재합니다.
) else (
    echo [*] 가상환경 생성 중...
    python -m venv venv
    if errorlevel 1 (
        echo [!] 가상환경 생성 실패. Python이 설치되어 있는지 확인하세요.
        pause
        exit /b 1
    )
    echo [✓] 가상환경 생성 완료
)

echo.

:: 가상환경 활성화
echo [*] 가상환경 활성화 중...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [!] 가상환경 활성화 실패
    pause
    exit /b 1
)
echo [✓] 가상환경 활성화 완료

echo.

:: pip 업그레이드
echo [*] pip 업그레이드 중...
python -m pip install --upgrade pip

echo.

:: requirements.txt 설치
echo [*] 필수 패키지 설치 중...
echo     (시간이 다소 걸릴 수 있습니다)
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [!] 일부 패키지 설치 실패. 로그를 확인하세요.
) else (
    echo.
    echo [✓] 기본 패키지 설치 완료
)

echo.

:: LongCat-Image 패키지 설치 (이미지 편집 기능)
echo [*] LongCat-Image 패키지 설치 중...
echo     (이미지 편집 기능에 필요합니다)
echo.
if exist "LongCat-Image\setup.py" (
    pip install -e ./LongCat-Image
    if errorlevel 1 (
        echo.
        echo [!] LongCat-Image 패키지 설치 실패.
    ) else (
        echo.
        echo [✓] LongCat-Image 패키지 설치 완료
    )
) else (
    echo [!] LongCat-Image 폴더를 찾을 수 없습니다.
    echo     git submodule update --init 을 실행하세요.
)

echo.
echo ========================================
echo    설정 완료!
echo    Run.bat으로 WebUI를 실행하세요.
echo ========================================
echo.
pause

