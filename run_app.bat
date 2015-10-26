@echo off
REM ------------------------------------------------------------
REM Clinical Eligibility Tool launcher
REM
REM This script assumes a local WinPython distribution is present
REM and launches the Flask UI in the default browser.
REM ------------------------------------------------------------

set BASE_DIR=%~dp0
set WINPYTHON_DIR=%BASE_DIR%WinPython

if not exist "%WINPYTHON_DIR%\python.exe" (
    echo WinPython not found.
    echo Please ensure WinPython is available at:
    echo %WINPYTHON_DIR%
    pause
    exit /b 1
)

cd /d "%BASE_DIR%"

"%WINPYTHON_DIR%\python.exe" app\ui\app.py

pause
