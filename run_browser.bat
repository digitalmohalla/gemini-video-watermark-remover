@echo off
cd /d "%~dp0"

:: Auto-run setup if binaries are missing
if not exist "bin\GeminiWatermarkTool-Video.exe" (
    echo First-time run detected. Downloading engine binaries...
    py setup.py
)

:: Launch in default browser without keeping the CMD window open
where pyw >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    start "" pyw app.py --browser
) else (
    start "" py app.py --browser
)

exit
