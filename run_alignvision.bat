@echo off
echo AlignVision CLI Application
echo ===========================
echo.
echo Available modes:
echo 1. OpenCV Window Mode (default)
echo 2. Web Streaming Mode
echo.

set /p choice="Select mode (1 or 2, or press Enter for OpenCV mode): "

if "%choice%"=="2" (
    echo Starting web streaming mode...
    echo Open your browser to http://localhost:5000
    echo Press Ctrl+C to stop
    .venv\Scripts\python.exe cli_main.py --mode web
) else (
    echo Starting OpenCV window mode...
    echo Click 4 points to define region, press 'c' to confirm, 'q' to quit
    .venv\Scripts\python.exe cli_main.py --mode opencv
)

pause