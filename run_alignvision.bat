@echo off
echo AlignVision - Computer Vision Alignment System
echo =============================================
echo.
echo Available modes:
echo 1. OpenCV Window Mode (traditional interface)
echo 2. Web Streaming Mode (browser interface)
echo.

set /p choice="Select mode (1 or 2, or press Enter for OpenCV): "

if "%choice%"=="2" (
    echo.
    echo Starting web streaming mode...
    echo Open your browser to: http://localhost:5000
    echo Press Ctrl+C to stop the server
    echo.
    python cli_main.py --mode web
) else (
    echo.
    echo Starting OpenCV window mode...
    echo - Click 4 points to define search region
    echo - Press 'c' to confirm setup
    echo - Press 'q' to quit
    echo.
    python cli_main.py --mode opencv
)

echo.
pause