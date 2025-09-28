#!/bin/bash

echo "AlignVision CLI Application"
echo "=========================="
echo
echo "Available modes:"
echo "1. OpenCV Window Mode (default)"
echo "2. Web Streaming Mode"
echo

read -p "Select mode (1 or 2, or press Enter for OpenCV mode): " choice

if [ "$choice" = "2" ]; then
    echo "Starting web streaming mode..."
    echo "Open your browser to http://localhost:5000"
    echo "Press Ctrl+C to stop"
    python cli_main.py --mode web
else
    echo "Starting OpenCV window mode..."
    echo "Click 4 points to define region, press 'c' to confirm, 'q' to quit"
    python cli_main.py --mode opencv
fi