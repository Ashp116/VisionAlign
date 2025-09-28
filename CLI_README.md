# AlignVision CLI

A command-line interface for the AlignVision computer vision system that supports both OpenCV window display and web streaming modes.

## Features

- **OpenCV Window Mode**: Traditional desktop application with mouse interaction
- **Web Streaming Mode**: Browser-based interface accessible via localhost
- **Real-time object detection and tracking**
- **Interactive region selection**
- **Adjustable detection thresholds**
- **Debug mode with detailed information**

## Installation

1. Make sure you have Python 3.8+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### OpenCV Window Mode (Default)

```bash
python cli_main.py
# or explicitly
python cli_main.py --mode opencv
```

**Controls:**
- Click 4 points on the video feed to define the search region
- Press 'c' to confirm setup and start tracking
- Press 'r' to reset points
- Press '+' or '-' to adjust detection threshold
- Press 'd' to toggle debug mode
- Press 'q' to quit

### Web Streaming Mode

```bash
# Default port (5000)
python cli_main.py --mode web

# Custom port
python cli_main.py --mode web --port 8080
```

Then open your browser and navigate to:
- `http://localhost:5000` (default)
- `http://localhost:8080` (custom port)

**Web Interface Features:**
- Click on the video feed to select region points
- Interactive buttons for all controls
- Real-time status updates
- Responsive design that works on desktop and mobile

## Requirements

- Python 3.8+
- OpenCV (cv2)
- NumPy
- Basler pylons camera SDK
- Flask (for web mode)

## File Structure

- `cli_main.py` - Main CLI application
- `main.py` - Original OpenCV-only version
- `reference.jpg` - Reference image for object detection (required)

## Examples

```bash
# Run with OpenCV window
python cli_main.py

# Run web server on default port
python cli_main.py --mode web

# Run web server on port 8080
python cli_main.py --mode web --port 8080

# Show help
python cli_main.py --help
```

## Troubleshooting

1. **No camera found**: Make sure your Basler camera is connected and the Pylon SDK is installed
2. **Reference image not found**: Ensure `reference.jpg` exists in the current directory
3. **Web mode not accessible**: Check that the specified port is not in use by another application
4. **Performance issues**: Try reducing the frame rate or resolution in the code

## Development

The CLI application is built with:
- **Argument parsing**: Uses `argparse` for clean command-line interface
- **Threading**: Separate threads for camera capture and web serving
- **Flask**: Lightweight web framework for streaming mode
- **OpenCV**: Computer vision processing and display

## Differences from Original

The CLI version includes:
- ✅ Command-line argument parsing
- ✅ Web streaming capability
- ✅ Browser-based controls
- ✅ Mobile-friendly web interface
- ✅ Better error handling and cleanup
- ✅ Modular code structure
- ✅ Real-time status updates in web mode