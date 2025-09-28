# AlignVision CLI Implementation Summary

## What Was Created

I've transformed your existing `main.py` AlignVision application into a comprehensive CLI tool that supports both traditional OpenCV window display and modern web streaming modes.

## New Files Created

### 1. `cli_main.py` - Main CLI Application
- **Full CLI interface** with argument parsing
- **Two modes**: OpenCV window and web streaming
- **Modular design** with separate functions for each mode
- **Threading support** for web mode camera capture
- **Web interface** with HTML template and controls
- **All original functionality** preserved from main.py

### 2. `CLI_README.md` - Documentation
- Complete usage instructions
- Feature overview
- Installation guide
- Troubleshooting section
- Examples and controls reference

### 3. `requirements.txt` - Dependencies
- Lists all Python package requirements
- Includes notes about Basler Pylon SDK

### 4. `run_alignvision.bat` - Windows Launcher
- Simple menu-driven interface
- No command-line knowledge required
- Automatically uses virtual environment

### 5. `run_alignvision.sh` - Unix/Linux Launcher
- Cross-platform shell script
- Same menu interface as Windows version

## Key Features

### OpenCV Window Mode
```bash
python cli_main.py --mode opencv
```
- **Identical to your original main.py** but with CLI structure
- Mouse-based point selection
- Keyboard controls (c, r, +, -, d, q)
- Real-time video processing
- Debug mode and threshold adjustment

### Web Streaming Mode
```bash
python cli_main.py --mode web --port 5000
```
- **Browser-based interface** accessible from any device
- **Click-to-select points** directly on the video stream
- **Real-time controls** via web buttons
- **Status updates** showing selected points and setup state
- **Mobile-friendly** responsive design
- **Multi-user access** (multiple browsers can view simultaneously)

## Usage Examples

### Simple Usage
```bash
# Default OpenCV mode
python cli_main.py

# Web streaming on default port (5000)
python cli_main.py --mode web

# Web streaming on custom port
python cli_main.py --mode web --port 8080
```

### Using the Launchers
```bash
# Windows
run_alignvision.bat

# Unix/Linux/Mac
./run_alignvision.sh
```

## Web Interface Features

When you run in web mode and open `http://localhost:5000`:

1. **Live video stream** from your Basler camera
2. **Point selection** by clicking directly on the video
3. **Interactive controls**:
   - Add Test Point (for testing)
   - Confirm Setup (when 4 points selected)
   - Reset Points
   - Reset Setup (return to point selection)
   - Increase/Decrease Threshold
   - Toggle Debug Mode
4. **Real-time status display**:
   - Points selected (X/4)
   - Setup completion status
   - Reference image loaded status
   - List of selected point coordinates
5. **Auto-refreshing status** every 2 seconds
6. **Responsive design** works on desktop and mobile

## Technical Implementation

### Architecture
- **Threaded design**: Camera capture runs in separate thread for web mode
- **Thread-safe**: Uses locks for frame sharing between threads
- **Error handling**: Proper cleanup of camera resources
- **Modular functions**: Easy to extend and maintain

### Web Technology Stack
- **Flask**: Lightweight Python web framework
- **MJPEG streaming**: Efficient video streaming to browsers
- **HTML5 + JavaScript**: Modern web interface
- **REST API endpoints**: For control interactions
- **Auto-refresh**: Real-time status updates

### Backwards Compatibility
- **Original functionality preserved**: All features from main.py work identically
- **Same dependencies**: Uses your existing Basler/OpenCV setup
- **Same reference system**: Still uses reference.jpg for object detection

## Benefits

### For Development
- **Easy testing**: Web mode allows remote testing from other devices
- **Better debugging**: Web interface shows status clearly
- **Professional appearance**: Clean web UI vs. console output

### For End Users
- **User-friendly**: Simple batch file launcher
- **Cross-platform**: Works on Windows, Linux, Mac
- **Remote access**: Web interface accessible from other devices on network
- **Mobile support**: Works on tablets/phones for monitoring

### For Deployment
- **Headless operation**: Web mode can run on servers without displays
- **Multi-user**: Multiple people can view the stream simultaneously
- **Integration ready**: Web API can be extended for other applications

## Next Steps

You can now run your AlignVision application in either mode:

1. **For development/local use**: Use OpenCV mode with familiar controls
2. **For remote monitoring**: Use web mode and access from any browser
3. **For easy sharing**: Share the localhost URL with others on your network

The CLI maintains 100% of your original functionality while adding modern web capabilities!