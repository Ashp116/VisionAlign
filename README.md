# AlignVision

A computer vision system for precise object alignment using Basler cameras. AlignVision provides both traditional OpenCV window display and modern web streaming interfaces for real-time object detection and alignment guidance.

## Features

- **Dual Interface Modes**: OpenCV window and web browser interfaces
- **Real-time Object Detection**: Multi-scale template matching with reference images
- **Interactive Region Selection**: Click-to-define search regions
- **Auto Exposure Control**: Automatic camera exposure adjustment
- **Image Quality Metrics**: Real-time brightness, contrast, saturation, and clipping analysis
- **Alignment Guidance**: Visual feedback for object positioning
- **Ford Motor Company Styling**: Professional web interface design

## Quick Start

### OpenCV Window Mode (Default)
```bash
python cli_main.py
```

### Web Streaming Mode
```bash
python cli_main.py --mode web
```
Then open `http://localhost:5000` in your browser.

## Installation

### Prerequisites
- Python 3.8+
- Basler Pylon Camera Software Suite
- Basler camera connected via USB

### Install Dependencies
```bash
pip install opencv-python numpy flask pypylon
```

### Setup Reference Image
Place your reference object image as `reference.jpg` in the project directory.

## Usage

### 1. OpenCV Window Mode
- Click 4 points on video feed to define search region
- Press 'c' to confirm setup and start tracking
- Press 'r' to reset points
- Press '+'/'-' to adjust detection threshold
- Press 'q' to quit

### 2. Web Streaming Mode
- Access via browser at `http://localhost:5000`
- Click on video feed to select region points
- Use web controls for all operations
- Monitor image quality metrics in real-time
- Enable auto exposure for optimal camera settings

## Project Structure

```
AlignVision/
├── cli_main.py              # Main CLI application
├── ae_hud.py                # Auto exposure & image metrics module
├── main.py                  # Original OpenCV-only version
├── reference.jpg            # Reference image for detection
├── camtest.py              # Camera testing utilities
├── capture_images.py       # Image capture tools
└── captures/               # Captured images directory
```

## Key Components

### cli_main.py
- Main application with dual-mode support
- Flask web server for streaming mode
- Object detection and alignment logic
- Ford-styled web interface
- Perfect alignment detection system

### ae_hud.py
- `AutoExposureController`: Automatic camera exposure management
- `ImageMetrics`: Image quality analysis (brightness, contrast, saturation, clipping)
- `HUDOverlay`: Visual overlay system for metrics display
- Utility functions for exposure analysis

## Web Interface Features

- **Live Video Stream**: Real-time camera feed
- **Interactive Controls**: Point selection, threshold adjustment, auto exposure
- **Quality Metrics Panel**: Real-time image analysis with master comparison
- **Status Indicators**: Detection quality, alignment status, system health
- **Perfect Alignment Detection**: Comprehensive success feedback
- **Mobile-Friendly**: Responsive design for all devices

## Command Line Options

```bash
python cli_main.py [options]

Options:
  --mode {opencv,web}    Interface mode (default: opencv)
  --port PORT           Web server port (default: 5000)
  --help               Show help message
```

## Image Quality Metrics

The system monitors four key image quality parameters:

1. **Clipping**: Percentage of over/under-exposed pixels
   - Green: <10%, Yellow: 10-20%, Red: >20%

2. **Brightness**: Mean pixel intensity compared to master
   - Good: <10 difference, Warning: 10-25, Poor: >25

3. **Contrast**: Standard deviation of pixel intensities
   - Good: >-10% change, Warning: -10% to -20%, Poor: <-20%

4. **Saturation**: Color saturation compared to master
   - Good: <15% change, Warning: 15-30%, Poor: >30%

## Perfect Alignment Detection

When both object detection confidence is high AND all image quality metrics are within acceptable ranges, the system displays "Perfect!" in the bottom right corner, indicating optimal alignment conditions.

## Development

### Architecture
- **Threaded Design**: Separate camera capture thread for web mode
- **Thread-Safe**: Proper locking for frame sharing
- **Modular Structure**: Separated concerns for maintainability
- **Error Handling**: Robust camera resource management

### Technology Stack
- **Computer Vision**: OpenCV for image processing
- **Camera Interface**: Basler Pylon SDK via pypylon
- **Web Framework**: Flask for streaming interface
- **Frontend**: HTML5 + JavaScript with real-time updates

## Troubleshooting

### Common Issues
1. **No camera found**: Ensure Basler camera is connected and Pylon SDK installed
2. **Reference image error**: Verify `reference.jpg` exists in project directory
3. **Web interface not accessible**: Check port availability and firewall settings
4. **Poor detection quality**: Adjust lighting conditions and camera position

### Performance Optimization
- Ensure adequate lighting for object detection
- Use appropriate reference image with good contrast
- Adjust detection threshold based on environmental conditions
- Enable auto exposure for consistent image quality

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with both OpenCV and web modes
5. Submit a pull request

## License

This project is part of a computer vision alignment system. Please ensure proper attribution when using or modifying the code.

## Changelog

### Latest Changes
- ✅ Added ae_hud.py module for auto exposure and image metrics
- ✅ Implemented perfect alignment detection system  
- ✅ Enhanced web interface with Ford Motor Company styling
- ✅ Added comprehensive image quality monitoring
- ✅ Fixed text overlap issues in status messages
- ✅ Improved clipping-focused quality assessment

## Support

For issues, questions, or contributions, please refer to the project's issue tracker or contact the development team.