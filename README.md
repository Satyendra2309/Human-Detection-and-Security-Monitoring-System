# Human Detection and Security Monitoring System

A sophisticated real-time human detection and security monitoring application built with Python, Flask, and OpenCV. The system provides advanced security features including person tracking, distance estimation, motion detection, and security zone monitoring.

## Features

### Core Functionality
- Real-time human detection using YOLOv3
- Accurate distance estimation from camera
- Person tracking with unique IDs
- Motion detection and alerts
- Live video feed with detection overlays

### Security Features
1. **Security Zones**
   - Three predefined security zones:
     - Restricted Area 1 (Primary security zone)
     - Restricted Area 2 (Secondary security zone)
     - Entry Point (Access monitoring)
   - Real-time violation detection
   - Customizable minimum distance thresholds
   - Visual indicators (Green: Safe, Red: Violation)

2. **Advanced Tracking**
   - Unique ID assignment for each detected person
   - Persistent tracking across frames
   - Track aging and management
   - Multiple person tracking support

3. **Alert System**
   - Real-time security alerts
   - Proximity warnings
   - Unauthorized entry detection
   - Motion detection alerts
   - Alert history logging

4. **Recording System**
   - Automatic incident recording
   - 30-second alert-triggered recordings
   - Organized file storage system
   - Timestamp-based file naming

## Requirements

- Python 3.7+
- OpenCV 4.x
- Flask
- NumPy
- Additional dependencies in `requirements.txt`

### Required Model Files
Place the following files in the `models` directory:
- `yolov3.weights` (237MB)
- `yolov3.cfg` (8.9KB)
- `coco.names` (704B)

You can download these files from:
- YOLOv3 weights: [https://pjreddie.com/media/files/yolov3.weights](https://pjreddie.com/media/files/yolov3.weights)
- YOLOv3 config: [https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg](https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg)
- COCO names: [https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names](https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Satyendra2309/Human-Detection-and-Security-Monitoring-System.git
cd Human-Detection-and-Security-Monitoring-System
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required model files and place them in the `models` directory:
```bash
mkdir models
cd models
# Download the model files from the links provided above
cd ..
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Access the web interface:
- Open a web browser and navigate to `http://localhost:5000`
- For remote access, use `http://<your-ip-address>:5000`

3. Monitor the interface:
- Live video feed with detection boxes
- Person tracking IDs and distances
- Security zone status
- Motion detection indicators
- Alert history

## Configuration

The application can be configured through `config.py`:
- Frame dimensions
- FPS settings
- Detection confidence threshold
- NMS threshold
- Security zone parameters
- Recording settings

## Directory Structure

```
human_detection_app/
├── app.py              # Main application file
├── config.py           # Configuration settings
├── requirements.txt    # Dependencies
├── models/            # YOLOv3 model files
├── logs/              # Application logs
├── recordings/        # Security incident recordings
├── alerts/           # Alert history
└── templates/        # Web interface templates
```

## Security Zones

The application includes three predefined security zones:
1. **Restricted Area 1**: `[[100, 100], [300, 100], [300, 300], [100, 300]]`
2. **Restricted Area 2**: `[[400, 100], [600, 100], [600, 300], [400, 300]]`
3. **Entry Point**: `[[250, 400], [450, 400], [450, 500], [250, 500]]`

Each zone can be customized by modifying the coordinates and minimum distance thresholds in `app.py`.

## Troubleshooting

1. **Camera Access Issues**
   - Ensure your webcam is properly connected
   - Check camera permissions
   - Verify no other application is using the camera

2. **Model Loading Errors**
   - Confirm all required model files are in the `models` directory
   - Verify file permissions
   - Check file integrity

3. **Performance Issues**
   - Adjust frame dimensions in config.py
   - Modify detection confidence threshold
   - Update security zone parameters

## Contributing

Contributions are welcome! Please feel free to submit pull requests.


## Acknowledgments

- YOLOv3 for object detection
- OpenCV community
- Flask framework

## Author

- Satyendra ([GitHub](https://github.com/Satyendra2309))

## Support

If you encounter any issues or have questions, please:
1. Check the troubleshooting section
2. Open an issue on GitHub
3. Contact the author 
