import os

class Config:
    # Camera settings
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 30

    # Model paths
    YOLO_WEIGHTS = os.path.join('models', 'yolov3.weights')
    YOLO_CONFIG = os.path.join('models', 'yolov3.cfg')
    COCO_NAMES = os.path.join('models', 'coco.names')

    # Detection parameters
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    
    # Distance estimation parameters
    # Average human height in meters
    HUMAN_HEIGHT = 1.7
    # Focal length (can be adjusted based on your camera)
    FOCAL_LENGTH = 615

    # Security settings
    MIN_DISTANCE_THRESHOLD = 2.0  # meters
    MOTION_DETECTION_THRESHOLD = 500  # pixels
    ALERT_COOLDOWN = 10  # seconds

    # Recording settings
    RECORD_DURATION = 30  # seconds
    MAX_RECORDINGS = 100  # maximum number of stored recordings

    # Logging
    LOG_FILE = 'logs/human_detection.log'
    MAX_LOG_SIZE = 10240  # 10KB
    BACKUP_COUNT = 10 