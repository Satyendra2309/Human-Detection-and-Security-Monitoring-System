from flask import Flask, Response, render_template, jsonify, send_file
import cv2
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import os
import threading
from datetime import datetime
from config import Config
import time
import queue

app = Flask(__name__)
app.config.from_object(Config)

# Setup logging
if not os.path.exists('logs'):
    os.mkdir('logs')
if not os.path.exists('recordings'):
    os.mkdir('recordings')
if not os.path.exists('alerts'):
    os.mkdir('alerts')

file_handler = RotatingFileHandler('logs/human_detection.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('Human Detection startup')

# Global statistics and state
detection_stats = {
    'humans_detected': 0,
    'avg_distance': 0,
    'total_distances': 0,
    'detection_count': 0,
    'alerts': [],
    'unauthorized_entries': 0,
    'last_motion_detected': None
}
stats_lock = threading.Lock()
alert_queue = queue.Queue()

class SecurityZone:
    def __init__(self, name, points, min_distance=2.0):
        self.name = name
        self.points = np.array(points, np.int32)
        self.min_distance = min_distance
        self.last_violation = None
        self.violation_cooldown = 10  # seconds

    def check_violation(self, point, distance):
        if cv2.pointPolygonTest(self.points, point, False) >= 0:
            if distance < self.min_distance:
                current_time = time.time()
                if (self.last_violation is None or 
                    current_time - self.last_violation > self.violation_cooldown):
                    self.last_violation = current_time
                    return True
        return False

class HumanDetector:
    def __init__(self):
        try:
            self.net = cv2.dnn.readNet(app.config['YOLO_WEIGHTS'], app.config['YOLO_CONFIG'])
            with open(app.config['COCO_NAMES'], "r") as f:
                self.classes = f.read().strip().split("\n")
            
            self.layer_names = self.net.getLayerNames()
            self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
            
            # Initialize motion detection
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=16, detectShadows=True)
            
            # Initialize security zones
            self.security_zones = [
                SecurityZone("Restricted Area 1", [[100, 100], [300, 100], [300, 300], [100, 300]], 2.0),
                SecurityZone("Restricted Area 2", [[400, 100], [600, 100], [600, 300], [400, 300]], 2.0),
                SecurityZone("Entry Point", [[250, 400], [450, 400], [450, 500], [250, 500]], 1.5)
            ]
            
            # Initialize tracking
            self.tracks = []  # List to store active tracks
            self.track_id = 0  # Counter for unique track IDs
            
            app.logger.info('Model and security features loaded successfully')
        except Exception as e:
            app.logger.error(f'Error loading model: {str(e)}')
            raise

    def estimate_distance(self, object_height_in_pixels):
        return (app.config['HUMAN_HEIGHT'] * app.config['FOCAL_LENGTH']) / object_height_in_pixels

    def detect_motion(self, frame):
        fg_mask = self.background_subtractor.apply(frame)
        kernel = np.ones((5,5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Minimum area threshold
                motion_detected = True
                break
                
        return motion_detected, fg_mask

    def update_tracks(self, detections):
        # Simple tracking based on IoU (Intersection over Union)
        if not self.tracks:
            # Initialize tracks if none exist
            for bbox in detections:
                self.tracks.append({
                    'id': self.track_id,
                    'bbox': bbox,
                    'age': 0
                })
                self.track_id += 1
        else:
            # Match detections with existing tracks
            matched_tracks = set()
            matched_detections = set()
            
            for i, track in enumerate(self.tracks):
                best_iou = 0.3  # IoU threshold
                best_match = -1
                
                for j, detection in enumerate(detections):
                    if j in matched_detections:
                        continue
                        
                    iou = self.calculate_iou(track['bbox'], detection)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = j
                
                if best_match != -1:
                    self.tracks[i]['bbox'] = detections[best_match]
                    self.tracks[i]['age'] = 0
                    matched_tracks.add(i)
                    matched_detections.add(best_match)
                else:
                    self.tracks[i]['age'] += 1
            
            # Remove old tracks
            self.tracks = [track for i, track in enumerate(self.tracks) 
                         if i in matched_tracks or track['age'] < 5]
            
            # Add new tracks
            for i, detection in enumerate(detections):
                if i not in matched_detections:
                    self.tracks.append({
                        'id': self.track_id,
                        'bbox': detection,
                        'age': 0
                    })
                    self.track_id += 1

    def calculate_iou(self, bbox1, bbox2):
        # Calculate intersection over union between two bounding boxes
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
        y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = bbox1[2] * bbox1[3]
        area2 = bbox2[2] * bbox2[3]
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0

    def check_security_zones(self, frame, x, y, w, h, distance):
        center_point = (x + w//2, y + h//2)
        
        for zone in self.security_zones:
            if zone.check_violation(center_point, distance):
                cv2.polylines(frame, [zone.points], True, (0, 0, 255), 2)
                alert_msg = f"Security Alert: Person detected in {zone.name} at distance {distance:.2f}m"
                alert_queue.put({
                    'message': alert_msg,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'zone': zone.name,
                    'distance': distance
                })
                with stats_lock:
                    detection_stats['unauthorized_entries'] += 1
                    detection_stats['alerts'].append(alert_msg)
                    if len(detection_stats['alerts']) > 10:
                        detection_stats['alerts'].pop(0)
            else:
                cv2.polylines(frame, [zone.points], True, (0, 255, 0), 2)

    def detect_humans(self, frame):
        try:
            height, width = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)

            bboxes, confidences = [], []
            total_distance = 0
            motion_detected, motion_mask = self.detect_motion(frame)

            if motion_detected:
                with stats_lock:
                    detection_stats['last_motion_detected'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > app.config['CONFIDENCE_THRESHOLD'] and class_id == 0:
                        center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                        w, h = int(detection[2] * width), int(detection[3] * height)
                        x, y = int(center_x - w / 2), int(center_y - h / 2)

                        bboxes.append([x, y, w, h])
                        confidences.append(float(confidence))

            indices = cv2.dnn.NMSBoxes(bboxes, confidences, 
                                     app.config['CONFIDENCE_THRESHOLD'], 
                                     app.config['NMS_THRESHOLD'])
            indices = indices.flatten() if len(indices) > 0 else []

            # Update tracking
            active_detections = [bboxes[i] for i in indices]
            self.update_tracks(active_detections)

            # Update statistics
            with stats_lock:
                detection_stats['humans_detected'] = len(indices)
                if indices:
                    total_distance = sum(self.estimate_distance(bboxes[i][3]) for i in indices)
                    detection_stats['total_distances'] += total_distance
                    detection_stats['detection_count'] += len(indices)
                    detection_stats['avg_distance'] = detection_stats['total_distances'] / detection_stats['detection_count']

            # Draw security zones
            for zone in self.security_zones:
                cv2.polylines(frame, [zone.points], True, (0, 255, 0), 2)

            # Draw detections and tracks
            for track in self.tracks:
                x, y, w, h = track['bbox']
                distance = self.estimate_distance(h)
                
                # Check security zones
                self.check_security_zones(frame, x, y, w, h, distance)
                
                # Draw detection box and distance
                color = (0, 0, 255) if distance < 2.0 else (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"ID: {track['id']} Dist: {distance:.2f}m", 
                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, color, 2)
                
                # Add proximity warning
                if distance < 2.0:
                    cv2.putText(frame, "WARNING: Close Proximity!", 
                              (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.7, (0, 0, 255), 2)

            # Add motion indicator
            if motion_detected:
                cv2.putText(frame, "Motion Detected", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                          1, (0, 0, 255), 2)

            return frame

        except Exception as e:
            app.logger.error(f'Error in detection: {str(e)}')
            return frame

detector = HumanDetector()

def generate_frames():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, app.config['FRAME_WIDTH'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, app.config['FRAME_HEIGHT'])
    cap.set(cv2.CAP_PROP_FPS, app.config['FPS'])

    # Initialize video writer for recording
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None
    recording = False
    recording_start = None

    try:
        while True:
            success, frame = cap.read()
            if not success:
                app.logger.error('Failed to grab frame')
                break

            frame = detector.detect_humans(frame)

            # Start recording if there are alerts
            if not alert_queue.empty() and not recording:
                recording = True
                recording_start = datetime.now()
                filename = f"recordings/alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.avi"
                out = cv2.VideoWriter(filename, fourcc, 20.0, 
                                    (frame.shape[1], frame.shape[0]))

            # Record frame if recording is active
            if recording and out is not None:
                out.write(frame)

            # Stop recording after 30 seconds
            if recording and (datetime.now() - recording_start).seconds > 30:
                recording = False
                out.release()
                out = None

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    except Exception as e:
        app.logger.error(f'Error in frame generation: {str(e)}')
    finally:
        cap.release()
        if out is not None:
            out.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    with stats_lock:
        return jsonify({
            'humans_detected': detection_stats['humans_detected'],
            'avg_distance': f"{detection_stats['avg_distance']:.2f}" if detection_stats['detection_count'] > 0 else "0.00",
            'unauthorized_entries': detection_stats['unauthorized_entries'],
            'last_motion': detection_stats['last_motion_detected'],
            'alerts': detection_stats['alerts'][-5:]  # Return last 5 alerts
        })

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000) 