from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading
import time
import cv2
import mediapipe as mp
import numpy as np
import joblib
import uuid
import eventlet
eventlet.monkey_patch()  # 必須這行以支援 WebSocket

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Mediapipe 初始化
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, max_num_faces=1,
    min_detection_confidence=0.7, min_tracking_confidence=0.7
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7,
                    min_tracking_confidence=0.7,
                    model_complexity=2)

# 已知參數
KNOWN_WIDTH = 12.0
FOCAL_LENGTH = 500

# 載入模型
clf = joblib.load("svm_model_V3_9.pkl")
scaler = joblib.load("scaler_V3_9.pkl")
with open("labels_V3_5.txt", 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# 儲存連線與處理緒
clients = {}

def compute_distances(landmarks):
    distances = []
    pairs = [
        (5, 2), (8, 7), (12, 11), (5, 8), (5, 7), (5, 0),
        (2, 7), (2, 8), (2, 0), (5, 11), (5, 12), (2, 11),
        (2, 12), (8, 11), (8, 12), (7, 11), (7, 12), (0, 8),
        (0, 7), (0, 11), (0, 12), (15, 9), (15, 10), (16, 9), (16, 10)
    ]
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    shoulder_dist = np.linalg.norm(
        np.array([left_shoulder.x, left_shoulder.y]) -
        np.array([right_shoulder.x, right_shoulder.y])
    )
    if shoulder_dist == 0:
        return None
    for pair in pairs:
        p1 = np.array([landmarks.landmark[pair[0]].x, landmarks.landmark[pair[0]].y])
        p2 = np.array([landmarks.landmark[pair[1]].x, landmarks.landmark[pair[1]].y])
        distance = np.linalg.norm(p1 - p2) / shoulder_dist
        distances.append(distance)
    return distances

def calculate_distance(left_eye, right_eye, image_width, image_height, focal):
    left_eye_x, left_eye_y = int(left_eye.x * image_width), int(left_eye.y * image_height)
    right_eye_x, right_eye_y = int(right_eye.x * image_width), int(right_eye.y * image_height)
    pixel_distance = np.sqrt((right_eye_x - left_eye_x)**2 + (right_eye_y - left_eye_y)**2)
    if pixel_distance == 0:
        return None
    return (KNOWN_WIDTH * focal) / pixel_distance

def detection_loop(sid):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    while sid in clients:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 姿勢辨識
        pose_results = pose.process(rgb_frame)
        label, confidence = "N/A", 0.0
        if pose_results.pose_landmarks:
            distances = compute_distances(pose_results.pose_landmarks)
            if distances:
                X = scaler.transform([distances])
                prediction = clf.predict(X)
                confidence = np.max(clf.predict_proba(X))
                label = labels[prediction[0]]

        # 距離估算
        face_results = face_mesh.process(rgb_frame)
        distance = None
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            distance = calculate_distance(left_eye, right_eye, frame.shape[1], frame.shape[0], FOCAL_LENGTH)

        socketio.emit('pose_data', {
            "label": label,
            "confidence": confidence,
            "distance": distance
        }, to=sid)

        socketio.sleep(1 / 10.0)  # 每秒最多傳送 10 次
    cap.release()

# 客戶端連線處理
@socketio.on('connect')
def on_connect():
    sid = request.sid
    clients[sid] = True
    thread = threading.Thread(target=detection_loop, args=(sid,))
    thread.start()
    print(f"[Connected] Client {sid}")

@socketio.on('disconnect')
def on_disconnect():
    sid = request.sid
    clients.pop(sid, None)
    print(f"[Disconnected] Client {sid}")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
