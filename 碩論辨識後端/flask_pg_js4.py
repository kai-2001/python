from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import time
import cv2
import mediapipe as mp
import numpy as np
import joblib
import uuid

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# 初始化 Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.7,min_tracking_confidence=0.7)

# 已知參數
KNOWN_DISTANCE = 50.0  # cm，參考距離
KNOWN_WIDTH = 12.0     # cm，兩眼之間的實際距離
FOCAL_LENGTH = 500     # 焦距（需根據相機校準）

# Dictionary to manage multiple detection processes
detection_threads = {}
detection_data = {}

# Initialize mediapipe Pose Detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=2
)


# Load the SVM model and scaler
model_filename = "svm_model_V3_10.pkl"
clf = joblib.load(model_filename)
scaler_filename = "scaler_V3_10.pkl"
scaler = joblib.load(scaler_filename)

# Load labels
label_file = "labels_V3_7.txt"
with open(label_file, 'r') as f:
    labels = f.readlines()
labels = [label.strip() for label in labels]

# def compute_distances(landmarks):
#     distances = []
#     pairs = [(5, 2), (8, 7), (12, 11), (5, 8),
#              (5, 7), (5, 0), (2, 7), (2, 8),
#              (2, 0), (5, 11), (5, 12), (2, 11),
#              (2, 12), (8, 11), (8, 12), (7, 11),
#              (7, 12), (0, 8), (0, 7), (0, 11),
#              (0, 12), (15, 9), (15, 10), (16, 9), (16, 10)]

#     left_eye = landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
#     right_eye = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]
#     eyes_center_x = int((left_eye.x + right_eye.x) / 2)
#     eyes_center_y = int((left_eye.y + right_eye.y) / 2)
#     nose_length = np.linalg.norm(np.array([eyes_center_x, eyes_center_y]) - np.array([landmarks.landmark[0].x, landmarks.landmark[0].y]))
#     reference_pair = (12, 11)
#     p_ref1 = np.array([landmarks.landmark[reference_pair[0]].x, landmarks.landmark[reference_pair[0]].y])
#     p_ref2 = np.array([landmarks.landmark[reference_pair[1]].x, landmarks.landmark[reference_pair[1]].y])
#     reference_distance = np.linalg.norm(p_ref1 - p_ref2)

#     for pair in pairs:
#         p1 = np.array([landmarks.landmark[pair[0]].x, landmarks.landmark[pair[0]].y])
#         p2 = np.array([landmarks.landmark[pair[1]].x, landmarks.landmark[pair[1]].y])
#         distance = np.linalg.norm(p1 - p2) / nose_length
#         distances.append(distance)

#     return distances
def compute_distances(landmarks):
    distances = []
    pairs = [
        (5,2), (8, 7), (12, 11), (5, 8), (5, 7), (5, 0), 
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
        return None  # 避免除以 0

    for pair in pairs:
        p1 = np.array([landmarks.landmark[pair[0]].x, landmarks.landmark[pair[0]].y])
        p2 = np.array([landmarks.landmark[pair[1]].x, landmarks.landmark[pair[1]].y])
        distance = np.linalg.norm(p1 - p2) / shoulder_dist
        distances.append(distance)
    return distances

def run_pose_detection(session_id):
    global detection_data
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    while session_id in detection_threads:
        ret, frame = cap.read()
        if not ret:
            break
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks
            distances = compute_distances(landmarks)
            distances = scaler.transform([distances])
            prediction = clf.predict(distances)
            confidence = np.max(clf.predict_proba(distances))
            label = labels[prediction[0]]
            detection_data[session_id] = {"label": label, "confidence": confidence}
            print(f"Session ID: {session_id}, Label: {label}, Confidence: {confidence}")
    cap.release()
    # Clean up session data when detection stops
    detection_data.pop(session_id, None)

# 計算距離的函數
def calculate_distance(left_eye, right_eye, image_width, image_height,focal):
    # 將 Mediapipe 的標記點轉換為像素座標
    left_eye_x, left_eye_y = int(left_eye.x * image_width), int(left_eye.y * image_height)
    right_eye_x, right_eye_y = int(right_eye.x * image_width), int(right_eye.y * image_height)

    # 計算像素距離
    pixel_distance = np.sqrt((right_eye_x - left_eye_x) ** 2 + (right_eye_y - left_eye_y) ** 2)

    # 避免除以零
    if pixel_distance == 0:
        return None

    # 根據公式計算實際距離
    distance = (KNOWN_WIDTH * focal) / pixel_distance
    return distance

@app.route('/start-detection', methods=['POST'])
def start_detection():
    session_id = str(uuid.uuid4())  # Generate a unique session ID
    if session_id not in detection_threads:
        detection_threads[session_id] = threading.Thread(target=run_pose_detection, args=(session_id,))
        detection_threads[session_id].start()
        return jsonify({"session_id": session_id, "message": "Detection started"}), 200
    else:
        return jsonify({"message": "Detection already running"}), 400

@app.route('/stop-detection', methods=['POST'])
def stop_detection():
    session_id = request.json.get('session_id')
    if session_id in detection_threads:
        detection_threads[session_id] = None  # Signal the thread to stop
        del detection_threads[session_id]
        return jsonify({"message": "Detection stopped"}), 200
    return jsonify({"message": "Invalid session ID"}), 400

@app.route('/get-pose', methods=['GET'])
def get_pose():
    session_id = request.args.get('session_id')
    pose_data = detection_data.get(session_id, {"label": "N/A", "confidence": 0.0})
    print(f"Fetching pose data for session ID: {session_id}, Data: {pose_data}")  # Print to verify data
    return jsonify(pose_data), 200

# @app.route('/process-image', methods=['POST'])
# def process_image():
    image_file = request.files['image']
    if not image_file:
        return jsonify({"label": "N/A", "confidence": 0.0, "error": "No image file provided"}), 400
    
    # Read image from file
    image = np.frombuffer(image_file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"label": "N/A", "confidence": 0.0, "error": "Image decoding failed"}), 400

    # Process image
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks
        distances = compute_distances(landmarks)
        distances = scaler.transform([distances])
        prediction = clf.predict(distances)
        confidence = np.max(clf.predict_proba(distances))
        label = labels[prediction[0]]
        return jsonify({"label": label, "confidence": confidence})
    
    return jsonify({"label": "N/A", "confidence": 0.0})

@app.route('/process-image', methods=['POST'])
def process_image():
    image_file = request.files.get('image')
    if image_file is None:
        return jsonify({"label": "N/A", "confidence": 0.0, "distance": None, "error": "No image file provided"}), 400

    try:
        image = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"label": "N/A", "confidence": 0.0, "distance": None, "error": f"Image decoding failed: {str(e)}"}), 400

    if image is None:
        return jsonify({"label": "N/A", "confidence": 0.0, "distance": None, "error": "Image decoding failed"}), 400

    try:
        threshold = int(request.form.get('threshold'))
    except:
        return jsonify({"label": "N/A", "confidence": 0.0, "distance": None, "error": "Threshold value is missing or invalid"}), 400

    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    distance1 = None
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]
        distance1 = calculate_distance(left_eye, right_eye, image.shape[1], image.shape[0], threshold)

    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks
        distances = compute_distances(landmarks)
        distances = scaler.transform([distances])
        prediction = clf.predict(distances)
        confidence = np.max(clf.predict_proba(distances))
        label = labels[prediction[0]]
        return jsonify({"label": label, "confidence": confidence, "distance": distance1})

    return jsonify({"label": "N/A", "confidence": 0.0, "distance": None})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
