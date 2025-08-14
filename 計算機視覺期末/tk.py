import cv2
import mediapipe as mp
import numpy as np
import joblib
import tkinter as tk
from PIL import Image, ImageTk

# Initialize mediapipe Hands Detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=2
)

# Load the SVM model and scaler
model_filename = "svm_model_V3.pkl"
clf = joblib.load(model_filename)
scaler_filename = "scaler_V3.pkl"
scaler = joblib.load(scaler_filename)

# Load labels
label_file = "labels_V3.txt"
with open(label_file, 'r') as f:
    labels = f.readlines()
labels = [label.strip() for label in labels]

def compute_distances(landmarks):
    distances = []
    pairs = [(5,2), (8, 7), (12, 11), (5, 8),
             (5, 7), (5, 0), (2, 7), (2, 8),
             (2, 0), (5, 11), (5, 12), (2, 11),
             (2, 12), (8, 11), (8, 12), (7, 11),
             (7, 12), (0, 8), (0, 7), (0, 11),
             (0, 12),(15, 9),(15, 10),(16, 9),(16, 10)]

    left_eye = landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
    right_eye = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]
    eyes_center_x = int((left_eye.x + right_eye.x) / 2)
    eyes_center_y = int((left_eye.y + right_eye.y) / 2)
    nose_length = np.linalg.norm(np.array([eyes_center_x, eyes_center_y]) - np.array([landmarks.landmark[0].x, landmarks.landmark[0].y]))
    reference_pair = (12, 11)
    p_ref1 = np.array([landmarks.landmark[reference_pair[0]].x, landmarks.landmark[reference_pair[0]].y])
    p_ref2 = np.array([landmarks.landmark[reference_pair[1]].x, landmarks.landmark[reference_pair[1]].y])
    reference_distance = np.linalg.norm(p_ref1 - p_ref2)
    
    for pair in pairs:
        p1 = np.array([landmarks.landmark[pair[0]].x, landmarks.landmark[pair[0]].y])
        p2 = np.array([landmarks.landmark[pair[1]].x, landmarks.landmark[pair[1]].y])
        distance = np.linalg.norm(p1 - p2) / nose_length
        distances.append(distance)

    return distances

def toggle_detection():
    global detecting
    detecting = not detecting
    
    if detecting:
        start_button.config(text="Stop Detection")
        cap.open(0)  # 開啟攝像頭
        update_frame()
    else:
        start_button.config(text="Start Detection")
        cap.release()  # 釋放攝像頭

def update_frame():
    ret, frame = cap.read()
    if ret:
        # 使用RGB格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks
            distances = compute_distances(landmarks)
            distances = scaler.transform([distances])

            prediction = clf.predict(distances)
            confidence = np.max(clf.predict_proba(distances))

            label = labels[prediction[0]]
            display_text = f"Pose: {label} ({confidence*100:.2f}%)"

            cv2.putText(rgb_frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            mp.solutions.drawing_utils.draw_landmarks(rgb_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # 將 OpenCV 圖像轉換為 Tkinter 支援的格式
        img = Image.fromarray(rgb_frame)
        img_tk = ImageTk.PhotoImage(image=img)

        # 更新 Tkinter 視窗的圖像
        panel.img_tk = img_tk
        panel.config(image=img_tk)

        # 循環呼叫此函數
        if detecting:
            panel.after(10, update_frame)

# 初始化 Tkinter 視窗
root = tk.Tk()
root.title("Pose Detection Demo")

# 創建 Tkinter 元件
panel = tk.Label(root)
panel.pack(padx=10, pady=10)

# 增加一個「開始」按鈕
detecting = False
start_button = tk.Button(root, text="Start Detection", command=toggle_detection)
start_button.pack(pady=10)

# 初始化攝像頭
cap = cv2.VideoCapture(0)

# 啟動 Tkinter 主迴圈
root.mainloop()

# 釋放攝像頭資源
cap.release()
