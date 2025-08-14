import cv2
import mediapipe as mp
import math

# 初始化 Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# 已知參考寬度（例如人臉的眼間距，單位毫米）
KNOWN_DISTANCE = 100.0  # cm
KNOWN_WIDTH = 12.0  # cm（例如，兩眼之間的實際距離）

# 焦距 F 的計算
FOCAL_LENGTH = 500  # 這個數值需要根據你的相機校準測試來調整

# 開啟攝像頭
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # 獲取左眼和右眼的節點（示例：用於計算距離）
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]

            # 計算眼睛之間的距離（以像素為單位）
            h, w, _ = frame.shape
            left_eye_x, left_eye_y = int(left_eye.x * w), int(left_eye.y * h)
            right_eye_x, right_eye_y = int(right_eye.x * w), int(right_eye.y * h)

            pixel_distance = math.sqrt((right_eye_x - left_eye_x) ** 2 + (right_eye_y - left_eye_y) ** 2)

            # 使用公式計算實際距離
            if pixel_distance != 0:
                distance = (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_distance
                cv2.putText(frame, f"Distance: {distance:.2f} cm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 畫出眼睛的位置
            cv2.circle(frame, (left_eye_x, left_eye_y), 5, (0, 255, 0), -1)
            cv2.circle(frame, (right_eye_x, right_eye_y), 5, (0, 255, 0), -1)

    cv2.imshow('Face Distance', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
