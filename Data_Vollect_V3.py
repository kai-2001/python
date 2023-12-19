import cv2
import mediapipe as mp
import numpy as np
import time
import os

# Initialize mediapipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.7,  # 调整检测置信度的阈值
    min_tracking_confidence=0.7,  # 调整追踪置信度的阈值
    model_complexity=2
)

def compute_distances(landmarks):
    distances = []
    
    # Define pairs for distance calculation
    pairs = [(5,2), (8, 7), (12, 11), (5, 8),
             (5, 7), (5, 0), (2, 7), (2, 8),
             (2, 0), (5, 11), (5, 12), (2, 11),
             (2, 12), (8, 11), (8, 12), (7, 11),
             (7, 12), (0, 8), (0, 7), (0, 11),
             (0, 12),(15, 9),(15, 10),(16, 9),(16, 10)]
    left_eye = landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE]
    right_eye = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE]
    eyes_center_x = int((left_eye.x + right_eye.x)  / 2)
    eyes_center_y = int((left_eye.y + right_eye.y)  / 2)
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

# Ask user for filename
filename = input("Please enter the filename for data: ")
save_path = "dataset_V3/" + filename

# Check if the 'dataset' directory exists, if not, create it
if not os.path.exists("dataset_V3"):
    os.makedirs("dataset_V3")

cap = cv2.VideoCapture(0)
data_collection = []

collecting = False
start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    if not collecting:
        cv2.putText(frame, "Press SPACE to start data collection", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        elapsed_time = int(time.time() - start_time)
        remaining_time = 10 - elapsed_time
        cv2.putText(frame, f"Time left: {remaining_time} seconds", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        if elapsed_time >= 10:
            break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks and collecting:
        landmarks = results.pose_landmarks
        distances = compute_distances(landmarks)
        data_collection.append(distances)

    cv2.imshow("Data Collection", frame)
    key = cv2.waitKey(1)
    
    if key == 32 and not collecting:
        collecting = True
        start_time = time.time()
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# Convert the data_collection list to numpy array and save
try:
    existing_data = np.load(f"{save_path}.npy")
    # 將新數據附加到現有數據
    updated_array = np.vstack((existing_data, np.array(data_collection)))
    # 將更新的數據保存回文件
    np.save(save_path, np.array(updated_array))

except FileNotFoundError:
    # 如果文件不存在，直接保存數據
    np.save(save_path, np.array(data_collection))
# np.save(save_path, np.array(data_collection))
print(f"Data saved to {save_path}")