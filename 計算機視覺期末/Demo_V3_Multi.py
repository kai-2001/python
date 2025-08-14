import cv2
import mediapipe as mp
import numpy as np
import joblib

# Initialize mediapipe Hands Detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.7,  # 调整检测置信度的阈值
    min_tracking_confidence=0.7,  # 调整追踪置信度的阈值
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

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
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
            display_text = f"Pose: {label} ({confidence*100:.2f}%)"

            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # To visualize the landmarks of the pose
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Demo', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break

    cap.release()
    cv2.destroyAllWindows()