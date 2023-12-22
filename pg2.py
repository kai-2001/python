import cv2
import mediapipe as mp
import numpy as np
import joblib
import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE, MOUSEBUTTONDOWN

pygame.mixer.pre_init(44100, -16, 2, 2048)  # 調整參數以減少延遲
pygame.mixer.init()

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
    pairs = [(5, 2), (8, 7), (12, 11), (5, 8),
             (5, 7), (5, 0), (2, 7), (2, 8),
             (2, 0), (5, 11), (5, 12), (2, 11),
             (2, 12), (8, 11), (8, 12), (7, 11),
             (7, 12), (0, 8), (0, 7), (0, 11),
             (0, 12), (15, 9), (15, 10), (16, 9), (16, 10)]

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

def draw_button(screen, rect, color, text, font_color, hover_color, font_size=20):

    # 檢查游標是否在按鈕範圍內
    mouse_x, mouse_y = pygame.mouse.get_pos()
    is_hovered = rect.collidepoint(mouse_x, mouse_y)

    # 根據游標位置選擇文字顏色和按鈕顏色
    text_color =  font_color
    button_color = hover_color if is_hovered else color

    pygame.draw.rect(screen, button_color, rect)

    font = pygame.font.Font(None, font_size)
    text_surface = font.render(text, True, text_color)
    text_rect = text_surface.get_rect(center=rect.center)
    screen.blit(text_surface, text_rect)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detection_enabled = False  # 初始狀態為未開始偵測

    # Initialize Pygame
    pygame.init()
    window_size = (640, 480)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Pose Detection with Pygame")

    clock = pygame.time.Clock()
    start_time = None
    is_playing = False
    alert_channel = None  # 初始化音效通道

    start_button_rect = pygame.Rect(400, 400, 100, 50)
    stop_button_rect = pygame.Rect(520, 400, 100, 50)



    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks and detection_enabled:
            landmarks = results.pose_landmarks
            distances = compute_distances(landmarks)
            distances = scaler.transform([distances])

            prediction = clf.predict(distances)
            confidence = np.max(clf.predict_proba(distances))

            label = labels[prediction[0]]
            display_text = f"Pose: {label} ({confidence*100:.2f}%)"

            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 240, 0), 2, cv2.LINE_AA)
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 检测是否趴著
            if label == 'p'or label == 'c':
                if start_time is None:
                    start_time = pygame.time.get_ticks()  # 開始計時
                else:
                    elapsed_time = (pygame.time.get_ticks() - start_time) / 1000.0  # 換算成秒
                    if elapsed_time >= 5:
                        # 趴著持續 5 秒，發出警告
                        cv2.putText(frame, "ALERT: Not upright for 5 seconds!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 200), 2, cv2.LINE_AA)
                        if not is_playing:
                            alert_channel = pygame.mixer.Sound('./hand_svm/w1.mp3').play(loops=-1)
                            is_playing = True
            else:
                start_time = None  # 重新開始計時
                is_playing = False  # 重置播放狀態
                if alert_channel is not None:
                    alert_channel.stop()  # 停止音效播放

        # Convert the frame to Pygame surface
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.flip(frame, axis=1)
        frame = np.rot90(frame)

        frame = pygame.surfarray.make_surface(frame)
        screen.blit(frame, (0, 0))

        draw_button(screen, start_button_rect, "#7173D0", "Start", (255, 255, 255), "#ABAEE3")
        draw_button(screen, stop_button_rect, "#B4403C", "Stop", (255, 255, 255), "#DA9090")


        pygame.display.flip()
        clock.tick(60)  # 控制畫面更新速率為每秒 60 次

        for event in pygame.event.get():
            if event.type == QUIT:
                cap.release()
                cv2.destroyAllWindows()
                pygame.quit()
                exit()
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    cap.release()
                    cv2.destroyAllWindows()
                    pygame.quit()
                    exit()
            elif event.type == MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if start_button_rect.collidepoint(mouse_x, mouse_y):
                    detection_enabled = True
                elif stop_button_rect.collidepoint(mouse_x, mouse_y):
                    detection_enabled = False
                    start_time = None  # 重新開始計時
                    is_playing = False  # 重置播放狀態
                    if alert_channel is not None:
                        alert_channel.stop()  # 停止音效播放
