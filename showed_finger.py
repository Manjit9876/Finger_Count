import cv2
import mediapipe as mp
import time
import math

# === Utility: Angle Calculation ===
def calculate_angle(a, b, c):
    """Returns angle at point b (in degrees) between points a-b-c."""
    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) -
        math.atan2(a[1] - b[1], a[0] - b[0])
    )
    ang = abs(ang)
    if ang > 180:
        ang = 360 - ang
    return ang

# === Mediapipe Setup ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# === Webcam Setup ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

prev_time = 0
tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky tips

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    total_fingers_all_hands = 0

    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hand_label = results.multi_handedness[idx].classification[0].label
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = img.shape
            lm_list = [(id, int(lm.x * w), int(lm.y * h)) for id, lm in enumerate(hand_landmarks.landmark)]

            fingers_up = []

            if lm_list:
                # === Thumb angle ===
                a = lm_list[1][1:]
                b = lm_list[2][1:]
                c = lm_list[3][1:]
                thumb_angle = calculate_angle(a, b, c)

                fingers_up.append(1 if thumb_angle > 150 else 0)

                # === Other fingers ===
                for i in range(1, 5):
                    tip_y = lm_list[tip_ids[i]][2]
                    pip_y = lm_list[tip_ids[i] - 2][2]
                    fingers_up.append(1 if tip_y < pip_y else 0)

                total_fingers = sum(fingers_up)
                total_fingers_all_hands += total_fingers

                # Show hand count
                cv2.putText(img, f'{hand_label} Hand: {total_fingers}', (10, 80 + idx * 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)

    # Show total count
    cv2.putText(img, f'Total Fingers: {total_fingers_all_hands}', (10, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

    # FPS calculation
    curr_time = time.time()
    fps = 1 / max((curr_time - prev_time), 1e-5)
    prev_time = curr_time

    cv2.putText(img, f'FPS: {int(fps)}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Finger Counter", img)

    # 🔑 Exit on 'q' or ESC
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
