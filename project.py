import cv2
import numpy as np
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


filters = [None, "GRAYSCALE", "SEPIA", "NEGATIVE", "BLUR"]
current_filter = 0

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()


last_action_time = 0
debounce_time = 1


def apply_filter(frame, filter_type):
    if filter_type == "GRAYSCALE":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif filter_type == "SEPIA":
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        sepia_frame = cv2.transform(frame, sepia_filter)
        return np.clip(sepia_frame, 0, 255).astype(np.uint8)
    elif filter_type == "NEGATIVE":
        return cv2.bitwise_not(frame)
    elif filter_type == "BLUR":
        return cv2.GaussianBlur(frame, (15, 15), 0)
    return frame

while True:
    success, img = cap.read()
    if not success:
        print("Error: Could not read frame.")
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            frame_height, frame_width, _ = img.shape


            thumb_x, thumb_y = int(landmarks[mp_hands.HandLandmark.THUMB_TIP].x * frame_width), int(landmarks[mp_hands.HandLandmark.THUMB_TIP].y * frame_height)
            index_x, index_y = int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame_width), int(landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame_height)
            middle_x, middle_y = int(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * frame_width), int(landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * frame_height)
            ring_x, ring_y = int(landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].x * frame_width), int(landmarks[mp_hands.HandLandmark.RING_FINGER_TIP].y * frame_height)
            pinky_x, pinky_y = int(landmarks[mp_hands.HandLandmark.PINKY_TIP].x * frame_width), int(landmarks[mp_hands.HandLandmark.PINKY_TIP].y * frame_height)


            cv2.circle(img, (thumb_x, thumb_y), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (index_x, index_y), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (middle_x, middle_y), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (ring_x, ring_y), 10, (255, 255, 0), cv2.FILLED)
            cv2.circle(img, (pinky_x, pinky_y), 10, (255, 0, 255), cv2.FILLED)

            current_time = time.time()

            if abs(thumb_x - index_x) < 30 and abs(thumb_y - index_y) < 30:
                if current_time - last_action_time > debounce_time:
                    cv2.putText(img, "Picture Captured!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imwrite(f"picture_{int(time.time())}.jpg", img)
                    print("Picture saved!")
                    last_action_time = current_time

            elif (abs(thumb_x - middle_x) < 30 and abs(thumb_y - middle_y) < 30) or \
                 (abs(thumb_x - ring_x) < 30 and abs(thumb_y - ring_y) < 30) or \
                 (abs(thumb_x - pinky_x) < 30 and abs(thumb_y - pinky_y) < 30):
                if current_time - last_action_time > debounce_time:
                    current_filter = (current_filter + 1) % len(filters)
                    print(f"Switched to filter: {filters[current_filter]}")
                    last_action_time = current_time

    filtered_img = apply_filter(img, filters[current_filter])

    if filters[current_filter] == "GRAYSCALE":
        cv2.imshow("Gesture-Controlled Photo App", cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR))
    else:
        cv2.imshow("Gesture-Controlled Photo App", filtered_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
