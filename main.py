import cv2
from datetime import datetime
import mediapipe as mp

image_capture_folder = "captures/"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

while True:
    result, frame = cap.read()
    if not result:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
        # Draw hand landmarks on the image
            for lm in hand_landmarks.landmark:
                # Extract landmark positions (lm.x, lm.y, lm.z)
                # Draw landmarks on the image
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    cv2.imshow("What's cookin', good lookin'?", frame)

    # append date and time to capture name
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    capture_name = f"capture_{timestamp}.png"

    # save image with appended capture name
    cv2.imwrite(f"{image_capture_folder}{capture_name}", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()