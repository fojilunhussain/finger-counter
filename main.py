import cv2
from datetime import datetime
import mediapipe as mp

image_capture_folder = "captures/"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

fingertip_ids = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)

while True:
    result, frame = cap.read()
    if not result:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            finger_states = []
            
            # Detecting handedness

            for hand, handedness in enumerate(results.multi_handedness):
                label = handedness.classification[0].label
                score = handedness.classification[0].score

                # Print handedness information for each detected hand
                print(f"Hand {hand + 1}: {label} hand, Confidence: {score:.2f}")

                # Defining finger coordinates

                if label == "Left":
                    if results.multi_hand_landmarks[hand].landmark[4].x > results.multi_hand_landmarks[hand].landmark[3].x:
                        print("{label} thumb is up")
                        finger_states.append(1)
                if label == "Right":
                    if results.multi_hand_landmarks[hand].landmark[4].x < results.multi_hand_landmarks[hand].landmark[3].x:
                        print("{label} thumb is up")
                        finger_states.append(1)

                for tip_id in fingertip_ids:
                    if tip_id != 4 and results.multi_hand_landmarks[hand].landmark[tip_id].y < results.multi_hand_landmarks[hand].landmark[tip_id - 1].y and results.multi_hand_landmarks[hand].landmark[tip_id].y < results.multi_hand_landmarks[hand].landmark[tip_id - 2].y:
                        print("Finger is up")
                        finger_states.append(1)

            # Extract and draw landmarks on the image
            
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            finger_count = finger_states.count(1)
            cv2.putText(frame, f"Finger count: {finger_count}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    cv2.imshow("What's cookin', good lookin'?", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()