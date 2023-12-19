import cv2
from datetime import datetime
import mediapipe as mp

image_capture_folder = "captures/"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
fingertip_ids = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)

while True:
    result, frame = cap.read()
    if not result:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(image)

    if results.multi_hand_landmarks:
        # hand_orientation = {
        #     "left_palm_visible": True,
        #     "right_palm_visible": True
        # }

        for hand_landmarks in results.multi_hand_landmarks:
            finger_states = []

            # Determine hand orientation

            # if results.multi_hand_landmarks[0].landmark[0].x > results.multi_hand_landmarks[0].landmark[1].x:
            #     hand_orientation["right_palm_visible"] = True


            # Defining finger coordinates
            
            if results.multi_hand_landmarks[0].landmark[4].x > results.multi_hand_landmarks[0].landmark[3].x:
                print("Thumb is up")
                finger_states.append(1)

            for tip_id in fingertip_ids:
                if tip_id != 4 and results.multi_hand_landmarks[0].landmark[tip_id].y < results.multi_hand_landmarks[0].landmark[tip_id - 1].y and results.multi_hand_landmarks[0].landmark[tip_id].y < results.multi_hand_landmarks[0].landmark[tip_id - 2].y:
                    print("Finger is up")
                    finger_states.append(1)


            # Extract and draw landmarks on the image

            for id, hand_lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(hand_lm.x * w), int(hand_lm.y * h)

                if id in fingertip_ids:
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
                else:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            finger_count = finger_states.count(1)
            cv2.putText(frame, f"Finger count: {finger_count}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    cv2.imshow("What's cookin', good lookin'?", frame)

    # now = datetime.now()
    # timestamp = now.strftime("%Y%m%d_%H%M%S")
    # capture_name = f"capture_{timestamp}.png"

    # cv2.imwrite(f"{image_capture_folder}{capture_name}", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()