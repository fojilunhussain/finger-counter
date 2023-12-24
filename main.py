import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

fingertip_ids = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)

def count_fingers(finger_states):
    return finger_states.count(1)

def draw_landmarks(frame, hand_landmarks):
    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

def calculate_finger_angles(hand_landmarks):
    # Define connections for each finger
    finger_connections = [
        [0, 1, 2, 3, 4], # Thumb
        [0, 5, 6, 7, 8], # Index
        [0, 9, 10, 11, 12], # Middle
        [0, 13, 14, 15, 16], # Ring
        [0, 17, 18, 19, 20] # Pinky
    ]

    finger_states = []

    for finger in finger_connections:
        # Get landmark positions for the finger connections
        finger_points = [hand_landmarks.landmark[i] for i in finger]

         # Calculate angles between the connections
        angles = []
        for i in range(len(finger_points) - 1):
            # Calculate vectors between consecutive landmarks
            v1 = [finger_points[i + 1].x - finger_points[i].x, finger_points[i + 1].y - finger_points[i].y]
            v2 = [finger_points[i + 2].x - finger_points[i + 1].x, finger_points[i + 2].y - finger_points[i + 1].y]

            # Calculate the angle between vectors using atan2 (or arctan)
            angle = np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2))
            angles.append(np.degrees(angle))

        # Use angle thresholds to determine finger state (raised or not)
        finger_raised = all(angle > cv2.threshold for angle in angles)
        finger_states.append(finger_raised)
        print(finger_states)

    return finger_states

def determine_finger_state(multi_hand_landmarks, label, finger_states):
    if label == "Left":
        if multi_hand_landmarks[4].x > multi_hand_landmarks[3].x:
            print(f"{label} thumb is up")
            finger_states.append(1)
    if label == "Right":
        if multi_hand_landmarks[4].x < multi_hand_landmarks[3].x:
            print(f"{label} thumb is up")
            finger_states.append(1)

    for tip_id in fingertip_ids:
        if tip_id != 4 and multi_hand_landmarks[tip_id].y < multi_hand_landmarks[tip_id - 1].y and multi_hand_landmarks[tip_id].y < multi_hand_landmarks[tip_id - 2].y:
            print("Finger is up")
            finger_states.append(1)

def detect_handedness(finger_states, results):
    for hand, handedness in enumerate(results.multi_handedness):
        label = handedness.classification[0].label
        score = handedness.classification[0].score

        print(f"Hand {hand + 1}: {label} hand, Confidence: {score:.2f}")

        # determine_finger_state(results.multi_hand_landmarks[hand].landmark, label, finger_states)
        calculate_finger_angles(results.multi_hand_landmarks[hand].landmark)

def process_frame(frame):
    flipped_frame = cv2.flip(frame, 1) # TODO: Test with different webcams.
    image = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2RGB)

    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            finger_states = []
            
            detect_handedness(finger_states, results)

            draw_landmarks(flipped_frame, hand_landmarks)

            finger_count = count_fingers(finger_states)
            cv2.putText(flipped_frame, f"Finger count: {finger_count}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    cv2.imshow("What's cookin', good lookin'?", flipped_frame)

def main():
    while True:
        result, frame = cap.read()
        if not result:
            break

        process_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
