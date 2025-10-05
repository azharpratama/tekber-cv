import cv2
import mediapipe as mp
import os
from dotenv import load_dotenv

load_dotenv()

cam = int(os.environ.get("cam", 0))
cap = cv2.VideoCapture(cam)
hands = mp.solutions.hands.Hands()
draws = mp.solutions.drawing_utils

while True:
    _, frame = cap.read()

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_obj = hands.process(frameRGB)

    if hand_obj.multi_hand_landmarks:
        for hand_landmarks in hand_obj.multi_hand_landmarks:
            draws.draw_landmarks(
                frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
            )

    cv2.imshow("Hand Skeleton", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
