import cv2
import os
from dotenv import load_dotenv

load_dotenv()

cam = int(os.environ.get("cam", 0))
camera = cv2.VideoCapture(cam)

while True:
    _, frame = camera.read()

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
