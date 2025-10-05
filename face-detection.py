import cv2
import os
from dotenv import load_dotenv

load_dotenv()

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
clf = cv2.CascadeClassifier(cascade_path)

cam = int(os.environ.get("cam", 0))
camera = cv2.VideoCapture(cam)

while True:
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(30, 30),
    )

    for x, y, width, height in faces:
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
