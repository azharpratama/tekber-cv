import cv2
import os
from dotenv import load_dotenv

load_dotenv()

recognizer = cv2.face.LBPHFaceRecognizer.create()
recognizer.read("face-model.yml")  # face model from face_training.py
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_COMPLEX

id = 0
names = ["None", "Izzat"]
cam = int(os.environ.get("cam", 0))
cap = cv2.VideoCapture(cam)

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        id, confidence = recognizer.predict(gray[y : y + h, x : x + w])

        if confidence < 100:
            id = names[id]
        else:
            id = "unknown"
        confidence = "{}%".format(round(100 - confidence))

        cv2.putText(frame, str(id), (x + 5, y - 5), font, 1, (255, 0, 0), 1)
        cv2.putText(
            frame, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1
        )

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()