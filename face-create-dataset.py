import cv2
import os
from dotenv import load_dotenv

load_dotenv()

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = int(os.environ.get("cam", 0))
cap = cv2.VideoCapture(cam)
dataset_path = "dataset/"

if not os.path.exists(dataset_path):
    os.mkdir(dataset_path)


# Automatically assign next available person ID
names_file = "names.txt"
if os.path.exists(names_file):
    with open(names_file, "r") as f:
        ids = [int(line.split(":")[0]) for line in f if ":" in line and line.split(":")[0].isdigit()]
    person_id = max(ids) + 1 if ids else 0
else:
    person_id = 0

print(f"Assigned person ID: {person_id}")
print("Enter person name: ")
person_name = input()

# Save the name mapping
with open(names_file, "a") as f:
    f.write(f"{person_id}:{person_name}\n")

count = 0  # count for image name id

print(f"Capturing images for {person_name} (ID: {person_id})")
print("Press 'q' to quit or wait for 30 images to be captured")

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1
        cv2.imwrite(
            dataset_path + "Person-" + str(person_id) + "-" + str(count) + ".jpg",
            gray[y : y + h, x : x + w],
        )

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) == ord("q"):
        break
    elif count == 30:  # stop when 30 photos have been taken
        break

cap.release()
cv2.destroyAllWindows()