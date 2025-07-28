import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path to the known faces folder
path = 'known_faces'
images = []
classNames = []

# Load all images and class names
myList = os.listdir(path)
for img_name in myList:
    curImg = cv2.imread(f'{path}/{img_name}')
    images.append(curImg)
    classNames.append(os.path.splitext(img_name)[0])

# Function to encode all known faces
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(img)
        if enc:
            encodeList.append(enc[0])
    return encodeList

# Function to mark attendance
def markAttendance(name):
    file_path = 'attendance.csv'
    
    # If the file doesn't exist, create it with headers
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write('Name,Time\n')
    
    with open(file_path, 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        if name not in nameList:
            now = datetime.now()
            timeString = now.strftime('%H:%M:%S')
            f.write(f'\n{name},{timeString}')

# Encode known faces
encodeListKnown = findEncodings(images)
print('Encoding complete. Starting webcam...')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)  # Resize for faster processing
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Find face locations and encodings in the current frame
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Compare with known encodings
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].capitalize()
            print(f"Recognized: {name}")
            y1, x2, y2, x1 = faceLoc
            # Scale back to original size
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            # Draw rectangle and name
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # Mark attendance
            markAttendance(name)

    cv2.imshow('Webcam Face Recognition', img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
