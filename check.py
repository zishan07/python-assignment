#libraries used face_recognition dlib, cmake, numpy, opencv-python, transitions

import cv2
import numpy as np
import face_recognition
import os
from transitions import Machine


class Detect(object):
    states = ['active', 'inactive']

    def __init__(self, name):

        self.name = name
        self.machine = Machine(model=self, states=Detect.states, initial='inactive')
        if matches[matchIndex]:
            self.machine.add_transition(trigger='person', source='inactive', dest='active')

        else:
            self.machine.add_transition(trigger='nonperson', source='inactive', dest='inactive')


path = 'faces'
images = []
names = []
myList = os.listdir(path)
print(myList)
for le in myList:
    curImage = cv2.imread(f'{path}/{le}')
    images.append(curImage)
    names.append(os.path.splitext(le)[0])
print(names)
# print(len(images))


def findencoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findencoding(images)
print(encodeListKnown)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:

    # cap = cv2.VideoCapture(0)
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)
        motion = Detect("Motion")
        if matches[matchIndex]:
            name = names[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 255), 2)
            motion.person()
        else:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, 'unknown', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
            motion.nonperson()

        print(motion.state)
    cv2.imshow('Webcam', img)
    # show webcam image

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    # press q for quitting task

cap.release()

cv2.destroyAllWindows()

