import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2 as cv
import numpy as np
import uuid
import face_recognition #얼굴
import mediapipe as mp #손
import time
import math

#웹캠
cap = cv.VideoCapture(0)

#손
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#거리구하기 함수
def dist(x1, y1, x2, y2) :
    return math.sqrt(math.pow(x1-x2, 2)) + math.sqrt(math.pow(y1-y2, 2))

compareIndex = [[18,4], [6,8], [10,12], [14,16], [18,20]]
open = [False, False, False, False, False]
gesture = [[True, True, True, True, True, "Hand!"]]

#시선탐지용 변수
COUNTER = 0
TOTAL_BLINKS = 0
CLOSED_EYES_FRAME = 3
cameraID = 0
videoPath = "Video/Your Eyes Independently_Trim5.mp4"
FRAME_COUNTER = 0
START_TIME = time.time()
FPS = 0

#얼굴인식용 변수
image = face_recognition.load_image_file("./img/girl.jpg")
encoding = face_recognition.face_encodings(image)[0]

known_face_encodings = [
    encoding
]
known_face_names = [
    "Name",
]

#얼굴변수 초기화
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:

    # hand recognition★

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
        ret, frame = cap.read()
        h, w, c = frame.shape

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = cv.flip(frame, 1)
        frame.flags.writeable = False
        results = hands.process(frame)
        frame.flags.writeable = True
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for i in range(0, 5):
                    open[i] = dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[compareIndex[i][0]].x, handLms.landmark[compareIndex[i][0]].y) < dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[compareIndex[i][1]].x, handLms.landmark[compareIndex[i][1]].y)
            
                print(open)   #손 동작 확인

                text_x = (handLms.landmark[0].x * w)
                text_y = (handLms.landmark[0].y * h)

                for i in range(0, len(gesture)) :
                    flag = True
                    for j in range(0, 5) :
                        if (gesture[i][j] != open[j]) :
                            flag = False
                    if (flag == True) :
                        cv.putText(frame, gesture[i][5], (round(text_x)- 50, round(text_y) -250), cv.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)

                mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)


            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )


        # face recognition★
        
        if process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]
        
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []

            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Person"

                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame

        # Display the results

        for (top, right, bottom, left), name in zip(face_locations, face_names):
           
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv.rectangle(frame, (left, top), (right, bottom), (189, 137, 170), 2)

            # Draw a label with a name below the face
            cv.rectangle(frame, (left, bottom - 35), (right, bottom), (189, 137, 170), cv.FILLED)
            font = cv.FONT_HERSHEY_DUPLEX
            cv.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display
        cv.imshow('Video', frame)

        # 'q' -> Exit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break