import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2 as cv
import numpy as np
import uuid
import face_recognition #얼굴
import mediapipe as mp #손
import time

#웹캠
cap = cv.VideoCapture(0)

#손
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#시선탐지용 변수
COUNTER = 0
TOTAL_BLINKS = 0
CLOSED_EYES_FRAME = 3
cameraID = 0
videoPath = "Video/Your Eyes Independently_Trim5.mp4"
FRAME_COUNTER = 0
START_TIME = time.time()
FPS = 0

#손탐지용 변수
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

#얼굴인식용 변수
image = face_recognition.load_image_file("c:/Users/luna/Desktop/girl.jpg")
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

    with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
        ret, frame = cap.read()

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = cv.flip(frame, 1)
        frame.flags.writeable = False
        results = hands.process(frame)
        frame.flags.writeable = True
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        print(results)
        
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )

        # Only process every other frame of video to save time
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

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
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

        # Display the resulting image
        cv.imshow('Video', frame)
        #cv.imshow('Video', image)

        # Hit 'q' on the keyboard to quit!
        if cv.waitKey(1) & 0xFF == ord('q'):
            break