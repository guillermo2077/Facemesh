import cv2
import mediapipe as mp
import time
import numpy as np
import keras.models
from keras.preprocessing.image import img_to_array

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
exp_classifier = keras.models.load_model('assets//exp_classifier.h5')


def proc_classify(face_img):
    # gray
    gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
    # array (480, 640, 1)
    if isinstance(face_cascade.detectMultiScale(gray, 1.15, 3), tuple):
        return 'Mascarilla'
    else:
        for (x, y, w, h) in face_cascade.detectMultiScale(gray, 1.15, 3):
            roi_gray = gray[y:y + w, x:x + h]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            image_pixels = img_to_array(roi_gray)
            image_pixels = np.expand_dims(image_pixels, axis=0)
            image_pixels /= 255

            model_output = np.argmax(exp_classifier.predict(image_pixels), axis=1)
            return str(emotions[model_output[0]])


emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
prevTime = 0

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(1)

with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_mesh.process(image)

        # Draw the face mesh annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

        currTime = time.time()

        if currTime - prevTime >= 1:
            prevTime = currTime
            em_to_print = proc_classify(image)

        cv2.putText(image, f'Emotion: {em_to_print}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)
        cv2.imshow('MediaPipe FaceMesh', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release
