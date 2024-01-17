import cv2
import dlib
import numpy as np
import sys
import time


face_detector = dlib.get_frontal_face_detector()
face_feature_predictor = dlib.shape_predictor("src/face_based/shape_predictor_68_face_landmarks.dat")

while True:
    length = int.from_bytes(sys.stdin.buffer.read(4))
    data_bytes = sys.stdin.buffer.read(length)
    data = np.frombuffer(data_bytes, dtype=np.uint8)
    img = data.reshape((480, 640, 3))

    time.sleep(5)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_boxes = face_detector(gray_img)
    features = np.zeros((len(face_boxes), 68, 2), dtype=np.int32)
    for i, face_box in enumerate(face_boxes):
        landmarks = face_feature_predictor(gray_img, box=face_box)
        features[i, :, :] = np.array([[p.x, p.y] for p in landmarks.parts()])
        
    output_bytes = features.tobytes()
    length = len(output_bytes)
    sys.stdout.buffer.write(int.to_bytes(length, length=4))
    if length > 0:
        sys.stdout.buffer.write(output_bytes)
    sys.stdout.flush()