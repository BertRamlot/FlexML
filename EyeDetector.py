import cv2
import ctypes

class Eye():
    def __init__(self, eye_rect, eye_im):
        # eye rect is relative to the screen, all vals are within [0,1]
        self.x = eye_rect[0]
        self.y = eye_rect[1]
        self.w = eye_rect[2]
        self.h = eye_rect[3]

        self.im = eye_im

    def __lt__(self, other_eye) -> bool:
        return self.x < other_eye.x

class EyePair():
    def __init__(self, left_eye: Eye, right_eye: Eye):
        self.left_eye = left_eye
        self.right_eye = right_eye

class EyeDetector():
    def __init__(self):
        print("Loading eye model")
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        print("Starting camera")
        self.cap = cv2.VideoCapture(0)

        self.last_img = None
        self.last_eye_pairs = []
    
    def valid_eyes_found(self) -> bool:
        return len(self.last_eye_pairs) > 0

    def update(self):
        ret, self.last_img = self.cap.read()
        gray = cv2.cvtColor(self.last_img, cv2.COLOR_BGR2GRAY)
        video_height, video_width = gray.shape

        eyes = []
        for ex,ey,w,h in self.eye_cascade.detectMultiScale(gray, minNeighbors=10):
            eye_im = self.last_img[ex:ex+w, ey:ey+h]
            rel_rect = (ex/video_width, ey/video_height, w/video_width, h/video_height)
            eyes.append(Eye(rel_rect, eye_im))
        eyes.sort()

        if len(eyes) % 2 != 0:
            self.last_eye_pairs = []
        else:
            self.last_eye_pairs = [EyePair(eyes[i], eyes[i+1]) for i in range(0, len(eyes), 2)]

        # self.last_img = self.last_img[:, round(0.25*self.last_img.shape[1]):round(0.75*self.last_img.shape[1])]
        # t0 = time.time()
        # print("c: {} s".format(time.time()-t0))
