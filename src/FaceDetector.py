import cv2
import dlib


class Face():
    def __init__(self, face_im, tl_rx: float, tl_ry: float, rw: float, rh: float, features_rx, features_ry):
        self.im = face_im
        self.tl_rx = tl_rx
        self.tl_ry = tl_ry
        self.rw = rw
        self.rh = rh
        self.features_rx = features_rx
        self.features_ry = features_ry
        self.rx = sum(self.features_rx[36:48])/12
        self.ry = sum(self.features_ry[36:48])/12

    def get_eye_im(self, eye_type):
        if eye_type == "left":
            start_feature_index = 36
        elif eye_type == "right":
            start_feature_index = 42
        else:
            raise RuntimeError("Invalid eye_type:", eye_type)
        eye_features_rx = self.features_rx[start_feature_index:start_feature_index+6]
        eye_features_ry = self.features_ry[start_feature_index:start_feature_index+6]
        min_x, max_x = round(min(eye_features_rx)*self.im.shape[1]), round(max(eye_features_rx)*self.im.shape[1])
        min_y, max_y = round(min(eye_features_ry)*self.im.shape[0]), round(max(eye_features_ry)*self.im.shape[0])
        return self.im[min_y:max_y, min_x:max_x]


class FaceDetector():
    def __init__(self, cap):
        self.cap = cap
        print("Loading face/feature model")
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_feature_predictor = dlib.shape_predictor("src/shape_predictor_68_face_landmarks.dat")
        self.video_width = round(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = round(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Camera dim: ({}x{})".format(self.video_width, self.video_height))
        self.last_img = None
        self.last_faces = []

    def valid_faces_found(self) -> bool:
        return len(self.last_faces) > 0

    def update(self):
        _, self.last_img = self.cap.read()
        gray_img = cv2.cvtColor(self.last_img, cv2.COLOR_BGR2GRAY)

        faces = []
        for approx_face_box in self.face_detector(gray_img):
            landmarks = self.face_feature_predictor(gray_img, box=approx_face_box)
            approx_x_parts = [p.x for p in landmarks.parts()]
            approx_y_parts = [p.y for p in landmarks.parts()]
            fx, fy = min(approx_x_parts), min(approx_y_parts)
            dx_parts = [x - fx for x in approx_x_parts]
            dy_parts = [y - fy for y in approx_y_parts]
            fw, fh = max(dx_parts), max(dy_parts)

            features_rx = [dx/fw for dx in dx_parts]
            features_ry = [dy/fh for dy in dy_parts]

            tl_rx, tl_ry = fx/self.video_width, fy/self.video_height
            rw, rh = fw/self.video_width, fh/self.video_height

            face_im = self.last_img[fy:fy+fh, fx:fx+fw]
            if face_im.size == 0:
                continue
            faces.append(Face(face_im, tl_rx, tl_ry, rw, rh, features_rx, features_ry))

        self.last_faces = faces

        # self.last_img = self.last_img[:, round(0.25*self.last_img.shape[1]):round(0.75*self.last_img.shape[1])]
        # t0 = time.time()
        # print("c: {} s".format(time.time()-t0))