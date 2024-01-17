import numpy as np
import cv2
import dlib
from PyQt6.QtCore import QObject, pyqtSlot, pyqtSignal

from src.Sample import Sample


class FaceSample(Sample):
    def __init__(self, sample: Sample, features: np.ndarray):
        self.__dict__.update(sample.__dict__)
        self.features = features
    
    @staticmethod
    def from_metadata(metadata) -> "FaceSample":
        img_path, type, pos_x, pos_y = metadata[:4]
        return FaceSample(Sample(img_path, None, type, (pos_x, pos_y)), metadata[4:])

    def get_extra_metadata(self) -> list:
        return list(self.features.flatten())
    
    def get_extra_metadata_headers(self) -> list[str]:
        return [f"fx_{i}" for i in range(68)] + [f"fy_{i}" for i in range(68)]
    
    def get_face_img(self) -> np.ndarray:
        min_x, min_y = self.features.min(axis=0)
        max_x, max_y = self.features.max(axis=0)
        return self.get_img()[min_y:max_y, min_x:max_x]

    def get_eye_im(self, eye_type) -> np.ndarray:
        if eye_type == "left":
            idx = 36
        elif eye_type == "right":
            idx = 42
        else:
            raise RuntimeError("Invalid eye_type:", eye_type)
        
        min_x, min_y = self.features[idx:idx+6].min(axis=0)
        max_x, max_y = self.features[idx:idx+6].max(axis=0)
        return self.get_img()[min_y:max_y, min_x:max_x]

class FaceSampleConvertor(QObject):
    face_samples = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        print("Loading face/feature model")
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_feature_predictor = dlib.shape_predictor("src/face_based/shape_predictor_68_face_landmarks.dat")

    @pyqtSlot(Sample)
    def convert_sample(self, sample: Sample):
        img = sample.get_img()
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for approx_face_box in self.face_detector(gray_img):
            landmarks = self.face_feature_predictor(gray_img, box=approx_face_box)
            features = np.array([[p.x, p.y] for p in landmarks.parts()])
            self.face_samples.emit(FaceSample(sample, features))
