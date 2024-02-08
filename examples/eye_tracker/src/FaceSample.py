import logging
import numpy as np
import dlib
import cv2
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSlot, pyqtSignal

from examples.eye_tracker.src.GazeSample import GazeSample, GazeSampleCollection


class FaceSample(GazeSample):
    """GazeSample that has extra features that define the contour (incl. eyes, mouth, ...) of a SINGLE face."""
    EYE_DIMENSIONS = np.array([60, 24])

    def __init__(self, sample: GazeSample, features: np.ndarray):
        self.__dict__.update(sample.__dict__)
        self.features = features

    def get_metadata(self) -> list:
        return super().get_metadata() + list(self.features.flatten())

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
        
        # This is another way (different results)
        min_real_x, min_real_y = self.features[idx:idx+6].min(axis=0)
        max_real_x, max_real_y = self.features[idx:idx+6].max(axis=0)
        real_eye_width = max_real_x - min_real_x
        real_eye_height = max_real_y - min_real_y

        if real_eye_width > 1.5 * FaceSample.EYE_DIMENSIONS[0]:
            logging.warn("Real eye width detected that is much larger than crop heigth")
        if real_eye_height > 1.5 * FaceSample.EYE_DIMENSIONS[1]:
            logging.warn("Real eye height detected that is much larger than crop heigth")
        center = np.round(self.features[idx:idx+6].mean(axis=0)).astype(np.int32)
        min_crop_x, min_crop_y = center - FaceSample.EYE_DIMENSIONS//2
        max_crop_x, max_crop_y = center + (FaceSample.EYE_DIMENSIONS - FaceSample.EYE_DIMENSIONS//2)
        
        img = self.get_img()

        return img[min_crop_y:max_crop_y, min_crop_x:max_crop_x]

class FaceSampleCollection(GazeSampleCollection):
    def __init__(self, path: Path):
        super().__init__(path)

    def from_metadata(self, metadata) -> FaceSample:
        gaze_sample = super().from_metadata(metadata)
        return FaceSample(gaze_sample, np.asarray(metadata[6:], dtype=np.int32).reshape(-1, 2))

    def get_metadata_headers(self) -> list[str]:
        extra_headers = [f"fx_{i}" for i in range(68)] + [f"fy_{i}" for i in range(68)]
        return super().get_metadata_headers() + extra_headers

class GazeToFaceSampleConvertor(QObject):
    """Converts a GazeSample to a FaceSample by running a face detector ('shape_predictor_68_face_landmarks.dat')."""

    face_samples = pyqtSignal(object)

    def __init__(self, shape_predictor_path: Path):
        super().__init__()
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_feature_predictor = dlib.shape_predictor(str(shape_predictor_path))

    @pyqtSlot(GazeSample)
    def convert_sample(self, sample: GazeSample):
        img = sample.get_img()        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_boxes = self.face_detector(gray_img)
        for face_box in face_boxes:
            landmarks = self.face_feature_predictor(gray_img, box=face_box)
            face_features = np.array([[p.x, p.y] for p in landmarks.parts()])
            face_sample = FaceSample(sample, face_features)
            self.face_samples.emit(face_sample)
