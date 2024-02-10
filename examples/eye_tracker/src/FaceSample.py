import logging
import numpy as np
import dlib
import cv2
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSlot, pyqtSignal

from examples.eye_tracker.src.GazeSample import GazeSample, GazeSampleCollection


class FaceSample(GazeSample):
    """GazeSample that has extra features that define the contour (incl. eyes, mouth, ...) of a SINGLE face."""

    # TODO: This is hard coded, would be nice if the model would adaptively change
    # E.g. if you detect that the average eye dimension is substantially different from the current EYE_DIMENSIONS,
    #      you could set the new EYE_DIMENSIONS, retrain the model, ...
    EYE_DIMENSIONS = np.array([24, 60]) # [y, x]

    def __init__(self, sample: GazeSample, face_id: int, features: np.ndarray):
        self.__dict__.update(sample.__dict__)
        self.face_id = face_id
        self.features = features

    def get_metadata(self) -> list:
        return super().get_metadata() + [self.face_id] + list(self.features.flatten())

    def get_face_img(self) -> np.ndarray:
        min_y, min_x = self.features.min(axis=0)
        max_y, max_x = self.features.max(axis=0)
        return self.get_img()[min_y:max_y, min_x:max_x]

    def get_eye_img(self, eye_type: str) -> np.ndarray:
        if eye_type == "left":
            idx = 36
        elif eye_type == "right":
            idx = 42
        else:
            raise RuntimeError("Invalid eye_type:", eye_type)
        
        real_eye_dims = self.features[idx:idx+6].max(axis=0) - self.features[idx:idx+6].min(axis=0)
        if (real_eye_dims > 2.0 * FaceSample.EYE_DIMENSIONS).all():
            logging.warn(f"Real eye size is much larger than model size, consider changing 'FaceSample.EYE_DIMENSIONS' to approx: {real_eye_dims}")
        elif (real_eye_dims < 0.3 * FaceSample.EYE_DIMENSIONS).all():
            logging.warn(f"Real eye size is much smaller than model size, consider changing 'FaceSample.EYE_DIMENSIONS' to approx: {real_eye_dims}")
        
        center = np.round(self.features[idx:idx+6].mean(axis=0)).astype(np.int32)
        min_crop_y, min_crop_x = center - FaceSample.EYE_DIMENSIONS//2
        max_crop_y, max_crop_x = center - FaceSample.EYE_DIMENSIONS//2 + FaceSample.EYE_DIMENSIONS

        img = self.get_img()
        return img[min_crop_y:max_crop_y, min_crop_x:max_crop_x]

class FaceSampleCollection(GazeSampleCollection):
    def __init__(self, path: Path):
        super().__init__(path)

    def from_metadata(self, metadata) -> FaceSample:
        gaze_sample = super().from_metadata(metadata)
        face_id = metadata[6]
        features = np.asarray(metadata[7:], dtype=np.int32).reshape(-1, 2)
        return FaceSample(gaze_sample, face_id, features)

    def get_metadata_headers(self) -> list[str]:
        feature_headers = ["face_id"] + [f"f{axis}_{i}" for i in range(68) for axis in ["y", "x"]]
        return super().get_metadata_headers() + feature_headers

class GazeToFaceSamplesConvertor(QObject):
    """Converts a GazeSample to a number of FaceSamples by running a face detector ('shape_predictor_68_face_landmarks.dat')."""

    face_samples = pyqtSignal(FaceSample)

    def __init__(self, shape_predictor_path: Path):
        super().__init__()
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_feature_predictor = dlib.shape_predictor(str(shape_predictor_path))

    @pyqtSlot(GazeSample)
    def convert_sample(self, sample: GazeSample):
        img = sample.get_img()        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_boxes = self.face_detector(gray_img)
        features_per_face = []
        for face_box in face_boxes:
            landmarks = self.face_feature_predictor(gray_img, box=face_box)
            face_features = np.array([[p.y, p.x] for p in landmarks.parts()])
            features_per_face.append(face_features)

        features_per_face.sort(key=lambda features: features[:, 1].mean())
        for i, face_features in enumerate(features_per_face):
            # TODO: This isn't proper face detection, the face_id is just the x position order of the face. This does work pretty well though.
            self.face_samples.emit(FaceSample(sample, i, face_features))
