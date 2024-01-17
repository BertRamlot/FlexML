import numpy as np
import cv2
import dlib
import multiprocessing
import queue
import subprocess
import sys
import pickle
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
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "src.face_based.face_detector"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    @pyqtSlot(Sample)
    def convert_sample(self, sample: Sample):
        img_bytes = sample.get_img().tobytes()
        img_shape = sample.get_img().shape
        print("CS", img_shape, len(img_bytes))

        try:
            self.proc.stdin.write(int.to_bytes(len(img_bytes), length=4))
            self.proc.stdin.write(img_bytes)
            self.proc.stdin.flush()

            length = int.from_bytes(self.proc.stdout.read(4))
            if length > 0:
                result_bytes = self.proc.stdout.read(length)
                result = np.frombuffer(result_bytes, dtype=np.int32).reshape((-1, 68, 2))
                for i in range(result.shape[0]):
                    new_sample = FaceSample(sample, result[i])
                    self.face_samples.emit(new_sample)
        except Exception as e:
            print(f"Error communicating with subprocess: {e}")

            # Capture and print any errors from the subprocess
            err_output = self.proc.stderr.read()
            if err_output:
                print(f"Subprocess error\n{err_output.decode('utf-8')}")
        finally:
            self.proc.terminate()
            self.proc.wait()