import time
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal

from FlexML.Sample import Sample


class SourceThread(QThread):
    new_item = pyqtSignal(np.ndarray)

    def __init__(self, timeout: float):
        super().__init__(None)
        self.timeout = timeout

    def run(self):
        while not self.is_done():
            success, item = self.get()
            if success:
                self.new_item.emit(item)
            if self.timeout:
                time.sleep(self.timeout)

    def get(self) -> tuple[bool, object]:
        raise NotImplementedError()
    
    def is_done(self) -> bool:
        return False

class WebcamSourceThread(SourceThread):
    def __init__(self, timeout: int, index: int = 0):
        super().__init__(timeout)
        print(f"Starting camera {index}... (this can take a while, I don't know why)")
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    def __del__(self):
        self.cap.release()

    def get(self) -> tuple[bool, np.ndarray]:
        return self.cap.read()

class DatasetSource(SourceThread):
    def __init__(self, dataset_path: Path):
        super().__init__(0.0)
        self.metadata_imgs = pd.read_csv(dataset_path / "metadata.csv")
        self.img_path = dataset_path / "raw"
        self.curr_index = 0

    def get(self) -> tuple[bool, Sample]:
        if self.curr_index >= len(self.metadata_imgs):
            return (False, None)
        sample = Sample.from_metadata(self.metadata_imgs[self.curr_index])
        self.curr_index += 1
        return (True, sample)

    def is_done(self) -> bool:
        return self.curr_index >= len(self.metadata_imgs)