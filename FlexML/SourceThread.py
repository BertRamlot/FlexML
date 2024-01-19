import numpy as np
import cv2
import time
import pandas as pd
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal, QEventLoop

from FlexML.Sample import Sample


class SourceThread(QThread):
    new_item = pyqtSignal(object)

    def __init__(self, timeout: float):
        super().__init__(None)
        self.timeout = timeout

    def run(self):
        self.exec()

    def exec(self):
        while not self.is_done():
            t0 = time.time()
            success, item = self.get()
            if success:
                self.new_item.emit(item)
            self.eventDispatcher().processEvents(QEventLoop.ProcessEventsFlag.AllEvents)
            t1 = time.time()
            sleep_ms = int(1000*(self.timeout - (t1 - t0)))
            if sleep_ms > 0:
                self.msleep(sleep_ms)

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