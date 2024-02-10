import logging
import numpy as np
import cv2
import mss
import time
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal, QEventLoop


class SourceThread(QThread):
    new_object = pyqtSignal(object)

    def __init__(self, timeout: float):
        super().__init__(None)
        self.timeout = timeout

    def run(self):
        while not self.is_done():
            t0 = time.time()
            success, item = self.get()
            if success:
                self.new_object.emit(item)
            # TODO: normally source threads don't receive events (unless followup obj are living on its thread)
            # do we need event loop?
            # self.eventDispatcher().processEvents(QEventLoop.ProcessEventsFlag.AllEvents)
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
        logging.info(f"Starting webcam capture (index={index}) ...")
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        logging.info(f"Finished starting webcam capture (index={index})")

    def __del__(self):
        self.cap.release()

    def get(self) -> tuple[bool, np.ndarray]:
        return self.cap.read()

class VideoFileSourceThread(SourceThread):
    def __init__(self, timeout: int, path: Path):
        super().__init__(timeout)
        self.cap = cv2.VideoCapture(str(path))

    def __del__(self):
        self.cap.release()

    def get(self) -> tuple[bool, np.ndarray]:
        return self.cap.read()

class ScreenSourceThread(SourceThread):
    def __init__(self, timeout: int, monitor: dict[str, int] | tuple[int, int, int, int] | None):
        super().__init__(timeout)
        self.sct = mss.mss()
        self.monitor = self.sct.monitors()[1] if monitor is None else monitor

    def __del__(self):
        self.sct.release()

    def get(self) -> tuple[bool, np.ndarray]:
        sct_img = self.sct.grab()
        return sct_img    
