import logging
import numpy as np
import cv2
import mss
import time
from pathlib import Path
from PyQt6.QtCore import QThread, QObject, QTimer, QEventLoop, pyqtSignal


class SourceObject(QObject):
    new_object = pyqtSignal(object)

    def __init__(self, timeout: float, use_seperate_thread: bool):
        super().__init__()
        self.timeout = timeout
        self._thread = None
        self._timer = None
        if use_seperate_thread:
            self._thread = QThread()
            self.moveToThread(self._thread)
            self._thread.started.connect(self._thread_run)
        else:
            self._timer = QTimer()
            self._timer.timeout.connect(self._timer_run)
    
    def _thread_run(self):
        while not self.is_done():
            t0 = time.time()
            success, item = self.get()
            if success:
                self.new_object.emit(item)
            self._thread.eventDispatcher().processEvents(QEventLoop.ProcessEventsFlag.AllEvents)
            t1 = time.time()
            sleep_ms = int(1000 * (self.timeout - (t1 - t0)))
            if sleep_ms > 0:
                self._thread.msleep(sleep_ms)        
    
    def _timer_run(self) -> bool:
        if self.is_done():
            self._timer.stop()

        success, item = self.get()
        if success:
            self.new_object.emit(item)
            
    def start(self):
        if self._timer is not None:
            timeout_ms = int(1000 * self.timeout)
            self._timer.start(timeout_ms)
        if self._thread is not None:
            self._thread.start()
        
    def get(self) -> tuple[bool, object]:
        raise NotImplementedError()
    
    def is_done(self) -> bool:
        return False

class WebcamSourceObject(SourceObject):
    def __init__(self, timeout: int, index: int = 0):
        super().__init__(timeout, True)
        logging.info(f"Starting webcam capture (index={index}) ...")
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        logging.info(f"Finished starting webcam capture (index={index})")

    def __del__(self):
        self.cap.release()

    def get(self) -> tuple[bool, np.ndarray]:
        return self.cap.read()

    def is_done(self) -> bool:
        # TODO: check if the webcam still works/exists
        return False

class VideoFileSourceObject(SourceObject):
    def __init__(self, timeout: int, path: Path):
        super().__init__(timeout, True)
        self.cap = cv2.VideoCapture(str(path))
        self.reached_end = False

    def __del__(self):
        self.cap.release()

    def get(self) -> tuple[bool, np.ndarray]:
        ret, frame = self.cap.read()
        if not ret:
            self.reached_end = True
        return ret, frame
    
    def is_done(self) -> bool:
        return self.reached_end

class ScreenSourceObject(SourceObject):
    def __init__(self, timeout: int, monitor: dict[str, int] | tuple[int, int, int, int] | None = None):
        super().__init__(timeout, False)
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1] if monitor is None else monitor

    def __del__(self):
        self.sct.close()

    def get(self) -> tuple[bool, np.ndarray]:
        sct_img = self.sct.grab(self.monitor)
        return True, np.asarray(sct_img)
