import time
import random
import numpy as np
import ctypes
import cv2
import win32api
import pandas as pd
from pathlib import Path
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from src.Sample import Sample


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

class SimpleBallSourceThread(SourceThread):
    """Ball that moves straight at a constant speed. When touching the edge of the screen, it bounces in a random direction."""

    def __init__(self, timeout: int):
        super().__init__(timeout)
        self.screen_bounds = np.array([ctypes.windll.user32.GetSystemMetrics(i) for i in range(2)], dtype=np.int32)

        self.ball_time = None
        self.ball_pos = self.screen_bounds/2.0
        self.ball_vel = self.screen_bounds/7.0

    def get(self) -> tuple[bool, tuple[float, float]]:
        now_time = time.time()

        if self.ball_time is None:
            self.ball_time = now_time
        dt = now_time - self.ball_time
        if dt > 0.01:
            # Update ball properties
            self.ball_time = now_time
            self.ball_pos += self.ball_vel*dt
            out_of_bounds_mask = (self.ball_pos < 0) | (self.ball_pos > self.screen_bounds)
            self.ball_vel *= np.where(out_of_bounds_mask, -1, 1)
            self.ball_pos = np.clip(self.ball_pos, 0, self.screen_bounds)
            if out_of_bounds_mask.any():
                rnd_angle = random.random()*np.pi/2
                vel_mag = np.linalg.norm(self.ball_vel)
                self.ball_vel = np.copysign(vel_mag * np.array([np.cos(rnd_angle), np.sin(rnd_angle)]), self.ball_vel)

        return (True, self.ball_pos)
    
class FeedbackBallSourceThread(SourceThread):
    """Ball that is drawn pulled towards areas with few samples and/or high errors."""
    # TODO
    pass

class ClickListenerSourceThread(SourceThread):
    """Records where the mouse clicks."""

    def __init__(self, timeout: int, buttons = [1, 2]):
        super().__init__(timeout)
        self.button_states = { k : 1 for k in buttons}

    def get(self) -> tuple[bool, tuple[float, float]]:
        max_screen_dim = np.array([ctypes.windll.user32.GetSystemMetrics(i) for i in range(2)], dtype=np.int32).max()

        new_button_states = { i:win32api.GetKeyState(i) for i in range(1, 3)}

        button = -1 
        for k in self.button_states:
            if self.button_states[k] == new_button_states[k]:
                continue
            if new_button_states[k] < 0:
                button = k
                break
        self.button_states = new_button_states
        
        screen_pos = np.array(win32api.GetCursorPos())
        return (button != -1, *screen_pos/max_screen_dim)

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