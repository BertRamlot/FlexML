import numpy as np
import time
import random
import ctypes
import win32api
from PyQt6.QtCore import pyqtSlot, QMutex

from FlexML.Sample import Sample
from FlexML.SourceThread import SourceThread


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

        return (True, self.ball_pos.astype(np.int32))
    
class FeedbackBallSourceThread(SourceThread):
    """Ball that is drawn pulled towards areas with few samples and/or high errors."""

    def __init__(self, timeout: int):
        super().__init__(timeout)
        self.screen_bounds = np.array([ctypes.windll.user32.GetSystemMetrics(i) for i in range(2)], dtype=np.int32)

        self.ball_time = None
        self.ball_pos = self.screen_bounds/2.0
        self.ball_vel = self.screen_bounds/7.0
        self.min_speed = self.screen_bounds.max()/12.0
        self.max_speed = self.screen_bounds.max()/5.0

        self.mutex = QMutex()
        self.error_map: dict[Sample, float] = {}

    @pyqtSlot(list, np.ndarray)
    def update_sample_errors(self, samples: list[Sample], losses: np.ndarray):
        self.mutex.lock()
        for sample, loss in zip(samples, losses):
            if sample.type in ["train", "val"]:
                if sample.type == "train":
                    # Don't use training loss, only validation loss
                    loss = None
                if loss is None and sample in self.error_map:
                    continue
                self.error_map[sample] = loss
        self.mutex.unlock()

    def get_force_vector(self, position: np.ndarray):
        # We do not like the center
        center_force = position / self.screen_bounds - 0.5*np.ones((2,))
        center_force = 1000* center_force / (1e-2 + np.linalg.norm(center_force))

        self.mutex.lock()
        # repelling force to prevent oversampling
        over_sample_force = np.zeros((2,))
        for sample, loss in self.error_map.items():
            dist = sample.gt - position
            over_sample_force += np.copysign(self.screen_bounds.max()**1.5/(10.0 + abs(dist))**2, -dist)
        # attracting force towards high loss
        loss_force = np.zeros((2,))
        for sample, loss in self.error_map.items():
            if loss is not None:
                dist = sample.gt - position
                loss_force += np.copysign(loss**2/(10.0 + abs(dist)), dist)
        
        total_force = np.zeros((2,))
        total_force += center_force
        total_force += over_sample_force
        total_force += loss_force
        if len(self.error_map) > 0:
            total_force /= len(self.error_map)
        self.mutex.unlock()
        total_force += np.random.random((2,))*self.max_speed/10.0
        # print(over_sample_force.round(), loss_force.round())
        return total_force

    def get(self) -> tuple[bool, tuple[float, float]]:
        now_time = time.time()
        if self.ball_time is None:
            self.ball_time = now_time
        dt = now_time - self.ball_time
        if dt > 0.01:
            # Update ball properties
            self.ball_time = now_time
            new_ball_vel = self.ball_vel + self.get_force_vector(self.ball_pos)*dt
            if np.linalg.norm(new_ball_vel) > self.max_speed:
                new_ball_vel = self.max_speed * (new_ball_vel / np.linalg.norm(new_ball_vel))
            elif np.linalg.norm(new_ball_vel) < self.min_speed:
                new_ball_vel = self.ball_vel
            self.ball_vel = new_ball_vel
            self.ball_pos += self.ball_vel*dt

            # Deal with hitting the edge of the screen
            out_of_bounds_mask = (self.ball_pos < 0) | (self.ball_pos > self.screen_bounds)
            self.ball_pos = np.clip(self.ball_pos, 0, self.screen_bounds)
            self.ball_vel *= np.where(out_of_bounds_mask, -1, 1)
            # self.ball_vel = np.array([0, 0])

        return (True, self.ball_pos.astype(np.int32))

class ClickListenerSourceThread(SourceThread):
    """Records where the mouse clicks."""

    def __init__(self, timeout: int, buttons = [1, 2]):
        super().__init__(timeout)
        self.button_states = { k : 1 for k in buttons}

    def get(self) -> tuple[bool, tuple[float, float]]:
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
        return (button != -1, *screen_pos)
