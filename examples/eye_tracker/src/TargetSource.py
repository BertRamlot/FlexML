import logging
import numpy as np
import time
import random
import win32api
from PyQt6.QtCore import pyqtSlot, QMutex

from FlexML.Sample import Sample
from FlexML.SourceObject import SourceObject


class SimpleBallSourceObject(SourceObject):
    """
    A ball that moves straight at a constant speed and bounces when touching the edge of the screen.
    The bounce is biased towards the sides/corners of the screen.
    """
    
    MIN_TIME_BETWEEN_UPDATES = 0.01

    def __init__(self, screen_dims: np.ndarray, timeout: float):
        """
        Initializes the SimpleBallSourceObject with screen dimensions and timeout.

        Args:
            screen_dims (np.ndarray): The dimensions of the screen, shape is (2,) with format (height, width).
            timeout (float): The time in seconds between updates.
        """
        if timeout < SimpleBallSourceObject.MIN_TIME_BETWEEN_UPDATES:
            logging.warning("Timeout ({}) is smaller than the set minimum ({}), this will cause unneeded overhead"
                            .format(timeout, SimpleBallSourceObject.MIN_TIME_BETWEEN_UPDATES))
        
        super().__init__(timeout, False)
        self.screen_dims = screen_dims
        self.ball_time = None
        self.ball_pos = self.screen_dims / 2.0
        self.ball_vel = self.screen_dims / np.array([5.0, 10.0])

    def get(self) -> tuple[bool, np.ndarray]:
        now_time = time.time()
        if self.ball_time is None:
            self.ball_time = now_time
        dt = now_time - self.ball_time
        if dt >= SimpleBallSourceObject.MIN_TIME_BETWEEN_UPDATES:
            # Update ball properties
            self.ball_time = now_time
            self.ball_pos += self.ball_vel*dt
            out_of_bounds_state = 1*(self.ball_pos > self.screen_dims) - 1*(self.ball_pos < 0)
            if (out_of_bounds_state != 0).any():
                # Bounce logic, this is more complex due to the desired bias towards the sides
                if (out_of_bounds_state != 0).all():
                    # Corner hit, choice random side
                    out_of_bounds_state[random.getrandbits(1)] = 0

                if out_of_bounds_state[0] == -1: # Top hit
                    base_angle = np.pi
                    clock_wise = (self.ball_vel < 0).all()
                elif out_of_bounds_state[1] == 1: # Right hit
                    base_angle = np.pi * 3.0/2.0
                    clock_wise = (self.ball_vel[0] < 0) and (self.ball_vel[1] > 0)
                elif out_of_bounds_state[0] == 1: # Bottom hit
                    base_angle = 0.0
                    clock_wise = (self.ball_vel > 0).all()
                elif out_of_bounds_state[1] == -1: # Left hit
                    base_angle = np.pi / 2.0
                    clock_wise = (self.ball_vel[0] > 0) and (self.ball_vel[1] < 0)

                # Random angle is biased towards staying to the sides
                rnd_angle = (random.random()**2)*np.pi/2
                if clock_wise:
                    target_angle = base_angle - rnd_angle
                else:
                    target_angle = base_angle + np.pi + rnd_angle
                
                vel_mag = np.linalg.norm(self.ball_vel)
                self.ball_vel = vel_mag * np.array([np.sin(target_angle), np.cos(target_angle)])

            self.ball_pos = np.clip(self.ball_pos, 0, self.screen_dims)

        return (True, self.ball_pos.astype(np.int32))
    
class FeedbackBallSourceObject(SourceObject):
    """
    A ball that avoids oversampled areas and moves towards areas with high loss.
    """

    MIN_TIME_BETWEEN_UPDATES = 0.01

    def __init__(self, screen_dims: np.ndarray, timeout: int):
        if timeout < FeedbackBallSourceObject.MIN_TIME_BETWEEN_UPDATES:
            logging.warning("Timeout ({}) is smaller than the set minimum ({}), this will cause unneeded overhead"
                            .format(timeout, FeedbackBallSourceObject.MIN_TIME_BETWEEN_UPDATES))
        
        super().__init__(timeout, False)
        self.screen_dims = screen_dims
        self.ball_time = None
        self.ball_pos = self.screen_dims / 2.0
        self.ball_vel = self.screen_dims / np.array([5.0, 10.0])
        self.min_speed = self.screen_dims.max() / 12.0
        self.max_speed = self.screen_dims.max() / 5.0
        self.error_map: dict[Sample, float] = {}

    @pyqtSlot(list, np.ndarray)
    def update_sample_errors(self, samples: list[Sample], losses: np.ndarray):
        """
        Updates the criterion loss for a list of samples.

        Args:
            samples (list[Sample]): List of N samples.
            losses (np.ndarray): Loss per sample, shape is (N,).
        """
        for sample, loss in zip(samples, losses):
            # Ignore "test" type
            if sample.type in ["train", "val"]:
                if sample.type == "train":
                    # Don't use training loss, only validation loss
                    loss = None
                if loss is None and sample in self.error_map:
                    continue
                self.error_map[sample] = loss

    def _get_force_vector_on_ball(self):
        # We do not like the center
        center_dist = (self.ball_pos - self.screen_dims/2.0) / self.screen_dims.max()
        center_dist_norm = max(1e-5, np.linalg.norm(center_dist))
        center_force = (center_dist/center_dist_norm) / (0.01 + center_dist_norm)

        # Repelling force to prevent oversampling
        over_sample_force = np.zeros((2,))
        # Attracting force towards high loss
        loss_force = np.zeros((2,))
        max_loss = max((v for v in self.error_map.values() if v is not None), default=1.0)
        for sample, loss in self.error_map.items():
            to_sample_vec = (sample.ground_truth - self.ball_pos) / self.screen_dims.max()
            to_sample_vec_norm = max(1e-5, np.linalg.norm(to_sample_vec))
            over_sample_force += - (to_sample_vec/to_sample_vec_norm) * 1/(0.1 + to_sample_vec_norm)
            if loss is not None:
                loss_force += (to_sample_vec/to_sample_vec_norm) * (loss/max_loss)**2/(0.1 + to_sample_vec_norm)

        over_sample_force /= max(1, len(self.error_map))
        loss_force /= max(1, sum(1 for loss in self.error_map.values() if loss is not None))
               
        total_force = np.sum(
            [
                0.02*center_force,
                over_sample_force,
                2.0*loss_force,
                np.random.uniform(-1, 1, size=(2,))
            ],
            axis=0
        ) * 0.1 * self.screen_dims.max()
        return total_force

    def get(self) -> tuple[bool, tuple[float, float]]:
        now_time = time.time()
        if self.ball_time is None:
            self.ball_time = now_time
        dt = now_time - self.ball_time
        if dt >= FeedbackBallSourceObject.MIN_TIME_BETWEEN_UPDATES:
            # Update ball properties
            self.ball_time = now_time
            new_ball_vel = self.ball_vel + self._get_force_vector_on_ball()*dt
            if np.linalg.norm(new_ball_vel) > self.max_speed:
                new_ball_vel = self.max_speed * (new_ball_vel / np.linalg.norm(new_ball_vel))
            elif np.linalg.norm(new_ball_vel) < self.min_speed:
                new_ball_vel = self.ball_vel
            self.ball_vel = new_ball_vel
            self.ball_pos += self.ball_vel*dt

            # Deal with hitting the edge of the screen
            out_of_bounds_mask = (self.ball_pos < 0) | (self.ball_pos > self.screen_dims)
            self.ball_pos = np.clip(self.ball_pos, 0, self.screen_dims)
            self.ball_vel *= np.where(out_of_bounds_mask, -1, 1)

        return (True, self.ball_pos.astype(np.int32))

class ClickListenerSourceObject(SourceObject):
    """
    Records where the mouse clicks.
    """

    def __init__(self, timeout: int, buttons = [1, 2]):
        super().__init__(timeout, False)
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
        
        x, y = win32api.GetCursorPos()
        return (button != -1, np.array([y, x]))
