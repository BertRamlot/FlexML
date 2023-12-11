import cv2
import time
import random
import csv
import numpy as np
from pathlib import Path

from src.FaceDetector import FaceDetector

class DataGenerator():
    def __init__(self, dataset_path: Path, buffer_size: int):
        self.images_dir_path = dataset_path / "raw"
        self.images_dir_path.mkdir(parents=True, exist_ok=True)
        self.meta_data_path = dataset_path / "meta_data.csv"

        self.start_time = time.time()
        self.last_register_time = 0
        self.buffer_size = buffer_size
        self._buffer = []

    def flush(self) -> None:
        while len(self._buffer) > 0:
            self.save_oldest_buffer_item()

    def clear_buffer(self) -> None:
        self._buffer = []


    @staticmethod
    def generate_filename(parent_path: Path, x: int, y: int):
        file_name = None
        i = 0
        while file_name is None or (parent_path / file_name).is_file():
            file_name = f"{x:.5f}_{y:.5f}_{i}.jpg"
            i += 1
        return file_name

    def save_oldest_buffer_item(self) -> None:
        face, x, y = self._buffer[0]
        self._buffer = self._buffer[1:]

        face_img_name = DataGenerator.generate_filename(self.images_dir_path, x, y)

        # Saving raw images
        cv2.imwrite(str((self.images_dir_path / face_img_name).absolute()), face.im)

        # Saving meta data to csv
        prec = 6
        training_percentage = 0.9
        meta_data = [
            0 if random.random() < training_percentage else 1,
            face_img_name,
            round(x, prec),
            round(y, prec),
            round(face.tl_rx, prec),
            round(face.tl_ry, prec),
            round(face.rw, prec),
            round(face.rh, prec)
        ]
        meta_data += [round(rx, prec) for rx in face.features_rx]
        meta_data += [round(ry, prec) for ry in face.features_ry]

        file_exists = self.meta_data_path.is_file()
        with open(self.meta_data_path, "a+", newline="", encoding="UTF8") as f:
            writer = csv.writer(f)
            if not file_exists:
                headers  = ["testing", "face_file_name", "x_screen", "y_screen", "tl_rx", "tl_ry", "rw", "rh"]
                headers += [f"fx_{i}" for i in range(len(face.features_rx))]
                headers += [f"fy_{i}" for i in range(len(face.features_ry))]
                writer.writerow(headers)
            writer.writerow(meta_data)

    def register_sample(self, face, x: int, y: int):
        self._buffer.append((face, x, y))

        if len(self._buffer) >= self.buffer_size:
            self.save_oldest_buffer_item()

        self.last_register_time = time.time()

    def get_target_position(self):
        raise NotImplementedError()

class BallDataGenerator(DataGenerator):
    def __init__(self, dataset_path: Path, buffer_size: int):
        DataGenerator.__init__(self, dataset_path, buffer_size)
        self.ball_pos = np.array([0.5, 0.5], dtype=np.float32)
        self.ball_vel = np.array([0.15, 0.15], dtype=np.float32)
        
        self.last_update_time = None

    # Override        
    def get_target_position(self):
        if self.last_update_time is None:
            self.last_update_time = time.time()
        dt = time.time() - self.last_update_time
        if dt > 0.01:
            # Update ball position
            self.ball_pos += self.ball_vel*dt
            out_of_bounds_mask = (self.ball_pos < 0) | (self.ball_pos > 1)
            self.ball_vel *= np.where(out_of_bounds_mask, -1, 1)
            self.ball_pos = np.clip(self.ball_pos, 0, 1)
            if out_of_bounds_mask.any():
                rnd_angle = random.random()*np.pi/2
                vel_mag = np.linalg.norm(self.ball_vel)
                self.ball_vel = np.copysign(vel_mag * np.array([np.cos(rnd_angle), np.sin(rnd_angle)]), self.ball_vel)

            self.last_update_time = time.time()

        time_since_start = time.time() - self.start_time
        time_since_register = time.time() - self.last_register_time
        return (self.ball_pos, time_since_register > 0.5 and time_since_start > 5)
