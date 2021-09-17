import os
import cv2
import ctypes
import random
import csv

from numpy import eye

from EyeDetector import EyeDetector

class DataGenerator():
    def __init__(self, buffer_size: int):
        self.img_folder = "data/raw/"
        self.training_csv_path = "data/training_meta_data.csv"
        self.testing_csv_path = "data/testing_meta_data.csv"
        if not os.path.exists(self.img_folder):
            os.makedirs(self.img_folder)

        self.buffer_size = buffer_size
        self._buffer = []

        user32 = ctypes.windll.user32
        self.screen_width, self.screen_heigth = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

        self.eye_detector = EyeDetector()
    
    def flush(self) -> None:
        while self.buffer_length() > 0:
            self.save_oldest_buffer_item()

    def clear_buffer(self) -> None:
        self._buffer = []

    def buffer_length(self) -> int:
        return len(self._buffer)

    def save_oldest_buffer_item(self) -> None:
        eye_pair, x, y = self._buffer[0]
        self._buffer = self._buffer[1:]

        left_eye_img_path = "L_{}_{}.jpg".format(x,y)
        right_eye_img_path = "R_{}_{}.jpg".format(x,y)


        meta_data = [
            left_eye_img_path, right_eye_img_path,
            round(x/self.screen_width,4),
            round(y/self.screen_heigth,4), 
            round(eye_pair.left_eye.x/self.screen_width,4), 
            round(eye_pair.left_eye.y/self.screen_heigth,4), 
            round(eye_pair.left_eye.w/self.screen_width,4), 
            round(eye_pair.left_eye.h/self.screen_heigth,4),
            round(eye_pair.right_eye.x/self.screen_width,4), 
            round(eye_pair.right_eye.y/self.screen_heigth,4), 
            round(eye_pair.right_eye.w/self.screen_width,4), 
            round(eye_pair.right_eye.h/self.screen_heigth,4)
            ]

        # Saving meta data to csv
        csv_path = self.training_csv_path if random.random() < 0.9 else self.testing_csv_path
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a+", newline='',encoding="UTF8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'left_eye_file_name', 'right_eye_file_name', 'x_screen', 'y_screen', 
                    'l_x', 'l_y', 'l_w', 'l_h',
                    'r_x', 'r_y', 'r_w', 'r_h'
                ])
            writer.writerow(meta_data)

        # Saving raw images
        cv2.imwrite(self.img_folder + left_eye_img_path, eye_pair.left_eye.im)
        cv2.imwrite(self.img_folder + right_eye_img_path, eye_pair.left_eye.im)
    
    
    def register_eye_position(self, x: int, y: int) -> bool: # return true if valid eyepair(s) found
        self.eye_detector.update()

        if not self.eye_detector.valid_eyes_found():
            return False

        for eye_pair in self.eye_detector.last_eye_pairs:
            self._buffer.append((eye_pair, x, y))

        if self.buffer_length() >= self.buffer_size:
            self.save_oldest_buffer_item()
        
        return True
