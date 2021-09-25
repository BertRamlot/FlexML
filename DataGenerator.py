import os
import cv2
import ctypes
import random
import csv
import cv2

from FaceDetector import FaceDetector

class DataGenerator():
    def __init__(self, data_folder_name: str, buffer_size: int):
        self.img_folder = "{}/raw/".format(data_folder_name)
        self.all_csv_path = "{}/meta_data.csv".format(data_folder_name)
        if not os.path.exists(self.img_folder):
            os.makedirs(self.img_folder)

        self.buffer_size = buffer_size
        self._buffer = []

        user32 = ctypes.windll.user32
        self.screen_width, self.screen_heigth = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        
        print("Starting camera")
        self.cap = cv2.VideoCapture(0)
        self.face_detector = FaceDetector(self.cap)
    
    def exit(self):
        self.cap.release()

    def flush(self) -> None:
        while self.buffer_length() > 0:
            self.save_oldest_buffer_item()

    def clear_buffer(self) -> None:
        self._buffer = []

    def buffer_length(self) -> int:
        return len(self._buffer)

    @staticmethod
    def generate_filename(parent_dir: str, x: int, y: int):
        path = None
        i = 0
        while path is None or os.path.exists(parent_dir + path):
            path = "{}_{}_{}.jpg".format(x, y, i)
            i += 1
        return path

    def save_oldest_buffer_item(self) -> None:
        face, x, y = self._buffer[0]
        self._buffer = self._buffer[1:]

        face_img_path = DataGenerator.generate_filename(self.img_folder, x, y)

        # Saving raw images
        cv2.imwrite(self.img_folder + face_img_path, face.im)

        # Saving meta data to csv
        prec = 6
        training_percentage = 0.9
        meta_data = [
            0 if random.random() < training_percentage else 1,
            face_img_path,
            round(x/self.screen_width,  prec),
            round(y/self.screen_heigth, prec),
            round(face.tl_rx, prec),
            round(face.tl_ry, prec),
            round(face.rw, prec),
            round(face.rh, prec)
        ]
        meta_data += [round(rx, prec) for rx in face.features_rx]
        meta_data += [round(ry, prec) for ry in face.features_ry]

        headers = ['testing', 'face_file_name', 'x_screen', 'y_screen', 'tl_rx', 'tl_ry', 'rw', 'rh']
        headers += ['fx_{}'.format(i) for i in range(len(face.features_rx))]
        headers += ['fy_{}'.format(i) for i in range(len(face.features_ry))]
        
        file_exists = os.path.isfile(self.all_csv_path)
        with open(self.all_csv_path, "a+", newline='',encoding="UTF8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(headers)
            writer.writerow(meta_data)

    # return true if valid eyepair(s) found
    def register_eye_position(self, x: int, y: int) -> bool:
        self.face_detector.update()

        if not self.face_detector.valid_faces_found():
            return False

        for face in self.face_detector.last_faces:
            self._buffer.append((face, x, y))

        if self.buffer_length() >= self.buffer_size:
            self.save_oldest_buffer_item()
        
        return True
