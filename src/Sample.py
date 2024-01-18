import random
import cv2
import numpy as np
from pathlib import Path
import csv
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot


class Sample():
    def __init__(self, img_path: str, img: np.ndarray, type: str, gt_pos: tuple[float, float]):
        self.img_path = img_path
        # TODO
        self.saved = img_path is not None
        self._img = img
        self.type = type if type else random.choices(["train", "val", "test"], [0.7, 0.2, 0.1])[0]
        self.gt_pos = gt_pos

    def get_img(self) -> np.ndarray:
        if self._img is None:
            if self.img_path is None:
                raise RuntimeError("'_img' or 'img_path' should be specified")
            self._img = np.asarray(cv2.imread(self.img_path))
        return self._img
    
    def save(self, dataset_path: Path):
        if self.saved:
            return
        
        # TODO
        JPEG_QUALITY = 50

        images_dir_path = dataset_path / "raw"
        images_dir_path.mkdir(parents=True, exist_ok=True)
        if self.img_path is None:
            x, y = self.gt_pos
            file_name = None
            i = 0
            while file_name is None or (images_dir_path / file_name).is_file():
                file_name = "{}_{}_{}.jpg".format(f"{x:.1f}", f"{y:.1f}", i)
                i += 1
            self.img_path = file_name

        absolute_img_path = (images_dir_path / self.img_path).absolute()
        cv2.imwrite(str(absolute_img_path), self.get_img(), [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

        metadata_path = dataset_path / "metadata.csv"
        file_exists = metadata_path.is_file()
        with open(metadata_path, "a+", newline="", encoding="UTF8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(self.get_metadata_headers())
            writer.writerow(self.get_metadata())
        
        self.saved = True


    def get_extra_metadata(self) -> list:
        return []
    
    def get_extra_metadata_headers(self) -> list[str]:
        return []

    def get_metadata(self) -> list:
        return [self.img_path, self.type, self.gt_pos[0], self.gt_pos[1]] + self.get_extra_metadata()
    
    def get_metadata_headers(self) -> list[str]:
        return ["img_path", "type", "x", "y"] + self.get_extra_metadata_headers()

# SampleMuxer?
class SampleGenerator(QObject):
    new_sample = pyqtSignal(Sample)

    def __init__(self):
        super().__init__(None)
        self.last_img = None
        self.last_label = None

    @pyqtSlot(np.ndarray)
    def set_last_label(self, label: np.ndarray, publish: bool = False):
        self.last_label = label
        if publish:
            self.publish_sample()

    @pyqtSlot(np.ndarray)
    def set_last_img(self, img: np.ndarray, publish: bool = True):
        self.last_img = img
        if publish:
            self.publish_sample()

    def publish_sample(self):
        sample = Sample(None, self.last_img, None, self.last_label)
        self.new_sample.emit(sample)

class DatasetDrain(QObject):
    def __init__(self, dataset_path: Path):
        super().__init__(None)
        self.dataset_path = dataset_path
    
    @pyqtSlot(Sample)
    def sink(self, obj: object):
        obj.save(self.dataset_path)
