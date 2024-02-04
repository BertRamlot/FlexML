import numpy as np
import ctypes
import cv2
import uuid
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from FlexML.MetadataSample import MetadataSample, MetadataSampleCollection


class GazeSample(MetadataSample):
    """Sample defined by an image (normally just a webcam frame), a screen position, and the corresponding screen dimensions."""

    def __init__(self, img_path: Path, img: np.ndarray, type: str, gt: object, window_dims = None):
        super().__init__(type, gt)
        self.img_path = img_path
        self._img = img
        if window_dims is None:
            # TODO
            self.window_dims = np.array([ctypes.windll.user32.GetSystemMetrics(i) for i in range(2)], dtype=np.int32)
        else:
            self.window_dims = np.asarray(window_dims)

    def get_img(self) -> np.ndarray:
        if self._img is None:
            if self.img_path is None:
                raise RuntimeError("'_img' or 'img_path' should be specified")
            self._img = np.asarray(cv2.imread(str(self.img_path)))
        return self._img
    
    def get_metadata(self) -> list:
        x, y = self.gt
        return [self.img_path.name, self.type, *self.window_dims, x, y]
    

class GazeSampleCollection(MetadataSampleCollection):

    def __init__(self, path: Path):
        super().__init__(path)
    
    def from_metadata(self, metadata) -> GazeSample:
        img_name, type, win_x, win_y, pos_x, pos_y = metadata[:6]
        img_path = self.dataset_path / "raw" / Path(img_name)
        return GazeSample(img_path, None, type, (pos_x, pos_y), window_dims=(win_x, win_y))
    
    def get_metadata_headers(self) -> list[str]:
        return ["type", "img_name", "win_x", "win_y", "x", "y"] + self.get_extra_metadata_headers()

    def save_sample(self, sample: GazeSample, jpeg_quality: int = 50):

        images_dir_path = self.dataset_path / "raw"
        images_dir_path.mkdir(parents=True, exist_ok=True)
        if self.img_path is None:
            self.img_path = f"{uuid.uuid4()}.jpg"

        cv2.imwrite(
            str((images_dir_path / self.img_path).absolute()),
            sample.get_img(),
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        )


class GazeSampleMuxer(QObject):
    new_sample = pyqtSignal(GazeSample)

    def __init__(self):
        super().__init__(None)
        self.last_img = None
        self.last_label = None

    @pyqtSlot(np.ndarray)
    def set_last_label(self, label: np.ndarray):
        self.last_label = label
    
    @pyqtSlot(np.ndarray)
    def set_last_img(self, img: np.ndarray):
        self.last_img = img
        
        sample = GazeSample(None, self.last_img, None, self.last_label)
        self.new_sample.emit(sample)
