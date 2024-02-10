import numpy as np
import cv2
import uuid
import collections
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from FlexML.MetadataSample import MetadataSample, MetadataSampleCollection


class GazeSample(MetadataSample):
    """Sample defined by an image (normally just a webcam frame), a screen position, and the corresponding screen dimensions."""

    def __init__(self, type: str, screen_position: tuple[int|float, int|float], screen_dims: np.ndarray, img_path: Path | None, img: np.ndarray | None):
        super().__init__(type, screen_position)
        self.img_path = img_path
        self._img = img
        self.screen_dims = screen_dims

    def get_img(self) -> np.ndarray:
        if self._img is None:
            if self.img_path is None:
                raise RuntimeError("'_img' or 'img_path' should be specified")
            self._img = np.asarray(cv2.imread(str(self.img_path)))
        return self._img
    
    def get_metadata(self) -> list:
        if self.gt is None:
            x, y = (None, None)
        else:
            x, y = self.gt
        return [self.img_path.name, self.type, *self.screen_dims, x, y]
    

class GazeSampleCollection(MetadataSampleCollection):

    def __init__(self, path: Path):
        super().__init__(path)
    
    def from_metadata(self, metadata) -> GazeSample:
        img_name, type, win_x, win_y, pos_x, pos_y = metadata[:6]
        img_path = self.dataset_path / "raw" / Path(img_name)
        return GazeSample(type, (pos_x, pos_y), np.array([win_x, win_y]), img_path, None)
    
    def get_metadata_headers(self) -> list[str]:
        return ["type", "img_name", "win_x", "win_y", "x", "y"]

    @pyqtSlot(MetadataSample)
    def add_sample(self, sample: MetadataSample):
        images_dir_path = self.dataset_path / "raw"
        images_dir_path.mkdir(parents=True, exist_ok=True)
        if sample.img_path is None:
            sample.img_path = images_dir_path / f"{uuid.uuid4()}.jpg"

        jpeg_quality: int = 50
        cv2.imwrite(
            str(sample.img_path),
            sample.get_img(),
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        )

        super().add_sample(sample)

class GazeSampleMuxer(QObject):    
    new_sample = pyqtSignal(GazeSample)

    def __init__(self, type_supplier: collections.abc.Callable, screen_dims: np.ndarray):
        super().__init__()
        self.last_img = None
        self.last_label = None
        self.type_supplier = type_supplier
        # You could also make this dynamic like 'img' and 'label' if needed
        self.screen_dims = screen_dims

    @pyqtSlot(np.ndarray)
    def set_last_label(self, label: np.ndarray):
        self.last_label = label
    
    @pyqtSlot(np.ndarray)
    def set_last_img(self, img: np.ndarray):
        self.last_img = img
        
        sample = GazeSample(self.type_supplier(), self.last_label, self.screen_dims, None, self.last_img)
        self.new_sample.emit(sample)
