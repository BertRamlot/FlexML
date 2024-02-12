import numpy as np
import cv2
import uuid
import collections
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from FlexML.MetadataSample import MetadataSample, MetadataSampleCollection


class GazeSample(MetadataSample):
    """
    Sample defined by an image (normally just a webcam frame), a screen position, and the corresponding screen dimensions.
    """

    def __init__(self, type: str, screen_position: tuple[int|float, int|float], screen_dims: np.ndarray, img_path: Path | None, img: np.ndarray | None, creation_time: float = None):
        super().__init__(type, screen_position, creation_time)
        self.img_path = img_path
        self._img = img
        self.screen_dims = screen_dims # [h, w]

    def get_img(self) -> np.ndarray:
        """
        Retrieves the image data as a numpy array. Loaded from disk using the 'img_path' if not yet loaded.

        Returns:
            np.ndarray: A numpy array representing the image data.
                
        Raises:
            RuntimeError: If neither '_img' nor 'img_path' has been specified.
        """
        if self._img is None:
            if self.img_path is None:
                raise RuntimeError("'_img' or 'img_path' should be specified")
            self._img = np.asarray(cv2.imread(str(self.img_path)))
        return self._img
    
    def get_metadata(self) -> list:
        if self.ground_truth is None:
            y, x = (None, None)
        else:
            y, x = self.ground_truth
        h, w = self.screen_dims
        return [self.img_path.name, self.type, self.creation_time, h, w, y, x]
    

class GazeSampleCollection(MetadataSampleCollection):
    
    JPEG_QUALITY = 70

    def __init__(self, path: Path):
        super().__init__(path)
    
    def from_metadata(self, metadata) -> GazeSample:
        img_name, type, creation_time, win_y, win_x, pos_y, pos_x = metadata[:7]
        img_path = self.dataset_path / "raw" / Path(img_name)
        return GazeSample(type, (pos_y, pos_x), np.array([win_y, win_x]), img_path, None, creation_time)
    
    def get_metadata_headers(self) -> list[str]:
        return ["type", "img_name", "creation_time", "win_y", "win_x", "y", "x"]

    @pyqtSlot(MetadataSample)
    def add_sample(self, sample: MetadataSample):
        """
        Adds a MetadataSample to the collection by adding an entry to the csv and saving the gaze image.
        
        Args:
            sample (MetadataSample): Sample to be added.
        """
        images_dir_path = self.dataset_path / "raw"
        images_dir_path.mkdir(parents=True, exist_ok=True)
        if sample.img_path is None:
            sample.img_path = images_dir_path / f"{uuid.uuid4()}.jpg"

        cv2.imwrite(
            str(sample.img_path),
            sample.get_img(),
            [int(cv2.IMWRITE_JPEG_QUALITY), GazeSampleCollection.JPEG_QUALITY]
        )

        super().add_sample(sample)

class GazeSampleMuxer(QObject):
    """
    Creates and emits GazeSamples from its subcomponents.
    """
    new_sample = pyqtSignal(GazeSample)

    def __init__(self, type_supplier: collections.abc.Callable, screen_dims: np.ndarray):
        """
        Initialize the GazeSampleMuxer.

        Args:
            type_supplier (collections.abc.Callable): A callable object that returns the type for the next sample when called (w/ 0 args).
            screen_dims (np.ndarray): The dimensions of the screen, shape is (2,) with format (height, width).
        """
        super().__init__()
        self.last_ground_truth = None
        self.type_supplier = type_supplier
        self.screen_dims = screen_dims

    @pyqtSlot(np.ndarray)
    def update_ground_truth(self, ground_truth: np.ndarray):
        """
        Update the last known ground truth position of the gaze.

        Args:
            ground_truth (np.ndarray): An array representing the last known ground truth position, shape is (2,) with format (y, x)
        """
        self.last_ground_truth = ground_truth
    
    @pyqtSlot(np.ndarray)
    def update_img(self, img: np.ndarray):
        """
        Creates and emits a new gaze sample signal using the provided image data.

        Args:
            img (np.ndarray): An array representing the image data.
        """
        type = None if self.last_ground_truth is None else self.type_supplier()
        sample = GazeSample(type, self.last_ground_truth, self.screen_dims, None, img)
        self.new_sample.emit(sample)
        self.last_ground_truth = None
