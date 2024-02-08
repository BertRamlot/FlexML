import uuid
import random
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot


class Sample():
    """Base class for ML samples."""

    def __init__(self, type: str | None, gt: object | None):
        self.type = type if type is not None else random.choices(["train", "val", "test"], [0.8, 0.15, 0.05])[0]
        self.gt = gt

class SampleCollection(QObject):
    """Collection of samples."""
    new_sample = pyqtSignal(Sample)
    
    def __init__(self, dataset_path: Path):
        super().__init__()
        self.dataset_path = dataset_path
    
    @pyqtSlot(Sample)
    def add_sample(self, sample: Sample):
        raise NotImplementedError()
