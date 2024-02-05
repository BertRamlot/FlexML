from pathlib import Path
import random
from PyQt6.QtCore import QObject, pyqtSignal


class Sample():
    """Base class for ML samples."""

    def __init__(self, type: str, gt: object):
        self.type = type if type else random.choices(["train", "val", "test"], [0.8, 0.15, 0.05])[0]
        self.gt = gt


class SampleCollection(QObject):
    new_sample = pyqtSignal(Sample)
    
    def __init__(self, dataset_path: Path):
        super().__init__()
        self.dataset_path = dataset_path
    
    def add_sample(self, sample: Sample):
        raise NotImplementedError()
