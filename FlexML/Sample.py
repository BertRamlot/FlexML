import time
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot


class Sample():
    """Base class for ML samples."""

    def __init__(self, type: str | None, ground_truth: object | None, creation_time: float = None):
        self.type = type
        self.ground_truth = ground_truth
        self.creation_time = time.time() if creation_time is None else creation_time

class SampleCollection(QObject):
    """Collection of samples."""

    new_sample = pyqtSignal(Sample)
    
    def __init__(self, dataset_path: Path):
        super().__init__()
        self.dataset_path = dataset_path
    
    @pyqtSlot(Sample)
    def add_sample(self, sample: Sample):
        raise NotImplementedError()
