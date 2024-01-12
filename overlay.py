import sys
from pathlib import Path
from argparse import ArgumentParser
import torch
from PyQt5 import QtWidgets
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("Tensorboard found")
except ImportError:
    TENSORBOARD_FOUND = False
    print("Tensorboard not found")

from src.SourceThread import SimpleBallSourceThread, WebcamSourceThread
from src.ModelThread import ModelThread
from src.DatasetGenerator import SampleConvertor
from src.Sample import Sample

from src.face_based.FaceNetwork import FaceDataset
from src.face_based.FaceSample import FaceSampleConvertor

class SampleGenerator(QObject):
    new_sample = pyqtSignal(Sample)

    def __init__(self, dataset_path: Path, sample_convertor: SampleConvertor):
        self.dataset_path = dataset_path
        self.sample_convertor = sample_convertor
        self.last_img = None
        self.last_label = None

    @pyqtSlot(np.ndarray)
    def set_last_label(self, label: np.ndarray):
        self.last_label = label

    @pyqtSlot(np.ndarray)
    def set_last_img(self, img: np.ndarray):
        self.last_img = img
        sample = Sample(None, self.last_img, None, self.last_label)
        converted_samples = self.sample_convertor.convert_sample(sample)
        for sample in converted_samples:
            sample.save(self.dataset_path)
            self.new_sample.emit(sample)

def get_source_worker(uid: str|None):
    if uid is None:
        return None
    elif uid == "simple-ball":
        # Low timeout to keep GUI stutters to a minimum
        return SimpleBallSourceThread(0.03)
    elif uid == "webcam":
        # Slight timeout to prevent to many samples that are near equal
        return WebcamSourceThread(0.2)
    else:
        raise LookupError("Invalid source worker uid:", uid)

def get_network(uid: str):
    pass


if __name__ == "__main__":
    parser = ArgumentParser(description="Overlay script parameters")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--load_datasets", type=str, nargs="+", default=None)
    parser.add_argument("--save_dataset", type=str, default=None)
    parser.add_argument("--gt_source", type=str, default="simple-ball", choices=["simple-ball"])
    parser.add_argument("--img_source", type=str, default="webcam", choices=["webcam"])
    # Model selection
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model_type", type=str, defult="face", choices=["face"])
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument('--device', type=str, default="cuda")
    # Model training
    parser.add_argument("--live_train", type=bool, default=False)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args(sys.argv[1:])

    app = QtWidgets.QApplication([])

    # Create overlay
    from src.EyeTrackingOverlay import EyeTrackingOverlay
    window = EyeTrackingOverlay()
    window.showFullScreen()

    # Create model thread
    if args.model and args.dataset:
        model_thread = ModelThread(
            Path("models") / args.model,
            args.model_type,
            args.load_datasets,
            args.device,
            args.live_train
        )
        model_thread.predicted_samples.connect(window.register_inference_positions)
    else:
        model_thread = None

    # Create source threads
    if args.dataset:
        print("Dataset passed, generating data")
        gt_src_thread = get_source_worker(args.gt_source) 
        img_src_thread = get_source_worker(args.img_source) 

        dataset_path = Path("datasets") / args.dataset
        sample_generator = SampleGenerator(dataset_path, FaceSampleConvertor())
        if gt_src_thread:
            gt_src_thread.new_item.connect(sample_generator.set_last_label)
            gt_src_thread.new_item.connect(window.register_gt_position)
        if img_src_thread:
            img_src_thread.new_item.connect(sample_generator.set_last_img)


    if sample_generator and model_thread:
        sample_generator.new_sample.connect(model_thread.add_sample)

    if model_thread:
        model_thread.start()

    if gt_src_thread:
        gt_src_thread.start()
    if img_src_thread:
        img_src_thread.start()

    # Event loop
    sys.exit(app.exec_())
