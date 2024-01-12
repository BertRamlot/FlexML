import sys
from pathlib import Path
from argparse import ArgumentParser
from PyQt5 import QtWidgets

from src.SourceThread import SimpleBallSourceThread, WebcamSourceThread, DatasetSource
from src.ModelThread import ModelThread
from src.Sample import SampleGenerator

from src.face_based.FaceSample import FaceSampleConvertor


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
    parser.add_argument("--load_datasets", type=str, nargs="*", default=None)
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

    # Create dataset loaders
    if args.load_datasets:
        sample_sources = [DatasetSource(load_dataset_path) for load_dataset_path in args.load_datasets]
    else:
        sample_sources = []


    # Create model thread
    if args.model and args.dataset:
        model_thread = ModelThread(
            Path("models") / args.model,
            args.model_type,
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

    # Start and finish disk sources
    for sample_source in sample_sources:
        sample_source.start()
    for sample_source in sample_sources:
        sample_source.wait()

    # Start live sources
    if gt_src_thread:
        gt_src_thread.start()
    if img_src_thread:
        img_src_thread.start()

    # Event loop
    sys.exit(app.exec_())
