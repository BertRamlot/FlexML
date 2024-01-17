import sys
from pathlib import Path
from argparse import ArgumentParser
from PyQt6 import QtWidgets

from src.EyeTrackingOverlay import EyeTrackingOverlay
from src.SourceThread import SimpleBallSourceThread, WebcamSourceThread, DatasetSource
from src.ModelThread import ModelElement, ModelController
from src.Sample import SampleGenerator

from src.face_based.FaceSample import FaceSampleConvertor
from src.face_based.FaceNetwork import FaceSampleToTensor
from src.Sample import DatasetDrain

from src.Linker import link_elements


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
    parser.add_argument("--load_datasets", type=str, nargs="*", default=None)
    parser.add_argument("--save_dataset", type=str, default=None)
    parser.add_argument("--gt_source", type=str, default=None, choices=["simple-ball"])
    parser.add_argument("--img_source", type=str, default=None, choices=["webcam"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="face", choices=["face"])
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument("--live_train", type=bool, default=False)
    args = parser.parse_args(sys.argv[1:])


    app = QtWidgets.QApplication(sys.argv)

    # Create overlay
    overlay = EyeTrackingOverlay()

    # Create model thread
    if args.model:
        model_controller = ModelController(

        )
        model = ModelElement(
            Path("models") / args.model,
            args.model_type,
            args.device,
            args.live_train,
            [
                {
                    "type": "train",
                    "dataloader_kwargs": {
                        "batch_size": 32,
                        "shuffle": True
                    }
                },
                {
                    "type": "val"
                },
                {
                    "type": "test"
                }
            ]
        )
    else:
        model_controller = None
        model = None

    gt_src_thread = get_source_worker(args.gt_source) 
    img_src_thread = get_source_worker(args.img_source)
    sample_generator = SampleGenerator()
    
    sample_convertor = FaceSampleConvertor()
    face_sample_to_tensor = FaceSampleToTensor(args.device)

    dataset_drain = DatasetDrain(Path("datasets") / args.save_dataset) if args.save_dataset else None

    # Live pipeline
    link_elements(img_src_thread, sample_generator, sample_convertor, face_sample_to_tensor, model, overlay)
    link_elements(gt_src_thread, sample_generator)
    link_elements(sample_convertor, dataset_drain)
    # link_elements(model_thread, gt_src_thread)

    if model_controller:
        model_controller.start()
    
    # Load all data from disk
    if args.load_datasets:
        dataset_sources = [DatasetSource(load_dataset_path) for load_dataset_path in args.load_datasets]
        for dataset_source in dataset_sources:
            link_elements(dataset_source, sample_convertor)
        for dataset_source in dataset_sources:
            dataset_source.start()
        for dataset_source in dataset_sources:
            dataset_source.wait()

    # Start live sources
    if gt_src_thread:
        gt_src_thread.start()
    if img_src_thread:
        img_src_thread.start()

    # Event loop
    sys.exit(app.exec())
