import sys
import os
from pathlib import Path
from argparse import ArgumentParser
from PyQt6 import QtWidgets

from FlexML.SourceThread import WebcamSourceThread, DatasetSource
from FlexML.ModelThread import ModelElement, ModelController
from FlexML.Sample import SampleMuxer
from FlexML.Helper import BufferThread, AttributeSelector
from FlexML.Sample import DatasetDrain

from examples.eye_tracker.src.EyeTrackingOverlay import EyeTrackingOverlay
from examples.eye_tracker.src.TargetSource import SimpleBallSourceThread
from examples.eye_tracker.src.FaceSample import FaceSampleConvertor
from examples.eye_tracker.src.FaceNetwork import FaceSampleToTensor

from FlexML.Linker import link_elements


def get_source_worker(uid: str|None):
    if uid is None:
        return None
    elif uid == "simple-ball":
        # Low timeout to keep GUI stutters to a minimum
        return SimpleBallSourceThread(0.01)
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
    args = parser.parse_args(sys.argv[1:])

    module_directory = Path(__file__).resolve().parent


    app = QtWidgets.QApplication(sys.argv)

    # Create overlay
    overlay = EyeTrackingOverlay()

    # Create model thread
    if args.model:
        model_controller = ModelController(

        )
        model = ModelElement(
            module_directory / Path("models") / args.model,
            args.model_type,
            args.device,
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
    sample_generator = SampleMuxer()
    sample_generator.moveToThread(img_src_thread)


    import queue
    sample_buffer = BufferThread(queue.Queue(5))
    sample_convertor = FaceSampleConvertor(module_directory / "shape_predictor_68_face_landmarks.dat")
    sample_convertor.moveToThread(sample_buffer)
    face_sample_to_tensor = FaceSampleToTensor(args.device)

    dataset_drain = DatasetDrain(module_directory / Path("datasets") / args.save_dataset) if args.save_dataset else None

    # Live pipeline
    link_elements(img_src_thread, ("set_last_img", sample_generator), sample_buffer)
    link_elements(gt_src_thread, ("set_last_label", sample_generator))
    link_elements(sample_buffer, sample_convertor, dataset_drain)
    link_elements(sample_convertor, face_sample_to_tensor, model, overlay)
    link_elements(gt_src_thread, overlay)


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
