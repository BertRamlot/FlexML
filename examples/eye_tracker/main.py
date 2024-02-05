import sys
import os
from pathlib import Path
from argparse import ArgumentParser
from PyQt6 import QtWidgets
from PyQt6.QtCore import QThread

from FlexML.SourceThread import WebcamSourceThread
from FlexML.Model import ModelElement, ModelController
from FlexML.Helper import BufferThread, Filter

from examples.eye_tracker.src.EyeTrackingOverlay import EyeTrackingOverlay
from examples.eye_tracker.src.TargetSource import SimpleBallSourceThread
from examples.eye_tracker.src.FaceSample import GazeToFaceSampleConvertor
from examples.eye_tracker.src.FaceNetwork import FaceSampleToTrainPair
from examples.eye_tracker.src.GazeSample import GazeSampleMuxer, GazeSampleCollection
from examples.eye_tracker.src.FaceSample import FaceSampleCollection

from FlexML.Linker import link_elements


def get_source_worker(uid: str|None):
    if uid is None:
        return None
    elif uid == "simple-ball":
        # Low timeout to keep GUI stutters to a minimum
        return SimpleBallSourceThread(0.02)
    elif uid == "webcam":
        # Slight timeout to prevent to many samples that are near equal
        return WebcamSourceThread(1)
    else:
        raise LookupError("Invalid source worker uid:", uid)


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
        model_element = ModelElement(
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
        model_controller = ModelController(model_element)
        model_element.moveToThread(model_controller)
    else:
        model_controller = None
        model_element = None

    gt_src_thread = get_source_worker(args.gt_source) 
    img_src_thread = get_source_worker(args.img_source)
    sample_muxer = GazeSampleMuxer()
    # sample_muxer.moveToThread(img_src_thread)


    import queue
    sample_buffer = BufferThread(queue.Queue(5))
    gaze_to_face_sample = GazeToFaceSampleConvertor(module_directory / "shape_predictor_68_face_landmarks.dat")
    gaze_to_face_sample.moveToThread(sample_buffer)
    face_sample_to_train_pair = FaceSampleToTrainPair(args.device)

    # Live pipeline
    link_elements(gt_src_thread,  ("set_last_label", sample_muxer))
    link_elements(img_src_thread, ("set_last_img",   sample_muxer), sample_buffer, gaze_to_face_sample)
    link_elements(gaze_to_face_sample, face_sample_to_train_pair, model_controller, overlay)
    link_elements(gt_src_thread, overlay)

    # Save data to disk
    if args.save_dataset:
        save_path = module_directory / Path("datasets") / args.save_dataset
        print(f"Saving new samples in: {save_path}")
        filt = Filter(lambda s: s.gt is not None)
        save_coll = FaceSampleCollection(module_directory / Path("datasets") / args.save_dataset)
        link_elements(
            gaze_to_face_sample, 
            filt, 
            save_coll
        )
    else:
        print("Not saving new samples")
    
    # Load all data from disk
    if args.load_datasets:
        total_published_samples = 0
        load_data_colls = [FaceSampleCollection(module_directory / "datasets" / load_dataset_path) for load_dataset_path in args.load_datasets]
        for dataset_source in load_data_colls:
            link_elements(dataset_source, face_sample_to_train_pair)
        for dataset_source in load_data_colls:
            total_published_samples += dataset_source.publish_all_samples()
        print(f"Loaded {total_published_samples} samples.")
        # TODO: wait for all samples to be processes before starting the live threads?
    else:
        print("Not loading any datasets")
    
    if model_controller:
        model_controller.start()

    # Start live sources
    if img_src_thread:
        img_src_thread.start()
    if gt_src_thread:
        gt_src_thread.start()
 
    # Event loop
    print("Starting GUI loop")
    sys.exit(app.exec())
