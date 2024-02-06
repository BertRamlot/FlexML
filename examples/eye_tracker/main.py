import sys
import signal
from pathlib import Path
from argparse import ArgumentParser
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QCoreApplication, QEventLoop

from FlexML.SourceThread import WebcamSourceThread
from FlexML.Model import ModelElement, ModelController
from FlexML.Helper import BufferThread, Filter, Convertor
from FlexML.Linker import link_elements

from examples.eye_tracker.src.EyeTrackingOverlay import EyeTrackingOverlay
from examples.eye_tracker.src.TargetSource import SimpleBallSourceThread
from examples.eye_tracker.src.FaceSample import GazeToFaceSampleConvertor
from examples.eye_tracker.src.FaceNetwork import FaceSampleToTrainPair, face_sample_to_X_tensor
from examples.eye_tracker.src.GazeSample import GazeSampleMuxer
from examples.eye_tracker.src.FaceSample import FaceSampleCollection


def get_source_worker(uid: str|None):
    if uid is None:
        return None
    elif uid == "simple-ball":
        # Low timeout to keep GUI stutters to a minimum
        return SimpleBallSourceThread(0.02)
    elif uid == "webcam":
        # Slight timeout to prevent to many samples that are near equal
        return WebcamSourceThread(0.03)
    else:
        raise LookupError("Invalid source worker uid:", uid)


if __name__ == "__main__":
    parser = ArgumentParser(description="Overlay script parameters")
    parser.add_argument("--load_datasets", type=str, nargs="*", default=None)
    parser.add_argument("--save_dataset", type=str, default=None)
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--inference", action='store_true')
    parser.add_argument("--gt_source", type=str, default=None, choices=["simple-ball"])
    parser.add_argument("--img_source", type=str, default=None, choices=["webcam"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="face", choices=["face"])
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args(sys.argv[1:])

    module_directory = Path(__file__).resolve().parent

    # TODO: proper
    if args.gt_source or args.img_source or args.inference:
        app = QApplication(sys.argv)
        overlay = EyeTrackingOverlay()
    else:
        app = QCoreApplication(sys.argv)
        overlay = None

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
    if args.train:
        link_elements(gaze_to_face_sample, face_sample_to_train_pair, model_controller)
    link_elements(model_controller, model_element, ("register_inference_positions", overlay))
    link_elements(gt_src_thread, ("register_gt_position", overlay))

    if args.inference:
        inference_convertor = Convertor(lambda s: face_sample_to_X_tensor(s, args.device))
        link_elements(gaze_to_face_sample, inference_convertor, model_controller)

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
        if not args.train:
            print("WARNING: You passed datasets to load without enabling training.")
        else:
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
    print("Starting application event loop")
    if overlay:
        sys.exit(app.exec())
    else:
        # TODO: this is a hack to avoid calling "app.exec()" which prevents you from interupting (w/ "CTRL+C") when there is no GUI.
        # There is probably a better way to do this.
        global quiting
        quiting = False
        def quit_application(_1, _2):
            global quiting
            quiting = True
        signal.signal(signal.SIGINT, quit_application)
        import time
        while not quiting:
            app.thread().eventDispatcher().processEvents(QEventLoop.ProcessEventsFlag.AllEvents)
            time.sleep(0.01)
