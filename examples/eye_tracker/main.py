import sys
import numpy as np
import time
import logging
import signal
from pathlib import Path
from argparse import ArgumentParser
import torch
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QCoreApplication, QEventLoop, QObject, pyqtSignal, pyqtSlot

from FlexML.SourceThread import WebcamSourceThread
from FlexML.Sample import Sample
from FlexML.Model import ModelElement, ModelController
from FlexML.Helper import BufferThread, Filter
from FlexML.Linker import link_QObjects

from examples.eye_tracker.src.EyeTrackingOverlay import EyeTrackingOverlay
from examples.eye_tracker.src.TargetSource import SimpleBallSourceThread, FeedbackBallSourceThread
from examples.eye_tracker.src.FaceSample import GazeToFaceSampleConvertor
from examples.eye_tracker.src.FaceNetwork import FaceSampleToTrainPair, FaceSampleToInferencePair
from examples.eye_tracker.src.GazeSample import GazeSampleMuxer
from examples.eye_tracker.src.FaceSample import FaceSampleCollection
from examples.eye_tracker.src.FaceNetwork import FaceNetwork


def get_source_worker(uid: str|None):
    if uid is None:
        return None
    elif uid == "simple-ball":
        # Low timeout to keep GUI stutters to a minimum
        return SimpleBallSourceThread(0.02)
    elif uid == "feedback-ball":
        return FeedbackBallSourceThread(0.02)
    elif uid == "webcam":
        # Slight timeout to prevent to many samples that are near equal
        return WebcamSourceThread(0.03)
    else:
        raise LookupError("Invalid source worker uid:", uid)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    parser = ArgumentParser(description="Overlay script parameters")
    parser.add_argument("--load_datasets", type=str, nargs="*", default=None)
    parser.add_argument("--save_dataset", type=str, default=None)
    parser.add_argument("--train", action='store_true')
    parser.add_argument("--inference", action='store_true')
    parser.add_argument("--gt_source", type=str, default=None, choices=["simple-ball", "feedback-ball"])
    parser.add_argument("--img_source", type=str, default=None, choices=["webcam"])
    parser.add_argument("--model", type=str, default=None)
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
        model_path = module_directory / Path("models") / args.model
        pth_path = max(model_path.glob("epoch_*.pth"), default=None)
        if pth_path:
            logging.info(f"Loading existing model: {pth_path}")
            # TODO:
            checkpoint = torch.load(pth_path) # , map_location=torch.device('cpu'))
        else:
            logging.info("Creating new model")
            checkpoint = None
            model_path.mkdir(parents=True, exist_ok=True)

        model = FaceNetwork().to(args.device)
        model_element = ModelElement(
            model,
            torch.optim.Adam(model.parameters(), lr=1e-3),
            checkpoint,
            model_path,
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
            ],
            {
                "l1_x": lambda y1, y2: (y1[..., 0] - y2[..., 0]).abs(),
                "l1_y": lambda y1, y2: (y1[..., 1] - y2[..., 1]).abs(),
                "euclid": lambda y1, y2: (y1-y2).pow(2).sum(-1).sqrt(),
                "criterion": lambda y1, y2: (y1-y2).pow(2).sum(-1).sqrt()
            }
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
    # Enfore some time between samples to prevent:
    # (1) leakage btwn train/val/test datasets
    # (2) too similar samples within a dataset
    import time
    def new_train_samples_filter_func(filtr, sample):
        if sample.gt is None:
            return False
        if hasattr(filtr, "last_pass_time"):
            elapsed_since_last_pass = time.time() - filtr.last_pass_time
        else:
            elapsed_since_last_pass = None
        if elapsed_since_last_pass is None or elapsed_since_last_pass > 1.0:
            filtr.last_pass_time = time.time()
            return True
        return False
    new_train_samples = Filter(new_train_samples_filter_func)
    link_QObjects(gaze_to_face_sample, new_train_samples)

    # Live pipeline
    link_QObjects(gt_src_thread,  ("set_last_label", sample_muxer))
    link_QObjects(img_src_thread, ("set_last_img",   sample_muxer), sample_buffer, gaze_to_face_sample)
    link_QObjects(model_controller, (model_element, "inference_results"), ("register_inference_positions", overlay))
    link_QObjects(gt_src_thread, ("register_gt_position", overlay))

    # TODO: kinda hacky
    if isinstance(gt_src_thread, FeedbackBallSourceThread):
        link_QObjects((model_element, "sample_errors"), gt_src_thread)

        class TrainPairToSampleLoss(QObject):
            output = pyqtSignal(list, np.ndarray)
            @pyqtSlot(Sample, torch.Tensor, torch.Tensor)
            def on_input(self, sample: Sample, _, __):
                self.output.emit([sample], np.array([None]))
        train_pair_to_sample_loss = TrainPairToSampleLoss()
        link_QObjects(face_sample_to_train_pair, train_pair_to_sample_loss, gt_src_thread)    

    if args.train:
        link_QObjects(new_train_samples, face_sample_to_train_pair, model_controller)

    if args.inference:
        face_sample_to_inference_pair = FaceSampleToInferencePair(args.device)
        link_QObjects(gaze_to_face_sample, face_sample_to_inference_pair, model_controller)

    # Save data to disk
    if args.save_dataset:
        save_path = module_directory / Path("datasets") / args.save_dataset
        logging.info(f"Saving new samples in: {save_path}")
        save_coll = FaceSampleCollection(module_directory / Path("datasets") / args.save_dataset)
        link_QObjects( new_train_samples, save_coll)
    else:
        logging.info("Not saving new samples")
    
    # Load all data from disk
    if args.load_datasets:
        if not args.train:
            logging.warn("Load dataset(s) passed without enabling training: ignoring the load dataset(s)")
        else:
            total_published_samples = 0
            load_data_colls = [FaceSampleCollection(module_directory / "datasets" / load_dataset_path) for load_dataset_path in args.load_datasets]
            for dataset_source in load_data_colls:
                link_QObjects(dataset_source, face_sample_to_train_pair)
            for dataset_source in load_data_colls:
                sample_count = dataset_source.publish_all_samples()
                logging.info(f"Loaded {sample_count} samples from {dataset_source.dataset_path}")
                total_published_samples += sample_count
            logging.info(f"Total loaded samples: {total_published_samples}")
            # TODO: wait for all samples to be processes before starting the live threads?
    else:
        logging.info("Not loading any datasets")
    
    if model_controller:
        model_controller.start()

    # Start live sources
    if img_src_thread:
        img_src_thread.start()
    if gt_src_thread:
        gt_src_thread.start()
 
    # Event loop
    if overlay:
        logging.info("Starting default event loop")
        sys.exit(app.exec())
    else:
        # TODO: this is a hack to avoid calling "app.exec()" which prevents you from interupting (w/ "CTRL+C") when there is no GUI.
        # There is probably a better way to do this.
        logging.info("Starting headless event loop")
        global quiting
        quiting = False
        signal.signal(signal.SIGINT, lambda _, __: globals().update({'quitting': True}))
        while not quiting:
            app.thread().eventDispatcher().processEvents(QEventLoop.ProcessEventsFlag.AllEvents)
            time.sleep(0.01)
