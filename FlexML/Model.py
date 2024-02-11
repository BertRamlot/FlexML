import logging
import collections
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    logging.info("Tensorboard found")
except ImportError:
    TENSORBOARD_FOUND = False
    logging.info("Tensorboard not found")
from PyQt6.QtCore import QThread, QObject, QMutex, pyqtSignal, pyqtSlot, QWaitCondition

from FlexML.Sample import Sample


class DynamicDataset(Dataset):
    """Dataset that can be grow dynamically."""
    
    def __init__(self, type: str | None):
        self.type = type
        self.input_label_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        self.samples: list[Sample] = []

    def __len__(self) -> int:
        # Min size needs to be atleast one for pytorch DataLoader to not throw an error when using "shuffle=True"
        # Another way to resolve this is to write your own custom sampler:
        # https://stackoverflow.com/questions/70369070/can-a-pytorch-dataloader-start-with-an-empty-dataset
        return max(1, len(self.input_label_pairs))

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        # TODO: this error when size is 0, see the "__len__" comment.
        return self.input_label_pairs[idx]
    
    def add_pair(self, sample: Sample, X: torch.Tensor, y: torch.Tensor):
        self.input_label_pairs.append((len(self.samples), X, y))
        self.samples.append(sample)

    def get_sample(self, index: int) -> Sample:
        return self.samples[index]

class ModelElement(QObject):
    """Manages the pytorch model."""

    inference_results = pyqtSignal(list, np.ndarray)
    sample_errors = pyqtSignal(list, np.ndarray)

    # TODO: hard coded
    SAVE_EVERY_N_EPOCH = 50

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, checkpoint, model_path: str, 
                 device: str, dataset_configs: list[dict[str, object]], loss_functions: dict[str, collections.abc.Callable]):
        super().__init__()
        
        self.epoch = 0
        self.model = model
        self.optimizer = optimizer
        if checkpoint:
            self.epoch = checkpoint["epoch"]
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])        
        self.model_path = model_path
        self.device = device
        self.loss_functions = loss_functions
        if "criterion" in loss_functions:            
            self.model.train()
        else:
            logging.warning("No loss function found named 'criterion' => training epochs will not have an effect")
            self.model.eval()
        
        self.tb_writer = SummaryWriter(self.model_path / "logs") if TENSORBOARD_FOUND else None


        self.datasets: dict[str, DynamicDataset] = {}
        self.data_loaders: dict[str, DataLoader] = {}
        for config in dataset_configs:
            type = config["type"]
            dataset = DynamicDataset(type)
            if "dataloader_kwargs" not in config:
                config["dataloader_kwargs"] = {}
            self.datasets[type] = dataset
            self.data_loaders[type] = DataLoader(dataset, **config["dataloader_kwargs"])

    @pyqtSlot()
    def run_epoch(self) -> dict:            
        # Do epoch, start with the training dataset as this changes the model
        dataset_losses = {}
        for type, data_loader in sorted(self.data_loaders.items(), key=lambda t: t[0] == "train", reverse=True):
            do_grad = (type == "train")
            losses = { loss_fn_name: 0.0 for loss_fn_name in self.loss_functions }
            with torch.set_grad_enabled(do_grad):
                for sample_indices, X, y in data_loader:
                    X = X.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    samples = [self.datasets[type].get_sample(index) for index in sample_indices]
                    output = self.model(X)
                    for loss_fn_name, loss_fn in self.loss_functions.items():
                        loss = loss_fn(y, output)
                        loss_mean = loss.mean()
                        losses[loss_fn_name] += loss_mean.item()
                        if loss_fn_name == "criterion":
                            loss_per_sample = loss.cpu().detach().numpy()
                            self.sample_errors.emit(samples, loss_per_sample)
                            if do_grad:
                                self.optimizer.zero_grad()
                                loss_mean.backward()
                                self.optimizer.step()
            dataset_losses[type] = { loss_fn_name: losses[loss_fn_name] / len(data_loader) for loss_fn_name in losses }
        
        # Report results
        if self.tb_writer:
            for loss_fn_name in self.loss_functions:
                loss_fn_epoch_losses = { type: dataset_losses[type][loss_fn_name] for type in dataset_losses.keys() }
                self.tb_writer.add_scalars(f'{loss_fn_name}_loss', loss_fn_epoch_losses, self.epoch)

        dataset_type_order = list(self.datasets.keys())
        dataset_sizes_strings = [f"{len(self.datasets[type]):4}" for type in dataset_type_order]
        criterion_loss_strings = [f"{type}={dataset_losses[type]['criterion']:.4f}" for type in dataset_type_order]

        logging.info("epoch: {:5} | samples ({}): {} | losses: {}".format(
            self.epoch, ' '.join(dataset_type_order), ' '.join(dataset_sizes_strings), ', '.join(criterion_loss_strings)))

        # Checkpoint model
        if self.epoch % ModelElement.SAVE_EVERY_N_EPOCH == 0:
            self.model_path.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                },
                self.model_path / f"epoch_{self.epoch:0>6}.pth"
            )
        self.epoch += 1
        return dataset_losses

    @pyqtSlot(list, torch.Tensor)
    def run_inference(self, samples: list[Sample], input_tensor: torch.Tensor):
        with torch.no_grad():
            predictions = self.model(input_tensor).cpu().detach().numpy()
            self.inference_results.emit(samples, predictions)

class ModelController(QThread):
    inference_request = pyqtSignal(list, torch.Tensor)

    def __init__(self, model_element: ModelElement, initial_train_samples_req_before_training: int):
        super().__init__()
        self.model_element = model_element
        if "train" not in self.model_element.datasets:
            logging.warning("No 'train' dataset found, model controller will not train.")
        if "val" not in self.model_element.datasets:
            logging.warning("No 'val' dataset found, model controller will keep training once started.")

        self.inference_queue: list[tuple[Sample, torch.Tensor]] = []
        self.new_samples: list[tuple[Sample, torch.Tensor, torch.Tensor]] = []

        self.pause_training_till_n_train_samples = initial_train_samples_req_before_training

        self.mutex = QMutex()
        self.condition = QWaitCondition()

    def run(self):
        val_loss_per_epoch = []
        while True:
            # Wait for something to do
            self.mutex.lock()
            while not self.inference_queue and not self.new_samples and self.pause_training_till_n_train_samples:
                self.condition.wait(self.mutex)        
            inference_items, self.inference_queue = self.inference_queue, []
            new_samples, self.new_samples = self.new_samples, []
            self.mutex.unlock()
            # Do inference
            if len(inference_items) > 0:
                samples = [item[0] for item in inference_items]
                X = torch.stack([item[1] for item in inference_items])
                self.inference_request.emit(samples, X)

            # Add new samples
            for sample, X, y in new_samples:
                self.model_element.datasets[sample.type].add_pair(sample, X, y)
                if sample.type == "train":
                    self.pause_training_till_n_train_samples = max(0, self.pause_training_till_n_train_samples - 1)
            if self.pause_training_till_n_train_samples > 0:
                continue

            # Wait for new samples
            if len(val_loss_per_epoch) > 50:
                avg_recent_loss = np.array(val_loss_per_epoch[-25:]).mean()
                avg_less_recent_loss = np.array(val_loss_per_epoch[-50:-25]).mean()
                if avg_less_recent_loss < avg_recent_loss:
                    val_loss_per_epoch = []
                    self.pause_training_till_n_train_samples = int(0.3 * len(self.model_element.datasets["train"]))
                    logging.info(f"Waiting for {self.pause_training_till_n_train_samples} training samples ...")
                    continue

            # Do an epoch
            dataset_losses = self.model_element.run_epoch()
            if "val" in dataset_losses and "criterion" in dataset_losses["val"]:
                val_loss_per_epoch.append(dataset_losses["val"]["criterion"])
    
    @pyqtSlot(Sample, torch.Tensor)
    def request_inference(self, sample: Sample, X: torch.Tensor):
        self.mutex.lock()
        self.inference_queue.append((sample, X))
        self.condition.wakeAll()
        self.mutex.unlock()

    @pyqtSlot(Sample, torch.Tensor, torch.Tensor)
    def add_training_pair(self, sample: Sample, X: torch.Tensor, y: torch.Tensor):
        self.mutex.lock()
        self.new_samples.append((sample, X, y))
        self.condition.wakeAll()
        self.mutex.unlock()