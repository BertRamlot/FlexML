import logging
import collections
import numpy as np
from pathlib import Path
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
    """
    Dataset that can be grown dynamically.
    """
    
    def __init__(self, type: str | None):
        self.type = type
        # Min size needs to be atleast one for pytorch DataLoader to not throw an error when using "shuffle=True"
        # So some placeholder items are added that are later removed
        # Another way to resolve this is to write your own custom sampler
        # Also see: https://stackoverflow.com/questions/70369070/can-a-pytorch-dataloader-start-with-an-empty-dataset
        self.input_label_pairs: list[tuple[np.ndarray, np.ndarray]] = [(torch.empty((0,)), torch.empty((0,)))]
        self.samples: list[Sample] = [None]

    def __len__(self) -> int:
        return len(self.input_label_pairs)

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray]:
        return self.input_label_pairs[idx]
    
    def add_training_tuple(self, sample: Sample, X: torch.Tensor, y: torch.Tensor):
        if sample is None:
            raise ValueError("Can't add a sample that is None")
        if self.samples[0] is None:
            # Remove place holder items
            self.input_label_pairs = []
            self.samples = []
        self.input_label_pairs.append((len(self.samples), X, y))
        self.samples.append(sample)

    def get_sample(self, index: int) -> Sample:
        return self.samples[index]

class ModelElement(QObject):
    """
    High level abstraction around a pytorch model, optimizer, loss function, datasets, dataloaders, ...
    """

    inference_results = pyqtSignal(list, np.ndarray)
    sample_errors = pyqtSignal(list, np.ndarray)

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, checkpoint, model_path: Path, 
                 device: str, dataset_configs: list[dict[str, object]], loss_functions: dict[str, collections.abc.Callable]):
        super().__init__()
        """
        Initialize ModelElement.

        Args:
            model (torch.nn.Module): PyTorch model.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            checkpoint: Checkpoint for resuming training (set to None when starting from scratch).
            model_path (Path): Base folder of the model to save checkpoints and write logs.
            device (str): Device to run the model on (e.g., 'cpu', 'cuda').
            dataset_configs (list[dict[str, object]]): List of dataset configurations.
            loss_functions (dict[str, collections.abc.Callable]): Dictionary of loss functions, loss function used for training NEEDS to be called "criterion".
        """
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
    def save_checkpoint(self):
        """
        Saves the current state (epoch, model state, and optimizer state) as a checkpoint.
        """
        checkpoint_path = self.model_path / f"epoch_{self.epoch:0>6}.pth"
        logging.info(f"Saving checkpoint to: {checkpoint_path}")
        self.model_path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            },
            self.model_path / f"epoch_{self.epoch:0>6}.pth"
        )

    @pyqtSlot()
    def run_epoch(self) -> dict:
        """
        Runs a single training epoch (if a train dataset is provided).
        Afterwards, all other (non training) datasets do an epoch to calculate their respective losses.
        
        The results are logged, and written to a tensorboard if available.

        Returns:
            dict: Dictionary of losses for each dataset.
        """
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
                    if samples[0] is None:
                        # TODO: this is a hack to support DynamicDataset work, see DynamicDataset for a full explanation
                        continue
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

        self.epoch += 1
        return dataset_losses

    @pyqtSlot(list, torch.Tensor)
    def run_inference(self, samples: list[Sample], input_tensor: torch.Tensor):
        """
        Runs inference on samples using the provided input tensors and emits the results.

        Args:
            samples (list[Sample]): List of N samples.
            input_tensor (torch.Tensor): Input tensor for each sample, shape is (N, ...).
        """
        with torch.no_grad():
            predictions = self.model(input_tensor).cpu().detach().numpy()
            self.inference_results.emit(samples, predictions)

class ModelController(QThread):
    """
    Thread that manages a ModelElement, deciding when to do inference, training, saving, add new samples, wait for new samples, ...
    """
    inference_request = pyqtSignal(list, torch.Tensor)

    def __init__(self, model_element: ModelElement, initial_train_samples_req_before_training: int, save_checkpoint_every_n_epochs: int):
        """
        Initialize ModelController with the given model element and initial number of training samples required before training starts.

        Args:
            model_element (ModelElement): The model element to manage.
            initial_train_samples_req_before_training (int): The initial number of training samples required before training starts.
            save_checkpoint_every_n_epochs (int): The frequency (in epochs) at which checkpoints will be saved during training.
        """
        super().__init__()
        self.model_element = model_element
        if "train" not in self.model_element.datasets:
            logging.warning("No 'train' dataset found, model controller will not train.")
        if "val" not in self.model_element.datasets:
            logging.warning("No 'val' dataset found, model controller will keep training once started.")

        self.new_inference_tuples: list[tuple[Sample, torch.Tensor]] = []
        self.new_training_tuples: list[tuple[Sample, torch.Tensor, torch.Tensor]] = []

        self.pause_training_till_n_train_samples = initial_train_samples_req_before_training
        self.save_checkpoint_every_n_epochs = save_checkpoint_every_n_epochs

        self.mutex = QMutex()
        self.condition = QWaitCondition()

    def run(self):
        logging.info(f"Waiting for {self.pause_training_till_n_train_samples} initial samples before starting training...")
        val_loss_per_epoch = []
        while True:
            # Wait for something to do
            self.mutex.lock()
            while not self.new_inference_tuples and not self.new_training_tuples and self.pause_training_till_n_train_samples:
                self.condition.wait(self.mutex)        
            inference_tuples, self.new_inference_tuples = self.new_inference_tuples, []
            training_tuples, self.new_training_tuples = self.new_training_tuples, []
            self.mutex.unlock()
            
            # Do inference
            if len(inference_tuples) > 0:
                samples = [sample for sample, _ in inference_tuples]
                X = torch.stack([X for _, X in inference_tuples])
                self.inference_request.emit(samples, X)

            # Add new samples
            for sample, X, y in training_tuples:
                self.model_element.datasets[sample.type].add_training_tuple(sample, X, y)
                if sample.type == "train":
                    self.pause_training_till_n_train_samples = max(0, self.pause_training_till_n_train_samples - 1)
            if self.pause_training_till_n_train_samples > 0:
                continue

            # Wait for new samples
            # TODO: these constants are arbitrary
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
                
            # Save a checkpoint
            if self.model_element.epoch % self.save_checkpoint_every_n_epochs == 0:
                self.model_element.save_checkpoint()
    
    @pyqtSlot(Sample, torch.Tensor)
    def request_inference(self, sample: Sample, X: torch.Tensor):
        """
        Sends an inference request to the underlying ModelElement for an inference tuple (sample, input data).

        Args:
            sample (Sample): The sample to be infered.
            X (torch.Tensor): Input tensor of the sample.
        """
        self.mutex.lock()
        self.new_inference_tuples.append((sample, X))
        self.condition.wakeAll()
        self.mutex.unlock()

    @pyqtSlot(Sample, torch.Tensor, torch.Tensor)
    def add_training_tuple(self, sample: Sample, X: torch.Tensor, y: torch.Tensor):
        """
        Adds a training tuple (sample, input data, target data) to the underlying ModelElement's datasets.

        Args:
            sample (Sample): The sample to be added.
            X (torch.Tensor): Input tensor of the sample.
            y (torch.Tensor): Ground truth of the sample.
        """
        self.mutex.lock()
        self.new_training_tuples.append((sample, X, y))
        self.condition.wakeAll()
        self.mutex.unlock()
