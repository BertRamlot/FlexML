import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    logging.info("Tensorboard found")
except ImportError:
    TENSORBOARD_FOUND = False
    logging.info("Tensorboard not found")
from PyQt6.QtCore import QThread, QObject, QMutex, pyqtSignal, pyqtSlot, QWaitCondition, QEventLoop

from FlexML.Sample import Sample

def epoch_report(tb_writer, epoch, losses, datasets):
    # losses: { 'data_type': { 'loss_fn_name': 0.001 } }
    loss_function_ids = []
    for loss_fn_dict in losses.values():
        loss_function_ids += list(loss_fn_dict.keys())
    loss_function_ids = set(loss_function_ids)

    for loss_function_id in loss_function_ids:
        dict = {}
        for dataset_type in losses.keys():
            if loss_function_id in losses[dataset_type]:
                dict[dataset_type] = losses[dataset_type][loss_function_id]
        if tb_writer:
            tb_writer.add_scalars(f'{loss_function_id}_loss', dict, epoch)

    dataset_type_order = list(datasets.keys())
    dataset_sizes = [f"{len(datasets[dataset_type]):4}" for dataset_type in dataset_type_order]
    criterion_loss_strings = [f"{dataset_type}={losses[dataset_type]['criterion']:.4f}" for dataset_type in dataset_type_order]


    logging.info(f"{epoch:6} | samples ({' '.join(dataset_type_order)}): {' '.join(dataset_sizes)} | losses: {', '.join(criterion_loss_strings)}")


class DynamicDataset(Dataset):
    def __init__(self, type: str | None):
        self.type = type
        self.input_label_pairs = []
        self.samples = []

    def __len__(self):
        # Min size needs to be atleast one for pytorch DataLoader to not throw an error when using "shuffle=True"
        # Another way to resolve this is to write your own custom sampler:
        # https://stackoverflow.com/questions/70369070/can-a-pytorch-dataloader-start-with-an-empty-dataset
        return max(1, len(self.input_label_pairs))

    def __getitem__(self, idx):
        return self.input_label_pairs[idx]
    
    def add_pair(self, sample: Sample, X: torch.Tensor, y: torch.Tensor):
        self.samples.append(sample)
        self.input_label_pairs.append((len(self.samples), X, y))

    def get_sample(self, index: int):
        return self.samples[index]

class ModelElement(QObject):
    """Manages the pytorch model."""

    inference_results = pyqtSignal(list, np.ndarray)

    SAVE_EVERY_N_EPOCH = 50

    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer, checkpoint, model_path: str, device: str, dataset_configs: list, loss_functions):
        """
        the loss function used for training is named "criterion"
        """
        super().__init__()
        
        self.epoch = 0
        self.model = model
        self.optimizer = optimizer
        self.model_path = model_path
        self.device = device
        self.loss_functions = loss_functions
        if "criterion" not in loss_functions:
            logging.warn("No loss function found named 'criterion', training epochs will not have an effect")
        self.tb_writer = SummaryWriter(self.model_path / "logs") if TENSORBOARD_FOUND else None

        if checkpoint:
            self.epoch = checkpoint["epoch"]
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.model.eval()

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
        all_losses = {}
        for type, data_loader in self.data_loaders.items():
            do_grad = (type == "train")
            losses = { name: 0.0 for name in self.loss_functions }
            with torch.set_grad_enabled(do_grad):
                for sample_indices, X, y in data_loader:
                    X = X.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)

                    output = self.model(X)
                    for name, loss_fn in self.loss_functions.items():
                        loss = loss_fn(output, y)
                        losses[name] += loss.item()
                        if do_grad and name == "criterion":
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

            for name in losses:
                losses[name] /= len(data_loader)

            all_losses[type] = losses
        
        # Report results
        epoch_report(self.tb_writer, self.epoch, all_losses, self.datasets)

        # Checkpoint model
        if self.epoch % ModelElement.SAVE_EVERY_N_EPOCH == 0:
            torch.save(
                {
                    'epoch': self.epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                },
                self.model_path / f"epoch_{self.epoch:0>6}.pth"
            )
        self.epoch += 1
        return all_losses

    @pyqtSlot(list, torch.Tensor)
    def run_inference(self, samples: list[Sample], input_tensor: torch.Tensor):
        with torch.no_grad():
            predictions = self.model(input_tensor).cpu().detach().numpy()
            self.inference_results.emit(samples, predictions)


class ModelController(QThread):
    inference_request = pyqtSignal(list, torch.Tensor)

    def __init__(self, model_element: ModelElement):
        super().__init__()
        self.model_element = model_element

        self.inference_queue = []
        self.new_samples = []

        # TODO: Make sure the dataset has atleast N ele per
        self.wait_train_for_n_samples = 50

        self.mutex = QMutex()
        self.condition = QWaitCondition()

    def run(self):
        loss_per_epoch = []
        while True:
            inference_items = []
            self.mutex.lock()
            while not self.inference_queue and not self.new_samples and self.wait_train_for_n_samples:
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
                    self.wait_train_for_n_samples = max(0, self.wait_train_for_n_samples -1)
            if self.wait_train_for_n_samples > 0:
                continue

            # Wait for new samples
            if len(loss_per_epoch) > 50:
                avg_less_recent_loss = np.array(loss_per_epoch[-50:-25]).mean()
                avg_recent_loss = np.array(loss_per_epoch[-25:]).mean()
                if avg_less_recent_loss < avg_recent_loss:
                    loss_per_epoch = []
                    self.wait_train_for_n_samples = int(0.3 * len(self.model_element.datasets["train"]))
                    logging.info(f"Waiting for {self.wait_train_for_n_samples} training samples ...")
                    continue

            # Do an epoch
            if not self.model_element.model.training:
                self.model_element.model.train()
            
            all_losses = self.model_element.run_epoch()
            loss_per_epoch.append(all_losses["val"]["criterion"])
    
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