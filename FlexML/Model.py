
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("Tensorboard found")
except ImportError:
    TENSORBOARD_FOUND = False
    print("Tensorboard not found")
from PyQt6.QtCore import QThread, QObject, QMutex, pyqtSignal, pyqtSlot, QWaitCondition, QEventLoop

from examples.eye_tracker.src.FaceNetwork import FaceNetwork


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


    print(f"{epoch:6} | samples ({' '.join(dataset_type_order)}): {' '.join(dataset_sizes)} | losses: {', '.join(criterion_loss_strings)}")


class DynamicDataset(Dataset):
    def __init__(self, type: str|None = None):
        self.type = type
        self.input_label_pairs = []

    def __len__(self):
        # Min size needs to be atleast one for pytorch DataLoader to not throw an error when using "shuffle=True"
        # Another way to resolve this is to write your own custom sampler:
        # https://stackoverflow.com/questions/70369070/can-a-pytorch-dataloader-start-with-an-empty-dataset
        return max(1, len(self.input_label_pairs))

    def __getitem__(self, idx):
        return self.input_label_pairs[idx]
    
    def add_pair(self, X, y):
        self.input_label_pairs.append((X, y))

class ModelElement(QObject):
    """Manages the pytorch model."""

    inference_results = pyqtSignal(tuple)

    def __init__(self, model_path: str, model_type: str, device: str, dataset_configs: list):
        super().__init__()

        self.model_path = model_path
        self.device = device

        # Load model if it exists
        pth_path = max(model_path.glob("epoch_*.pth"), default=None)
        if pth_path:
            print("Loading existing model:", pth_path)
            checkpoint = torch.load(pth_path)
            self.epoch = checkpoint["epoch"]
            if model_type != checkpoint["model_type"]:
                print("WARNING: provided 'model_type' does not correspond with checkpoint 'model_type'")
            self.model_type = checkpoint["model_type"]
        else:
            checkpoint = None
            self.epoch = 0
            self.model_type = model_type
            model_path.mkdir(parents=True, exist_ok=True)
        
        # TODO: use 'self.model_type'
        model = FaceNetwork().to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        model.eval()

        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.model = model
        self.loss_functions = {
            "l1_x": lambda y1, y2: (y1[..., 0] - y2[..., 0]).abs().mean(),
            "l1_y": lambda y1, y2: (y1[..., 1] - y2[..., 1]).abs().mean(),
            "euclid": lambda y1, y2: (y1-y2).pow(2).sum(-1).sqrt().mean(),
            "criterion": lambda y1, y2: (y1-y2).pow(2).sum(-1).sqrt().mean()
        }

        self.tb_writer = SummaryWriter(model_path / "logs") if TENSORBOARD_FOUND else None

        
        self.datasets = {}
        self.data_loaders = {}
        for config in dataset_configs:
            t = config["type"]
            dataset = DynamicDataset(type=t)
            if "dataloader_kwargs" not in config:
                config["dataloader_kwargs"] = {}
            data_loader = DataLoader(dataset, **config["dataloader_kwargs"])
            self.datasets[t] = dataset
            self.data_loaders[t] = data_loader

    @pyqtSlot(int)
    def run_epoch(self, epoch: int) -> dict:
        # Train/Val/Test epoch
        all_losses = {}
        for type, data_loader in self.data_loaders.items():
            do_grad = (type == "train")
            losses = { name: 0.0 for name in self.loss_functions}
            with torch.set_grad_enabled(do_grad):
                for X, y in data_loader:
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
        epoch_report(self.tb_writer, epoch, all_losses, self.datasets)

        # Checkpoint model
        if epoch % 50 == 0:
            torch.save(
                {
                    'model_type': self.model_type,
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                },
                self.model_path / f"epoch_{epoch:0>6}.pth"
            )
        return all_losses

    @pyqtSlot(torch.Tensor)
    def run_inference(self, input_tensor: torch.Tensor):
        with torch.no_grad():
            predictions = self.model(input_tensor).cpu().detach().numpy()
            self.inference_results.emit(predictions)


class ModelController(QThread):
    inference_request = pyqtSignal(torch.Tensor)

    def __init__(self, model_element: ModelElement):
        super().__init__()
        self.model_element = model_element
        self.inference_queue = [] # TODO: not a queue, better name

        self.mutex = QMutex()
        self.condition = QWaitCondition()
        self.wait_train_for_n_samples = 30

    def run(self):
        self.exec()

    def exec(self):
        loss_per_epoch = []
        epoch = 0
        while True:
            self.eventDispatcher().processEvents(QEventLoop.ProcessEventsFlag.AllEvents)
            self.mutex.lock()

            if len(self.inference_queue) > 0:
                model_input = torch.stack(self.inference_queue)
                print("INFERENCE ON", len(model_input), "ELEMENTS")
                self.inference_request.emit(model_input)

            if self.wait_train_for_n_samples > 0:
                self.condition.wait(self.mutex)
                self.mutex.unlock()
                continue

            if len(loss_per_epoch) > 50:
                avg_less_recent_loss = np.array(loss_per_epoch[-50:-25]).mean()
                avg_recent_loss = np.array(loss_per_epoch[-25:]).mean()
                if avg_less_recent_loss < avg_recent_loss:
                    # No recent improvements
                    loss_per_epoch = []
                    self.wait_train_for_n_samples = int(0.3 * len(self.model_element.datasets["train"]))
                    print(f"Waiting for {self.wait_train_for_n_samples} training samples ...")
                    self.condition.wait(self.mutex)
                    self.mutex.unlock()
                    continue

            # train
            if not self.model_element.model.training:
                self.model_element.model.train()
            
            all_losses = self.model_element.run_epoch(epoch)
            loss_per_epoch.append(all_losses["val"]["criterion"])
            epoch += 1

            self.mutex.unlock()
    
    @pyqtSlot(torch.Tensor)
    def request_inference(self, X: torch.Tensor):
        self.mutex.lock()
        self.inference_queue.append(X)
        self.condition.wakeAll()
        self.mutex.unlock()

    @pyqtSlot(torch.Tensor, torch.Tensor, str)
    def add_training_pair(self, X: torch.Tensor, y: torch.Tensor, type: str):
        self.mutex.lock()
        if type not in self.model_element.datasets:
            raise ValueError("Unexpected type:", type)
        self.model_element.datasets[type].add_pair(X, y)
        if type == "train" and self.wait_train_for_n_samples > 0:
            self.wait_train_for_n_samples -= 1
            if self.wait_train_for_n_samples == 0:
                self.condition.wakeAll()
        self.mutex.unlock()
