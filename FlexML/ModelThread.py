
from pathlib import Path
from PyQt6.QtCore import QThread, QObject, QMutex, pyqtSignal, pyqtSlot
import torch
from torch.utils.data import DataLoader, Dataset
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("Tensorboard found")
except ImportError:
    TENSORBOARD_FOUND = False
    print("Tensorboard not found")

from examples.eye_tracker.src.FaceNetwork import FaceNetwork


def test_epoch(test_loader, model, device, loss_functions):
    losses = { name:0.0 for name in loss_functions}
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            output = model(X)
            for name, loss_fn in loss_functions.items():
                loss = loss_fn(output, y)
                losses[name] += loss.item()
   
    for name in losses:
        losses[name] /= len(test_loader)
        
    return losses

def train_epoch(train_loader, model, device, loss_functions, optimizer):
    if "criterion" not in loss_functions:
        raise KeyError("No criterion found")
    
    losses = { name:0.0 for name in loss_functions}
    for X, y in train_loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        output = model(X)
        for name, loss_fn in loss_functions.items():
            loss = loss_fn(output, y)
            losses[name] += loss.item()
            if name == "criterion":
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    for name in losses:
        losses[name] /= len(train_loader)

    return losses 

def training_report(tb_writer, epoch, train_losses, val_losses, test_losses):
    for name in train_losses.keys() & val_losses.keys(): # & test_losses.keys():
        dict = {}
        if name in train_losses:
            dict["train"] = train_losses[name]
        if name in val_losses:
            dict["val"] = val_losses[name]
        if tb_writer:
            tb_writer.add_scalars(f'{name}_loss', dict, epoch)

    print(f"{epoch:6} | train_loss={train_losses['criterion']:.5f}, val_loss={test_losses['criterion']:.5f}")


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

class ModelController(QThread):
    def __init__(self):
        super().__init__()
        pass

    def run(self):
        pass

class ModelElement(QObject):
    """Manages the pytorch model."""

    predicted_samples = pyqtSignal(tuple)

    def __init__(self, model_path: str, model_type: str, device: str, dataset_configs: list):
        super().__init__()

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
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        model.eval()

        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.inference_queue_mutex = QMutex()
        self.inference_queue = []
        self.model = model
        self.loss_functions = {
            "l1_x": lambda y1, y2: (y1[..., 0] - y2[..., 0]).abs().mean(),
            "l1_y": lambda y1, y2: (y1[..., 1] - y2[..., 1]).abs().mean(),
            "euclid": lambda y1, y2: (y1-y2).pow(2).sum(-1).sqrt().mean(),
            "criterion": lambda y1, y2: (y1-y2).pow(2).sum(-1).sqrt().mean()
        }

        self.tb_writer = SummaryWriter(model_path / "logs") if TENSORBOARD_FOUND else None

        
        self.data_mutex = QMutex()
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

    @pyqtSlot(torch.Tensor)
    def request_inference(self, X: torch.Tensor):

        self.inference_queue_mutex.lock()
        self.inference_queue.append(X)
        self.inference_queue_mutex.unlock()

    @pyqtSlot(torch.Tensor, torch.Tensor, str)
    def add_training_pair(self, X: torch.Tensor, y: torch.Tensor, type: str):
        self.data_mutex.lock()
        if type not in self.datasets:
            raise ValueError("Unexpected type:", type)
        if not self.model.training:
            self.model.train()
        self.datasets[type].add_pair(X, y)
        self.data_mutex.unlock()
    
    def run_epoch(self, epoch: int):
        losses = {}
        for type, data_loader in self.data_loaders.values():
            losses[type] = train_epoch(data_loader, self.model, self.device, self.loss_functions, self.optimizer)
        
        training_report(self.tb_writer, epoch, losses)

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

    def run_inference(self):
        # Do inference on the new samples
        self.new_samples_mutex.lock()
        if self.inference_queue:
            with torch.no_grad():
                all_input = torch.stack(self.inference_queue)
                predictions = self.model(all_input).cpu().detach().numpy()
                self.predicted_samples.emit(zip(self.inference_queue, predictions))
            self.inference_queue = []
        self.new_samples_mutex.unlock()
