
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, pyqtSlot
import torch
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("Tensorboard found")
except ImportError:
    TENSORBOARD_FOUND = False
    print("Tensorboard not found")

from src.Sample import Sample

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


class ModelThread(QThread):
    """Manages the pytorch model."""

    predicted_samples = pyqtSignal(tuple)

    def __init__(self, model_path: str, model_type: str, dataset_paths: list[Path], device: str, training: bool):
        super(ModelThread, self).__init__()

        self.device = device

        # Load model if it exists
        pth_path = max(model_path.glob("epoch_*.pth"), default=None)
        if pth_path:
            print("Loading existing model:", pth_path)
            checkpoint = torch.load(pth_path)
            self.epoch = checkpoint["epoch"]
            if model_type != checkpoint["model_type"]:
                print("WARNING: provided 'model_type' does not correspond with checkpoint 'model_type'")
            model_type = checkpoint["model_type"]
        else:
            model_path.mkdir(parents=True, exist_ok=True)
        
        model = FaceNetwork().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        if checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        model.train(training)

        self.new_samples = []
        self.new_samples_mutex = QMutex()
        self.model = model

        self.tb_writer = SummaryWriter(model_path / "logs") if TENSORBOARD_FOUND else None

        print("Loading datasets ... ", end="")
        self.train_dataset = FaceDataset(dataset_paths, self.device, type="train")
        self.val_dataset = FaceDataset(dataset_paths, self.device, type="val")
        self.test_dataset = FaceDataset(dataset_paths, self.device, type="test")
        print(f"{len(self.train_dataset)}/{len(self.val_dataset)}/{len(self.test_dataset)} train/val/test samples loaded")

        self.train_data_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_data_loader = DataLoader(self.val_dataset, batch_size=1)
        self.test_data_loader = DataLoader(self.test_dataset, batch_size=1)

    @pyqtSlot(Sample)
    def request_inference(self, sample: Sample):
        self.new_samples.append(sample)

    @pyqtSlot(Sample)
    def add_sample(self, sample: Sample):
        # TODO:
        # self.new_samples_mutex.lock()
        if sample.type == "train":
            self.train_dataset.add_sample(sample)
        elif sample.type == "val":
            self.val_dataset.add_sample(sample)
        elif sample.type == "test":
            self.test_dataset.add_sample(sample)
        else:
            raise ValueError("Unexpected type:", sample.type)
        # self.new_samples_mutex.unlock()
    
    def run(self):
        loss_functions = {
            "l1_x": lambda y1, y2: (y1[..., 0] - y2[..., 0]).abs().mean(),
            "l1_y": lambda y1, y2: (y1[..., 1] - y2[..., 1]).abs().mean(),
            "euclid": lambda y1, y2: (y1-y2).pow(2).sum(-1).sqrt().mean(),
            "criterion": lambda y1, y2: (y1-y2).pow(2).sum(-1).sqrt().mean()
        }

        while True:
            # TODO: wait for more samples

            train_losses = train_epoch(self.train_data_loader, self.model, self.device, loss_functions, self.optimizer)
            val_losses = test_epoch(self.val_data_loader, self.model, self.device, loss_functions)
            test_losses = test_epoch(self.test_data_loader, self.model, self.device, loss_functions)
            
            training_report(self.tb_writer, epoch, train_losses, val_losses, test_losses)

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
            epoch += 1

            # Do inference on the new samples
            self.new_samples_mutex.lock()
            if self.new_samples:
                with torch.no_grad():
                    faces_input = torch.stack([FaceDataset.face_sample_to_tensor(sample, self.device) for sample in self.new_samples])
                    predictions = self.model(faces_input).cpu().detach().numpy()
                    self.predicted_samples.emit(zip(self.new_samples, predictions))
                self.new_samples = []
            self.new_samples_mutex.unlock()
