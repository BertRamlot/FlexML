import sys
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("Tensorboard found")
except ImportError:
    TENSORBOARD_FOUND = False
    print("Tensorboard not found")


from src.FaceNeuralNetwork import FaceNeuralNetwork, FaceDataset


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

def train_one_epoch(train_loader, model, device, loss_functions, optimizer):
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

def training_report(tb_writer, epoch, train_losses, test_losses):
    for name in train_losses.keys() & test_losses.keys():
        dict = {}
        if name in train_losses:
            dict["train"] = train_losses[name]
        if name in test_losses:
            dict["test"] = test_losses[name]
        if tb_writer:
            tb_writer.add_scalars(f'{name}_loss', dict, epoch)

    print(f"{epoch:6} | train_loss={train_losses['criterion']:.5f}, test_loss={test_losses['criterion']:.5f}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args(sys.argv[1:])


    model_path = Path("models") / args.model_name
    model_path.mkdir(parents=True, exist_ok=True)
    dataset_path = Path("datasets") / args.dataset
    if not dataset_path.exists():
        raise FileNotFoundError("dataset not found")

    tb_writer = SummaryWriter(model_path) if TENSORBOARD_FOUND else None

    print("Loading training data set ... ", end="")
    train_dataset = FaceDataset(dataset_path, args.device, testing=False)
    train_data_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    print(f"{len(train_dataset)} samples loaded")

    print("Loading testing data set ... ", end="")
    test_dataset = FaceDataset(dataset_path, args.device, testing=True)
    test_data_loader = DataLoader(test_dataset, batch_size=1)
    print(f"{len(test_dataset)} samples loaded")

    model = FaceNeuralNetwork().to(args.device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    last_epoch_file = max(model_path.glob("epoch_*.pth"), default=None)
    if last_epoch_file is None:
        print("No existing model found")
        epoch = 0
    else:
        print("Loading existing model:", last_epoch_file)
        checkpoint = torch.load(last_epoch_file)       
        epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print("Starting training")
    loss_functions = {
        "L1_x": lambda y1, y2: (y1[..., 0] - y2[..., 0]).abs().mean(),
        "L1_y": lambda y1, y2: (y1[..., 1] - y2[..., 1]).abs().mean(),
        "euclid": lambda y1, y2: (y1-y2).pow(2).sum(-1).sqrt().mean(),
        "criterion": torch.nn.MSELoss()
    }
    max_epochs = 1e7 if args.max_epochs is None else args.max_epochs
    while epoch < max_epochs:
        train_losses = train_one_epoch(train_data_loader, model, args.device, loss_functions, optimizer)
        test_losses = test_epoch(test_data_loader, model, args.device, loss_functions)
        
        training_report(tb_writer, epoch, train_losses, test_losses)

        if epoch % 50 == 0 or epoch+1 == max_epochs:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                model_path / f"epoch_{epoch:0>6}.pth"
            )
        
        epoch += 1
    