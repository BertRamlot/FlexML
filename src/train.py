import os
import sys
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


def test_epoch(test_loader, model, criterion, device):
    epoch_loss = 0
    with torch.no_grad():
        for X, y in test_dataloader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            output = model(X)
            loss = criterion(output, y)
            epoch_loss += loss.item()
    
    return epoch_loss / len(test_loader)

def train_one_epoch(train_loader, model, criterion, optimizer, device):
    epoch_loss = 0
    for batch, (X, y) in enumerate(train_loader):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        output = model(X)
        loss = criterion(output, y)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return epoch_loss / len(train_dataloader)

def training_report(tb_writer, epoch, train_loss, test_loss):
    if tb_writer:
        tb_writer.add_scalar('train/l2_loss', train_loss, epoch)
        tb_writer.add_scalar('test/l2_loss', test_loss, epoch)

    print(f"{epoch} | train_loss={train_loss}, test_loss={test_loss}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_set_name", type=str, required=True)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-2)
    args = parser.parse_args(sys.argv[1:])

    model_path = Path("models") / args.model_name
    data_set_path = Path("data_sets") / args.data_set_name
    os.makedirs(model_path, exist_ok = True)
    os.makedirs(data_set_path, exist_ok = True)

    tb_writer = SummaryWriter(model_path) if TENSORBOARD_FOUND else None

    # loss_fn = torch.nn.L1Loss()
    loss_fn = torch.nn.MSELoss()

    print("Loading training data set")
    train_data_set = FaceDataset(data_set_path, args.device, testing=False)
    train_dataloader = DataLoader(train_data_set, batch_size=8, shuffle=True)

    print("Loading testing data set")
    test_data_set = FaceDataset(data_set_path, args.device, testing=True)
    test_dataloader = DataLoader(test_data_set, batch_size=1, shuffle=True)

    model = FaceNeuralNetwork().to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

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
    def criterion(output, target):
        clamped = torch.clamp(output, min=0.0, max=1.0)
        return loss_fn(clamped, target)

    max_epochs = 1e7 if args.max_epochs is None else args.max_epochs
    while epoch < max_epochs:
        model.train()
        train_loss = train_one_epoch(train_dataloader, model, criterion, optimizer, args.device)

        model.eval()
        test_loss = test_epoch(test_dataloader, model, criterion, args.device)
        
        training_report(tb_writer, epoch, train_loss, test_loss)

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },
            model_path / f"epoch_{epoch:0>6}.pth"
        )
        
        epoch += 1
    