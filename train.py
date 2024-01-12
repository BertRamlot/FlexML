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

from src.face_based.FaceNetwork import FaceNetwork, FaceDataset


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

    tb_writer = SummaryWriter(model_path / "logs") if TENSORBOARD_FOUND else None

    print("Loading training data set ... ", end="")
    train_dataset = FaceDataset(dataset_path, args.device, type="train")
    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print(f"{len(train_dataset)} samples loaded")

    print("Loading testing data set ... ", end="")
    test_dataset = FaceDataset(dataset_path, args.device, type="val")
    test_data_loader = DataLoader(test_dataset, batch_size=1)
    print(f"{len(test_dataset)} samples loaded")

    model = FaceNetwork().to(args.device)
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
        "criterion": lambda y1, y2: (y1-y2).pow(2).sum(-1).sqrt().mean()
        # "criterion": torch.nn.L1Loss()
        # "criterion": torch.nn.MSELoss()
    }
    max_epochs = 1e7 if args.max_epochs is None else args.max_epochs
    while epoch < max_epochs:
        train_losses = train_epoch(train_data_loader, model, args.device, loss_functions, optimizer)
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
    