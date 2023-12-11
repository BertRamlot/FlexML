import sys
from pathlib import Path
from argparse import ArgumentParser
import torch
from PyQt5 import QtWidgets

from src.EyeTrackingOverlay import EyeTrackingOverlay


if __name__ == "__main__":
    parser = ArgumentParser(description="Demo script parameters")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])

    if args.model:
        print("Model passed, doing inference")
        from src.FaceNeuralNetwork import FaceNeuralNetwork

        model_path = Path("models") / args.model
        if args.epoch is None:
            pth_path = max(model_path.glob("epoch_*.pth"), default=None)
        else:
            pth_path = model_path / f"epoch_{args.epoch}.pth"
        if pth_path is None or not pth_path.exists():
            raise FileNotFoundError("Failed to find model specified")
        
        checkpoint = torch.load(pth_path)
        model = FaceNeuralNetwork().to(args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    else:
        model = None

    if args.dataset:
        print("Dataset passed, generating data")
        import numpy as np
        import ctypes
        from src.DataGenerator import BallDataGenerator

        screen_dims = np.array([ctypes.windll.user32.GetSystemMetrics(i) for i in range(2)], dtype=np.float32)
        screen_dims /= screen_dims.max()
        data_generator = BallDataGenerator(Path("datasets") / args.dataset, 200, screen_dims)
    else:
        data_generator = None


    app = QtWidgets.QApplication(sys.argv)
    window = EyeTrackingOverlay(args.device, model=model, data_generator=data_generator)
    window.showFullScreen()
    sys.exit(app.exec_())