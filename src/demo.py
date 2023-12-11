import ctypes
import sys
from pathlib import Path
from argparse import ArgumentParser
from PyQt5 import QtGui, QtWidgets, QtCore
import cv2
import torch
import numpy as np

from src.FaceNeuralNetwork import FaceDataset, FaceNeuralNetwork
from src.FaceDetector import FaceDetector


class EyeTrackingOverlay(QtWidgets.QMainWindow):
    def __init__(self, model, history_amount: int, device):
        QtWidgets.QMainWindow.__init__(self, None, QtCore.Qt.WindowStaysOnTopHint)

        self.screen_dims = np.array([ctypes.windll.user32.GetSystemMetrics(i) for i in range(2)])
        self.setGeometry(0, 0, *self.screen_dims)
        self.setStyleSheet("background:transparent")
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)

        self.device = device
        self.history_amount = history_amount
        self.pred_locations_history = [[] for _ in range(history_amount)]

        self.model = model

        print("Starting camera ...")
        self.cap = cv2.VideoCapture(0)
        self.faceDetector = FaceDetector(self.cap)

        timer = QtCore.QTimer(self, timeout=self.update_eyes)
        timer.start()

    #region Painting
    def paintEvent(self, e):
        qp = QtGui.QPainter()
        qp.begin(self)
        self.drawPredictedLocations(qp)
        qp.end()

    def drawPredictedLocations(self, qp):
        colors = [QtCore.Qt.blue, QtCore.Qt.green, QtCore.Qt.red, QtCore.Qt.yellow]

        pen = QtGui.QPen(QtCore.Qt.green, 4, QtCore.Qt.SolidLine)

        if len(self.pred_locations_history[-1]) == 0:
            r = 200
            x, y = self.screen_dims // 2
            qp.setPen(pen)
            qp.drawEllipse(x-r, y-r, 2*r, 2*r)
            qp.drawText(x-50, y, "No valid face(s) found")
        else:
            r = 30
            for i, pred_loc in enumerate(self.pred_locations_history[-1]):
                pen.setColor(colors[i % len(colors)])
                qp.setPen(pen)
                qp.drawEllipse(pred_loc[0]-r, pred_loc[1]-r, 2*r, 2*r)
    #endregion

    def exit(self):
        self.cap.release()

    @QtCore.pyqtSlot()
    def update_eyes(self):
        self.faceDetector.update()

        pred_locations = []
        if self.faceDetector.valid_faces_found():
            faces_input = torch.stack([FaceDataset.face_to_tensor(face, self.device) for face in self.faceDetector.last_faces])
            predictions = self.model(faces_input).cpu().detach().numpy()
            pred_locations = (np.clip(predictions, 0.0, 1.0) * self.screen_dims).round().astype(np.int32)
        self.pred_locations_history = self.pred_locations_history[1:]
        self.pred_locations_history.append(pred_locations)

        self.update()

if __name__ == "__main__":
    parser = ArgumentParser(description="Demo script parameters")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument("--epoch", type=int, default=None)
    args = parser.parse_args(sys.argv[1:])

    model_path = Path("models") / args.model_name
    if args.epoch is None:
        pth_path = max(model_path.glob("epoch_*.pth"), default=None)
    else:
        pth_path = model_path / f"epoch_{args.epoch}.pth"
    checkpoint = torch.load(pth_path)
    model = FaceNeuralNetwork().to(args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    app = QtWidgets.QApplication(sys.argv)
    window = EyeTrackingOverlay(model, 10, args.device)
    window.showFullScreen()
    sys.exit(app.exec_())