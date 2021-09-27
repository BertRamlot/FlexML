import ctypes
import sys
from PyQt5 import QtGui, QtWidgets, QtCore
import cv2
import torch

from src.FaceNeuralNetwork import FaceDataset, FaceNeuralNetwork
from src.FaceDetector import FaceDetector


class EyeTrackingOverlay(QtWidgets.QMainWindow):
    def __init__(self, model_name: str, history_amount: int):
        QtWidgets.QMainWindow.__init__(self, None, QtCore.Qt.WindowStaysOnTopHint)

        user32 = ctypes.windll.user32
        self.screen_width, self.screen_heigth = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        print("Detected screen dims: {}x{}".format(self.screen_width, self.screen_heigth))

        self.history_amount = history_amount
        self.pred_locations_history = [[] for _ in range(history_amount)]

        self.initNetwork(model_name)
        print("Starting camera")
        self.cap = cv2.VideoCapture(0)
        self.initFaceDetection(self.cap)
        self.initUI()

        timer = QtCore.QTimer(self, timeout=self.update_eyes, interval=1)
        timer.start()

    #region Init
    def initNetwork(self, model_name: str):
        self.device = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(self.device))

        self.model = FaceNeuralNetwork().to(self.device)

        checkpoint = torch.load(model_name)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def initFaceDetection(self, cap):
        self.faceDetector = FaceDetector(cap)

    def initUI(self):
        self.setGeometry(0, 0, self.screen_width, self.screen_heigth)
        self.setStyleSheet("background:transparent")
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)
    #endregion
    
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
            qp.setPen(pen)
            qp.drawEllipse(round(self.screen_width/2)-r, round(self.screen_heigth/2)-r, 2*r, 2*r)
            qp.drawText(round(self.screen_width/2), round(self.screen_heigth/2), "No valid face(s) found")
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
            preds = self.model(faces_input)
            clamped_preds = torch.clamp(preds, min=0.0, max=1.0)
            pred_locations = [(round(cp[0].item()*self.screen_width), round(cp[1].item()*self.screen_heigth)) for cp in clamped_preds]

        self.pred_locations_history = self.pred_locations_history[1:]
        self.pred_locations_history.append(pred_locations)

        self.update()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = EyeTrackingOverlay('models/model_weights_p_B_abs.pth', 10)
    window.showFullScreen()
    # window.exit()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()