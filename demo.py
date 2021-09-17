import ctypes
import sys
from PyQt5 import QtGui, QtWidgets, QtCore
import torch

from EyeNeuralNetwork import EyeDataset, EyeNeuralNetwork
from EyeDetector import EyeDetector

class EyeTrackingOverlay(QtWidgets.QMainWindow):
    def __init__(self, model_name: str):
        QtWidgets.QMainWindow.__init__(self, None, QtCore.Qt.WindowStaysOnTopHint)

        user32 = ctypes.windll.user32
        self.screen_width, self.screen_heigth = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        print("Detected screen dims: {}x{}".format(self.screen_width, self.screen_heigth))

        self.left_eye = None
        self.right_eye = None

        self.initNetwork(model_name)
        self.initEyeDetection()
        self.initUI()

        timer = QtCore.QTimer(self, timeout=self.update_eyes, interval=10)
        timer.start()

    def initNetwork(self, model_name: str):
        self.device = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(self.device))

        self.model = EyeNeuralNetwork().to(self.device)

        checkpoint = torch.load(model_name)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def initEyeDetection(self):
        self.eyeDetector = EyeDetector()

    def initUI(self):
        self.setGeometry(0, 0, self.screen_width, self.screen_heigth)
        self.setStyleSheet("background:transparent")
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)

    @QtCore.pyqtSlot()
    def update_eyes(self):
        self.eyeDetector.update()

        if not self.eyeDetector.valid_eyes_found():
            self.left_eye = None
            self.right_eye = None
        else:
            # Todo: multiple eyes
            processed_eyes = EyeDataset.eye_pair_to_tensor(self.eyeDetector.last_eye_pairs[0], self.device).unsqueeze(dim=0)
            preds = self.model(processed_eyes)
            clamped_preds = torch.clamp(preds, min=0.0, max=1.0)
            pred_locations = [(round(cp[0].item()*self.screen_width), round(cp[1].item()*self.screen_heigth)) for cp in clamped_preds]
            # pred_locations = [(round(cp[0].item()*self.screen_width), 200) for cp in clamped_preds]
            for i, pred_loc in enumerate(pred_locations):
                if i == 0:
                    self.left_eye = pred_loc
                else:
                    self.right_eye = pred_loc
            # print("  {} -> pred_locations: {}".format(clamped_preds, pred_locations)) #, end='\r')
        self.update()

    def paintEvent(self, e):
        qp = QtGui.QPainter()
        qp.begin(self)
        self.drawPredictedEyes(qp)
        qp.end()

    def drawPredictedEyes(self, qp):
        pen = QtGui.QPen(QtCore.Qt.green, 4, QtCore.Qt.SolidLine)

        if self.left_eye is None and self.right_eye is None:
            r = 200
            qp.setPen(pen)
            qp.drawEllipse(round(self.screen_width/2)-r, round(self.screen_heigth/2)-r, 2*r, 2*r)
            qp.drawText(round(self.screen_width/2), round(self.screen_heigth/2), "No valid eyes found")
        else:
            r = 30
            if not self.left_eye is None:
                qp.setPen(pen)
                qp.drawEllipse(self.left_eye[0]-r, self.left_eye[1]-r, 2*r, 2*r)
                for dx in [0, 2*r]:
                    for dy in [0, 2*r]:
                        qp.drawText(self.left_eye[0]-r+dx, self.left_eye[1]-r+dy, "L")

            if not self.right_eye is None:
                pen.setColor(QtCore.Qt.blue)
                qp.setPen(pen)
                qp.drawEllipse(self.right_eye[0]-r, self.right_eye[1]-r, 2*r, 2*r)
                for dx in [0, 2*r]:
                    for dy in [0, 2*r]:
                        qp.drawText(self.right_eye[0]-r+dx, self.right_eye[1]-r+dy, "R")

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = EyeTrackingOverlay('model_weights.pth')
    window.showFullScreen()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()