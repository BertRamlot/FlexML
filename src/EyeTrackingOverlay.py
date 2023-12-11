import ctypes
import cv2
import numpy as np
import torch
from PyQt5 import QtGui, QtWidgets, QtCore

from src.FaceDetector import FaceDetector
from src.FaceNeuralNetwork import FaceDataset, FaceNeuralNetwork
from src.DataGenerator import DataGenerator


class EyeTrackingOverlay(QtWidgets.QMainWindow):
    def __init__(self, device=None, model: FaceNeuralNetwork = None, data_generator: DataGenerator = None):
        QtWidgets.QMainWindow.__init__(self, None, QtCore.Qt.WindowStaysOnTopHint)

        # Overlay setup
        screen_dims = np.array([ctypes.windll.user32.GetSystemMetrics(i) for i in range(2)], dtype=np.int32)
        self.max_screen_dim = screen_dims.max()
        self.setGeometry(0, 0, *screen_dims)
        self.setStyleSheet("background:transparent")
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)

        # Inference setup
        self.device = device
        self.model = model
        self.prediction_history = [[] for _ in range(5)]

        # Data generation setup
        self.data_generator = data_generator

        # Face detection setup
        print("Starting camera ... (this can take a while, I don't know why)")
        self.cap = cv2.VideoCapture(0)
        self.face_detector = FaceDetector(self.cap)
        self.face_detector_timer = QtCore.QTimer(self, timeout=self.detect_faces, interval=0.1)
        self.face_detector_timer.start()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        super(EyeTrackingOverlay, self).keyPressEvent(event)
        if event.key() == ord('Q'):
            print("Closing EyeTrackingOverlay")
            self.face_detector_timer.stop()
            if self.data_generator:
                self.data_generator.flush()
                self.cap.exit()
            self.close()
        elif event.key() == ord('C'):
            print("Cancelling recent captures")
            if self.data_generator:
                self.data_generator.clear_buffer()

    def paintEvent(self, e):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.data_generator:
            self.drawTargetLocation(qp)
        if self.model:
            self.drawPredictedLocations(qp)
        qp.end()

    def drawTargetLocation(self, qp):
        position, capture = self.data_generator.get_target_position()
        outer_ring_color = QtCore.Qt.yellow if capture else QtCore.Qt.blue
        inner_ring_color = QtCore.Qt.red if capture else QtCore.Qt.red

        x, y = (position * self.max_screen_dim).round().astype(np.int32)

        r = 6
        pen = QtGui.QPen(outer_ring_color, 8, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        qp.drawEllipse(x-r, y-r, 2*r, 2*r)

        r = 2
        pen = QtGui.QPen(inner_ring_color, 2, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        qp.drawEllipse(x-r, y-r, 2*r, 2*r)

    def drawPredictedLocations(self, qp):
        colors = (QtCore.Qt.blue, QtCore.Qt.green, QtCore.Qt.red, QtCore.Qt.yellow)
        pen = QtGui.QPen(QtCore.Qt.green, 4, QtCore.Qt.SolidLine)

        if len(self.prediction_history[-1]) == 0:
            r = 200
            x, y = self.screen_dims // 2
            qp.setPen(pen)
            qp.drawEllipse(x-r, y-r, 2*r, 2*r)
            qp.drawText(x-50, y, "No valid face(s) found")

        # Inference
        r = 30
        for i, pred_loc in enumerate(self.prediction_history[-1]):
            pen.setColor(colors[i % len(colors)])
            qp.setPen(pen)
            qp.drawEllipse(pred_loc[0]-r, pred_loc[1]-r, 2*r, 2*r)

    def exit(self):
        self.cap.release()

    @QtCore.pyqtSlot()
    def detect_faces(self):
        capture = False
        if self.data_generator:
            target, capture = self.data_generator.get_target_position()
        if self.model:
            capture = True

        if capture:
            self.face_detector.update()
            if self.face_detector.faces_found():
                # Inference
                if self.model:
                    pred_locations = []
                    faces_input = torch.stack([FaceDataset.face_to_tensor(face, self.device) for face in self.face_detector.last_faces])
                    predictions = self.model(faces_input).cpu().detach().numpy()
                    pred_locations = (predictions * self.max_screen_dim).round().astype(np.int32)
                    self.prediction_history = self.prediction_history[1:] + [pred_locations]

                # Data generation
                if self.data_generator:
                    for face in self.face_detector.last_faces:
                        self.data_generator.register_sample(face, *target)

        self.update()
