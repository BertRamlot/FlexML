import logging
import numpy as np
import time
from PyQt6 import QtGui, QtWidgets, QtCore

from examples.eye_tracker.src.FaceSample import FaceSample


class EyeTrackingOverlay(QtWidgets.QMainWindow):

    # TODO: hard coded
    GROUND_TRUTH_HISTORY_LENGTH = 20
    INFERENCE_HISTORY_LENGTH = 50
    INFERENCE_HISTORY_TIME = 1.0

    def __init__(self, window_dims: np.ndarray):
        super().__init__(flags=QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.WindowStaysOnTopHint)
        self.window_dims = window_dims # [h, w]
        self.ground_truth_history: list[np.ndarray] = []
        self.inference_history: dict[int, tuple[list[float, np.ndarray]]] = {}

        self.setGeometry(0, 0, self.window_dims[1], self.window_dims[0])
        self.setStyleSheet("background:transparent")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.showFullScreen()

    @QtCore.pyqtSlot(np.ndarray)
    def register_gt_position(self, pos: np.ndarray):
        self.ground_truth_history = self.ground_truth_history[1-EyeTrackingOverlay.GROUND_TRUTH_HISTORY_LENGTH:] + [pos]
        self.update()

    @QtCore.pyqtSlot(list, np.ndarray)
    def register_inference_positions(self, face_samples: list[FaceSample], predictions: np.ndarray):
        now_time = time.time()
        for face_sample, prediction in zip(face_samples, predictions):
            if face_sample.face_id in self.inference_history:
                min_time = now_time - EyeTrackingOverlay.INFERENCE_HISTORY_TIME
                new_history = [(time, pos) for time, pos in self.inference_history[face_sample.face_id] if time >= min_time]
            else: 
                new_history = []
            new_history.append((now_time, prediction))
            self.inference_history[face_sample.face_id] = new_history
        self.update()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        super().keyPressEvent(event)
        if event.key() == ord("q"):
            logging.info("Closing overlay")
            self.close()

    def paintEvent(self, e):
        qp = QtGui.QPainter()
        qp.begin(self)
        self.draw_target_location(qp)
        self.draw_predicted_locations(qp)
        qp.end()

    def draw_target_location(self, qp: QtGui.QPainter):
        if not self.ground_truth_history:
            return
        
        # Draws the path of the ground_truth
        # polygon = QtGui.QPolygon()
        # for point in self.ground_truth_history:
        #     polygon.append(QtCore.QPoint(*np.flip(point)))
        # pen = QtGui.QPen(QtCore.Qt.GlobalColor.green, 1, QtCore.Qt.PenStyle.SolidLine)
        # qp.setPen(pen)
        # qp.drawPolyline(polygon)
        
        # Draw the target (ball):
        position = self.ground_truth_history[-1]
        y, x = position.round().astype(np.int32)
        # - Draw outer ring
        r = 10
        pen = QtGui.QPen(QtCore.Qt.GlobalColor.yellow, 8, QtCore.Qt.PenStyle.SolidLine)
        qp.setPen(pen)
        qp.drawEllipse(x-r, y-r, 2*r, 2*r)
        # - Draw inner ring
        r = 5
        pen = QtGui.QPen(QtCore.Qt.GlobalColor.red, 2, QtCore.Qt.PenStyle.SolidLine)
        qp.setPen(pen)
        qp.drawEllipse(x-r, y-r, 2*r, 2*r)

    def draw_predicted_locations(self, qp: QtGui.QPainter):        
        colors = (QtCore.Qt.GlobalColor.blue, QtCore.Qt.GlobalColor.green, QtCore.Qt.GlobalColor.red, QtCore.Qt.GlobalColor.yellow)
        pen = QtGui.QPen(QtCore.Qt.GlobalColor.green, 4, QtCore.Qt.PenStyle.SolidLine)

        now_time = time.time()
        for face_id in self.inference_history:
            if len(self.inference_history[face_id]) == 0:
                continue
            times, positions = zip(*self.inference_history[face_id])
            
            # Exponential average over the current prediction_history
            # This reduces the temporal instabilities: higher alpha => higher weight towards fresher samples
            alpha = 8.0
            factors = np.exp((np.array(times) - now_time) * alpha)
            factors /= factors.sum()
            # print(factors)
            predicted_pos = (factors[:, np.newaxis] * positions).sum(axis=0) 
            # Put out of bound coordinates to the edge of the screen. This is not done in the training loop
            predicted_pos = (np.clip(predicted_pos, 0.0, 1.0) * self.window_dims.max()).round().astype(np.int32)

            # Draw inference circle
            r = 15
            y, x = predicted_pos
            pen.setColor(colors[face_id % len(colors)])
            qp.setPen(pen)
            qp.drawEllipse(x-r, y-r, 2*r, 2*r)
