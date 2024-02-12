import numpy as np
import time
from PyQt6 import QtGui, QtWidgets, QtCore

from examples.eye_tracker.src.FaceSample import FaceSample


class EyeTrackingOverlay(QtWidgets.QMainWindow):
    # TODO: hard coded
    GROUND_TRUTH_HISTORY_LENGTH = 20
    PREDICTION_HISTORY_TIME_WINDOW = 1.0

    def __init__(self, window_dims: np.ndarray):
        """
        Initialize the EyeTrackingOverlay.

        Args:
            window_dims (np.ndarray): Dimensions of the overlay window, shape is (2,) with format (height, width).
        """
        super().__init__(flags=QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.WindowStaysOnTopHint)
        self.window_dims = window_dims # [h, w]
        self.ground_truth_history: list[np.ndarray] = []
        self.prediction_history: dict[int, list[tuple[float, np.ndarray]]] = {}

        self.setGeometry(0, 0, *np.flip(self.window_dims))
        self.setStyleSheet("background:transparent")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.showFullScreen()

    @QtCore.pyqtSlot(np.ndarray)
    def register_ground_truth(self, pos: np.ndarray):
        """
        Register the ground truth position.

        Args:
            pos (np.ndarray): Ground truth position, shape is (2,) with format (y, x).
        """
        self.ground_truth_history = self.ground_truth_history[-(EyeTrackingOverlay.GROUND_TRUTH_HISTORY_LENGTH-1):] + [pos]
        self.update()

    @QtCore.pyqtSlot(list, np.ndarray)
    def register_predictions(self, face_samples: list[FaceSample], predictions: np.ndarray):
        """
        Register predictions for a list of face samples.

        Args:
            face_samples (list[FaceSample]): List of N face samples.
            predictions (np.ndarray): Prediction per face sample, shape is (N, 2) where each row is a (y, x) position.
        """
        for face_sample, prediction in zip(face_samples, predictions):
            if face_sample.face_id in self.prediction_history:
                min_time = face_sample.creation_time - EyeTrackingOverlay.PREDICTION_HISTORY_TIME_WINDOW
                new_history = [(time, pos) for time, pos in self.prediction_history[face_sample.face_id] if time >= min_time]
            else: 
                new_history = []
            new_history.append((face_sample.creation_time, prediction))
            self.prediction_history[face_sample.face_id] = new_history
        self.update()

    def paintEvent(self, e):
        qp = QtGui.QPainter()
        qp.begin(self)
        self.draw_target_location(qp)
        self.draw_predicted_locations(qp)
        qp.end()

    def draw_target_location(self, qp: QtGui.QPainter):
        if not self.ground_truth_history:
            return
        
        # Draws the path of the ground truth
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
        for face_id in self.prediction_history:
            if len(self.prediction_history[face_id]) == 0:
                continue
            times, positions = zip(*self.prediction_history[face_id])
            
            # Exponential average over the current prediction_history to reduce temporal instabilities.
            # higher alpha => higher weights for newer samples
            alpha = 10.0
            factors = np.exp((np.array(times) - now_time) * alpha)
            factors /= factors.sum()
            predicted_pos = (factors[:, np.newaxis] * positions).sum(axis=0) 
            # Put out of bound coordinates to the edge of the screen. This is not done in the training loop
            predicted_pos = (np.clip(predicted_pos, 0.0, 1.0) * self.window_dims.max()).round().astype(np.int32)

            # Draw inference circle
            r = 15
            y, x = predicted_pos
            pen.setColor(colors[face_id % len(colors)])
            qp.setPen(pen)
            qp.drawEllipse(x-r, y-r, 2*r, 2*r)
