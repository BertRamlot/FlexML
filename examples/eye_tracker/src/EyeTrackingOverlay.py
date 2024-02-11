import logging
import numpy as np
import time
from PyQt6 import QtGui, QtWidgets, QtCore

from examples.eye_tracker.src.FaceSample import FaceSample

class EyeTrackingOverlay(QtWidgets.QMainWindow):

    GROUND_TRUTH_HISTORY_LENGTH = 20
    INFERENCE_HISTORY_LENGTH = 50

    def __init__(self, window_dims: np.ndarray):
        super().__init__(flags=QtCore.Qt.WindowType.FramelessWindowHint | QtCore.Qt.WindowType.WindowStaysOnTopHint)
        self.window_dims = window_dims # [h, w]
        self.gt_history = []
        self.inference_history: dict[int, tuple[float, list[np.ndarray]]] = {}

        self.setGeometry(0, 0, self.window_dims[1], self.window_dims[0])
        self.setStyleSheet("background:transparent")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.showFullScreen()

    @QtCore.pyqtSlot(np.ndarray)
    def register_gt_position(self, pos: np.ndarray):
        self.gt_history = self.gt_history[1-EyeTrackingOverlay.GROUND_TRUTH_HISTORY_LENGTH:] + [pos]
        self.update()

    @QtCore.pyqtSlot(list, np.ndarray)
    def register_inference_positions(self, face_samples: list[FaceSample], predictions: np.ndarray):
        now_time = time.time()
        for face_sample, prediction in zip(face_samples, predictions):
            if face_sample.face_id in self.inference_history:
                last_inference_time, old_history = self.inference_history[face_sample.face_id]
                if last_inference_time + 1.0 > now_time:
                    new_history = old_history[1-EyeTrackingOverlay.INFERENCE_HISTORY_LENGTH:] + [prediction]
                else:
                    new_history = [prediction]
            else: 
                new_history = [prediction]
            self.inference_history[face_sample.face_id] = (time.time(), new_history)
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
        if not self.gt_history:
            return
        position = self.gt_history[-1]
        outer_ring_color = QtCore.Qt.GlobalColor.yellow
        inner_ring_color = QtCore.Qt.GlobalColor.red

        y, x = position.round().astype(np.int32)

        r = 10
        pen = QtGui.QPen(outer_ring_color, 8, QtCore.Qt.PenStyle.SolidLine)
        qp.setPen(pen)
        qp.drawEllipse(x-r, y-r, 2*r, 2*r)

        r = 5
        pen = QtGui.QPen(inner_ring_color, 2, QtCore.Qt.PenStyle.SolidLine)
        qp.setPen(pen)
        qp.drawEllipse(x-r, y-r, 2*r, 2*r)

    def draw_predicted_locations(self, qp: QtGui.QPainter):        
        colors = (QtCore.Qt.GlobalColor.blue, QtCore.Qt.GlobalColor.green, QtCore.Qt.GlobalColor.red, QtCore.Qt.GlobalColor.yellow)
        pen = QtGui.QPen(QtCore.Qt.GlobalColor.green, 4, QtCore.Qt.PenStyle.SolidLine)

        now_time = time.time()
        for face_id in self.inference_history:
            last_update_time, prediction_history = self.inference_history[face_id]
            if last_update_time + 1.0 < now_time:
                continue
            # Exponential average over the current prediction_history
            # This reduces the temporal instabilities
            alpha = 0.5
            predicted_pos = np.array([alpha*((1-alpha)**i)*p for i, p in enumerate(reversed(prediction_history))]).sum(axis=0)
            predicted_pos /= 1-(1-alpha)**len(prediction_history)
            # Put out of bound coordinates to the edge of the screen. This is not done in the training loop
            predicted_pos = (np.clip(predicted_pos, 0.0, 1.0) * self.window_dims.max()).round().astype(np.int32)

            # Draw inference circle
            r = 15
            y, x = predicted_pos
            pen.setColor(colors[face_id % len(colors)])
            qp.setPen(pen)
            qp.drawEllipse(x-r, y-r, 2*r, 2*r)
