import numpy as np
import ctypes
from PyQt6 import QtGui, QtWidgets, QtCore


class EyeTrackingOverlay(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__(None, QtCore.Qt.WindowType.WindowStaysOnTopHint)

        self.window_dims = np.array([ctypes.windll.user32.GetSystemMetrics(i) for i in range(2)], dtype=np.int32)
        self.setGeometry(0, 0, *self.window_dims)
        self.setStyleSheet("background:transparent")
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.showFullScreen()

        self.gt_history = []
        self.inference_history = []

    @QtCore.pyqtSlot(tuple)
    def register_gt_position(self, pos):
        self.gt_history.append(pos)
        if len(self.gt_history) > 5:
            self.gt_history = self.gt_history[-5:]
        self.update()

    @QtCore.pyqtSlot(list)
    def register_inference_positions(self, positions):
        self.inference_history.append(positions * self.window_dims.max())
        if len(self.inference_history) > 5:
            self.inference_history = self.inference_history[-5:]
        self.update()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        super().keyPressEvent(event)
        if event.key() == ord("q"):
            print("Closing EyeTrackingOverlay")
            self.close()

    def paintEvent(self, e):
        qp = QtGui.QPainter()
        qp.begin(self)
        self.draw_target_location(qp)
        self.draw_predicted_locations(qp)
        qp.end()

    def draw_target_location(self, qp):
        if not self.gt_history:
            return
        position = self.gt_history[-1]
        outer_ring_color = QtCore.Qt.yellow
        inner_ring_color = QtCore.Qt.red

        x, y = position.round().astype(np.int32)

        r = 10
        pen = QtGui.QPen(outer_ring_color, 8, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        qp.drawEllipse(x-r, y-r, 2*r, 2*r)

        r = 5
        pen = QtGui.QPen(inner_ring_color, 2, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        qp.drawEllipse(x-r, y-r, 2*r, 2*r)

    def draw_predicted_locations(self, qp):
        if not self.inference_history:
            return
        
        colors = (QtCore.Qt.blue, QtCore.Qt.green, QtCore.Qt.red, QtCore.Qt.yellow)
        pen = QtGui.QPen(QtCore.Qt.green, 4, QtCore.Qt.SolidLine)

        if len(self.inference_history[-1]) == 0:
            r = 200
            x, y = self.screen_dims // 2
            qp.setPen(pen)
            qp.drawEllipse(x-r, y-r, 2*r, 2*r)
            qp.drawText(x-50, y, "No valid face(s) found")

        # Inference circles
        r = 30
        for i, pred_loc in enumerate(self.inference_history[-1]):
            x, y = (pred_loc * self.window_dims.max()).round().astype(np.int32)

            pen.setColor(colors[i % len(colors)])
            qp.setPen(pen)
            qp.drawEllipse(x-r, y-r, 2*r, 2*r)
