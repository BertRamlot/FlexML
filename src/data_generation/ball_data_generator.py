import sys
import random
import ctypes
import time
import math
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from PyQt5 import QtGui, QtWidgets, QtCore

from src.data_generation.DataGenerator import DataGenerator


class BallTrackingOverlay(QtWidgets.QMainWindow):
    def __init__(self, dataGenerator: DataGenerator):
        QtWidgets.QMainWindow.__init__(self, None, QtCore.Qt.WindowStaysOnTopHint)

        self.screen_dims = np.array([ctypes.windll.user32.GetSystemMetrics(i) for i in range(2)], dtype=np.int32)
        self.setGeometry(0, 0, *self.screen_dims)
        self.setStyleSheet("background:transparent")
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)

        self.dataGenerator = dataGenerator

        self.ball_pos = np.array([0.5, 0.5], dtype=np.float32)
        self.ball_vel = np.array([0.15, 0.15], dtype=np.float32)
        
        self.last_update_time = None
        self.start_time = None
        self.last_capture_time = 0
        
        self.timer = QtCore.QTimer(self, timeout=self.time_step)
        self.timer.start()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        super(BallTrackingOverlay, self).keyPressEvent(event)
        if event.key() == ord('Q'):
            print("Closing ball tracker")
            self.timer.stop()
            self.dataGenerator.flush()
            self.dataGenerator.exit()
            self.close()
        elif event.key() == ord('C'):
            print("Cancelling recent captures")
            self.dataGenerator.clear_buffer()

    @QtCore.pyqtSlot()
    def time_step(self) -> None:
        if self.start_time is None:
            self.start_time = time.time()

        elapsed_time = time.time() - self.start_time
        time_since_capture = time.time() - self.last_capture_time
        if elapsed_time > 5 and time_since_capture > 0.3: 
            self.last_capture_time = time.time()
            succes = self.dataGenerator.register_eye_position(self.ball_pos[0], self.ball_pos[1])
            print("\rEye status: {}".format("OK  " if succes else "ERR  "), end='')

        dt = 0.01
        if self.last_update_time is not None:
            time.sleep(max(0, self.last_update_time + 0.01 - time.time()))
        self.last_update_time = time.time()

        self.ball_pos += self.ball_vel*dt
        out_of_bounds_mask = (self.ball_pos < 0) | (self.ball_pos > 1)
        self.ball_vel *= np.where(out_of_bounds_mask, -1, 1)
        self.ball_pos = np.clip(self.ball_pos, 0, 1)
        if out_of_bounds_mask.any():
            rnd_angle = random.random()*math.pi/2
            vel_mag = np.linalg.norm(self.ball_vel)
            self.ball_vel = np.copysign(vel_mag * np.array([np.cos(rnd_angle), np.sin(rnd_angle)]), self.ball_vel)

        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        qp = QtGui.QPainter()
        qp.begin(self)
        self.drawBall(qp)
        qp.end()

    def drawBall(self, qp: QtGui.QPainter) -> None:
        elapsed_time = time.time() - self.start_time
        outer_ring_color = QtCore.Qt.yellow if elapsed_time > 5 else QtCore.Qt.blue
        inner_ring_color = QtCore.Qt.red if elapsed_time > 5 else QtCore.Qt.red

        x, y = (self.ball_pos * self.screen_dims).round().astype(np.int32)

        r = 6
        pen = QtGui.QPen(outer_ring_color, 8, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        qp.drawEllipse(x-r, y-r, 2*r, 2*r)

        r = 2
        pen = QtGui.QPen(inner_ring_color, 2, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        qp.drawEllipse(x-r, y-r, 2*r, 2*r)

if __name__ == "__main__":
    parser = ArgumentParser(description="Ball data generation script parameters")
    parser.add_argument("--data_set_name", type=str, required=True)
    args = parser.parse_args(sys.argv[1:])

    app = QtWidgets.QApplication(sys.argv)

    data_generator = DataGenerator(Path("data_sets") / args.data_set_name, 200)
    window = BallTrackingOverlay(data_generator)

    window.showFullScreen()
    sys.exit(app.exec_())
