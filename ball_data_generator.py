import sys
import random
from PyQt5 import QtGui, QtWidgets, QtCore
import ctypes
import time
import math

from DataGenerator import DataGenerator


class BallTrackingOverlay(QtWidgets.QMainWindow):
    def __init__(self, data_folder_file_name: str):
        QtWidgets.QMainWindow.__init__(self, None, QtCore.Qt.WindowStaysOnTopHint)

        user32 = ctypes.windll.user32
        self.screen_width, self.screen_heigth = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        self.step_count = 0

        self.dataGenerator = DataGenerator(data_folder_file_name, 200)

        self.ball_pos = [random.randint(0, self.screen_width), random.randint(0, self.screen_heigth)]
        self.ball_vel = [5, 5]

        self.initUI()

        timer = QtCore.QTimer(self, timeout=self.time_step, interval=1)
        timer.start()

    def initUI(self) -> None:
        self.setGeometry(0, 0, self.screen_width, self.screen_heigth)
        self.setStyleSheet("background:transparent")
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground, True)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        super(BallTrackingOverlay, self).keyPressEvent(event)
        if event.key() == ord('Q'):
            print("Closing ball tracker")
            self.dataGenerator.flush()
            self.dataGenerator.exit()
            self.close()
        elif event.key() == ord('C'):
            print("Cancelling recent captures")
            self.dataGenerator.clear_buffer()

    @QtCore.pyqtSlot()
    def time_step(self) -> None:
        if self.step_count > 100: 
            if self.step_count % 3 == 0:
                succes = self.dataGenerator.register_eye_position(self.ball_pos[0], self.ball_pos[1])
                print("\rEye status: {}".format("OK  " if succes else "ERR  "), end='')
        else:
            time.sleep(0.030)
        dp_max = 5
        bounced = False

        self.ball_pos[0] = round(self.ball_pos[0] + self.ball_vel[0]) # + random.randint(-dp_max, dp_max))
        if self.ball_pos[0] < 0:
            self.ball_pos[0] = 0
            self.ball_vel[0] *= -1
            bounced = True
        elif self.ball_pos[0] >= self.screen_width:
            self.ball_pos[0] = self.screen_width-1
            self.ball_vel[0] *= -1
            bounced = True

        self.ball_pos[1] = round(self.ball_pos[1] + self.ball_vel[1]) # + random.randint(-dp_max, dp_max))
        if self.ball_pos[1] < 0:
            self.ball_pos[1] = 0
            self.ball_vel[1] *= -1
            bounced = True
        elif self.ball_pos[1] >= self.screen_heigth:
            self.ball_pos[1] = self.screen_heigth-1
            self.ball_vel[1] *= -1
            bounced = True

        if bounced:
            rnd_angle = random.random()*math.pi/2
            vel_mag = math.sqrt(math.pow(self.ball_vel[0], 2) + math.pow(self.ball_vel[1], 2))
            self.ball_vel[0] = math.copysign(math.cos(rnd_angle)*vel_mag, self.ball_vel[0])
            self.ball_vel[1] = math.copysign(math.sin(rnd_angle)*vel_mag, self.ball_vel[1])

        self.step_count += 1

        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        qp = QtGui.QPainter()
        qp.begin(self)
        self.drawBall(qp)
        qp.end()

    def drawBall(self, qp: QtGui.QPainter) -> None:
        outer_ring_color = QtCore.Qt.yellow if self.step_count > 100 else QtCore.Qt.blue
        inner_ring_color = QtCore.Qt.red if self.step_count > 100 else QtCore.Qt.red

        r = 6
        pen = QtGui.QPen(outer_ring_color, 8, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        qp.drawEllipse(self.ball_pos[0]-r, self.ball_pos[1]-r, 2*r, 2*r)

        r = 2
        pen = QtGui.QPen(inner_ring_color, 2, QtCore.Qt.SolidLine)
        qp.setPen(pen)
        qp.drawEllipse(self.ball_pos[0]-r, self.ball_pos[1]-r, 2*r, 2*r)

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = BallTrackingOverlay("data_up_B")
    window.showFullScreen()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
