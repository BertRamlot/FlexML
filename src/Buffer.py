import queue
from PyQt6.QtCore import QThread, pyqtSlot, pyqtSignal


class BufferThread(QThread):
    new_item = pyqtSignal(object)

    def __init__(self, queue: queue.Queue):
        super().__init__()
        self._queue = queue

    @pyqtSlot(object)
    def push(self, item):
        try:
            self._queue.put_nowait(item)
            print("New len=", self._queue.qsize())
        except queue.Full:
            pass

    def run(self):
        while True:
            item = self._queue.get()
            self.new_item.emit(item)
            self._queue.task_done()

