import queue

from PyQt6.QtCore import QThread, QObject, pyqtSignal, pyqtSlot


# TODO: support primitives
class ConstantSource(QThread):
    new_item = pyqtSignal(object)

    def __init__(self, obj: object):
        super().__init__()
        self.obj = obj

    def run(self):
        while True:
            self.new_item.emit(self.obj)


class BufferThread(QThread):
    new_item = pyqtSignal(object)

    def __init__(self, queue: queue.Queue):
        super().__init__(None)
        self._queue = queue
        self.start()

    @pyqtSlot(object)
    def push(self, item):
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            pass

    def run(self):
        while True:
            item = self._queue.get()
            self.new_item.emit(item)
            self._queue.task_done()

class AttributeSelector(QObject):
    output_attribute = pyqtSignal(object)

    def __init__(self, attribute_name: str, call_if_callable: bool = False):
        self.attribute_name = attribute_name
        self.call_if_callable = call_if_callable

    @pyqtSlot(object)
    def input_objec(self, object: object):
        attr = getattr(object, self.attribute_name)
        if self.call_if_callable and callable(attr):
            attr = attr()
        self.output_attribute.emit(attr)

class Muxer(QObject):
    output = pyqtSignal(object)

    def __init__(self):
        super().__init__()

class Demuxer(QObject):
    output = pyqtSignal()

    def __init__(self):
        super().__init__()

