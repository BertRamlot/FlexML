import queue

from PyQt6.QtCore import QThread, QObject, QEventLoop, pyqtSignal, pyqtSlot


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
        self.exec()

    def exec(self):
        while True:
            item = self._queue.get()
            self.new_item.emit(item)
            self._queue.task_done()
            self.eventDispatcher().processEvents(QEventLoop.ProcessEventsFlag.AllEvents)

class AttributeSelector(QObject):
    output_attribute = pyqtSignal(object)

    def __init__(self, attribute_name: str, call_if_callable: bool = False):
        self.attribute_name = attribute_name
        self.call_if_callable = call_if_callable

    @pyqtSlot(object)
    def input_object(self, object: object):
        attr = getattr(object, self.attribute_name)
        if self.call_if_callable and callable(attr):
            attr = attr()
        self.output_attribute.emit(attr)

class Convertor(QObject):
    output = pyqtSignal(object)

    def __init__(self, function) -> None:
        super().__init__()
        self.function = function
    
    def on_input(self, object: object):
        self.output.emit(self.function(object))

class Filter(QObject):
    output = pyqtSignal(object)
    
    def __init__(self, predicate) -> None:
        super().__init__()
        self.predicate = predicate
    
    def on_input(self, object: object):
        if self.predicate(object):
            self.output.emit(object)
