import queue
import collections

from PyQt6.QtCore import QThread, QObject, QEventLoop, pyqtSignal, pyqtSlot


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
            # TODO: remove?
            self.eventDispatcher().processEvents(QEventLoop.ProcessEventsFlag.AllEvents)

class AttributeSelector(QObject):
    """Emits an attribute of the incomming object."""

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
    """Converts the incomming objects using the specified function and emits the result."""

    output = pyqtSignal(object)

    def __init__(self, function: collections.abc.Callable) -> None:
        super().__init__()
        self.function = function
    
    @pyqtSlot(object)
    def on_input(self, object: object):
        self.output.emit(self.function(object))

class Filter(QObject):
    """Emits an incoming object only if the predicate evaluates to True."""

    output = pyqtSignal(object)
    
    def __init__(self, predicate: collections.abc.Callable):
        super().__init__()
        self.predicate = predicate
    
    @pyqtSlot(object)
    def on_input(self, object):
        if self.predicate(self, object):
            self.output.emit(object)
