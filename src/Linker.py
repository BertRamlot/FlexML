from typing import Callable
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QMetaMethod, QMetaObject

from src.Sample import SampleGenerator
from src.face_based.FaceSample import FaceSampleConvertor



def get_non_pyqt_methods(obj: QObject, method_type: QMetaMethod.MethodType, func_name: str|None) -> list[tuple[str, Callable, QMetaMethod]]:    
    candidate_methods = {}
    for subcls in type(obj).mro():
        if subcls.__module__.startswith("PyQt6."):
            # Ignore all native PyQt signals
            break
        for key, value in sorted(vars(subcls).items()):
            if not callable(value):
                continue
            if func_name is not None and key != func_name:
                continue
            candidate_methods[key] = getattr(obj, key)

    methods = []
    meta_obj = obj.metaObject()
    for i in range(meta_obj.methodCount()):
        meta_method_obj = meta_obj.method(i)
        method_name = meta_method_obj.methodSignature().data().decode().partition("(")[0]
        if method_name not in candidate_methods:
            continue
        if meta_method_obj.methodType() != method_type:
            continue
        methods.append((method_name, candidate_methods[method_name], meta_method_obj))
    return methods

"""
element should be any of the following:
- QObject
- (str, QObject)
- (QObject, str)
- (str, QObject, str)
"""
# TODO: no differentiation btwn different objects, only objects and primitives
# slots are linked using the decorator types: "pyqtSlot(...)"
def link_elements(*elements):
    # Unify input
    tuple_elements = []
    for element in elements:
        if element is None:
            tuple_elements.append((None, None, None))
            continue
        if isinstance(element, tuple):
            if len(element) == 3:
                tuple_elements.append(element)
                continue
            if len(element) == 2:
                if isinstance(element[0], QObject):
                    obj, signal_name = element
                    tuple_elements.append((None, obj, signal_name))
                    continue
                if isinstance(element[1], QObject):
                    slot_name, obj = element
                    tuple_elements.append((slot_name, obj, None))
                    continue
            raise ValueError("Invalid tuple (Wrong length):", element)
        if isinstance(element, QObject):
            obj = element
            tuple_elements.append((None, obj, None))
            continue
        raise ValueError("Invalid element (Invalid type):", element)

    
    for i in range(1, len(tuple_elements)):
        _, send_obj, send_signal_name = tuple_elements[i-1]
        rcv_slot_name, rcv_obj, _ = tuple_elements[i]

        if send_obj is None or rcv_obj is None:
            print(f"Skipping connection: {send_obj} {rcv_obj}")
            continue

        signals = get_non_pyqt_methods(send_obj, QMetaMethod.MethodType.Signal, send_signal_name)
        slots = get_non_pyqt_methods(rcv_obj, QMetaMethod.MethodType.Slot, rcv_slot_name)
        
        for signal_name, signal_method, signal_meta_method in signals:
            signal_param_types = signal_meta_method.parameterTypes()
            matching_slots = []
            for slot_name, slot_method, slot_meta_method in slots:
                slot_param_types = slot_meta_method.parameterTypes()
                if len(signal_param_types) != len(slot_param_types):
                    continue
                if all(signal_param_types[i] == slot_param_types[i] for i in range(len(signal_param_types))):
                    matching_slots.append(slot_method)
            if len(matching_slots) == 0:
                print(f"Failed to link (no match) {send_obj} and {rcv_obj}")
                continue
            elif len(matching_slots) > 1:
                print("WARNING: multiple valid signal/slot combinations found!")
            print("Conncection:")
            print(f"- '{signal_method}'")
            print(f"- '{matching_slots[0]}'")
            signal_method.connect(matching_slots[0])
            