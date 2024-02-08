import logging
from collections.abc import Callable
from PyQt6.QtCore import QObject, QMetaMethod


def get_non_pyqt_methods(qobject: QObject, method_type: QMetaMethod.MethodType, func_name: str | None) -> list[tuple[str, Callable, QMetaMethod]]:
    """
    Get non-PyQt methods of a QObject instance.

    Args:
        qobject (QObject): The QObject instance.
        method_type (QMetaMethod.MethodType): Type of methods to retrieve (Signal or Slot).
        func_name (str, None): Name of the method to filter (optional).

    Returns:
        list[tuple[str, Callable, QMetaMethod]]: List of method tuples containing method name, method reference, and QMetaMethod.
    """
    candidate_methods = {}
    for subcls in type(qobject).mro():
        if subcls.__module__.startswith("PyQt6."):
            # Ignore all native PyQt signals
            break
        for variable_name, variable in vars(subcls).items():
            if not callable(variable):
                continue
            if func_name is not None and variable_name != func_name:
                continue
            candidate_methods[variable_name] = getattr(qobject, variable_name)

    methods = []
    meta_obj = qobject.metaObject()
    for i in range(meta_obj.methodCount()):
        meta_method_obj = meta_obj.method(i)
        method_name = meta_method_obj.methodSignature().data().decode().partition("(")[0]
        if method_name not in candidate_methods:
            continue
        if meta_method_obj.methodType() != method_type:
            continue
        methods.append((method_name, candidate_methods[method_name], meta_method_obj))
    return methods

def link_QObjects(*elements: QObject | tuple[str, QObject] | tuple[QObject, str] | tuple[str, QObject, str] | None):
    """
    Link signals and slots between QObject instances.

    Args:
        elements: List of elements to link. Each element can be any of the following formats:
                    - QObject instance
                    - (QObject instance, signal name)
                    - (slot name, QObject instance)
                    - (slot name, QObject instance, signal name)
    """
    # Unify input
    unified_elements = []
    for ele in elements:
        if ele is None:
            slot_name, qobject, signal_name = None, None, None
        elif isinstance(ele, QObject):
            slot_name, qobject, signal_name = None, ele, None
        elif isinstance(ele, tuple) and len(ele) == 3 and isinstance(ele[0], str) and isinstance(ele[1], QObject) and isinstance(ele[2], str):
            slot_name, qobject, signal_name = ele
        elif isinstance(ele, tuple) and len(ele) == 2 and isinstance(ele[0], QObject) and isinstance(ele[1], str):
            slot_name, qobject, signal_name = None, *ele
        elif isinstance(ele, tuple) and len(ele) == 2 and isinstance(ele[0], str) and isinstance(ele[1], QObject):
            slot_name, qobject, signal_name = *ele, None
        else:
            raise ValueError("Invalid element format:", ele)
        unified_elements.append((slot_name, qobject, signal_name))

    for i in range(len(unified_elements)-1):
        _, send_obj, send_signal_name = unified_elements[i]
        rcv_slot_name, rcv_obj, _ = unified_elements[i+1]

        if send_obj is None or rcv_obj is None:
            logging.info(f"Skipping connection between {send_obj} and {rcv_obj}")
            continue

        signals = get_non_pyqt_methods(send_obj, QMetaMethod.MethodType.Signal, send_signal_name)
        slots = get_non_pyqt_methods(rcv_obj, QMetaMethod.MethodType.Slot, rcv_slot_name)
        
        for signal_name, signal_method, signal_meta_method in signals:
            signal_param_types = signal_meta_method.parameterTypes()
            matching_slots = [slot_method for _, slot_method, slot_meta_method in slots if signal_param_types == slot_meta_method.parameterTypes()]
            # Duplicates can exist due to overriding a slot in a base-class with a slot
            matching_slots = list(dict.fromkeys(matching_slots))
            if len(matching_slots) == 0:
                logging.error(f"Failed to link (no match found) between {send_obj} and {rcv_obj}")
                continue
            elif len(matching_slots) > 1:
                logging.warn(f"Multiple valid signal/slot combinations found between {send_obj} and {rcv_obj}")

            logging.debug(f"Connecting {signal_method} with {matching_slots[0]}")
            signal_method.connect(matching_slots[0])
            