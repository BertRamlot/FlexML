import logging
from collections.abc import Callable
from PyQt6.QtCore import QObject, QMetaMethod, QByteArray


def get_non_pyqt_methods(qobject: QObject, method_type: QMetaMethod.MethodType, func_name: str | None) -> list[tuple[Callable, list[QByteArray]]]:
    """
    Get non-PyQt methods of a QObject instance.

    Args:
        qobject (QObject): The QObject instance.
        method_type (QMetaMethod.MethodType): Type of methods to retrieve (Signal or Slot).
        func_name (str, None): Name of the method to filter (optional).

    Returns:
        list[tuple[Callable, list[QByteArray]]]: List of method tuples containing method reference and parameter types.
    """
    candidate_methods = {}
    for subcls in type(qobject).mro():
        if subcls.__module__.startswith("PyQt6."):
            # Ignore all native PyQt variables
            break
        for variable_name, variable in vars(subcls).items():
            if not callable(variable):
                continue
            if func_name is not None and variable_name != func_name:
                continue
            if variable_name in candidate_methods:
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
        methods.append((candidate_methods[method_name], meta_method_obj.parameterTypes()))
        del candidate_methods[method_name]
    return methods

def link_QObjects(*elements: QObject | tuple[str, QObject] | tuple[QObject, str] | tuple[str, QObject, str] | None):
    """
    Link signals and slots between QObject instances.

    Args:
        elements: List of elements to link. Each element can be any of the following formats:
        - None or QObject instance
        - (None or QObject instance, signal name)
        - (slot name, None or QObject instance)
        - (slot name, None or QObject instance, signal name)
    """
    # Unify input
    unified_elements = []
    for ele in elements:
        if ele is None or isinstance(ele, QObject):
            slot_name, qobject, signal_name = None, ele, None
        elif isinstance(ele, tuple):
            if len(ele) == 3 and isinstance(ele[0], str) and (ele[1] is None or isinstance(ele[1], QObject)) and isinstance(ele[2], str):
                slot_name, qobject, signal_name = ele
            elif len(ele) == 2 and (ele[0] is None or isinstance(ele[0], QObject)) and isinstance(ele[1], str):
                slot_name, qobject, signal_name = None, *ele
            elif len(ele) == 2 and isinstance(ele[0], str) and (ele[1] is None or isinstance(ele[1], QObject)):
                slot_name, qobject, signal_name = *ele, None
            else:
                raise ValueError("Invalid tuple format:", ele)
        else:
            raise ValueError("Invalid element format:", ele)
        unified_elements.append((slot_name, qobject, signal_name))

    # Link elements
    for i in range(len(unified_elements)-1):
        _, send_obj, send_signal_name = unified_elements[i]
        rcv_slot_name, rcv_obj, _ = unified_elements[i+1]

        if send_obj is None or rcv_obj is None:
            logging.debug(f"Skipping connection between {send_obj} and {rcv_obj}")
            continue

        signals = get_non_pyqt_methods(send_obj, QMetaMethod.MethodType.Signal, send_signal_name)
        slots = get_non_pyqt_methods(rcv_obj, QMetaMethod.MethodType.Slot, rcv_slot_name)
        
        for signal_method, signal_param_types in signals:
            matching_slots = [slot_method for slot_method, slot_param_types in slots if signal_param_types == slot_param_types]
            if len(matching_slots) == 0:
                logging.error(f"Failed to link (no match found) between {send_obj} and {rcv_obj}")
                logging.debug(f"{send_obj} signals: {signals}")
                logging.debug(f"{rcv_obj} slots: {slots}")
                continue
            elif len(matching_slots) > 1:
                logging.warning(f"Multiple valid signal/slot combinations found between {send_obj} and {rcv_obj}")

            slot_method = matching_slots[0]
            logging.debug(f"Connecting {signal_method} with {slot_method}")
            signal_method.connect(slot_method)
            