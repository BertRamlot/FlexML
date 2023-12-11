import sys
import time
import ctypes
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import win32api


from src.data_generation.DataGenerator import DataGenerator


if __name__ == "__main__":
    parser = ArgumentParser(description="Click data generation script parameters")
    parser.add_argument("--data_set_name", type=str, required=True)
    args = parser.parse_args(sys.argv[1:])
    dataGenerator = DataGenerator(Path("data_sets") / args.data_set_name, 1)

    screen_dims = np.array([ctypes.windll.user32.GetSystemMetrics(i) for i in range(2)], dtype=np.int32)

    print("Started listening for clicks!")
    prev_states = [1, 1]
    while True:
        time.sleep(0.001)

        new_states = [win32api.GetKeyState(i) for i in range(1, 3)]
        next_index = -1 
        for i in range(len(new_states)):
            if prev_states[i] != new_states[i] and new_states[i] < 0:
                next_index = i
                break
        prev_states = new_states
        
        if next_index == -1:
            continue
        
        # Valid mouse click detected
        screen_pos = np.array(win32api.GetCursorPos())
        dataGenerator.register_eye_position(*screen_pos/screen_dims)
        dataGenerator.flush()
