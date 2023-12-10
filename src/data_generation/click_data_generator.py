import sys
import time
import win32api
from pathlib import Path
from argparse import ArgumentParser

from src.data_generation.DataGenerator import DataGenerator


if __name__ == "__main__":
    parser = ArgumentParser(description="Click data generation script parameters")
    parser.add_argument("--data_set_name", type=str, required=True)
    args = parser.parse_args(sys.argv[1:])
    dataGenerator = DataGenerator(Path("data_sets") / args.data_set_name, 5)

    prev_states = [1, 1]
    while True:
        time.sleep(0.001)

        # Detect Mouse clicks one time each
        new_states = [win32api.GetKeyState(i) for i in range(1, 3)]
        next_index = next((i for i in range(len(new_states)) if prev_states[i] != new_states[i] and new_states[i] < 0), -1)
        prev_states = new_states
        
        if next_index == -1:
            continue
        
        # Valid mouse click detected
        x, y = win32api.GetCursorPos()
        dataGenerator.register_eye_position(x, y)

    dataGenerator.flush()
    dataGenerator.exit()