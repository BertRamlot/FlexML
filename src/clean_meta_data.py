import pandas as pd
import numpy as np
import os


data_folder_name = "data_p_B"

print("Removing meta data without pictures")
csv_path = data_folder_name + "/meta_data.csv"
data = pd.read_csv(csv_path)
mask = np.array([os.path.exists(data_folder_name + "/raw/" + path) for path in data['face_file_name']])
data = data[mask]
data.to_csv(csv_path, index=False)
print("{:30} -> {} rows removed".format(csv_path, 0 if len(mask) == 0 else np.sum(np.invert(mask))))

print("\nRemoving pictures without meta data")
