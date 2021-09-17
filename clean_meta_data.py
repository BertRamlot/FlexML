import pandas as pd
import numpy as np
import os

def main():
    print("Removing Meta-Data without pictures")
    ANNOTATION_CSV_FILES = ["data/training_meta_data.csv", "data/testing_meta_data.csv"]
    for csv_path in ANNOTATION_CSV_FILES:
        data = pd.read_csv(ANNOTATION_CSV_FILES[0])
        mask = np.array([os.path.exists("data/raw/" + path) for path in data.iloc[:,0]])
        data = data[mask]
        data.to_csv(csv_path, index=False)
        print("{:30} -> {} rows removed".format(csv_path, 0 if len(mask) == 0 else np.sum(np.invert(mask))))

    print("\nRemoving pictures without Meta-Data")
if __name__ == "__main__":
    main()