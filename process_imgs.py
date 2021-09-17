from EyeDetector import EyeDetector
import os
import cv2
import pandas as pd

from EyeNeuralNetwork import EyeDataset


def main():
    RAW_FOLDER = "data/raw/"
    PROCESSED_FOLDER = "data/processed"
    ANNOTATIONS_FILES = ["data/testing_meta_data.csv", "data/training_meta_data.csv"]

    if not os.path.exists(PROCESSED_FOLDER):
        os.makedirs(PROCESSED_FOLDER)

    for annotation_file in ANNOTATIONS_FILES:
        for index, row in pd.read_csv(annotation_file, sep=',').iterrows():
            in_path = os.path.join(RAW_FOLDER, row[0])
            out_path = os.path.join(PROCESSED_FOLDER, row[0])

            img = cv2.imread(in_path)

            processed_img = EyeDataset.pre_process_img(img)

            cv2.imwrite(out_path, processed_img)

if __name__ == "__main__":
    main()