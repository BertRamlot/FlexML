from pathlib import Path
import cv2
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from src.FaceDetector import Face


class FaceDataset(Dataset):
    def __init__(self, dataset_path: Path, device, testing=None):
        self.meta_data_imgs = pd.read_csv(dataset_path / "meta_data.csv")
        if testing is not None:
            self.meta_data_imgs = self.meta_data_imgs[self.meta_data_imgs['testing'] == testing]
        
        self.img_path = dataset_path / "raw"
        self.device = device

        self.input_label_pairs = []
        for i in range(len(self.meta_data_imgs)):
            meta_data = self.meta_data_imgs.iloc[i]
            face = self.get_face(meta_data)
            X = FaceDataset.face_to_tensor(face, self.device)
            y = torch.tensor([meta_data['x_screen'], meta_data['y_screen']], device=self.device).float()
            self.input_label_pairs.append((X, y))

    def __len__(self):
        return len(self.meta_data_imgs)

    def __getitem__(self, idx):
        return self.input_label_pairs[idx]

    def get_face(self, meta_data) -> Face:
        face_img = cv2.imread(str((self.img_path / meta_data['face_file_name']).absolute()))
        tl_rx, tl_ry, rw, rh = meta_data['tl_rx'], meta_data['tl_ry'], meta_data['rw'], meta_data['rh']
        features_rx = [meta_data[f'fx_{i}'] for i in range(68)]
        features_ry = [meta_data[f'fy_{i}'] for i in range(68)]
        return Face(face_img, tl_rx, tl_ry, rw, rh, features_rx, features_ry)

    @staticmethod
    def pre_process_img(img):
        processed_img = cv2.resize(img, (40, 10), interpolation=cv2.INTER_CUBIC)
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.normalize(processed_img, None, 0, 255, cv2.NORM_MINMAX)
        return processed_img
    
    @staticmethod
    def face_to_tensor(face: Face, device: str) -> torch.Tensor:
        left_eye_tensor = torch.from_numpy(np.asarray(FaceDataset.pre_process_img(face.get_eye_im("left")))).float()
        right_eye_tensor = torch.from_numpy(np.asarray(FaceDataset.pre_process_img(face.get_eye_im("right")))).float()
        tensor_vals = []
        tensor_vals.append(left_eye_tensor.flatten()/torch.mean(left_eye_tensor))
        tensor_vals.append(right_eye_tensor.flatten()/torch.mean(right_eye_tensor))
        tensor_vals.append(torch.Tensor([face.rx, face.ry, face.rw, face.rh]))
        # tensor_vals.append(torch.Tensor(face.features_rx))
        # tensor_vals.append(torch.Tensor(face.features_ry))
        return torch.cat(tensor_vals, dim=0).float().to(device=device)


class FaceNeuralNetwork(nn.Module):
    def __init__(self):
        super(FaceNeuralNetwork, self).__init__()

        self.meta_data_size = 4 + 68*2
        self.input_eye_size = 40*10

        out_eye_size = 4

        def gen_eye_stack():
            return nn.Sequential(
                nn.Conv2d(1, 5, kernel_size=(5, 5), stride=(2, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2)),
                nn.Flatten(),
                nn.Linear(8*5, out_eye_size),
                nn.ReLU(),
            )
        self.left_eye_stack = gen_eye_stack()
        self.right_eye_stack = gen_eye_stack()

        self.main_stack = nn.Sequential(
            nn.Linear(out_eye_size*2+self.meta_data_size, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )

    def forward(self, x):
        left_eye_out = self.left_eye_stack(x[:,:self.input_eye_size].reshape(-1,1,10,40))
        right_eye_out = self.right_eye_stack(x[:,self.input_eye_size:2*self.input_eye_size].reshape(-1,1,10,40))

        main_input = torch.cat((left_eye_out, right_eye_out, x[:,-self.meta_data_size:]), 1)
        output = self.main_stack(main_input)
        return torch.clamp(output, min=0.0, max=1.0)