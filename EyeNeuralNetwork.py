import cv2
import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.io import read_image

from EyeDetector import EyePair, Eye


class EyeDataset(Dataset):
    def __init__(self, annotations_file, img_dir, device, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.device = device
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        meta_data = self.img_labels.iloc[idx, 2:]

        # input
        left_eye_img = cv2.imread(os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]))
        right_eye_img = cv2.imread(os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]))
        eye_rect_l = [meta_data['l_x'], meta_data['l_y'], meta_data['l_w'], meta_data['l_h']]
        eye_rect_r = [meta_data['r_x'], meta_data['r_y'], meta_data['r_w'], meta_data['r_h']]

        eye_pair = EyePair(Eye(eye_rect_l, left_eye_img), Eye(eye_rect_r, right_eye_img))
        input = EyeDataset.eye_pair_to_tensor(eye_pair, self.device)

        # label
        label = torch.tensor([meta_data['x_screen'], meta_data['y_screen']]).float()
        return input, label

    @staticmethod
    def pre_process_img(img):
        processed_img = cv2.resize(img, (50, 50), interpolation=cv2.INTER_CUBIC)
        # processed_img = processed_img[5:25:,:]
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.normalize(processed_img, None, 0, 255, cv2.NORM_MINMAX)
        return processed_img

    @staticmethod
    def get_input_size() -> int:
        return 2*50*50 + 8

    @staticmethod
    def eye_pair_to_tensor(eye_pair: EyePair, device: str) -> torch.Tensor:
        # np_arrays = [np.asarray(EyeNeuralNetwork.pre_process_img(img)) for img in imgs] 
        # torch.from_numpy(np.stack(np_arrays)).to(device).float()/255
        left_eye_tensor = torch.from_numpy(np.asarray(EyeDataset.pre_process_img(eye_pair.left_eye.im)))
        right_eye_tensor = torch.from_numpy(np.asarray(EyeDataset.pre_process_img(eye_pair.right_eye.im)))

        tensor_vals = []
        tensor_vals.append(left_eye_tensor.flatten()/torch.max(left_eye_tensor))
        tensor_vals.append(right_eye_tensor.flatten()/torch.max(right_eye_tensor))
        tensor_vals.append(torch.Tensor([
            eye_pair.left_eye.x,  eye_pair.left_eye.y,  eye_pair.left_eye.w,  eye_pair.left_eye.h,
            eye_pair.right_eye.x, eye_pair.right_eye.y, eye_pair.right_eye.w, eye_pair.right_eye.h,
            ]))

        return torch.cat(tensor_vals, dim=0).float().to(device=device)


class EyeNeuralNetwork(nn.Module):
    def __init__(self):
        super(EyeNeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(EyeDataset.get_input_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)