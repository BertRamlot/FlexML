import cv2
import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from FaceDetector import Face


class FaceDataset(Dataset):
    def __init__(self, annotations_file, img_dir, device, testing=None):
        self.annotations_file = annotations_file
        self.meta_data_imgs = pd.read_csv(annotations_file)
        if testing is not None:
            self.meta_data_imgs = self.meta_data_imgs[self.meta_data_imgs['testing'] == testing]
        self.img_dir = img_dir
        self.device = device

        self.input_label_pairs = []
        for i in range(len(self.meta_data_imgs)):
            meta_data = self.meta_data_imgs.iloc[i]
            face = self.get_face(meta_data)
            input = FaceDataset.face_to_tensor(face, self.device)
            label = torch.tensor([meta_data['x_screen'], meta_data['y_screen']], device=self.device).float()
            self.input_label_pairs.append((input, label))

    def __len__(self):
        return len(self.meta_data_imgs)

    def __getitem__(self, idx):
        return self.input_label_pairs[idx]

    def get_face(self, meta_data) -> Face:
        face_img = cv2.imread(os.path.join(self.img_dir, meta_data['face_file_name']))
        tl_rx, tl_ry, rw, rh = meta_data['tl_rx'], meta_data['tl_ry'], meta_data['rw'], meta_data['rh']
        features_rx = [meta_data['fx_' + str(i)] for i in range(68)] #todo
        features_ry = [meta_data['fy_' + str(i)] for i in range(68)] #todo
        return Face(face_img, tl_rx, tl_ry, rw, rh, features_rx, features_ry)

    @staticmethod
    def pre_process_img(img):
        processed_img = cv2.resize(img, (30, 10), interpolation=cv2.INTER_CUBIC)
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.normalize(processed_img, None, 0, 255, cv2.NORM_MINMAX)
        return processed_img
    
    @staticmethod
    def face_to_tensor(face: Face, device: str) -> torch.Tensor:
        left_eye_tensor = torch.from_numpy(np.asarray(FaceDataset.pre_process_img(face.get_eye_im(True))))
        right_eye_tensor = torch.from_numpy(np.asarray(FaceDataset.pre_process_img(face.get_eye_im(False))))

        tensor_vals = []
        tensor_vals.append(left_eye_tensor.flatten()/torch.max(left_eye_tensor))
        tensor_vals.append(right_eye_tensor.flatten()/torch.max(right_eye_tensor))
        tensor_vals.append(torch.Tensor([face.rx, face.ry, face.rw, face.rh]))
        tensor_vals.append(torch.Tensor(face.features_rx))
        tensor_vals.append(torch.Tensor(face.features_ry))
        return torch.cat(tensor_vals, dim=0).float().to(device=device)


class FaceNeuralNetwork(nn.Module):
    def __init__(self):
        super(FaceNeuralNetwork, self).__init__()

        self.meta_data_size = 4 + 68*2

        self.input_eye_size = 30*10
        inter_eye_size = 150
        out_eye_size = 50
        
        
        inter_main_size = round((out_eye_size*2+self.meta_data_size+2)/2)

        self.left_eye_stack = nn.Sequential(
            nn.Linear(self.input_eye_size, inter_eye_size),
            nn.ReLU(),
            nn.Linear(inter_eye_size, inter_eye_size),
            nn.ReLU(),
            nn.Linear(inter_eye_size, out_eye_size)
        )

        self.right_eye_stack = nn.Sequential(
            nn.Linear(self.input_eye_size, inter_eye_size),
            nn.ReLU(),
            nn.Linear(inter_eye_size, inter_eye_size),
            nn.ReLU(),
            nn.Linear(inter_eye_size, out_eye_size)
        )

        self.main_stack = nn.Sequential(
            nn.Linear(out_eye_size*2+self.meta_data_size, inter_main_size),
            nn.ReLU(),
            nn.Linear(inter_main_size, inter_main_size),
            nn.ReLU(),
            nn.Linear(inter_main_size, 2)
        )

    def forward(self, x):
        left_eye_out = self.left_eye_stack(x[:,:self.input_eye_size])
        right_eye_out = self.right_eye_stack(x[:,self.input_eye_size:2*self.input_eye_size])

        main_input = torch.cat((left_eye_out, right_eye_out, x[:,-self.meta_data_size:]), 1)
        return self.main_stack(main_input)