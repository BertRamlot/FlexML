import cv2
import numpy as np
import torch
from torch import nn
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from src.face_based.FaceSample import FaceSample


class FaceSampleToTensor(QObject):
    output_tensor = pyqtSignal(torch.Tensor)

    def __init__(self, device: str):
        super().__init__(None)
        self.device = device

    @pyqtSlot(FaceSample)
    def to_tensor(self, sample: FaceSample) -> torch.Tensor:
        def pre_process_img(img):
            processed_img = cv2.resize(img, (40, 10), interpolation=cv2.INTER_CUBIC)
            processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
            processed_img = cv2.normalize(processed_img, None, 0, 255, cv2.NORM_MINMAX)
            return np.asarray(processed_img)
        left_eye_tensor = torch.from_numpy(pre_process_img(sample.get_eye_im("left"))).float()
        right_eye_tensor = torch.from_numpy(pre_process_img(sample.get_eye_im("right"))).float()
        
        tensor_vals = []
        tensor_vals.append(left_eye_tensor.flatten()/torch.mean(left_eye_tensor))
        tensor_vals.append(right_eye_tensor.flatten()/torch.mean(right_eye_tensor))
        
        screen_max_dim = max(sample.get_img()[:2].shape)
        rel_features = sample.features / screen_max_dim
        face_rel_tl_xy = np.min(sample.features, axis=0) / screen_max_dim
        face_rel_wh = np.array(sample.get_face_img().shape[:2]) / screen_max_dim
        eyes_center = sample.features[36:48].mean(axis=0) / screen_max_dim

        tensor_vals.append(torch.Tensor(rel_features).flatten())
        tensor_vals.append(torch.Tensor(face_rel_tl_xy).flatten())
        tensor_vals.append(torch.Tensor(face_rel_wh).flatten())
        tensor_vals.append(torch.Tensor(eyes_center).flatten())
        
        tensor = torch.cat(tensor_vals, dim=0).float().to(device=self.device)
        self.output_tensor.emit(tensor)

class FaceNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.metadata_size = 4 + 68*2
        self.input_eye_size = 40*10

        out_eye_size = 4

        def gen_eye_stack():
            return nn.Sequential(
                nn.Conv2d(1, 3, kernel_size=(5, 5), stride=(2, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                nn.Flatten(),
                nn.Linear(27, out_eye_size),
                nn.ReLU(),
            )
        self.left_eye_stack = gen_eye_stack()
        self.right_eye_stack = gen_eye_stack()

        self.main_stack = nn.Sequential(
            nn.Linear(out_eye_size*2+self.metadata_size, 40),
            nn.ReLU(),
            nn.Linear(40, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )

    def forward(self, x):
        left_eye_out = self.left_eye_stack(x[:,:self.input_eye_size].reshape(-1,1,10,40))
        right_eye_out = self.right_eye_stack(x[:,self.input_eye_size:2*self.input_eye_size].reshape(-1,1,10,40))

        main_input = [left_eye_out, right_eye_out]
        if self.metadata_size > 0:
            main_input.append(x[:,-self.metadata_size:])
        main_input = torch.cat(main_input, 1)
        output = self.main_stack(main_input)
        return torch.clamp(output, min=0.0, max=1.0)
