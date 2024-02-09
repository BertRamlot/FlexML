import cv2
import numpy as np
import torch
from torch import nn
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from examples.eye_tracker.src.FaceSample import FaceSample


def face_sample_to_y_tensor(sample: FaceSample, device):
    screen_max_dim = sample.screen_dims.max()
    return torch.tensor(sample.gt).to(device=device) / screen_max_dim

def face_sample_to_X_tensor(sample: FaceSample, device):
    def pre_process_img(img):
        processed_img = img
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
        return np.asarray(processed_img)/255.0
    left_eye_tensor = torch.from_numpy(pre_process_img(sample.get_eye_im("left"))).float()
    right_eye_tensor = torch.from_numpy(pre_process_img(sample.get_eye_im("right"))).float()

    """
    cv2.imshow("FULL", sample.get_img())
    cv2.imshow("FACE", sample.get_face_img())
    cv2.imshow("EYE_L", pre_process_img(sample.get_eye_im("left")))
    cv2.imshow("EYE_R", pre_process_img(sample.get_eye_im("right")))
    """
    # cv2.waitKey(0) 

    tensor_vals = []
    tensor_vals.append(left_eye_tensor.flatten())
    tensor_vals.append(right_eye_tensor.flatten())
    
    # TODO: do this another way
    screen_max_dim = max(sample.get_img()[:2].shape)
    rel_features = sample.features / screen_max_dim
    face_rel_tl_xy = np.min(sample.features, axis=0) / screen_max_dim
    face_rel_wh = np.array(sample.get_face_img().shape[:2]) / screen_max_dim
    eyes_center = sample.features[36:48].mean(axis=0) / screen_max_dim

    tensor_vals.append(torch.Tensor(rel_features).flatten())
    tensor_vals.append(torch.Tensor(face_rel_tl_xy).flatten())
    tensor_vals.append(torch.Tensor(face_rel_wh).flatten())
    tensor_vals.append(torch.Tensor(eyes_center).flatten())
    
    X = torch.cat(tensor_vals, dim=0).float().to(device=device)
    return X


class FaceSampleToTrainPair(QObject):
    output_train_pair = pyqtSignal(FaceSample, torch.Tensor, torch.Tensor)

    def __init__(self, device: str):
        super().__init__()
        self.device = device

    @pyqtSlot(FaceSample)
    def to_train_pair(self, sample: FaceSample):
        if sample.gt is None:
            return
        X = face_sample_to_X_tensor(sample, self.device)
        y = face_sample_to_y_tensor(sample, self.device)
        self.output_train_pair.emit(sample, X, y)

class FaceSampleToInferencePair(QObject):
    output_tensor = pyqtSignal(FaceSample, torch.Tensor)

    def __init__(self, device: str):
        super().__init__(None)
        self.device = device

    @pyqtSlot(FaceSample)
    def to_tensor(self, sample: FaceSample):
        X = face_sample_to_X_tensor(sample, self.device)
        self.output_tensor.emit(sample, X)

class FaceNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.metadata_size = 0 # 4 + 68*2

        out_eye_size = 4

        def gen_eye_stack():
            return nn.Sequential(
                nn.Conv2d(1, 4, kernel_size=(5, 5)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                nn.Flatten(),
                nn.Linear(4*280, out_eye_size), # 108
                nn.ReLU(),
            )
        self.left_eye_stack = gen_eye_stack()
        self.right_eye_stack = self.left_eye_stack # gen_eye_stack()

        self.main_stack = nn.Sequential(
            nn.Linear(out_eye_size*2+self.metadata_size, 2)
        )

    def forward(self, x):
        eye_size = FaceSample.EYE_DIMENSIONS.prod()

        left_eye_input = x[:,:eye_size].reshape(-1,1,*FaceSample.EYE_DIMENSIONS)
        right_eye_input = x[:,eye_size:2*eye_size].reshape(-1,1,*FaceSample.EYE_DIMENSIONS)

        main_input = [self.left_eye_stack(left_eye_input), self.right_eye_stack(right_eye_input)]
        if self.metadata_size > 0:
            main_input.append(x[:,-self.metadata_size:])
        main_input = torch.cat(main_input, 1)
        output = self.main_stack(main_input)
        return output
