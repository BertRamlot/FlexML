import cv2
import numpy as np
import torch
from torch import nn
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from examples.eye_tracker.src.FaceSample import FaceSample


class FaceSampleTensorfier(QObject):
    """
    Adds tensors to FaceSamples.
    """
    train_tuples = pyqtSignal(FaceSample, torch.Tensor, torch.Tensor)
    inference_tuples = pyqtSignal(FaceSample, torch.Tensor)

    def __init__(self, device: str):
        super().__init__()
        self.device = device

    def face_sample_to_X_tensor(self, sample: FaceSample) -> torch.Tensor:
        def pre_process_img(img):
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return np.asarray(gray_img)/255.0
        left_eye_tensor = torch.from_numpy(pre_process_img(sample.get_eye_img("left"))).float()
        right_eye_tensor = torch.from_numpy(pre_process_img(sample.get_eye_img("right"))).float()

        """
        cv2.imshow("FULL", sample.get_img())
        cv2.imshow("FACE", sample.get_face_img())
        cv2.imshow("EYE_L", pre_process_img(sample.get_eye_im("left")))
        cv2.imshow("EYE_R", pre_process_img(sample.get_eye_im("right")))
        # cv2.waitKey(0) 
        """

        tensor_vals = []
        
        # Eye data
        tensor_vals.append(left_eye_tensor.flatten())
        tensor_vals.append(right_eye_tensor.flatten())
        
        # Meta data
        img_max_dim = max(sample.get_img().shape[:2])
        face_dims = np.array(sample.get_face_img().shape[:2]) / img_max_dim
        eyes_center = sample.features[36:48].mean(axis=0) / img_max_dim
        tensor_vals.append(torch.Tensor(face_dims).flatten())
        tensor_vals.append(torch.Tensor(eyes_center).flatten())
        
        return torch.cat(tensor_vals, dim=0).float().to(device=self.device)

    @pyqtSlot(FaceSample)
    def add_X_tensor(self, sample: FaceSample):
        """
        Emits an inference tuple consisting of a FaceSample and its input tensor X.

        Args:
            sample (FaceSample): FaceSample instance used to generate the input tensor X.
        """
        X = self.face_sample_to_X_tensor(sample)
        self.inference_tuples.emit(sample, X)

    @pyqtSlot(FaceSample)
    def add_X_y_tensors(self, sample: FaceSample):
        """
        Emits a training tuple consisting of FaceSample, input tensor X, and output tensor y.

        Args:
            sample (FaceSample): FaceSample instance used to generate the input tensor X and output tensor y.
        """
        if sample.ground_truth is None:
            return
        X = self.face_sample_to_X_tensor(sample)
        y = torch.tensor(sample.ground_truth).to(device=self.device) / sample.screen_dims.max()
        self.train_tuples.emit(sample, X, y)

class FaceNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.metadata_size = 4

        out_eye_size = 4

        def gen_eye_stack():
            return nn.Sequential(
                nn.Conv2d(1, 4, kernel_size=(5, 5)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
                nn.Flatten(),
                nn.Linear(4*280, out_eye_size),
                nn.ReLU(),
            )
        self.left_eye_stack = gen_eye_stack()
        self.right_eye_stack = self.left_eye_stack # gen_eye_stack()

        self.main_stack = nn.Sequential(
            nn.Linear(out_eye_size*2+self.metadata_size, 2)
        )

    def forward(self, X: torch.Tensor):
        eye_size = FaceSample.EYE_DIMENSIONS.prod()

        left_eye_input = X[:,:eye_size].reshape(-1, 1, *FaceSample.EYE_DIMENSIONS)
        right_eye_input = X[:,eye_size:2*eye_size].reshape(-1, 1, *FaceSample.EYE_DIMENSIONS)

        main_input = [self.left_eye_stack(left_eye_input), self.right_eye_stack(right_eye_input)]
        if self.metadata_size > 0:
            main_input.append(X[:, 2*eye_size:2*eye_size+self.metadata_size])
        main_input = torch.cat(main_input, 1)
        output = self.main_stack(main_input)
        return output
