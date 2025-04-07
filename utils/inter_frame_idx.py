from utils.utils import morph_open

import torch
from kornia.color import rgb_to_grayscale

import cv2
import numpy as np

class FlowEstimation:
    def __init__(self, flow_estimator: str = "farneback"):
        assert flow_estimator in ["farneback", "dualtvl1"], "Flow estimator must be one of [farneback, dualtvl1]"

        if flow_estimator == "farneback":
            self.flow_estimator = self.OptFlow_Farneback
        elif flow_estimator == "dualtvl1":
            self.flow_estimator = self.OptFlow_DualTVL1
        else:
            raise NotImplementedError

    def OptFlow_Farneback(self, I0: torch.Tensor, I1: torch.Tensor) -> torch.Tensor:
        device = I0.device
        
        I0 = I0.cpu().clamp(0, 1) * 255
        I1 = I1.cpu().clamp(0, 1) * 255

        batch_size = I0.shape[0]
        for i in range(batch_size):
            I0_np = I0[i].permute(1, 2, 0).numpy().astype(np.uint8)
            I1_np = I1[i].permute(1, 2, 0).numpy().astype(np.uint8)

            I0_gray = cv2.cvtColor(I0_np, cv2.COLOR_BGR2GRAY)
            I1_gray = cv2.cvtColor(I1_np, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(I0_gray, I1_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).float()
            if i == 0:
                flows = flow
            else:
                flows = torch.cat((flows, flow), dim = 0)

        return flows.to(device)

    def OptFlow_DualTVL1(
        self, 
        I0: torch.Tensor, 
        I1: torch.Tensor,
        tau: float = 0.25,
        lambda_: float = 0.15,
        theta: float = 0.3,
        scales_number: int = 5,
        warps: int = 5,
        epsilon: float = 0.01,
        inner_iterations: int = 30,
        outer_iterations: int = 10,
        scale_step: float = 0.8,
        gamma: float = 0.0
    ) -> torch.Tensor:
        optical_flow = cv2.optflow.createOptFlow_DualTVL1()
        optical_flow.setTau(tau)
        optical_flow.setLambda(lambda_)
        optical_flow.setTheta(theta)
        optical_flow.setScalesNumber(scales_number)
        optical_flow.setWarpingsNumber(warps)
        optical_flow.setEpsilon(epsilon)
        optical_flow.setInnerIterations(inner_iterations)
        optical_flow.setOuterIterations(outer_iterations)
        optical_flow.setScaleStep(scale_step)
        optical_flow.setGamma(gamma)

        device = I0.device
        
        I0 = I0.cpu().clamp(0, 1) * 255
        I1 = I1.cpu().clamp(0, 1) * 255

        batch_size = I0.shape[0]
        for i in range(batch_size):
            I0_np = I0[i].permute(1, 2, 0).numpy().astype(np.uint8)
            I1_np = I1[i].permute(1, 2, 0).numpy().astype(np.uint8)

            I0_gray = cv2.cvtColor(I0_np, cv2.COLOR_BGR2GRAY)
            I1_gray = cv2.cvtColor(I1_np, cv2.COLOR_BGR2GRAY)

            flow = optical_flow.calc(I0_gray, I1_gray, None)
            flow = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).float()
            if i == 0:
                flows = flow
            else:
                flows = torch.cat((flows, flow), dim = 0)

        return flows.to(device)
    
    def __call__(self, I1: torch.Tensor, I0: torch.Tensor) -> torch.Tensor:
        return self.flow_estimator(I1, I0)

def get_inter_frame_temp_index(
    I0: torch.Tensor, 
    It: torch.Tensor, 
    I1: torch.Tensor, 
    flow0tot: torch.Tensor, 
    flow1tot: torch.Tensor, 
    k: int = 5, 
    threshold: float = 2e-2
) -> torch.Tensor:

    I0_gray = rgb_to_grayscale(I0)
    It_gray = rgb_to_grayscale(It)
    I1_gray = rgb_to_grayscale(I1)

    mask0tot = morph_open(It_gray - I0_gray, k=k)
    mask1tot = morph_open(I1_gray - It_gray, k=k)

    mask0tot = (abs(mask0tot) > threshold).to(torch.uint8)
    mask1tot = (abs(mask1tot) > threshold).to(torch.uint8)

    flow_mag0tot = torch.sqrt(flow0tot[:, 0, :, :]**2 + flow0tot[:, 1, :, :]**2).unsqueeze(1)
    flow_mag1tot = torch.sqrt(flow1tot[:, 0, :, :]**2 + flow1tot[:, 1, :, :]**2).unsqueeze(1)

    norm0tot = (flow_mag0tot*mask0tot).squeeze(1)
    norm1tot = (flow_mag1tot*mask1tot).squeeze(1)
    d0tot = torch.sum(norm0tot, dim = (1, 2)) 
    d1tot = torch.sum(norm1tot, dim = (1, 2))
    
    return d0tot / (d0tot + d1tot + 1e-12)