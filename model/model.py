import torch
import torch.nn as nn
from torch.nn.functional import interpolate

import math
from tqdm import tqdm

from modules.feature_extactor import Extractor
from modules.half_warper import HalfWarper
from modules.cupy_module.nedt import NEDT
from modules.flow_models.flow_models import (
    RAFTFineFlow,
    PWCFineFlow
)
from modules.synthesizer import Synthesis

class FeatureWarper(nn.Module):
    def __init__(
        self,
        in_channels: int = 3, 
        channels: list[int] = [32, 64, 128, 256],
    ):
        super().__init__()    
        channels = [in_channels + 1] + channels
        
        self.half_warper = HalfWarper()
        self.feature_extractor = Extractor(channels)
        self.nedt = NEDT()

    def forward(
        self, 
        I0: torch.Tensor, 
        I1: torch.Tensor, 
        flow0to1: torch.Tensor, 
        flow1to0: torch.Tensor, 
        tau: torch.Tensor = None
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        assert tau.shape == (I0.shape[0], 2), "tau shape must be (batch, 2)"

        flow0tot = tau[:, 0][:, None, None, None] * flow0to1
        flow1tot = tau[:, 1][:, None, None, None] * flow1to0

        I0 = torch.cat([I0, self.nedt(I0)], dim=1)
        I1 = torch.cat([I1, self.nedt(I1)], dim=1)

        z0to1, z1to0 = HalfWarper.z_metric(I0, I1, flow0to1, flow1to0)
        base0, base1 = self.half_warper(I0, I1, flow0tot, flow1tot, z0to1, z1to0)
        warped0, warped1 = [base0], [base1]

        features0 = self.feature_extractor(I0)
        features1 = self.feature_extractor(I1)

        for feat0, feat1 in zip(features0, features1):
            f0 = interpolate(flow0tot, size=feat0.shape[2:], mode='bilinear', align_corners=False)
            f1 = interpolate(flow1tot, size=feat0.shape[2:], mode='bilinear', align_corners=False)
            z0 = interpolate(z0to1, size=feat0.shape[2:], mode='bilinear', align_corners=False)
            z1 = interpolate(z1to0, size=feat0.shape[2:], mode='bilinear', align_corners=False)
            w0, w1 = self.half_warper(feat0, feat1, f0, f1, z0, z1)
            warped0.append(w0)
            warped1.append(w1)
        return warped0, warped1

class MultiInputResShift(nn.Module):
    def __init__(
        self,
        kappa: float=2.0,
        p: float =0.3, 
        min_noise_level: float=0.04,
        etas_end: float=0.99, 
        timesteps: int=15,
        flow_model: str = 'raft',
        flow_kwargs: dict = {},
        warping_kwargs: dict = {},
        synthesis_kwargs: dict = {}
    ):
        super().__init__()

        self.timesteps = timesteps
        self.kappa = kappa
        self.eta_partition = None

        sqrt_eta_1 = min(min_noise_level / kappa, min_noise_level, math.sqrt(0.001))
        b0 = math.exp(1/float(timesteps - 1) * math.log(etas_end/sqrt_eta_1))
        base = torch.ones(timesteps)*b0
        beta = ((torch.linspace(0,1,timesteps))**p)*(timesteps-1)
        sqrt_eta = torch.pow(base, beta) * sqrt_eta_1

        self.register_buffer("sqrt_sum_eta", sqrt_eta)
        self.register_buffer("sum_eta", sqrt_eta**2)

        sum_prev_eta = torch.roll(self.sum_eta, 1)
        sum_prev_eta[0] = 0
        self.register_buffer("sum_prev_eta", sum_prev_eta)

        self.register_buffer("sum_alpha", self.sum_eta - self.sum_prev_eta)

        self.register_buffer("backward_mean_c1", self.sum_prev_eta / self.sum_eta)
        self.register_buffer("backward_mean_c2", self.sum_alpha / self.sum_eta)
        self.register_buffer("backward_std", self.kappa*torch.sqrt(self.sum_prev_eta*self.sum_alpha/self.sum_eta))

        if flow_model == 'raft':
            self.flow_model = RAFTFineFlow(**flow_kwargs).requires_grad_(False).eval()
        elif flow_model == 'pwc':
            self.flow_model = PWCFineFlow(**flow_kwargs).requires_grad_(False).eval()
        else:
            raise ValueError(f"Flow model {flow_model} not supported")

        self.feature_warper = FeatureWarper(**warping_kwargs)
        self.synthesis = Synthesis(**synthesis_kwargs)

    def forward_process(
        self, 
        x: torch.Tensor | None, 
        Y: list[torch.Tensor], 
        tau: torch.Tensor | float | None, 
        t: torch.Tensor | int
    ) -> torch.Tensor:
        assert tau.shape == (x.shape[0], len(Y)), "tau shape must be (batch, len(Y))"
        
        if tau is None:
            tau: torch.Tensor = torch.full((x.shape[0], len(Y)), 0.5, device=x.device, dtype=x.dtype)
        elif isinstance(tau, float):
            assert tau >= 0 and tau <= 1, "tau must be between 0 and 1"
            tau: torch.Tensor = torch.cat([
                torch.full((x.shape[0], 1), tau, device=x.device, dtype=x.dtype),
                torch.full((x.shape[0], 1), 1 - tau, device=x.device, dtype=x.dtype)
            ], dim=1)
        if not torch.is_tensor(t):
            t: torch.Tensor = torch.tensor([t], device=x.device, dtype=torch.long)
        if x is None:
            x: torch.Tensor = torch.zeros_like(Y[0])
        
        eta = self.sum_eta[t][:, None] * tau
        eta = eta[:, :, None, None, None].transpose(0, 1)

        e_i = torch.stack([y - x for y in Y])
        mean = x + (eta*e_i).sum(dim=0)

        sqrt_sum_eta = self.sqrt_sum_eta[t][:, None, None, None]
        std = self.kappa*sqrt_sum_eta
        epsilon = torch.randn_like(x)

        return mean + std*epsilon

    @torch.inference_mode()
    def reverse_process(
        self, 
        Y: list[torch.Tensor],
        tau: torch.Tensor | float, 
        flows: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        y = Y[0]
        batch, device, dtype = y.shape[0], y.device, y.dtype
        
        if isinstance(tau, float):
            assert tau >= 0 and tau <= 1, "tau must be between 0 and 1"
            tau: torch.Tensor = torch.cat([
                torch.full((batch, 1), tau, device=device, dtype=dtype),
                torch.full((batch, 1), 1 - tau, device=device, dtype=dtype)
            ], dim=1)
        if flows is None:
           flow0to1, flow1to0 = self.flow_model(Y[0], Y[1])
        else:
            flow0to1, flow1to0 = flows
        warp0to1, warp1to0 = self.feature_warper(Y[0], Y[1], flow0to1, flow1to0, tau)

        T = torch.tensor([self.timesteps-1,] * batch, device=device, dtype=torch.long)
        x = self.forward_process(torch.zeros_like(Y[0]), [warp0to1[0][:, :3], warp1to0[0][:, :3]], tau, T)

        pbar = tqdm(total=self.timesteps, desc="Reversing Process")
        for i in reversed(range(self.timesteps)):
            t = torch.ones(batch, device = device, dtype=torch.long) * i 
            
            predicted_x0 = self.synthesis(x, warp0to1, warp1to0, t)
            
            mean_c1 = self.backward_mean_c1[t][:, None, None, None]  
            mean_c2 = self.backward_mean_c2[t][:, None, None, None]
            std = self.backward_std[t][:, None, None, None]
            
            eta = self.sum_eta[t][:, None] * tau
            prev_eta = self.sum_prev_eta[t][:, None] * tau
            eta = eta[:, :, None, None, None].transpose(0, 1)
            prev_eta = prev_eta[:, :, None, None, None].transpose(0, 1)
            e_i = torch.stack([y - predicted_x0 for y in Y])

            mean = (
                mean_c1*(x + (eta*e_i).sum(dim=0)) 
                + mean_c2*predicted_x0 
                - (prev_eta*e_i).sum(dim=0)
            )

            x = mean + std*torch.randn_like(x)
            pbar.update(1)
        pbar.close()
        return x

    # Training Step Only
    def forward(
        self, 
        I0: torch.Tensor, 
        It: torch.Tensor, 
        I1: torch.Tensor, 
        flow1to0: torch.Tensor | None = None, 
        flow0to1: torch.Tensor | None = None, 
        tau: torch.Tensor | None = None,
        t: torch.Tensor | None = None
    ) -> torch.Tensor:

        if tau is None:
            tau = torch.full((It.shape[0], 2), 0.5, device=It.device, dtype=It.dtype)

        if flow0to1 is None or flow1to0 is None:
            flow0to1, flow1to0 = self.flow_model(I0, I1)
        
        if t is None:
            t = torch.randint(low=1, high=self.timesteps, size=(It.shape[0],), device=It.device, dtype=torch.long)
        
        warp0to1, warp1to0 = self.feature_warper(I0, I1, flow0to1, flow1to0, tau)
        x_t = self.forward_process(It, [warp0to1[0][:, :3], warp1to0[0][:, :3]], tau, t)

        predicted_It = self.synthesis(x_t, warp0to1, warp1to0, t)
        return predicted_It
