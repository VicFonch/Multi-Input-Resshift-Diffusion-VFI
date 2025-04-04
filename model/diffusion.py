import torch
import torch.nn as nn

import math
from tqdm import tqdm

class MultiInputResShiftSheduler(nn.Module):
    def __init__(
        self,
        kappa: float=2.0,
        p: float =0.3, 
        min_noise_level: float=0.04,
        etas_end: float=0.99, 
        timesteps: int=15
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

    def forward_process(self, x, Y, tau, t):
        assert tau.shape == (x.shape[0], len(Y)), "tau shape must be (batch, len(Y))"
        
        if tau is None:
            tau = torch.full((x.shape[0], len(Y)), 0.5, device=x.device, dtype=x.dtype)
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=x.device, dtype=torch.long)
        if x is None:
            x = torch.zeros_like(Y[0])
        
        eta = self.sum_eta[t][:, None] * tau
        eta = eta[:, :, None, None, None].transpose(0, 1)

        e_i = torch.stack([y - x for y in Y])
        mean = x + (eta*e_i).sum(dim=0)

        sqrt_sum_eta = self.sqrt_sum_eta[t][:, None, None, None]
        std = self.kappa*sqrt_sum_eta
        epsilon = torch.randn_like(x)

        return mean + std*epsilon

    @torch.inference_mode()
    def reverse_process(self, model, Y, tau, **kwargs):
        if not isinstance(Y, list):
            Y = [Y]
        
        y = Y[0]
        batch, device = y.shape[0], y.device
        
        assert tau.shape == (batch, len(Y)), "tau shape must be (batch, len(Y))"
        model.eval()

        T = torch.tensor([self.timesteps-1,] * batch, device=device, dtype=torch.long)
        x = self.forward_process(torch.zeros_like(Y[0]), Y, tau, T)

        pbar = tqdm(total=self.timesteps, desc="Reversing Process")
        for i in reversed(range(self.timesteps)):
            t = torch.ones(batch, device = device, dtype=torch.long) * i 
            
            predicted_x0 = model(x, Y, tau, t, **kwargs)
            
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