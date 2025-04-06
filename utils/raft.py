import torch
from torchvision.models.optical_flow import raft_large 
from modules.flow_models.raft.rfr_new import RAFT

def raft_flow(I0, I1, data_domain="animation", device = 'cuda'):
    if I0.dtype != torch.float32 or I1.dtype != torch.float32:
        I0 = I0.to(torch.float32)
        I1 = I1.to(torch.float32)
    if data_domain == "animation":
        raft = RAFT().requires_grad_(False).eval().to(device)
    elif data_domain == "photorealism":
        raft = raft_large().requires_grad_(False).eval().to(device)
    else:
        raise ValueError("data_domain must be either 'animation' or 'photorealism'")
    return raft(I0, I1) if data_domain == "animation" else raft(I0, I1)[-1]