import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.color import rgb_to_lab

from utils.utils import morph_open

from modules.cupy_module.softsplat import FunctionSoftsplat

class HalfWarper(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def backward_wrapping(img, flow, resample='bilinear', padding_mode='border', align_corners=False):
        if len(img.shape)!=4: img = img[None,]
        if len(flow.shape)!=4: flow = flow[None,]
        
        q = 2 * flow / torch.tensor([
            flow.shape[-2], flow.shape[-1],
        ], device=flow.device, dtype=torch.float)[None,:,None,None]
        
        q = q + torch.stack(torch.meshgrid(
            torch.linspace(-1,1, flow.shape[-2]),
            torch.linspace(-1,1, flow.shape[-1]),
        ))[None,].to(flow.device)
        
        if img.dtype!=q.dtype:
            img = img.type(q.dtype)

        return F.grid_sample(
            img,
            q.flip(dims=(1,)).permute(0,2,3,1).contiguous(),
            mode=resample, # nearest, bicubic, bilinear
            padding_mode=padding_mode,  # border, zeros, reflection
            align_corners=align_corners,
        )
    
    @staticmethod
    def forward_warpping(img, flow, mode='softmax', metric=None, mask=True):
        if len(img.shape)!=4: img = img[None,]
        if len(flow.shape)!=4: flow = flow[None,]
        if metric is not None and len(metric.shape)!=4: metric = metric[None,]
        flow = flow.flip(dims=(1,))
        if img.dtype!=torch.float32:
            img = img.type(torch.float32)
        if flow.dtype!=torch.float32:
            flow = flow.type(torch.float32)
        if metric is not None and metric.dtype!=torch.float32:
            metric = metric.type(torch.float32)
        
        assert img.device==flow.device
        if metric is not None: assert img.device==metric.device
        if img.device.type=='cpu':
            img = img.to('cuda')
            flow = flow.to('cuda')
            if metric is not None: metric = metric.to('cuda')
        
        if mask:
            bs,ch,h,w = img.shape
            img = torch.cat([img, torch.ones(bs,1,h,w, dtype=img.dtype, device=img.device)], dim=1)
        
        return FunctionSoftsplat(img, flow, metric, mode)
    
    @staticmethod
    def z_metric(img0, img1, flow1to0, flow0to1):
        img0 = rgb_to_lab(img0[:,:3])
        img1 = rgb_to_lab(img1[:,:3])
        z1to0 = -0.1*(img1 - HalfWarper.backward_wrapping(img0, flow1to0)).norm(dim=1, keepdim=True)
        z0to1 = -0.1*(img0 - HalfWarper.backward_wrapping(img1, flow0to1)).norm(dim=1, keepdim=True)
        return z1to0, z0to1
    
    def forward(self, I0, I1, flow1tot, flow0tot, z1to0 = None, z0to1 = None, k = 5, mask=True):
        if z1to0 is None or z0to1 is None:
            z1to0, z0to1 = self.z_metric(I0, I1, flow1tot, flow0tot)

        # image warping
        fw0to1 = HalfWarper.forward_warpping(I0, flow0tot, mode='softmax', metric=z0to1, mask=True)
        fw1to0 = HalfWarper.forward_warpping(I1, flow1tot, mode='softmax', metric=z1to0, mask=True)

        wrapped_image0tot = fw0to1[:,:-1] 
        wrapped_image1tot = fw1to0[:,:-1]
        mask0tot = morph_open(fw0to1[:,-1:], k=k)
        mask1tot = morph_open(fw1to0[:,-1:], k=k)

        base0 = mask0tot*wrapped_image0tot + (1 - mask0tot)*wrapped_image1tot
        base1 = mask1tot*wrapped_image1tot + (1 - mask1tot)*wrapped_image0tot

        if mask:
            base0 = torch.cat([base0, mask0tot], dim=1)
            base1 = torch.cat([base1, mask1tot], dim=1)
        return base0, base1