import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from modules.synthesizer import Synthesis
from modules.feature_extactor import Extractor
from modules.half_warper import HalfWarper
from modules.cupy_module.nedt import NEDT
from modules.flow_model import RAFTFineFlow, PWCFineFlow

class Warping(nn.Module):
    def __init__(
        self,
        in_channels=4, 
        channels=[32, 64, 128, 256],
    ):
        super().__init__()    
        channels = [in_channels] + channels
        
        self.half_warper = HalfWarper()
        self.feature_extractor = Extractor(channels)
        # self.feature_extractor = VGGExtractor(
        #     layers_to_extract = [8, 15, 22]
        # ).requires_grad_(False).eval()
        self.nedt = NEDT()

    def forward(self, I0, I1, flow0to1, flow1to0, t = 0.5):

        if isinstance(t, torch.Tensor):
            t = t[:,None,None,None]
        flow1tot = (1 - t)*flow1to0
        flow0tot = t*flow0to1

        features0 = self.feature_extractor(I0)
        features1 = self.feature_extractor(I1)

        I0 = torch.cat([I0, self.nedt(I0)], dim=1)
        I1 = torch.cat([I1, self.nedt(I1)], dim=1)

        z1to0, z0to1 = HalfWarper.z_metric(I0, I1, flow1to0, flow0to1)
        base0, base1 = self.half_warper(I0, I1, flow1tot, flow0tot, z1to0, z0to1)
        warped0, warped1 = [base0], [base1]
        for feat0, feat1 in zip(features0, features1):
            f0 = interpolate(flow1tot, size=feat0.shape[2:], mode='bilinear', align_corners=False)
            f1 = interpolate(flow0tot, size=feat0.shape[2:], mode='bilinear', align_corners=False)
            z0 = interpolate(z1to0, size=feat0.shape[2:], mode='bilinear', align_corners=False)
            z1 = interpolate(z0to1, size=feat0.shape[2:], mode='bilinear', align_corners=False)
            w0, w1 = self.half_warper(feat0, feat1, f0, f1, z0, z1)
            warped0.append(w0)
            warped1.append(w1)
        return warped0, warped1
  
class SofsplatResshift(nn.Module):
    def __init__(
        self,
        in_channels=3,
        channels=[32, 64, 128, 256],
        temb_channels=256,
        flow_model='raft',
        heads = 1,
        window_size = 7,
        window_attn = True,
        grid_attn = True,
        expansion_rate = 4,
        num_conv_blocks = 2,
        dropout = 0.0
    ):
        super(SofsplatResshift, self).__init__()

        assert flow_model in ['raft', 'pwc'], 'Flow estimation must be in [raft, pwc]'

        if flow_model == 'raft':
            self.flow_model = RAFTFineFlow().requires_grad_(False).eval()
        else:
            self.flow_model = PWCFineFlow().requires_grad_(False).eval()
        self.warping = Warping(in_channels, channels)
        self.synthesis = Synthesis(in_channels, 
                                   channels,
                                   temb_channels,
                                   heads = heads,
                                   window_size = window_size,
                                   window_attn = window_attn,
                                   grid_attn = grid_attn,
                                   expansion_rate = expansion_rate,
                                   num_conv_blocks = num_conv_blocks,
                                   dropout = dropout)
        # Evaluation of the optical flow
        self.flow0to1 = None
        self.flow1to0 = None

    def forward(self, x, Y, tau, temb, flows=None):
        if flows is None:
            flow0to1, flow1to0 = self.flow_model(Y[0], Y[1])
            self.flow0to1, self.flow1to0 = flow0to1, flow1to0
        else:
            flow0to1, flow1to0 = flows
        warp0to1, warp1to0 = self.warping(Y[0], Y[1], flow0to1, flow1to0, tau[:, 0])
        synt = self.synthesis(x, warp0to1, warp1to0, temb)
        return synt