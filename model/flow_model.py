import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from modules.cupy_module import correlation
from modules.half_warper import HalfWarper
from modules.feature_extactor import Extractor

from raft.rfr_new import RAFT

class Decoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.syntesis = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, img1, img2, residual):
        width = img1.shape[3] and img2.shape[3]
        height = img1.shape[2] and img2.shape[2]

        if residual is None:
            corr = correlation.FunctionCorrelation(tenOne=img1, tenTwo=img2)
            main = torch.cat([img1, corr], 1)
        else:
            flow = interpolate(input=residual, 
                               size=(height, width), 
                               mode='bilinear', 
                               align_corners=False) / \
                                float(residual.shape[3]) * float(width)
            backwarp_img = HalfWarper.backward_wrapping(img=img2, flow=flow)
            corr = correlation.FunctionCorrelation(tenOne=img1, tenTwo=backwarp_img)
            main = torch.cat([img1, corr, flow], 1)

        return self.syntesis(main)

class PWCFineFlow(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = Extractor([3, 16, 32, 64, 96, 128, 192]) 

        self.decoders = nn.ModuleList([
            Decoder(16 + 81 + 2),
            Decoder(32 + 81 + 2),
            Decoder(64 + 81 + 2),
            Decoder(96 + 81 + 2),
            Decoder(128 + 81 + 2),
            Decoder(192 + 81)
        ])

    def forward(self, img1, img2):
        width = img1.shape[3] and img2.shape[3]
        height = img1.shape[2] and img2.shape[2]

        feats1 = self.feature_extractor(img1)
        feats2 = self.feature_extractor(img2)

        forward = None
        backward = None

        for i in reversed(range(len(feats1))):
            forward = self.decoders[i](feats1[i], feats2[i], forward)
            backward = self.decoders[i](feats2[i], feats1[i], backward)

        forward = interpolate(input=forward, 
                              size=(height, width), 
                              mode='bilinear', 
                              align_corners=False) * \
                                 (float(width) / float(forward.shape[3]))
        backward = interpolate(input=backward,
                                 size=(height, width),
                                 mode='bilinear',
                                 align_corners=False) * \
                                  (float(width) / float(backward.shape[3]))

        return forward, backward    


class RAFTFineFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.raft = RAFT()

    def forward(self, img1, img2):
        forward = self.raft(img1, img2)
        backward = self.raft(img2, img1)
        return forward, backward