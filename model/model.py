import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from modules.basic_layers import (
    SinusoidalPositionalEmbedding,
    DobleConv,
    MaxViTBlock,
    Downsample,
    Upsample
) 
from modules.feature_extactor import Extractor
from modules.half_warper import HalfWarper
from modules.cupy_module.nedt import NEDT
from modules.flow_model import (
    RAFTFineFlow,
    PWCFineFlow
)

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

class UnetDownBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            temb_channels=128,
            heads = 1, 
            window_size = 7,
            window_attn = True,
            grid_attn = True,
            expansion_rate = 4,
            num_conv_blocks = 2,
            dropout = 0.0
        ):
        super(UnetDownBlock, self).__init__()
        self.pool = Downsample(
            in_channels = in_channels,
            out_channels = in_channels,
            use_conv = True
        )
        in_channels = 3 * in_channels + 2
        self.conv = nn.ModuleList([
            DobleConv(
                in_channels = in_channels if i == 0 else out_channels,
                out_channels = out_channels,
                emb_channels = temb_channels,
                gated_conv = True
            ) for i in range(num_conv_blocks)
        ])
        self.maxvit = MaxViTBlock(
            channels = out_channels,
            #latent_dim = out_channels // 6,
            heads = heads,
            window_size = window_size,
            window_attn = window_attn,
            grid_attn = grid_attn,
            expansion_rate = expansion_rate,
            dropout = dropout,
            emb_channels = temb_channels
        )

    def forward(self, x, warp0, warp1, temb):
        x = self.pool(x)
        x = torch.cat([x, warp0, warp1], dim=1)
        for conv in self.conv:
            x = conv(x, temb)
        x = self.maxvit(x, temb)
        return x

class UnetMiddleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        out_channels,
        temb_channels=128,
        heads = 1, 
        window_size = 7,
        window_attn = True,
        grid_attn = True,
        expansion_rate = 4,
        dropout = 0.0
    ):
        super(UnetMiddleBlock, self).__init__()

        self.middle_blocks = nn.ModuleList([
            DobleConv(
                in_channels = in_channels,
                out_channels = mid_channels,
                emb_channels = temb_channels,
                gated_conv = True
            ),
            MaxViTBlock(
                channels = mid_channels,
                #latent_dim = mid_channels // 6,
                heads = heads,
                window_size = window_size,
                window_attn = window_attn,
                grid_attn = grid_attn,
                expansion_rate = expansion_rate,
                dropout = dropout,
                emb_channels = temb_channels
            ),
            DobleConv(
                in_channels = mid_channels,
                out_channels = out_channels,
                emb_channels = temb_channels,
                gated_conv = True
            )
        ])

    def forward(self, x, temb):
        for block in self.middle_blocks:
            x = block(x, temb)
        return x
        
class UnetUpBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            temb_channels=128,
            heads = 1, 
            window_size = 7,
            window_attn = True,
            grid_attn = True,
            expansion_rate = 4,
            num_conv_blocks = 2,
            dropout = 0.0
        ):
        super(UnetUpBlock, self).__init__()
        in_channels = 2 * in_channels
        self.maxvit = MaxViTBlock(
            channels = in_channels,
            #latent_dim = in_channels // 6,
            heads = heads,
            window_size = window_size,
            window_attn = window_attn,
            grid_attn = grid_attn,
            expansion_rate = expansion_rate,
            dropout = dropout,
            emb_channels = temb_channels
        )
        self.upsample = Upsample(
            in_channels = in_channels,
            out_channels = in_channels,
            use_conv = True
        )
        self.conv = nn.ModuleList([
            DobleConv(
                in_channels if i == 0 else out_channels,
                out_channels,
                emb_channels = temb_channels, 
                gated_conv = True
            ) for i in range(num_conv_blocks)
        ])
        
    def forward(self, x, skip_connection, temb):
        x = torch.cat([x, skip_connection], dim=1)
        x = self.maxvit(x, temb)
        x = self.upsample(x)   
        for conv in self.conv:
            x = conv(x, temb)     
        return x

class Synthesis(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        temb_channels,
        heads = 1,
        window_size = 7,
        window_attn = True,
        grid_attn = True,
        expansion_rate = 4,
        num_conv_blocks = 2,
        dropout = 0.0
    ):
        super(Synthesis, self).__init__()


        self.t_pos_encoding = SinusoidalPositionalEmbedding(temb_channels)

        self.input_blocks = nn.ModuleList([
            nn.Conv2d(3*in_channels + 4, channels[0], kernel_size=3, padding=1),
            DobleConv(
                in_channels = channels[0],
                out_channels = channels[0],
                emb_channels = temb_channels,
                gated_conv = True
            )
        ])
        
        self.down_blocks = nn.ModuleList([
            UnetDownBlock(
                #3 * channels[i] + 2, 
                channels[i],
                channels[i + 1], 
                temb_channels,
                heads = heads,
                window_size = window_size,
                window_attn = window_attn,
                grid_attn = grid_attn,
                expansion_rate = expansion_rate,
                num_conv_blocks = num_conv_blocks,
                dropout = dropout,
            ) for i in range(len(channels) - 1)
        ])

        self.middle_block = UnetMiddleBlock(
            in_channels = channels[-1],
            mid_channels = channels[-1],
            out_channels = channels[-1],
            temb_channels = temb_channels,
            heads = heads,
            window_size = window_size,
            window_attn = window_attn,
            grid_attn = grid_attn,
            expansion_rate = expansion_rate,
            dropout = dropout,
        )

        self.up_blocks = nn.ModuleList([
            UnetUpBlock(
                channels[i + 1], 
                channels[i], 
                temb_channels,
                heads = heads,
                window_size = window_size,
                window_attn = window_attn,
                grid_attn = grid_attn,
                expansion_rate = expansion_rate,
                num_conv_blocks = num_conv_blocks,
                dropout = dropout,
            ) for i in reversed(range(len(channels) - 1))
        ])

        self.output_blocks = nn.ModuleList([
            DobleConv(
                in_channels = channels[0],
                out_channels = channels[0],
                emb_channels = temb_channels,
                gated_conv = True
            ),
            nn.Conv2d(channels[0], in_channels, kernel_size=3, padding=1)
        ])

    def forward(self, x, warp0, warp1, temb):
        temb = temb.unsqueeze(-1).type(torch.float)
        temb = self.t_pos_encoding(temb)

        x = self.input_blocks[0](torch.cat([x, warp0[0], warp1[0]], dim=1))
        x = self.input_blocks[1](x, temb)

        features = []
        for i, down_block in enumerate(self.down_blocks):
            x = down_block(x, warp0[i + 1], warp1[i + 1], temb)
            features.append(x)

        x = self.middle_block(x, temb)

        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x, features[-(i + 1)], temb) 

        x = self.output_blocks[0](x, temb)
        x = self.output_blocks[1](x)

        return x
  
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