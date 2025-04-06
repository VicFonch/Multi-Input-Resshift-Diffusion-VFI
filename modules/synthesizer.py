import torch
import torch.nn as nn

from modules.basic_layers import (
    SinusoidalPositionalEmbedding,
    ResGatedBlock,
    MaxViTBlock,
    Downsample,
    Upsample
) 

class UnetDownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int = 128,
        heads: int = 1, 
        window_size: int = 7,
        window_attn: bool = True,
        grid_attn: bool = True,
        expansion_rate: int = 4,
        num_conv_blocks: int = 2,
        dropout: float = 0.0
    ):
        super(UnetDownBlock, self).__init__()
        self.pool = Downsample(
            in_channels = in_channels,
            out_channels = in_channels,
            use_conv = True
        )
        in_channels = 3 * in_channels + 2
        self.conv = nn.ModuleList([
            ResGatedBlock(
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

    def forward(
        self, 
        x: torch.Tensor, 
        warp0: torch.Tensor, 
        warp1: torch.Tensor, 
        temb: torch.Tensor
    ):
        x = self.pool(x)
        x = torch.cat([x, warp0, warp1], dim=1)
        for conv in self.conv:
            x = conv(x, temb)
        x = self.maxvit(x, temb)
        return x

class UnetMiddleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        temb_channels: int = 128,
        heads: int = 1, 
        window_size: int = 7,
        window_attn: bool = True,
        grid_attn: bool = True,
        expansion_rate: int = 4,
        dropout: float = 0.0
    ):
        super(UnetMiddleBlock, self).__init__()

        self.middle_blocks = nn.ModuleList([
            ResGatedBlock(
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
            ResGatedBlock(
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
        in_channels: int,
        out_channels: int,
        temb_channels: int = 128,
        heads: int = 1, 
        window_size: int = 7,
        window_attn: bool = True,
        grid_attn: bool = True,
        expansion_rate: int = 4,
        num_conv_blocks: int = 2,
        dropout: float = 0.0
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
            ResGatedBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                emb_channels = temb_channels, 
                gated_conv = True
            ) for i in range(num_conv_blocks)
        ])
        
    def forward(
        self, 
        x: torch.Tensor, 
        skip_connection: torch.Tensor, 
        temb: torch.Tensor
    ):
        x = torch.cat([x, skip_connection], dim=1)
        x = self.maxvit(x, temb)
        x = self.upsample(x)   
        for conv in self.conv:
            x = conv(x, temb)     
        return x

class Synthesis(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: list[int],
        temb_channels: int,
        heads: int = 1,
        window_size: int = 7,
        window_attn: bool = True,
        grid_attn: bool = True,
        expansion_rate: int = 4,
        num_conv_blocks: int = 2,
        dropout: float = 0.0
    ):
        super(Synthesis, self).__init__()


        self.t_pos_encoding = SinusoidalPositionalEmbedding(temb_channels)

        self.input_blocks = nn.ModuleList([
            nn.Conv2d(3*in_channels + 4, channels[0], kernel_size=3, padding=1),
            ResGatedBlock(
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
            ResGatedBlock(
                in_channels = channels[0],
                out_channels = channels[0],
                emb_channels = temb_channels,
                gated_conv = True
            ),
            nn.Conv2d(channels[0], in_channels, kernel_size=3, padding=1)
        ])

    def forward(
        self, 
        x: torch.Tensor, 
        warp0: list[torch.Tensor], 
        warp1: list[torch.Tensor], 
        temb: torch.Tensor
    ):
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