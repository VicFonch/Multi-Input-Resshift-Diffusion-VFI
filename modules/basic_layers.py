import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

class GroupNorm(nn.Module):
    def __init__(self, in_channels, num_groups=32):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)
    
    def forward(self, x):
        return self.gn(x)

class AdaLayerNorm(nn.Module):
    def __init__(self, channels, cond_channels=0, return_scale_shift=True):
        super(AdaLayerNorm, self).__init__()
        self.norm = nn.LayerNorm(channels) 
        self.return_scale_shift = return_scale_shift
        if cond_channels != 0:
            if return_scale_shift:
                self.proj = nn.Linear(cond_channels, channels * 3, bias=False)
            else:
                self.proj = nn.Linear(cond_channels, channels * 2, bias=False)
            nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x, cond = None):
        x = self.norm(x)
        if cond is None:
            return x

        def expand_dims(tensor, dims):
            for dim in dims:
                tensor = tensor.unsqueeze(dim)
            return tensor
        
        dims = list(range(1, len(x.shape) - 1))  

        if self.return_scale_shift:
            gamma, beta, sigma = self.proj(cond).chunk(3, dim=-1)
            gamma, beta, sigma = [expand_dims(t, dims) for t in (gamma, beta, sigma)]
            return x * (1 + gamma) + beta, sigma
        else:
            gamma, beta = self.proj(cond).chunk(2, dim=-1)
            gamma, beta = [expand_dims(t, dims) for t in (gamma, beta)]
            return x * (1 + gamma) + beta

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, emb_dim = 256):
        super(SinusoidalPositionalEmbedding, self).__init__()
        self.channels = emb_dim

    def forward(self, t):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, self.channels, 2, device=t.device).float() / self.channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, self.channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, self.channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

class GatedConv2d(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 padding=1, 
                 bias=False):
        super(GatedConv2d, self).__init__()
        self.gate_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.feature_conv = nn.Conv2d(in_channels, 
                                      out_channels, 
                                      kernel_size=kernel_size, 
                                      padding=padding,
                                      bias=bias)

    def forward(self, x):
        gate = torch.sigmoid(self.gate_conv(x))
        feature = F.silu(self.feature_conv(x)) 
        return gate * feature

class ResGatedBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 mid_channels=None, 
                 num_groups=32,
                 residual=True,
                 emb_channels=None,
                 gated_conv=False):
        super().__init__()
        self.residual = residual
        self.emb_channels = emb_channels
        if not mid_channels:
            mid_channels = out_channels
        
        if gated_conv: conv2d = GatedConv2d
        else: conv2d = nn.Conv2d

        self.conv1 = conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = GroupNorm(mid_channels, num_groups=num_groups)
        self.nonlienrity = nn.SiLU()
        if emb_channels:
            self.emb_proj = nn.Linear(emb_channels, mid_channels)
        self.conv2 = conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = GroupNorm(out_channels, num_groups=num_groups)

        if in_channels != out_channels:
            self.skip = conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def double_conv(self, x, emb):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.nonlienrity(x)
        if emb is not None and self.emb_channels is not None:
            x = x + self.emb_proj(emb)[:,:,None,None]
        x = self.conv2(x)
        return self.norm2(x)

    def forward(self, x, emb = None):
        if self.residual:
            if hasattr(self, 'skip'):
                return F.silu(self.skip(x) + self.double_conv(x, emb))
            return F.silu(x + self.double_conv(x, emb))
        else:
            return self.double_conv(x, emb)
        
class Downsample(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 use_conv: bool=True):
        super().__init__()

        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0)
        else:
            assert in_channels == out_channels
            self.conv = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (0, 1, 0, 1)
        hidden_states = F.pad(x, pad, mode="constant", value=0)
        return self.conv(hidden_states) if self.use_conv else self.conv(x)

class Upsample(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 use_conv: bool=True):
        super().__init__()

        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, 
                          scale_factor = (2, 2) if x.dim() == 4 else (1, 2, 2), 
                          mode='nearest')
        return self.conv(x) if self.use_conv else x

class FeedForward(nn.Module):
    def __init__(self, dim, emb_channels, expansion_rate = 4, dropout = 0.0):
        super().__init__()
        inner_dim = int(dim * expansion_rate)
        self.norm = AdaLayerNorm(dim, emb_channels)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
        self.__init_weights()

    def __init_weights(self):
        nn.init.xavier_uniform_(self.net[0].weight)
        nn.init.xavier_uniform_(self.net[3].weight)

    def forward(self, x, emb = None):
        x, sigma = self.norm(x, emb)
        return self.net(x) * sigma

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        emb_channels = 512,
        dim_head = 32,
        dropout = 0.,
        window_size = 7
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'
        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5
        self.norm = AdaLayerNorm(dim, emb_channels)

        self.to_q = nn.Linear(dim, dim, bias = False)
        self.to_k = nn.Linear(dim, dim, bias = False)
        self.to_v = nn.Linear(dim, dim, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)
        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x, emb = None):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        x, sigma = self.norm(x, emb)
        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')
    
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v)) # split heads

        q = q * self.scale
        sim = torch.einsum('b h i d, b h j d -> b h i j', q, k) # sim
        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')# add positional bias
        attn = self.attend(sim) # attention
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v) # aggregate

        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width) # merge heads
        out = self.to_out(out) # combine heads out
        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width) * sigma

class MaxViTBlock(nn.Module):
    def __init__(
        self,
        channels,
        emb_channels = 512,
        heads = 1,
        window_size = 8,
        window_attn = True,
        grid_attn = True,
        expansion_rate = 4,
        dropout = 0.0,
    ):
        super(MaxViTBlock, self).__init__()
        dim_head = channels // heads
        layer_dim = dim_head * heads
        w = window_size

        self.window_attn = window_attn
        self.grid_attn = grid_attn

        if window_attn:
            self.wind_rearrange_forward = Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1 = w, w2 = w)  # block-like attention
            self.wind_attn = Attention(
                dim = layer_dim,
                emb_channels = emb_channels,
                dim_head = dim_head, 
                dropout = dropout, 
                window_size = w
            )

            self.wind_ff = FeedForward(dim = layer_dim, 
                                       emb_channels = emb_channels,
                                       expansion_rate = expansion_rate,
                                       dropout = dropout)
            self.wind_rearrange_backward = Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)')

        if grid_attn:
            self.grid_rearrange_forward = Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1 = w, w2 = w)  # grid-like attention
            self.grid_attn = Attention(
                dim = layer_dim,
                emb_channels = emb_channels,
                dim_head = dim_head, 
                dropout = dropout, 
                window_size = w
            )
            self.grid_ff = FeedForward(dim = layer_dim, 
                                       emb_channels = emb_channels,
                                       expansion_rate = expansion_rate,
                                       dropout = dropout)
            self.grid_rearrange_backward = Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)')

    def forward(self, x, emb = None):
        if self.window_attn:
            x = self.wind_rearrange_forward(x)
            x = x + self.wind_attn(x, emb = emb)
            x = x + self.wind_ff(x, emb = emb)
            x = self.wind_rearrange_backward(x)
        if self.grid_attn:
            x = self.grid_rearrange_forward(x)
            x = x + self.grid_attn(x, emb = emb)
            x = x + self.grid_ff(x, emb = emb)
            x = self.grid_rearrange_backward(x)
        return x
