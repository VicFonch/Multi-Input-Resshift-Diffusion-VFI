import torch
import cupy
import kornia
import torch.nn as nn

from modules.cupy_module.cupy_utils import cupy_launch
# Code taken from https://github.com/ShuhongChen/eisai-anime-interpolator

_batch_edt_kernel = ('kernel_dt', '''
    extern "C" __global__ void kernel_dt(
        const int bs,
        const int h,
        const int w,
        const float diam2,
        float* data,
        float* output
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= bs*h*w) {
            return;
        }
        int pb = idx / (h*w);
        int pi = (idx - h*w*pb) / w;
        int pj = (idx - h*w*pb - w*pi);

        float cost;
        float mincost = diam2;
        for (int j = 0; j < w; j++) {
            cost = data[h*w*pb + w*pi + j] + (pj-j)*(pj-j);
            if (cost < mincost) {
                mincost = cost;
            }
        }
        output[idx] = mincost;
        return;
    }
''')

class NEDT(nn.Module):
    def __init__(self):
        super().__init__()

    def batch_edt(self, img, block=1024):
        # must initialize cuda/cupy after forking
        _batch_edt = cupy_launch(*_batch_edt_kernel)

        # bookkeeppingg
        if len(img.shape)==4:
            assert img.shape[1]==1
            img = img.squeeze(1)
            expand = True
        else:
            expand = False
        bs,h,w = img.shape
        diam2 = h**2 + w**2
        odtype = img.dtype
        grid = (img.nelement()+block-1) // block

        # first pass, y-axis
        data = ((1-img.type(torch.float32)) * diam2).contiguous()
        intermed = torch.zeros_like(data)
        _batch_edt(
            grid=(grid, 1, 1),
            block=(block, 1, 1),  # < 1024
            args=[
                cupy.int32(bs),
                cupy.int32(h),
                cupy.int32(w),
                cupy.float32(diam2),
                data.data_ptr(),
                intermed.data_ptr(),
            ],
        )
        
        # second pass, x-axis
        intermed = intermed.permute(0,2,1).contiguous()
        out = torch.zeros_like(intermed)
        _batch_edt(
            grid=(grid, 1, 1),
            block=(block, 1, 1),
            args=[
                cupy.int32(bs),
                cupy.int32(w),
                cupy.int32(h),
                cupy.float32(diam2),
                intermed.data_ptr(),
                out.data_ptr(),
            ],
        )
        ans = out.permute(0,2,1).sqrt()
        ans = ans.type(odtype) if odtype!=ans.dtype else ans

        if expand:
            ans = ans.unsqueeze(1)
        return ans

    def batch_dog(self, img, t=1.0, sigma=1.0, k=1.6, epsilon=0.01, kernel_factor=4, clip=True):
        # to grayscale if needed
        bs,ch,h,w = img.shape
        if ch in [3,4]:
            img = kornia.color.rgb_to_grayscale(img[:,:3])
        else:
            assert ch==1

        # calculate dog
        kern0 = max(2*int(sigma*kernel_factor)+1, 3)
        kern1 = max(2*int(sigma*k*kernel_factor)+1, 3)
        g0 = kornia.filters.gaussian_blur2d(
            img, (kern0,kern0), (sigma,sigma), border_type='replicate',
        )
        g1 = kornia.filters.gaussian_blur2d(
            img, (kern1,kern1), (sigma*k,sigma*k), border_type='replicate',
        )
        out = 0.5 + t*(g1 - g0) - epsilon
        out = out.clip(0,1) if clip else out
        return out
    
    def forward(
        self, img, t=2.0, sigma_factor=1/540, 
        k=1.6, epsilon=0.01,
        kernel_factor=4, exp_factor=540/15
    ):
        dog = self.batch_dog(
            img, t=t, sigma=img.shape[-2]*sigma_factor, k=k,
            epsilon=epsilon, kernel_factor=kernel_factor, clip=False,
        )
        edt = self.batch_edt((dog > 0.5).float())
        out = 1 - (-edt*exp_factor / max(edt.shape[-2:])).exp()
        return out