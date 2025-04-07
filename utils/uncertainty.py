import itertools
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as LPIPS
from utils.utils import denorm

def compute_lpips_variability(samples, net='alex', device='cuda'):
    loss_fn = LPIPS(net_type=net).to(device)
    loss_fn.eval()

    if samples.min() >= 0.0:
        samples = samples * 2 - 1  # Convertir [0, 1] → [-1, 1]

    N = samples.size(0)
    scores = []

    for i, j in itertools.combinations(range(N), 2):
        x = samples[i:i+1].to(device)
        y = samples[j:j+1].to(device)
        dist = loss_fn(denorm(x.clamp(-1, 1)), denorm(y.clamp(-1, 1)))
        scores.append(dist.item())

    return sum(scores) / len(scores)

def compute_pixelwise_correlation(samples):
    N, C, H, W = samples.shape
    samples_flat = samples.view(N, C, -1)  # (N, C, H*W)

    corrs = []

    for i, j in itertools.combinations(range(N), 2):
        x = samples_flat[i]  # (C, HW)
        y = samples_flat[j]  # (C, HW)

        mean_x = x.mean(dim=1, keepdim=True)
        mean_y = y.mean(dim=1, keepdim=True)

        x_centered = x - mean_x
        y_centered = y - mean_y

        numerator = (x_centered * y_centered).sum(dim=1)
        denominator = (x_centered.norm(dim=1) * y_centered.norm(dim=1)) + 1e-8

        corr = numerator / denominator  # (C,)
        corrs.append(corr.mean().item())

    return sum(corrs) / len(corrs)

def compute_dynamic_range(samples):
    max_vals, _ = samples.max(dim=0)  # (C, H, W)
    min_vals, _ = samples.min(dim=0)  # (C, H, W)
    
    dynamic_range = max_vals - min_vals  # (C, H, W)
    return dynamic_range.mean().item()