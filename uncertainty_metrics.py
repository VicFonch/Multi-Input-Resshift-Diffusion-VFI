import os
import random
import torch
import yaml
from PIL import Image
from argparse import ArgumentParser
from torchvision.transforms import Compose, ToTensor, Resize, Normalize

from model.model import MultiInputResShift
from utils.uncertainty import compute_lpips_variability, compute_pixelwise_correlation, compute_dynamic_range

mean = [0.5, 0.5, 0.5]
sd = [0.5, 0.5, 0.5]
transform_img = Compose([
    ToTensor(),
    Resize((256, 448)),
    Normalize(mean, sd)
])

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--root_path", type=str, default="_data/ATD-12K/test")
    parser.add_argument("--tau_val", type=float, default=0.5)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--by_category", type=bool, default=False)
    return parser.parse_args()

def load_model(config_path: str = "config.yaml", 
               checkpoint_path: str = "_checkpoint/muti_input_resshift.pth"
               ) -> MultiInputResShift:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    model = MultiInputResShift(**config["model"])
    model.load_state_dict(torch.load(checkpoint_path))
    return model.eval().requires_grad_(False).cuda()

def load_triplet_images(folder_path: str) -> list[torch.Tensor]:
    def load_and_transform(img_name: str) -> torch.Tensor:
        img = Image.open(os.path.join(folder_path, img_name)).convert("RGB")
        return transform_img(img).unsqueeze(0).cuda()
    return [load_and_transform(f"frame{i}.png") for i in [1, 2, 3]]

def generate_predictions(model: MultiInputResShift,
                         I0: torch.Tensor,
                         I1: torch.Tensor,
                         tau_val: float,
                         n: int = 10
                         ) -> torch.Tensor:
    batch = I0.shape[0]
    tau: torch.Tensor = torch.stack([
        torch.full((batch, 1), tau_val, device=I0.device, dtype=I0.dtype),
        torch.full((batch, 1), 1 - tau_val, device=I0.device, dtype=I0.dtype)
    ], dim=1)
    return torch.stack([model.reverse_process([I0, I1], tau).squeeze(1) for _ in range(n)], dim=0)

def evaluate_folder(model: MultiInputResShift, 
                    img_path: str, 
                    tau_val: float) -> tuple[torch.Tensor, float, float, float]:
    I0, _, I1 = load_triplet_images(img_path)
    predictions = generate_predictions(model, I0, I1, tau_val)
    gray_pred = ( # Convert to grayscale by standard RGB weights
        0.2989 * predictions[:, 0, :, :] 
        + 0.5870 * predictions[:, 1, :, :] 
        + 0.1140 * predictions[:, 2, :, :]
    )
    return (
        torch.std(gray_pred, dim=0).cpu(),
        compute_lpips_variability(predictions),
        compute_pixelwise_correlation(predictions),
        compute_dynamic_range(gray_pred)
    )

def evaluate_dataset(model: MultiInputResShift, 
                     root_path: str, 
                     category: str | None = None, 
                     num_samples: int = 100, 
                     tau_val: float = 0.5
                     ) -> dict[str, torch.Tensor]:
    if category is None:
        folders = [f for f in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, f))]
    else:
        folders = [f for f in os.listdir(root_path) if category in f]
    folders = random.sample(folders, min(len(folders), num_samples))

    sd_list, lpips_list, corr_list, minmax_list = [], [], [], []
    count = 0

    for folder in folders:
        if count >= num_samples:
            break
        img_path = os.path.join(root_path, folder)
        try:
            sd, lpips, corr, minmax = evaluate_folder(model, img_path, tau_val)
            sd_list.append(sd)
            lpips_list.append(lpips)
            corr_list.append(corr)
            minmax_list.append(minmax)
            count += 1
        except Exception as e:
            print(f"Error processing {folder}: {e}")

    return {
        "std": torch.stack(sd_list).mean(0),
        "lpips": torch.tensor(lpips_list).mean(),
        "correlation": torch.tensor(corr_list).mean(),
        "dynamic_range": torch.tensor(minmax_list).mean()
    }

def main():
    args = parse_args()
    model = load_model()

    def print_metrics(title, results):
        print(f"\n=== {title} ===")
        print(f"  Std Dev (mean over pixels) : {results['std'].mean().item():.4f}")
        print(f"  LPIPS Variability          : {results['lpips']:.4f}")
        print(f"  Pixelwise Correlation      : {results['correlation']:.4f}")
        print(f"  Dynamic Range              : {results['dynamic_range']:.4f}")

    if args.by_category:
        for category in ["Disney", "Japan"]:
            results = evaluate_dataset(
                model=model,
                root_path=args.root_path,
                num_samples=args.num_samples,
                category=category,
                tau_val=args.tau_val
            )
            print_metrics(f"Category: {category}", results)
    else:
        results = evaluate_dataset(
            model=model,
            root_path=args.root_path,
            category=None,
            num_samples=args.num_samples,
            tau_val=args.tau_val
        )
        print_metrics("All Categories", results)

if __name__ == "__main__":
    main()