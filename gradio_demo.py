import gradio as gr

from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import numpy as np
import imageio
import tempfile

from utils.utils import denorm
from model.hub import MultiInputResShiftHub

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiInputResShiftHub.from_pretrained("vfontech/Multiple-Input-Resshift-VFI")
model.requires_grad_(False).to(device).eval()

transform = Compose([
    Resize((256, 448)),
    ToTensor(),
    Normalize(mean=[0.5]*3, std=[0.5]*3),
])

def to_numpy(img_tensor: torch.Tensor) -> np.ndarray:
    img_np = denorm(img_tensor, mean=[0.5]*3, std=[0.5]*3).squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    return (img_np * 255).astype(np.uint8)

def interpolate(img0_pil: Image.Image, 
                img2_pil: Image.Image, 
                tau: float=0.5, 
                num_samples: int=1) -> tuple:
    img0 = transform(img0_pil.convert("RGB")).unsqueeze(0).to(device)
    img2 = transform(img2_pil.convert("RGB")).unsqueeze(0).to(device)

    try:
        if num_samples == 1:
            # Unique image
            img1 = model.reverse_process([img0, img2], tau)
            return Image.fromarray(to_numpy(img1)), None
        else:
            # Múltiples imágenes → video
            frames = [to_numpy(img0)]
            for t in np.linspace(0, 1, num_samples):
                img = model.reverse_process([img0, img2], float(t))
                frames.append(to_numpy(img))
            frames.append(to_numpy(img2))

            temp_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            imageio.mimsave(temp_path, frames, fps=8)
            return None, temp_path
    except Exception as e:
        print(f"Error during interpolation: {e}")
        return None, None
    
def build_demo() -> gr.Blocks:
    header = """
    <div style="text-align: center; padding: 1.5rem 0;">
        <h1 style="font-size: 2.4rem; margin-bottom: 0.5rem;">🎞️ Multi-Input ResShift Diffusion VFI</h1>
        <p style="font-size: 1.1rem; color: #444;">
            Efficient and stochastic video frame interpolation for hand-drawn animation.
        </p>

        <div style="display: flex; justify-content: center; flex-wrap: wrap; gap: 12px; margin: 1rem 0;">
            <a href="https://arxiv.org/pdf/2504.05402">
                <img src="https://img.shields.io/badge/arXiv-Paper-A42C25.svg" alt="arXiv">
            </a>
            <a href="https://huggingface.co/vfontech/Multiple-Input-Resshift-VFI">
                <img src="https://img.shields.io/badge/🤗-Model-ffbd45.svg" alt="HF">
            </a>
            <a href="https://colab.research.google.com/drive/1MGYycbNMW6Mxu5MUqw_RW_xxiVeHK5Aa#scrollTo=EKaYCioiP3tQ">
                <img src="https://img.shields.io/badge/Colab-Demo-green.svg" alt="Colab">
            </a>
            <a href="https://github.com/VicFonch/Multi-Input-Resshift-Diffusion-VFI">
                <img src="https://img.shields.io/badge/GitHub-Code-blue.svg?logo=github" alt="GitHub">
            </a>
        </div>

        <div style="max-width: 700px; margin: 0 auto; font-size: 0.96rem; color: #333;">
            <p style="margin-bottom: 0.5rem;"><strong>Usage:</strong></p>
            <ul style="list-style-type: none; padding: 0; line-height: 1.6;">
                <li>All images are resized to <strong>256×448</strong>.</li>
                <li>If <code>Number of Samples = 1</code>, generates a single interpolated frame using Tau.</li>
                <li>If <code>Number of Samples > 1</code>, Tau is ignored and a full interpolation sequence is generated.</li>
            </ul>
        </div>
    </div>
    """

    with gr.Blocks() as demo:
        gr.HTML(header)

        with gr.Row():
            img0 = gr.Image(type="pil", label="Initial Image (frame1)")
            img2 = gr.Image(type="pil", label="Final Image (frame3)")

        with gr.Row():
            tau = gr.Slider(0.0, 1.0, step=0.05, value=0.5, label="Tau Value (only if Num Samples = 1)")
            samples = gr.Slider(1, 20, step=1, value=1, label="Number of Samples")

        btn = gr.Button("Generate")

        with gr.Row():
            output_img = gr.Image(label="Interpolated Image (if num_samples = 1)")
            output_vid = gr.Video(label="Interpolation in video (if num_samples > 1)")

        btn.click(interpolate, inputs=[img0, img2, tau, samples], outputs=[output_img, output_vid])

        gr.Examples(
            examples=[
                ["_data/example_images/frame1.png", "_data/example_images/frame3.png", 0.5, 1],
            ],
            inputs=[img0, img2, tau, samples],
        )

    return demo

if __name__ == "__main__":
    demo = build_demo()
    #demo.launch(server_name="0.0.0.0", ssr_mode=False)
    demo.launch()