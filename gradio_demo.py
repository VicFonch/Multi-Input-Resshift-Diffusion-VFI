import gradio as gr

from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import numpy as np
import imageio
import tempfile

from utils.utils import denorm
from model.hub import MultiInputResShiftHub

model = MultiInputResShiftHub.from_pretrained("vfontech/Multiple-Input-Resshift-VFI")
model.requires_grad_(False).cuda().eval()

transform = Compose([
    Resize((256, 448)),
    ToTensor(),
    Normalize(mean=[0.5]*3, std=[0.5]*3),
])

def to_numpy(img_tensor):
    img_np = denorm(img_tensor, mean=[0.5]*3, std=[0.5]*3).squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    return (img_np * 255).astype(np.uint8)

def interpolate(img0_pil, img2_pil, tau, num_samples):
    img0 = transform(img0_pil.convert("RGB")).unsqueeze(0).cuda()
    img2 = transform(img2_pil.convert("RGB")).unsqueeze(0).cuda()

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

demo = gr.Interface(
    fn=interpolate,
    inputs=[
        gr.Image(type="pil", label="Initial Image (frame1)"),
        gr.Image(type="pil", label="Final Image (frame3)"),
        gr.Slider(0.0, 1.0, step=0.05, value=0.5, label="Tau Value (only if Num Samples = 1)"),
        gr.Slider(1, 15, step=1, value=1, label="Number of Samples"),
    ],
    outputs=[
        gr.Image(label="Interpolated Image (if num_samples = 1)"),
        gr.Video(label="Interpolation in video (if num_samples > 1)"),
    ],
    title="Multi-Input ResShift Diffusion VFI",
    description=(
        "📄 [arXiv Paper](https://arxiv.org/pdf/2504.05402) • "
        "🤗 [Model](https://huggingface.co/vfontech/Multiple-Input-Resshift-VFI) • "
        "🧪 [Colab](https://colab.research.google.com/drive/1MGYycbNMW6Mxu5MUqw_RW_xxiVeHK5Aa#scrollTo=EKaYCioiP3tQ) • "
        "🌐 [GitHub](https://github.com/VicFonch/Multi-Input-Resshift-Diffusion-VFI)\n\n"
        "Video interpolation using Conditional Residual Diffusion.\n"
        "- All images are resized to 256x448.\n"
        "- If `Number of Samples` = 1, generates only one intermediate image with the given Tau value.\n"
        "- If `Number of Samples` > 1, ignores Tau and generates a sequence of interpolated images."
    ),
    examples=[
        ["_data/example_images/frame1.png", "_data/example_images/frame3.png", 0.5],
    ],
)

if __name__ == "__main__":
    demo.queue(max_size=12)
    demo.launch(max_threads=1)