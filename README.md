# **Multi-Input ResShift Diffusion**

_(Waiting for acceptance)_

<div align="center">

### Time-adaptive Video Frame Interpolation based on Residual Diffusion

**Víctor Manuel Fonte Chávez · Jean-Bernard Hayet · Claudia Esteves**

<div align="center">
  <a href='https://arxiv.org/pdf/2504.05402'><img src='https://img.shields.io/badge/arXiv-2405.17933-b31b1b.svg'></a> 
  <a href='https://huggingface.co/vfontech/Multiple-Input-Resshift-VFI'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face%20-Model-yellow'></a>
  <a href='https://colab.research.google.com/drive/1MGYycbNMW6Mxu5MUqw_RW_xxiVeHK5Aa#scrollTo=EKaYCioiP3tQ'><img src='https://img.shields.io/badge/Colab-Demo-Green'></a>
  <a href='https://huggingface.co/spaces/vfontech/Multi-Input-Res-Diffusion-VFI'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face%20Space-Demo-g'></a>
</div>

<!-- [arXiv](https://arxiv.org/pdf/2504.05402) · [GitHub Repo](https://github.com/VicFonch/Multi-Input-Resshift-Diffusion-VFI) -->

**From CIMAT (Research Center in Mathematics)**  
Accepted at the **24th ACM SIGGRAPH / Eurographics Symposium on Computer Animation**

</div>

---

## 🧠 Overview

In this work, we propose a novel **diffusion-based approach** for **video frame interpolation (VFI)** tailored to **traditional hand-drawn animation**. We introduce three key contributions:

1. **Time-awareness**: Our model explicitly handles the interpolation timestep, which is also re-estimated during training to better adapt to the large temporal variations found in animation.

2. **Residual Diffusion**: We extend and generalize the **ResShift** diffusion scheme (originally proposed for super-resolution) to the VFI setting, achieving strong results with very few diffusion steps (≈10).

3. **Uncertainty Estimation**: By leveraging the stochastic nature of diffusion, we provide **pixel-wise uncertainty maps** to indicate where the model may be less confident.

We perform extensive evaluations against state-of-the-art methods and show that our model achieves superior results on animation video datasets.

---

## ⚙️ Setup

To install the required environment:

```bash
conda create -n multi-input-resshift python=3.12
conda activate multi-input-resshift
pip install -r requirements.txt
```

💡 **Note**: Make sure your system is compatible with **CUDA 12.4**. If not, install [CuPy](https://docs.cupy.dev/en/stable/install.html) according to your current CUDA version.

---

## 🧩 Pretrained Models

Download our pretrained weights from **[Our weights](https://huggingface.co/vfontech/Multiple-Input-Resshift-VFI/tree/main)** and place them inside the `_checkpoint/` directory.

If you plan to **train the model**, you must also download the pretrained weights for the **RFR module**. These can be found in the [AnimeInterp repository](https://github.com/lisiyao21/AnimeInterp) and should be placed in the `_pretrain_models/` folder.

---

## 🗂️ Dataset Structure: ATD-12K

To test and evaluate the model, download the **ATD-12K dataset** from [AnimeInterp](https://github.com/lisiyao21/AnimeInterp).  
Place the dataset inside the `_data/` folder and organize it as follows:

```
_data/ATD-12k/
├── train/
│   ├── Disney_xxxxxx/
│   │   ├── frame1.jpg
│   │   ├── frame2.jpg
│   │   └── frame3.jpg
│   ├── Japan_xxxxxx/
│   │   ├── frame1.jpg
│   │   ├── frame2.jpg
│   │   └── frame3.jpg
├── val/
│   └── ...
└── test/
    └── ...
```

📌 Each folder (e.g., `Disney_xxxxxx`) contains a sequence of 3 consecutive frames from a traditional animation video.

---

## 🚀 Usage

### 🏋️ Training

To train the model from scratch or fine-tune on your own data:

```bash
python train.py --config config.yaml --data_path _data/ATD-12K
```

Make sure to place the required `RFR` pretrained weights inside the `_pretrain_models/` directory, as previously indicated. Then, specify the corresponding path in the `config.yaml` file.

If you wish to perform fine-tuning, simply set the path to these weights in the `pretrained_model_path` parameter within the configuration file.

You can edit `config.yaml` to change training parameters such as learning rate, batch size, loss weights, and dataset paths.

---

### 🔍 Inference

You can run inference on a sequence of frames using:

```bash
python inference.py --img0_path path/to/img0 --img1_path path/to/img1 --output_path path/to/save --tau_val 0.5 --num_samples 1
```

- `--img0_path` and `--img1_path`: Paths to the input images (e.g., `frame1.png` and `frame3.png`)
- `--output_path`: Folder where the interpolated frame will be saved
- `--tau_val`: The interpolation time (e.g., `0.5` for halfway between the frames, `0.25` for closer to `img0`, and `0.75` for closer to `img1`)
- `--num_samples`: Number of stochastic samples to generate (useful for uncertainty estimation; set to `1` for deterministic output)

The output will be a predicted `frame2.png` saved in the specified output directory. If `--num_samples > 1`, the `--tau_val` argument will be ignored and a linear spacing of `num_samples` tau values between `0` and `1` (excluding endpoints) will be used instead.

---

## 🌐 Gradio Demo

You can interact with the model directly in your browser using the [Gradio demo hosted on Hugging Face Spaces](https://huggingface.co/spaces/vfontech/Multi-Input-Res-Diffusion-VFI) or if you have a Nvidia GPU in your local computer, you can run it using the following command:

```bash
python gradio_demo.py
```

You can upload an initial and a final image (e.g., `frame1` and `frame3`), and control the interpolation behavior with the following parameters:

- **Tau**: A float between 0 and 1 indicating the position of the interpolated frame between the two inputs. This is only used if `Number of Samples = 1`.
- **Number of Samples**: If greater than 1, the Tau value is ignored and a sequence of interpolated frames is generated and returned as an `.mp4` video.

All images are automatically resized to **256 × 448**, and the output adapts depending on the number of samples:

- If `num_samples = 1`, you will get a single interpolated image.
- If `num_samples > 1`, you will get a video showing the temporal interpolation.

This demo runs fully in the browser using GPU-backed inference on Hugging Face Spaces. Huge thanks to Hugging Face for the hardware support!

---

## 📊 Uncertainty Evaluation

We provide a script to evaluate the **uncertainty of the model's predictions** by generating multiple interpolated frames and analyzing their variability.

The script computes:

- **Per-pixel standard deviation**
- **LPIPS variability**
- **Pixelwise correlation**
- **Dynamic range**

Run the script with:

```bash
python evaluate_uncertainty.py --root_path _data/ATD-12K/test --tau_val 0.5 --num_samples 100
```

Add `--by_category True` to evaluate separately for animation styles like _Disney_ or _Japan_.

---

## 📚 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{chavez2025timeadaptivevideoframeinterpolation,
  title        = {Time-adaptive Video Frame Interpolation based on Residual Diffusion},
  author       = {Victor Fonte Chavez and Claudia Esteves and Jean-Bernard Hayet},
  year         = {2025},
  eprint       = {2504.05402},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url          = {https://arxiv.org/abs/2504.05402}
}
```
