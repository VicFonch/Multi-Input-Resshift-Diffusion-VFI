# **Multi-Input ResShift Diffusion**

_(Waiting for acceptance)_

<div align="center">

### Time-adaptive Video Frame Interpolation based on Residual Diffusion

**Víctor Manuel Fonte Chávez · Jean-Bernard Hayet · Claudia Esteves**

[arXiv (Coming Soon)]() · [GitHub Repo](https://github.com/VicFonch/Multi-Input-Resshift-Diffusion-VFI)

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
conda create -n resshift python=3.10
conda activate resshift
pip install -r requirements.txt
```

💡 **Note**: Make sure your system is compatible with **CUDA 12.4**. If not, install [CuPy](https://docs.cupy.dev/en/stable/install.html) according to your current CUDA version.

---

## 🧩 Pretrained Models

Download our pretrained weights from **[Our weights](#)** and place them inside the `_checkpoint/` directory.

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

### 🔍 Inference

Once you have downloaded the pretrained weights, you can run inference on a sequence of frames using:

```bash
python inference.py --input_dir path/to/sequence --output_dir path/to/save --tau 0.5
```

- `--input_dir`: Folder containing `frame1.png` and `frame3.png`
- `--tau`: The interpolation time (e.g., 0.5 for halfway, 0.25 for closer to frame1)

The output will be a predicted `frame2.png` saved in the specified output directory.

---

### 🏋️ Training

To train the model from scratch or fine-tune on your own data:

```bash
python train.py --config config.yaml
```

Make sure you have placed the required RFR weights in the `_pretrain_models/` folder as mentioned above and add the path in `config.yaml`

You can edit `config.yaml` to change training parameters such as learning rate, batch size, loss weights, and dataset paths.

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
