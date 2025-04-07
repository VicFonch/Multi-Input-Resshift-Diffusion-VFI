# **Multi Input Resshift Diffusion (Waiting for acceptance)**

<div align="center">
<h2>Time-adaptive Video Frame Interpolation based on Residual Diffusion</h2>

_**Victor Manuel Fonte Chavez | Jean-Bernard Hayet | Claudia Esteves**_

[arxiv](Waiting for acceptance) | [github](https://github.com/VicFonch/Multi-Input-Resshift-Diffusion-VFI)

<strong>From CIMAT (Research Center in Mathematics) at 24TH ACM SIGGRAPH / Eurographics Symposium on Computer Animation</strong>

## </div>

In this work, we propose a new diffusion-based method for video frame interpolation (VFI), in the context of traditional hand-made animation. We introduce three main contributions: The first is that we explicitly handle the interpolation time in our model, which we also re-estimate during the training process, to cope with the particularly large variations observed in the animation domain, compared to natural videos; The second is that we adapt and generalize a diffusion scheme called ResShift recently proposed in the super-resolution community to VFI, which allows us to perform a very low number of diffusion steps (in the order of $10$) to produce our estimates; The third is that we leverage the stochastic nature of the diffusion process to provide a pixel-wise estimate of the uncertainty on the interpolated frame, which could be useful to anticipate where the model may be wrong. We provide extensive comparisons with respect to state-of-the-art models and show that our model outperforms these models on animation videos.

---

## Dependencies

To install all the necessary dependencies, run the following command:

```bash
conda create -n resshift python=3.10
conda activate resshift
pip install -r requirements.txt
```

Note: Make sure you have CUDA 12.4 installed on your system, as the project uses CuPy with support for this version of CUDA. Otherwise, install CuPy for the version of CUDA you have on your computer.

---

## Our ATD-12k file structure

For evaluation and testing, download the ATD12k dataset from [AnimeInterp](https://github.com/lisiyao21/AnimeInterp). Copy the data into the `_data` folder and sort it as follows, following the steps described in the paper.

    _data/ATD-12k/
        train/
            Disney_xxxxxx/
                frame1.jpg
                frame2.jpg
                frane3.jpg
            Japan_xxxxxx/
                frame1.jpg
                frame2.jpg
                frane3.jpg
        val/
            ...
        test/
            ...

---
