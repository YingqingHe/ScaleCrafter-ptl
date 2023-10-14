# ScaleCrafter: Tuning-free Higher-Resolution Visual Generation with Diffusion Models


<div align="center">

 <a href=''><img src='https://img.shields.io/badge/ArXiv-2305.18247-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://yingqinghe.github.io/scalecrafter/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://github.com/YingqingHe/ScaleCrafter'><img src='https://img.shields.io/badge/diffuser version-code-blue'></a> 
 

_**[Yingqing He*](https://github.com/YingqingHe), [Shaoshu Yang*](), [Haoxin Chen](), [Xiaodong Cun](http://vinthony.github.io/), [Menghan Xia](https://menghanxia.github.io/), <br> 
[Yong Zhang<sup>#](https://yzhang2016.github.io), [Xintao Wang](https://xinntao.github.io/), [Ran He](), [Qifeng Chen<sup>#](https://cqf.io/), and [Ying Shan](https://scholar.google.com/citations?hl=zh-CN&user=4oXBp9UAAAAJ)**_

(* first author, # corresponding author)

</div>

## 🔆 Abstract
<b>TL; DR: 🤗🤗🤗 **ScaleCrafter:** A tuning-free approach that can generate images with resolution of 4096x4096 based on pre-trained diffusion models, which is 16 times higher than the original training resolution.</b>

> In this work, we investigate the capability of generating images from pre-trained diffusion models at much higher resolutions than the training image sizes. In addition, the generated images should have arbitrary image aspect ratios. When generating images directly at a higher resolution, 1024 x 1024, with the pre-trained Stable Diffusion using training images of resolution 512 x 512, we observe persistent problems of object repetition and unreasonable object structures. Existing works for higher-resolution generation, such as attention-based and joint-diffusion approaches, cannot well address these issues. As a new perspective, we examine the structural components of the U-Net in diffusion models and identify the crucial cause as the limited perception field of convolutional kernels. Based on this key observation, we propose a simple yet effective re-dilation that can dynamically adjust the convolutional perception field during inference. We further propose the dispersed convolution and noise-damped classifier-free guidance, which can enable ultra-high-resolution image generation (e.g., 4096 x 4096). Notably, our approach does not require any training or optimization. Extensive experiments demonstrate that our approach can address the repetition issue well and achieve state-of-the-art performance on higher-resolution image synthesis, especially in texture details. Our work also suggests that a pre-trained diffusion model trained on low-resolution images can be directly used for high-resolution visual generation without further tuning, which may provide insights for future research on ultra-high-resolution image and video synthesis.


## 📝 Changelog
- __[2023.10.12]__: 🔥 Release paper and source code.
<br>

## ⏳ TODO
- [ ] sampling scripts of other resolutions


## ⚙️ Setup
```bash
conda create -n scalecrafter-ptl python=3.8.5
conda activate scalecrafter
pip install -r requirements.txt
```
download checkpoint
```
mkdir checkpoints
wget -O checkpoints/v2-1_512-ema-pruned.ckpt https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt
```
---

## 💫 Inference
```
bash shellscripts/sample_1024x1024.sh
```

## 🔮 Tiled VAE decoding with SyncGN
<div>
Input prompt: "A corgi sits on a beach chair on a beautiful beach, with palm trees behind, high details"
Results:
<table class="center">
  <tr>
  <td><img src=assets/decode/1-wo-overlap-wo-gn.png width="320"></td>
  <td><img src=assets/decode/2-w-overlap-wo-gn.png width="320"></td>
  <td><img src=assets/decode/3-wo-overlap-w-gn.png width="320"></td>
  <td><img src=assets/decode/4-w-overlap-w-gn.png width="320"></td>
  <tr>
  <td style="text-align:center;" width="320">"w/o overlap; w/o syncGN"</td>
  <td style="text-align:center;" width="320">"w/ overlap; w/o syncGN"</td>
  <td style="text-align:center;" width="320">"w/o overlap; w/ syncGN"</td>
  <td style="text-align:center;" width="320">"w/ overlap; w/ syncGN"</td>
  <tr>
</table >

For sampling with overlapped tiled decoding and SyncGN, just add: `--tiled_decoding --sync_gn`. For disabling the overlapped tiling, set `--overlap 0`.

Full commands:

```
python scripts/txt2img.py \
--prompt "A corgi sits on a beach chair on a beautiful beach, with palm trees behind, high details" \
--ckpt checkpoints/v2-1_512-ema-pruned.ckpt \
--config configs/stable-diffusion/v2-inference.yaml \
--device cuda \
--ddim_eta 1 \
--scale 9 --seed 51 --H 1024 --W 1024 \
--dilate 2 --dilate_tau 20 --dilate_skip 4 \
--n_iter 1 --n_samples 1 \
--outdir $outdir \
--tiled_decoding --sync_gn
```
