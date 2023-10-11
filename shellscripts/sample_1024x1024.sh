

outdir="results"
ckpt=checkpoints/v2-1_512-ema-pruned.ckpt

python scripts/txt2img.py \
--prompt "A panda is surfing in the universe." \
--ckpt $ckpt \
--config configs/stable-diffusion/v2-inference.yaml \
--device cuda \
--ddim_eta 1 \
--scale 9 --seed 51 --H 1024 --W 1024 \
--dilate 2 --dilate_tau 20 --dilate_skip 4 \
--outdir $outdir
