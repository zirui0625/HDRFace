# -*- coding: utf-8 -*-
export HF_ENDPOINT="https://hf-mirror.com"
ckpt_path="./pretrained"
python inference_sd.py \
    --input_image "./data/celebA-Test-LR" \
    --output_dir "./results/HDRFace_sd_celebA" \
    --sr_path "./data/celebA-Test-sd-SR" \
    --pretrained_model_name_or_path ./preset/models/stable-diffusion-v2.1-base \
    --dino_path ./preset/models/dinov3-vitl16-pretrain-lvd1689m \
    --ckpt_path $ckpt_path \
    --gpu_ids 2 \
    --mixed_precision fp32 \
    --merge_lora

# ckpt_path="./exp/train/HDRFace_sd/checkpoint_100000"