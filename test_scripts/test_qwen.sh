set -euo pipefail
export qwen_path=/data/vjuicefs_ai_camera_pgroup_ql/public_data/FaceEnhance/pretrains/Qwen-Image

GPU_ID=0
ALIGN_METHOD="wavelet"

INPUT_PATH="./data/celebA-Test-LR"
SR_PATH="./data/celebA-Test-Qwen-SR"   
OUTPUT_PATH="./results/HDRFace_Qwen_celebA"
CKPT_PATH="./pretrained_qwen"
DINO_PATH="./preset/models/dinov3-vitl16-pretrain-lvd1689m"

python inference_qwen.py \
    --gpu_id        "${GPU_ID}" \
    --input_path    "${INPUT_PATH}" \
    --sr_path       "${SR_PATH}" \
    --output_path   "${OUTPUT_PATH}" \
    --ckpt_path     "${CKPT_PATH}" \
    --dino_path     "${DINO_PATH}" \
    --scale         1.0 \
    --cfg           1.0 \
    --align_method  "${ALIGN_METHOD}" \
    --suffix        "" \
    --first_resize_w 512 \
    --first_resize_h 512 \
    --tiled_size    512