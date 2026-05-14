"""
HDRFace Inference Script
"""
import argparse
import copy
import glob
import os
import random
import sys

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as Fun
import torchvision.transforms.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image
from peft import LoraConfig
from safetensors import safe_open
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

sys.path.append(os.getcwd())

from utils.fusion import SDFM
from utils.others import get_x0_from_noise
from utils.vaehook import perfcount
from utils.wavelet_color_fix import wavelet_color_fix

class HDRFace(nn.Module):

    def __init__(self, args, gpu_id: int, unet: nn.Module | None):
        super().__init__()

        self.args = args
        self.device = torch.device(f"cuda:{gpu_id}")
        self.noise_scheduler = DDIMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        self.vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae"
        )
        if args.merge_lora:
            self.unet = copy.deepcopy(unet)
        else:
            self.unet = UNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="unet"
            )
        self.weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32

        # DINO feature encoder
        self.img_processor = AutoImageProcessor.from_pretrained(args.dino_path)
        self.img_encoder_model = AutoModel.from_pretrained(args.dino_path)

        self._load_ckpt(args.ckpt_path)
        self.unet.to(self.device, dtype=self.weight_dtype)
        self.vae.to(self.device, dtype=self.weight_dtype)
        self.img_encoder_model.to(self.device, dtype=self.weight_dtype)

        self.timesteps = 399

    def _img_encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run the DINO image encoder and return the last hidden state."""
        return self.img_encoder_model(pixel_values=pixel_values).last_hidden_state

    def _load_ckpt(self, ckpt_path: str):
        """Load fusion module weights and LoRA weights."""
        if not self.args.cat_prompt_embedding:
            self.fusion = SDFM(
                token_dim=1024,
                hidden_dim=512,
                use_layernorm=True,
            )
            fusion_path = os.path.join(ckpt_path, "fusion.pth")
            state = torch.load(fusion_path, map_location="cpu")

            # Unwrap common checkpoint wrappers
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict) and any(k.startswith("module.") for k in state):
                state = {k[len("module."):]: v for k, v in state.items()}

            self.fusion.load_state_dict(state, strict=True)
            self.fusion = self.fusion.to(self.device, dtype=self.weight_dtype)

        if not self.args.merge_lora:
            pipe = StableDiffusionPipeline(
                self.vae, None, None,
                self.unet, self.noise_scheduler,
                None, None,
            )
            pipe.load_lora_weights(ckpt_path)
            self.unet = pipe.unet

    @perfcount
    @torch.no_grad()
    def forward(
        self,
        lq: torch.Tensor,
        lq_images: list[Image.Image],
        sr_images: list[Image.Image],
    ) -> torch.Tensor:

        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()

        with torch.cuda.stream(stream1):
            sr_pixels = self.img_processor(images=sr_images, return_tensors="pt")[
                "pixel_values"
            ].to(device=self.device, dtype=self.weight_dtype)
            lq_pixels = self.img_processor(images=lq_images, return_tensors="pt")[
                "pixel_values"
            ].to(device=self.device, dtype=self.weight_dtype)

            lr_tokens = self._img_encode(lq_pixels)
            sr_tokens = self._img_encode(sr_pixels)
            prompt_embeds, _ = self.fusion(sr_tokens, lr_tokens)

        with torch.cuda.stream(stream2):
            lq_latent = (
                self.vae.encode(lq.to(self.weight_dtype)).latent_dist.sample()
                * self.vae.config.scaling_factor
            )

        torch.cuda.synchronize()

        # --- UNet denoising step ---
        model_pred = self.unet(
            lq_latent, self.timesteps, encoder_hidden_states=prompt_embeds
        ).sample

        x0 = get_x0_from_noise(
            lq_latent.double(),
            model_pred.double(),
            self.alphas_cumprod.double(),
            self.timesteps,
        ).float()

        # --- Decode and normalise to [0, 1] ---
        output = self.vae.decode(
            x0.to(self.weight_dtype) / self.vae.config.scaling_factor
        ).sample.clamp(-1, 1)
        output = output * 0.5 + 0.5

        return output.clamp(0.0, 1.0)

def merge_lora_into_unet(args) -> UNet2DConditionModel:
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    alpha_scale = float(args.lora_alpha / args.lora_rank)

    lora_path = os.path.join(args.ckpt_path, "pytorch_lora_weights.safetensors")
    with safe_open(lora_path, framework="pt") as f:
        lora_state = {key: f.get_tensor(key) for key in f.keys()}

    unet_state = unet.state_dict()
    processed_keys: set[str] = set()

    for key in lora_state:
        # --- PEFT-style keys: lora_A / lora_B ---
        if "lora_A" in key:
            lora_a_key = key
            lora_b_key = key.replace("lora_A", "lora_B")
            unet_key = key.replace(".lora_A.weight", ".weight").replace("unet.", "")

            assert lora_b_key in lora_state and unet_key in unet_state
            W_A = lora_state[lora_a_key]
            W_B = lora_state[lora_b_key]
            W_orig = unet_state[unet_key]
            processed_keys.update([lora_a_key, lora_b_key])

            if W_orig.ndim == 4:
                # Conv2d weight: (out, in, kH, kW)
                out_ch, in_ch, kH, kW = W_orig.shape
                rank = W_A.shape[0]
                assert rank == args.lora_rank, (
                    f"Expected lora_rank={args.lora_rank}, got {rank}"
                )
                delta = torch.matmul(
                    W_B.view(out_ch, rank),
                    W_A.view(rank, -1),
                ).view(out_ch, in_ch, kH, kW)
            else:
                delta = torch.mm(W_B, W_A)

            unet_state[unet_key] = W_orig + alpha_scale * delta

        # --- Legacy-style keys: lora.up / lora.down ---
        elif "lora.up.weight" in key:
            lora_up_key = key
            lora_down_key = key.replace("lora.up.weight", "lora.down.weight")
            unet_key = key.replace(".lora.up.weight", ".weight").replace("unet.", "")

            assert lora_down_key in lora_state and unet_key in unet_state
            W_up = lora_state[lora_up_key]
            W_down = lora_state[lora_down_key]
            W_orig = unet_state[unet_key]
            processed_keys.update([lora_up_key, lora_down_key])

            if W_orig.ndim == 2:
                unet_state[unet_key] = W_orig + alpha_scale * torch.matmul(W_up, W_down)
            else:
                print(f"[Warning] Unhandled weight shape for '{unet_key}', skipping.")
                continue

    unprocessed = [k for k in lora_state if k not in processed_keys]
    if unprocessed:
        print("[Warning] The following LoRA keys were not merged:")
        for k in unprocessed:
            print(f"  - {k}")

    unet.load_state_dict(unet_state)
    print("[merge_lora_into_unet] Done.")
    return unet

def main_worker(
    unet: nn.Module | None,
    rank: int,
    gpu_id: int,
    image_names: list[str],
    weight_dtype: torch.dtype,
    args,
):
    torch.cuda.set_device(gpu_id)
    model = HDRFace(args, gpu_id, unet).to(gpu_id)

    for image_name in tqdm(image_names, desc=f"GPU {gpu_id}"):
        output_path = os.path.join(args.output_dir, os.path.basename(image_name))

        # Derive the corresponding super-resolved image path
        sr_image_name = os.path.join(args.sr_path, os.path.basename(image_name))

        lq_pil = Image.open(image_name).convert("RGB")
        sr_pil = Image.open(sr_image_name).convert("RGB")

        with torch.no_grad():
            lq = F.to_tensor(lq_pil).unsqueeze(0).to(gpu_id, dtype=weight_dtype) * 2 - 1
            if lq.shape[2] == lq.shape[3]:
                lq = Fun.interpolate(lq, (512, 512), mode="bilinear", align_corners=True)

            output = model(lq, [lq_pil], [sr_pil])
            transforms.ToPILImage()(output[0].cpu()).save(output_path)

def run_inference(args, unet: nn.Module | None):
    """Distribute inference across multiple GPUs using multiprocessing."""
    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(f"{args.input_image}/*.[jpJP][pnPN]*[gG]"))
    else:
        image_names = [args.input_image]

    print(f"[run_inference] {len(image_names)} images to process.")
    random.shuffle(image_names)

    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    num_gpus = len(args.gpu_ids)
    images_per_gpu = len(image_names) // num_gpus

    processes = []
    for rank, gpu_id in enumerate(args.gpu_ids):
        start = rank * images_per_gpu
        end = start + images_per_gpu if rank != num_gpus - 1 else len(image_names)
        p = mp.Process(
            target=main_worker,
            args=(unet, rank, gpu_id, image_names[start:end], weight_dtype, args),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def parse_args():
    parser = argparse.ArgumentParser(
        description="HDRFace"
    )

    # I/O
    io = parser.add_argument_group("I/O")
    io.add_argument("--input_image", "-i", type=str, required=True,
                    help="Path to input image or directory of images.")
    io.add_argument("--sr_path", type=str, required=True,
                    help="Directory containing sr images.")
    io.add_argument("--output_dir", "-o", type=str, required=True,
                    help="Directory to save restored images.")
    
    # Model paths
    model = parser.add_argument_group("Model paths")
    model.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                       help="Path to the base Stable Diffusion model (SD 2.1).")
    model.add_argument("--ckpt_path", type=str, required=True,
                       help="Path to the HDRFace checkpoint directory.")
    model.add_argument("--dino_path", type=str, required=True,
                       help="Path to the DINOv2 encoder model.")

    # Runtime
    rt = parser.add_argument_group("Runtime")
    rt.add_argument("--mixed_precision", type=str, choices=["fp16", "fp32"], default="fp32")
    rt.add_argument("--gpu_ids", nargs="+", type=int, default=[0],
                    help="GPU IDs to use for inference.")
    rt.add_argument("--seed", type=int, default=42)
    rt.add_argument("--process_size", type=int, default=512)

    # LoRA
    lora = parser.add_argument_group("LoRA")
    lora.add_argument("--merge_lora", action="store_true",
                      help="Merge LoRA weights into UNet before inference.")
    lora.add_argument("--lora_rank", type=int, default=16)
    lora.add_argument("--lora_alpha", type=float, default=16)

    # Ablation / experimental
    exp = parser.add_argument_group("Experimental")
    exp.add_argument("--cat_prompt_embedding", action="store_true",
                     help="Concatenate prompt embeddings instead of using the fusion module.")

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    total = len(glob.glob(f"{args.input_image}/*"))
    print(f"[HDRFace] Found {total} images in input directory.")

    unet = merge_lora_into_unet(args) if args.merge_lora else None
    run_inference(args, unet)
