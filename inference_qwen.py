"""
HDRFace Inference Script
"""
import argparse
import glob
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from generator import Generator
from utils.fusion import SDFM
from utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix

SUPPORTED_IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
DEFAULT_DINO_DIM = 1024
DEFAULT_DIT_TEXT_DIM = 3584  # Qwen / SD3 DiT text-stream dimension

class HDRFaceModel(nn.Module):
    def __init__(
        self,
        generator: Generator,
        dino_path: str,
        device: torch.device,
        weight_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        self.generator = generator
        self.device = device
        self.weight_dtype = weight_dtype

        print(f'[HDRFace] Loading DINOv3 from: {dino_path}')
        self.img_processor = AutoImageProcessor.from_pretrained(dino_path)
        self.img_encoder_model = AutoModel.from_pretrained(dino_path)

        self.fusion = SDFM(
            token_dim=DEFAULT_DINO_DIM,
            hidden_dim=512,
            use_layernorm=True,
        )

        dit_text_dim = self._get_dit_text_dim(generator)
        print(f'[HDRFace] DiT text dim: {dit_text_dim}, DINO dim: {DEFAULT_DINO_DIM}')
        if dit_text_dim != DEFAULT_DINO_DIM:
            print(f'[HDRFace] Adding projection layer: {DEFAULT_DINO_DIM} -> {dit_text_dim}')
            self.dino_projection: Optional[nn.Linear] = nn.Linear(DEFAULT_DINO_DIM, dit_text_dim)
        else:
            print('[HDRFace] No projection needed; dimensions already match.')
            self.dino_projection = None

        self.img_encoder_model.to(self.device, dtype=self.weight_dtype)
        self.fusion.to(self.device, dtype=self.weight_dtype)
        if self.dino_projection is not None:
            self.dino_projection.to(self.device, dtype=self.weight_dtype)

    @staticmethod
    def _get_dit_text_dim(generator: Generator) -> int:
        try:
            dit = generator.pipe.dit
            if hasattr(dit, 'txt_norm'):
                if hasattr(dit.txt_norm, 'normalized_shape'):
                    return dit.txt_norm.normalized_shape[0]
                if hasattr(dit.txt_norm, 'weight'):
                    return dit.txt_norm.weight.shape[0]

            if hasattr(dit, 'txt_in'):
                if hasattr(dit.txt_in, 'in_features'):
                    return dit.txt_in.in_features
                if hasattr(dit.txt_in, 'linear'):
                    if hasattr(dit.txt_in.linear, 'in_features'):
                        return dit.txt_in.linear.in_features
                    if hasattr(dit.txt_in.linear, 'weight'):
                        return dit.txt_in.linear.weight.shape[1]

            for name, module in dit.named_modules():
                if 'txt' in name.lower() and 'norm' in name.lower() and isinstance(module, nn.LayerNorm):
                    return module.normalized_shape[0]

        except Exception as exc:  
            print(f'[HDRFace] Warning: failed to detect text dim ({exc}); falling back to default.')

        return DEFAULT_DIT_TEXT_DIM

    def load_weights(self, ckpt_path: str) -> None:

        fusion_path = os.path.join(ckpt_path, 'fusion.pth')
        projection_path = os.path.join(ckpt_path, 'projection.pth')

        if not os.path.exists(fusion_path):
            fusion_path = self._find_alternative(ckpt_path, keyword='fusion')

        print(f'[HDRFace] Loading fusion weights from: {fusion_path}')
        fusion_state = self._load_state_dict(fusion_path)
        self.fusion.load_state_dict(fusion_state, strict=True)
        print('[HDRFace] Fusion weights loaded.')

        if self.dino_projection is None:
            return

        if not os.path.exists(projection_path):
            print(f'[HDRFace] Warning: projection checkpoint not found at {projection_path}')
            return

        print(f'[HDRFace] Loading projection weights from: {projection_path}')
        projection_state = self._load_state_dict(projection_path)
        self.dino_projection.load_state_dict(projection_state, strict=True)
        print('[HDRFace] Projection weights loaded.')

    @staticmethod
    def _find_alternative(ckpt_dir: str, keyword: str) -> str:
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f'Checkpoint directory does not exist: {ckpt_dir}')
        candidates = [f for f in os.listdir(ckpt_dir) if keyword in f.lower() and f.endswith('.pth')]
        if not candidates:
            available = '\n  - '.join(os.listdir(ckpt_dir))
            raise FileNotFoundError(
                f'No "{keyword}" checkpoint found in {ckpt_dir}. Available files:\n  - {available}'
            )
        return os.path.join(ckpt_dir, candidates[0])

    @staticmethod
    def _load_state_dict(path: str) -> dict:
        state = torch.load(path, map_location='cpu')
        if isinstance(state, dict) and 'state_dict' in state:
            state = state['state_dict']
        if isinstance(state, dict) and any(k.startswith('module.') for k in state):
            state = {k[len('module.'):]: v for k, v in state.items()}
        return state

    @torch.no_grad()
    def extract_dino_features(self, images) -> torch.Tensor:
        if not isinstance(images, list):
            images = [images]

        pixel_values = self.img_processor(
            images=images,
            return_tensors='pt',
        )['pixel_values'].to(device=self.device, dtype=self.weight_dtype)

        return self.img_encoder_model(pixel_values=pixel_values).last_hidden_state

    @torch.no_grad()
    def process_with_dino(
        self,
        lq_image: Image.Image,
        sr_image: Image.Image,
        verbose: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        lr_tokens = self.extract_dino_features(lq_image)
        sr_tokens = self.extract_dino_features(sr_image)

        prompt_embeds, alpha = self.fusion(sr_tokens, lr_tokens)

        if self.dino_projection is not None:
            prompt_embeds = self.dino_projection(prompt_embeds)

        b, seq_len, _ = prompt_embeds.shape
        prompt_emb_mask = torch.ones(b, seq_len, dtype=torch.bool, device=self.device)

        if verbose:
            print(f'  DINO LR tokens : {tuple(lr_tokens.shape)}')
            print(f'  DINO SR tokens : {tuple(sr_tokens.shape)}')
            print(f'  Prompt embeds  : {tuple(prompt_embeds.shape)}')
            if isinstance(alpha, torch.Tensor):
                print(f'  Fusion alpha   : [{alpha.min().item():.4f}, {alpha.max().item():.4f}]')

        return prompt_embeds, prompt_emb_mask, alpha


def collect_image_paths(input_path: str) -> List[str]:
    """Return a sorted list of image paths under ``input_path``."""
    if os.path.isdir(input_path):
        paths: List[str] = []
        for ext in SUPPORTED_IMAGE_EXTS:
            paths.extend(glob.glob(os.path.join(input_path, '**', f'*{ext}'), recursive=True))
        return sorted(set(paths))
    return [input_path]

def find_sr_image(lq_path: str, sr_dir: Optional[str], input_root: str) -> Optional[str]:
    if sr_dir is None:
        return None

    rel_path = (
        os.path.relpath(lq_path, input_root)
        if os.path.isdir(input_root) else os.path.basename(lq_path)
    )
    rel_stem = os.path.splitext(rel_path)[0]
    base_stem = os.path.splitext(os.path.basename(lq_path))[0]

    candidates: List[str] = []
    for ext in SUPPORTED_IMAGE_EXTS:
        candidates.append(os.path.join(sr_dir, rel_stem + ext))
        candidates.append(os.path.join(sr_dir, base_stem + ext))
    for ext in SUPPORTED_IMAGE_EXTS:
        candidates.extend(
            glob.glob(os.path.join(sr_dir, '**', base_stem + ext), recursive=True)
        )

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def build_output_path(
    img_path: str,
    input_root: str,
    output_dir: str,
    suffix: str,
    keep_structure: bool,
) -> str:
    """Build the destination file path for a given input image."""
    if keep_structure and os.path.isdir(input_root):
        rel_path = os.path.relpath(img_path, input_root)
    else:
        rel_path = os.path.basename(img_path)

    stem = os.path.splitext(rel_path)[0]
    return os.path.join(output_dir, f'{stem}{suffix}.png')


def parse_index_range(text: str) -> Tuple[int, Optional[int]]:
    """Parse a ``"start,end"`` argument into integer indices."""
    parts = text.split(',')
    start = int(parts[0]) if parts[0] else 0
    end = int(parts[1]) if len(parts) > 1 and parts[1] else None
    return start, end


def build_generator(qwen_path: str, ckpt_path: str, device: torch.device) -> Generator:
    """Construct and return the Qwen-Image generator on the requested device."""
    trained_ckpt = os.path.join(ckpt_path, 'pretrained.pth')
    transformer_files = sorted(
        glob.glob(os.path.join(qwen_path, 'transformer', 'diffusion_pytorch_model-*.safetensors'))
    )
    text_encoder_files = sorted(
        glob.glob(os.path.join(qwen_path, 'text_encoder', 'model-*.safetensors'))
    )
    vae_file = os.path.join(qwen_path, 'vae', 'diffusion_pytorch_model.safetensors')
    tokenizer_path = os.path.join(qwen_path, 'tokenizer')

    if not transformer_files or not text_encoder_files or not os.path.exists(vae_file):
        raise FileNotFoundError(f'Incomplete Qwen-Image checkpoint at: {qwen_path}')

    import json
    pretrained_weights = json.dumps([transformer_files, text_encoder_files, vae_file])

    print('=' * 60)
    print('Loading Qwen-Image generator')
    print('=' * 60)

    generator = Generator(
        torch_dtype=torch.bfloat16,
        pretrained_weights=pretrained_weights,
        tokenizer_path=tokenizer_path,
        learning_rate=0,
        use_gradient_checkpointing=False,
        pretrained_ckpt_path_gen=trained_ckpt,
    )
    generator.pipe.requires_grad_(False)
    generator = generator.to(device=device)
    generator.device = device
    generator.pipe.device = device
    print('[HDRFace] Generator ready.')
    return generator

@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for HDRFace inference.')
    device = torch.device('cuda:0') 
    torch.cuda.set_device(device)
    print(f'[HDRFace] Using GPU id={args.gpu_id} (mapped to cuda:0).')

    qwen_path = os.environ.get('qwen_path')
    if qwen_path is None:
        raise EnvironmentError('Please set the `qwen_path` environment variable.')

    generator = build_generator(qwen_path, args.ckpt_path, device)

    print('=' * 60)
    print('Initializing HDRFace')
    print('=' * 60)
    model = HDRFaceModel(generator=generator, dino_path=args.dino_path, device=device)
    if args.ckpt_path:
        model.load_weights(args.ckpt_path)
    else:
        print('[HDRFace] Warning: no fusion checkpoint provided.')

    if args.sr_path is not None:
        if not os.path.exists(args.sr_path):
            raise FileNotFoundError(f'SR directory does not exist: {args.sr_path}')
        print(f'[HDRFace] Using external SR images from: {args.sr_path}')
    else:
        print('[HDRFace] No SR path provided; will fall back to bicubic upsampling.')

    image_paths = collect_image_paths(args.input_path)
    print(f'[HDRFace] Found {len(image_paths)} input images.')

    start_idx, end_idx = parse_index_range(args.start_end)
    image_paths = image_paths[start_idx:end_idx] if end_idx is not None else image_paths[start_idx:]
    print(f'[HDRFace] Processing range [{start_idx}:{end_idx}]: {len(image_paths)} images.')

    os.makedirs(args.output_path, exist_ok=True)

    if image_paths:
        image_paths.insert(0, image_paths[0])

    tile_size = args.tiled_size // 8
    tile_stride = tile_size - tile_size // 4
    sr_found, sr_missing, processed = 0, 0, 0
    t_start = time.perf_counter()

    print('\n' + '=' * 60)
    print('Starting inference')
    print('=' * 60 + '\n')

    for idx, img_path in enumerate(tqdm(image_paths, desc='HDRFace')):
        if idx == 1:
            t_start = time.perf_counter()

        print('\n' + '=' * 60)
        print(f'[{idx}/{len(image_paths) - 1}] {os.path.basename(img_path)}')
        print('=' * 60)

        out_path = build_output_path(
            img_path=img_path,
            input_root=args.input_path,
            output_dir=args.output_path,
            suffix=args.suffix,
            keep_structure=args.keep_structure,
        )

        if os.path.exists(out_path) and not args.overwrite:
            print(f'  Skip (exists): {out_path}')
            continue

        print(f'  Input : {img_path}')
        print(f'  Output: {out_path}')
        lq_img = Image.open(img_path).convert('RGB')
        w, h = lq_img.size
        print(f'  LQ size: {w}x{h}')

        if args.first_resize_w is not None and args.first_resize_h is not None:
            target_w, target_h = args.first_resize_w, args.first_resize_h
        else:
            target_w, target_h = round(w * args.scale), round(h * args.scale)
        print(f'  Target size: {target_w}x{target_h}')

        sr_img_path = find_sr_image(img_path, args.sr_path, args.input_path)
        if sr_img_path:
            print(f'  SR found: {os.path.basename(sr_img_path)}')
            sr_img = Image.open(sr_img_path).convert('RGB')
            if sr_img.size != (target_w, target_h):
                print(f'  Resizing SR to target: {target_w}x{target_h}')
                sr_img = sr_img.resize((target_w, target_h), Image.LANCZOS)
            sr_found += 1
        else:
            if args.sr_path:
                print('  SR not found, falling back to bicubic upsampling.')
                sr_missing += 1
            else:
                print('  Using bicubic upsampling.')
            sr_img = lq_img.resize((target_w, target_h), Image.BICUBIC)

        print('\n  [1/4] Extracting DINOv3 features and fusing...')
        prompt_embeds, prompt_emb_mask, _ = model.process_with_dino(
            lq_image=lq_img,
            sr_image=sr_img,
        )

        # --- Step 2: DiT inference ---
        print(f'\n  [2/4] Running DiT inference (cfg={args.cfg})...')
        if not hasattr(generator, 'infer_with_dino_embeddings'):
            raise AttributeError(
                "Generator is missing `infer_with_dino_embeddings`. "
                "Please add this method to use DINO-based prompt injection."
            )

        res_img = generator.infer_with_dino_embeddings(
            prompt_emb=prompt_embeds,
            prompt_emb_mask=prompt_emb_mask,
            negative_prompt='',
            condition_image=lq_img,
            cfg_scale=args.cfg,
            fidelity=args.fidelity,
            tiled=True,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )

        print('\n  [3/4] Post-processing...')
        cropped = res_img.crop((0, 0, target_w, target_h))
        if args.align_method == 'adain':
            output_pil = adain_color_fix(target=cropped, source=sr_img)
            print('    Color fix: AdaIN')
        elif args.align_method == 'wavelet':
            output_pil = wavelet_color_fix(target=cropped, source=sr_img)
            print('    Color fix: wavelet')
        else:
            output_pil = cropped
            print('    Color fix: none')

        if args.resize_back:
            output_pil = output_pil.resize((w, h), Image.LANCZOS)
            print(f'    Resized back to {w}x{h}')

        print('\n  [4/4] Saving...')
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        output_pil.save(out_path)
        print(f'  Saved: {out_path}')

        if idx > 0:
            processed += 1

    if processed > 0:
        elapsed = time.perf_counter() - t_start
        print('\n' + '=' * 60)
        print('Done')
        print('=' * 60)
        print(f'  Processed images : {processed}')
        if args.sr_path:
            print(f'  SR images found  : {sr_found}')
            print(f'  SR images missing: {sr_missing}')
        print(f'  Total time       : {elapsed:.2f} s')
        print(f'  Average per image: {elapsed / processed:.2f} s')
        print(f'  Output directory : {args.output_path}')
        print('=' * 60 + '\n')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HDRFace"
    )

    parser.add_argument('--input_path', '-i', type=str, required=True,
                        help='Path to LQ image.')
    parser.add_argument('--sr_path', type=str, default=None,
                        help='Path to SR image.')
    parser.add_argument('--output_path', '-o', type=str,
                        help='Path to image output.')

    # Models
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Path to the pretrained checkpoint.')
    parser.add_argument('--dino_path', type=str,
                        help='Path to the local DINO checkpoint.')

    # Hardware
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU index to use.')

    # Inference parameters
    parser.add_argument('--scale', type=float, default=2.0,
                        help='Upscaling factor used when SR images are not provided.')
    parser.add_argument('--cfg', type=float, default=1.0,
                        help='Classifier-free guidance scale.')
    parser.add_argument('--fidelity', type=float, default=1.0,
                        help='Fidelity parameter in [0, 1].')
    parser.add_argument('--start_end', type=str, default='0,',
                        help='Index range for processing, e.g. "0,100" or "50,".')
    parser.add_argument('--align_method', type=str, default='adain',
                        choices=['adain', 'wavelet', 'none'],
                        help='Color alignment method applied after DiT inference.')

    # Image size
    parser.add_argument('--tiled_size', type=int, default=512,
                        help='Tile size used for tiled inference.')
    parser.add_argument('--first_resize_w', type=int, default=None,
                        help='Target width (overrides --scale when set together with height).')
    parser.add_argument('--first_resize_h', type=int, default=None,
                        help='Target height (overrides --scale when set together with width).')
    parser.add_argument('--resize_back', action='store_true',
                        help='Resize the final output back to the original LQ resolution.')

    # File handling
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix appended to output filenames.')
    parser.add_argument('--keep_structure', action='store_true',
                        help='Preserve the input directory structure under the output path.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing output files.')

    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())