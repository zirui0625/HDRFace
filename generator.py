"""
HDRFace generator: a Qwen-Image diffusion transformer wrapped with
dual-LoRA adapters for single-step face restoration.
"""

import json
import sys
from copy import deepcopy
from typing import Optional

import torch
from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline
from base import BaseModelForT2ILoRA  
from model_training.train import replace_linear_with_duallora 

LORA_RANK = 128
LORA_TARGET_PATTERNS = (
    'img_in',
    'img_mod.1',
    'attn.to_q',
    'attn.to_k',
    'attn.to_v',
    'to_out.0',
    'img_mlp.net.0.proj',
    'img_mlp.net.2',
)

START_TIMESTEP = 750
NUM_TIMESTEPS = 1000


class Generator(BaseModelForT2ILoRA):
    """Qwen-Image generator with dual-LoRA adapters."""
    def __init__(
        self,
        torch_dtype: torch.dtype = torch.bfloat16,
        pretrained_weights: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        learning_rate: float = 1e-4,
        use_gradient_checkpointing: bool = True,
        pretrained_ckpt_path_gen: Optional[str] = None,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        self.use_gradient_checkpointing = use_gradient_checkpointing

        model_configs = self._build_model_configs(pretrained_weights)
        tokenizer_config = ModelConfig(tokenizer_path) if tokenizer_path else None

        self.pipe = QwenImagePipeline.from_pretrained(
            torch_dtype=torch_dtype,
            device='cpu',
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
        )

        self.pipe.scheduler.set_timesteps(NUM_TIMESTEPS, training=True)
        self.pipe.freeze_except([])

        self.pipe.new_vae = deepcopy(self.pipe.vae)
        self._unfreeze_module(
            self.pipe.new_vae.encoder,
            target_cls=type(self.pipe.new_vae.encoder.conv_in),
        )

        self._add_dual_lora(self.pipe.dit, lora_rank=LORA_RANK)

        if pretrained_ckpt_path_gen:
            state_dict = torch.load(pretrained_ckpt_path_gen, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)

    @staticmethod
    def _build_model_configs(pretrained_weights: Optional[str]) -> list:
        if pretrained_weights is None:
            return []
        paths = json.loads(pretrained_weights)
        return [ModelConfig(path=p) for p in paths]

    @staticmethod
    def _unfreeze_module(model: torch.nn.Module, target_cls: type) -> None:
        for name, module in model.named_modules():
            if isinstance(module, target_cls) and 'time_conv' not in name:
                for param in module.parameters():
                    param.requires_grad = True

    @staticmethod
    def _add_dual_lora(model: torch.nn.Module, lora_rank: int) -> None:
        replace_linear_with_duallora(
            model,
            list(LORA_TARGET_PATTERNS),
            rank=lora_rank,
            alpha1=0,
            alpha2=lora_rank,
            use_fp8=True,
        )

    def configure_optimizers(self):
        trainable = filter(lambda p: p.requires_grad, self.pipe.parameters())
        return torch.optim.RMSprop(
            trainable,
            lr=self.learning_rate,
            alpha=0.9,    
            momentum=0.0, 
        )

    def forward(
        self,
        noisy_latents: torch.Tensor,
        condition_latent: torch.Tensor,
        timestep: torch.Tensor,
        prompt_emb: torch.Tensor,
        prompt_emb_mask: torch.Tensor,
    ) -> torch.Tensor:
        _, _, h, w = noisy_latents.shape
        return self.pipe.model_fn(
            self.pipe.dit,
            noisy_latents,
            condition_latent,
            timestep,
            prompt_emb,
            prompt_emb_mask,
            h * 8,
            w * 8,
            use_gradient_checkpointing=True,
        )

    def _single_step_denoise(
        self,
        inputs_shared: dict,
        posi_prompt_emb: dict,
        nega_prompt_emb: dict,
        cfg_scale: float,
        fidelity: float,
        tiled: bool,
        tile_size: int,
        tile_stride: int,
    ) -> torch.Tensor:
        lq_latents = inputs_shared['condition_latents']
        new_lq_latents = self.pipe.new_vae.encode(
            inputs_shared['condition_rgb'],
            tiled=tiled,
            tile_size=tile_size * 8,
            tile_stride=tile_stride * 8,
        )
        noise = inputs_shared['noise']

        # Fixed timestep for the denoising step.
        fixed_timestep_id = torch.randint(START_TIMESTEP, START_TIMESTEP + 1, (1,))
        fixed_timestep = self.pipe.scheduler.timesteps[fixed_timestep_id].to(device=self.device)
        one_step_sigma = self.pipe.scheduler.sigmas[fixed_timestep_id].to(
            dtype=torch.bfloat16, device=self.device,
        )

        # Optional fidelity-controlled noise injection on the condition latents.
        fidelity_timestep_id = int(START_TIMESTEP + fidelity * (NUM_TIMESTEPS - START_TIMESTEP) + 0.5)
        if fidelity_timestep_id != NUM_TIMESTEPS:
            fidelity_timestep_id = torch.randint(fidelity_timestep_id, fidelity_timestep_id + 1, (1,))
            fidelity_timestep = self.pipe.scheduler.timesteps[fidelity_timestep_id].to(device=self.device)
            lq_latents = self.pipe.scheduler.add_noise(lq_latents.detach(), noise, fidelity_timestep)

        noisy_latents = self.pipe.scheduler.add_noise(new_lq_latents.detach(), noise, fixed_timestep)
        _, _, h, w = noisy_latents.shape

        # Positive branch.
        noise_pred_posi = self.pipe.model_fn(
            self.pipe.dit, noisy_latents, lq_latents, fixed_timestep,
            **posi_prompt_emb,
            height=h * 8, width=w * 8,
            tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
        )

        # Optional negative branch for CFG.
        if cfg_scale != 1.0:
            noise_pred_nega = self.pipe.model_fn(
                self.pipe.dit, noisy_latents, lq_latents, fixed_timestep,
                **nega_prompt_emb,
                height=h * 8, width=w * 8,
                tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
            )
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        else:
            noise_pred = noise_pred_posi

        # One-step prediction.
        training_pred = noisy_latents + (0 - one_step_sigma) * noise_pred

        # Decode through the original VAE.
        image = self.pipe.vae.decode(
            training_pred,
            device=self.device,
            tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
        )
        return self.pipe.vae_output_to_image(image)

    @torch.no_grad()
    def infer(
        self,
        prompt: str,
        negative_prompt: str,
        condition_image,
        cfg_scale: float,
        fidelity: float,
        tiled: bool,
        tile_size: int,
        tile_stride: int,
    ):
        """Run inference using a text prompt."""
        inputs_posi = {'prompt': prompt}
        inputs_nega = {'negative_prompt': negative_prompt}
        inputs_shared = {
            'cfg_scale': cfg_scale,
            'input_image': None,
            'condition_image': condition_image,
            'height': condition_image.size[1],
            'width': condition_image.size[0],
            'seed': 42,
            'rand_device': self.device,
            'tiled': tiled,
            'tile_size': tile_size,
            'tile_stride': tile_stride,
        }

        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega,
            )

        posi_prompt_emb = {
            'prompt_emb': inputs_posi['prompt_emb'],
            'prompt_emb_mask': inputs_posi['prompt_emb_mask'],
        }
        nega_prompt_emb = {
            'prompt_emb': inputs_nega['prompt_emb'],
            'prompt_emb_mask': inputs_nega['prompt_emb_mask'],
        }

        return self._single_step_denoise(
            inputs_shared=inputs_shared,
            posi_prompt_emb=posi_prompt_emb,
            nega_prompt_emb=nega_prompt_emb,
            cfg_scale=cfg_scale,
            fidelity=fidelity,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )

    @torch.no_grad()
    def infer_with_dino_embeddings(
        self,
        prompt_emb: torch.Tensor,
        prompt_emb_mask: torch.Tensor,
        negative_prompt: str,
        condition_image,
        cfg_scale: float,
        fidelity: float,
        tiled: bool,
        tile_size: int,
        tile_stride: int,
    ):
       
        inputs_posi = {'prompt': ''}
        inputs_nega = {'negative_prompt': negative_prompt}
        inputs_shared = {
            'cfg_scale': cfg_scale,
            'input_image': None,
            'condition_image': condition_image,
            'height': condition_image.size[1],
            'width': condition_image.size[0],
            'seed': 42,
            'rand_device': self.device,
            'tiled': tiled,
            'tile_size': tile_size,
            'tile_stride': tile_stride,
        }

        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(
                unit, self.pipe, inputs_shared, inputs_posi, inputs_nega,
            )

        posi_prompt_emb = {
            'prompt_emb': prompt_emb.to(device=self.device, dtype=torch.bfloat16),
            'prompt_emb_mask': prompt_emb_mask.to(device=self.device),
        }
        nega_prompt_emb = {
            'prompt_emb': inputs_nega['prompt_emb'],
            'prompt_emb_mask': inputs_nega['prompt_emb_mask'],
        }

        return self._single_step_denoise(
            inputs_shared=inputs_shared,
            posi_prompt_emb=posi_prompt_emb,
            nega_prompt_emb=nega_prompt_emb,
            cfg_scale=cfg_scale,
            fidelity=fidelity,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )