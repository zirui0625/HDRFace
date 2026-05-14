# A single unified model that wraps both the generator and discriminator
from diffusers import UNet2DConditionModel
from models.autoencoder_kl import AutoencoderKL
from diffusers.utils import convert_unet_state_dict_to_peft
from transformers import CLIPTextModel
from accelerate.utils import broadcast
from peft import LoraConfig, set_peft_model_state_dict
from torch import nn
import torch 
import torch.nn.functional as F
import pyiqa
from pytorch_wavelets import DWTForward
from einops import rearrange
from safetensors.torch import load_file
import copy
from models.sd_guidance import SDGuidance
from utils.others import NoOpContext, get_prev_sample_from_noise, get_x0_from_noise, EdgeDetectionModel
from diffusers.utils.import_utils import is_xformers_available
from ArcFace.iresnet import create_arcface_embedding
from Codebook.vqvae_v3 import SDVQVAE
from torchvision import transforms
import numpy as np

def extract_and_combine_parts(img, locs):
    B, C, H, W = img.shape
    combined_img = torch.zeros_like(img)
    keys = ['loc_left_eye', 'loc_right_eye', 'loc_mouth']
    for b in range(B):
        for key in keys:
            if key in locs:
                x1, y1, x2, y2 = locs[key][b]
                x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
                part = img[b, :, y1:y2, x1:x2]
                combined_img[b, :, y1:y2, x1:x2] = part

    return combined_img

class SDUniModel(nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()

        self.args = args
        self.accelerator = accelerator
        self.guidance_model = SDGuidance(args, accelerator) 
        self.num_train_timesteps = self.guidance_model.num_train_timesteps
        self.conditioning_timestep = args.conditioning_timestep 
        self.use_fp16 = args.use_fp16 
        self.gradient_checkpointing = args.gradient_checkpointing 
        self.backward_simulation = args.backward_simulation 

        self.denoising_timestep = args.denoising_timestep 
        self.noise_scheduler = self.guidance_model.scheduler
        self.num_denoising_step = args.num_denoising_step 

        self.timestep_interval = self.denoising_timestep//self.num_denoising_step

        self.feedforward_model = UNet2DConditionModel.from_pretrained(
            args.model_id, subfolder="unet"
        ).float()
        self.unet_name = "feedforward_model"

        self.feedforward_model.requires_grad_(False)
        lora_target_modules = [
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
            "ff.net.0.proj", "ff.net.2",
            "conv1", "conv2", "conv_shortcut",
            "downsamplers.0.conv", "upsamplers.0.conv",
            "time_emb_proj",
        ]
        lora_config = LoraConfig(
            r=args.lora_rank,
            target_modules=lora_target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
        self.feedforward_model.add_adapter(lora_config)

        self.sdxl = args.sdxl 

        if self.sdxl or args.sdxl_guidance:
            self.add_time_ids = self.build_condition_input(args.resolution, accelerator)

        if args.lora_ckpt is not None:
            print(f"Initialize LoRA weights form {args.lora_ckpt}")
            self.load_lora_weights(args.lora_ckpt)


        if args.enable_xformers:
            if is_xformers_available():
                import xformers

                self.feedforward_model.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        if self.gradient_checkpointing:
            self.feedforward_model.enable_gradient_checkpointing()

        self.text_encoder = CLIPTextModel.from_pretrained(
            args.model_id, subfolder="text_encoder"
        ).to(accelerator.device)
        self.text_encoder.requires_grad_(False)

        self.alphas_cumprod = self.guidance_model.alphas_cumprod.to(accelerator.device)
        self.alphas = self.noise_scheduler.alphas.to(accelerator.device)
        self.betas = self.noise_scheduler.betas.to(accelerator.device)

        self.vae = AutoencoderKL.from_pretrained(
            args.model_id, 
            subfolder="vae"
        ).float().to(accelerator.device)
        if args.finetuned_encoder is not None:
            print(f"Load Encoder weights form {args.finetuned_encoder}")
            self.load_encoder_weights(args.finetuned_encoder)
        self.vae.requires_grad_(False) # May need train too, need add lora
        if args.train_vae:
            self.vae.requires_grad_(True)
            self.vae.train()


        if self.use_fp16 and not self.sdxl:
            self.vae.to(torch.float16)
        
        if self.args.codebook: 
            self.decoder = self.vae.decoder
            self.decoder_conv_in = copy.deepcopy(self.decoder.conv_in)
            self.decoder.requires_grad_(False)
            del self.decoder.conv_in
            if not self.args.freeze_codebook:
                self.decoder_conv_in.requires_grad_(True)

        self.network_context_manager = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.use_fp16 else NoOpContext()

        self.block_size = 8
        self.freq_quality = 85
        self.freq_mode = 'inv_gamma'
        self.freq_gamma = 1.0
        self.register_buffer("dct_mat", self._create_dct_matrix(self.block_size))
        self.register_buffer("freq_w", self._build_freq_weight(
            quality=self.freq_quality, mode=self.freq_mode, gamma=self.freq_gamma
        ))

        if self.args.codebook:
            resume_codebook = '/home/jkwang/data/DAEFR/experiments/logs/2024-10-23T01-45-13_SDvae_v3_edge/checkpoints/last.ckpt'
        
            state_dict = torch.load(resume_codebook, map_location='cpu', weights_only=False)["state_dict"]
            new_state_dict = {}

            with open("ckpt_keys.txt", "w") as f: 
                for key in state_dict:
                    if accelerator.is_main_process: 
                        print(f"Key: {key}", file=f)
                    if key.startswith('vqvae.'):
                        if accelerator.is_main_process: 
                            print(f" Original key: {key}", file=f)
                        new_key = key.replace('vqvae.', '')
                        if accelerator.is_main_process: 
                            print(f" New key: {new_key}", file=f)
                        new_state_dict[new_key] = state_dict[key]
            self.vqvae = SDVQVAE()  
            try:
                self.vqvae.load_state_dict(new_state_dict, strict=False)
                # If True: Missing key(s) in state_dict: "idx_pred_layer.0.weight", "idx_pred_layer.0.bias", "idx_pred_layer.1.weight".
                self.vqvae.train()
                self.vqvae.quantize.eval()
                self.vqvae.quantize.requires_grad_(False)
                print("Model VQVAE loaded successfully, with quantize layer in eval mode.")

                if self.args.freeze_codebook:
                    self.vqvae.requires_grad_(False)
                    print("Model VQVAE loaded successfully with all freezed, except the idx_predicter. ")
                    self.vqvae.idx_pred_layer.requires_grad_(True)
                    self.vqvae.idx_pred_layer.train()

            except Exception as e:
                print(f"Error loading state_dict: {e}")

            dim_embd, codebook_size = 512, 2048
            # (hw, b, n) -> (hw, b, codebook_size)
            

        if args.spatial_loss:
            if args.percep_weight > 0 or args.comp_lpips_loss_weight > 0:
                self.lpips_loss = pyiqa.create_metric('lpips', device=accelerator.device, as_loss=True)
            if args.high_freq_weight > 0 or args.high_freq_weight_dists > 0 or args.dists_weight > 0:
                self.dists_loss = pyiqa.create_metric('dists', device=accelerator.device, as_loss=True)
            self.edge_detection_model = EdgeDetectionModel().to(accelerator.device)
            self.edge_detection_model.requires_grad_(False)
            if args.percep_weight > 0 and args.edge_weight == 0:
                del self.edge_detection_model
            if args.arcface_spatial_loss_weight > 0:
                self.arcface_embedding = create_arcface_embedding()
                self.arcface_embedding.load_state_dict(torch.load("preset/models/ms1mv3_arcface_r50_fp16.pth"))
                self.arcface_embedding.to(accelerator.device)
                self.arcface_embedding.requires_grad_(False)
            if args.adaface_spatial_loss_weight > 0 or args.triplet_adaface_spatial_loss_weight > 0:
                import AdaFace.net as adaface_net
                self.adaface_models = {
                    'ir_50':"/data/user/jkwang/DFOSD/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt",
                }
                def load_pretrained_model(architecture='ir_50'):
                    # load model and pretrained statedict
                    assert architecture in self.adaface_models.keys()
                    model = adaface_net.build_model(architecture)
                    statedict = torch.load(self.adaface_models[architecture])['state_dict']
                    model_statedict = {key[6:]:val for key, val in statedict.items() if key.startswith('model.')}
                    model.load_state_dict(model_statedict)
                    model.eval()
                    return model
                self.adaface_model = load_pretrained_model('ir_50').to(accelerator.device)

    def to_input_pil(self, pil_rgb_image):
        np_img = np.array(pil_rgb_image)
        # print(f"{np_img.shape=}, {np_img.dtype=}, {np_img.max()=}, {np_img.min()=}") 
        # np_img.shape=(112, 112, 3), np_img.dtype=dtype('uint8'), np_img.max()=253, np_img.min()=2
        brg_img = ((np_img[:,:,::-1] / 255.) - 0.5) / 0.5
        tensor = torch.tensor([brg_img.transpose(2,0,1)]).float()
        return tensor
    def to_input_tensor(self, rgb_tensor):
        assert rgb_tensor.shape[2] == 3 # H W C, 112 112 3
        
        # print(f"{rgb_tensor.shape=}, {rgb_tensor.dtype=}, {rgb_tensor.max()=}, {rgb_tensor.min()=}") 
        # np_img.shape=(112, 112, 3), np_img.dtype=dtype('uint8'), np_img.max()=253, np_img.min()=2
        bgr_tensor = rgb_tensor.flip(-1)
        brg_img = (bgr_tensor - 0.5) / 0.5 
        # print(f"{brg_img.shape=}, {brg_img.dtype=}, {brg_img.max()=}, {brg_img.min()=}")
        # brg_img.shape=(112, 112, 3), dtype=dtype('float64'), max=1, min=-1
        tensor = brg_img.permute(2,0,1).unsqueeze(0)
        # print(f"{tensor.size()=}, {tensor.dtype=}, {tensor.max()=}, {tensor.min()=}")
        # 1,3,112,112 torch.float32 max=1, min=-1
        return tensor

    def load_lora_weights(self, lora_ckpt):
        lora_sd = load_file(lora_ckpt)
        lora_sd = {key.replace("unet.", ""): value for key, value in lora_sd.items()}
        unet_state_dict = convert_unet_state_dict_to_peft(lora_sd)
        incompatible_keys = set_peft_model_state_dict(self.feedforward_model, unet_state_dict, adapter_name="default")

    def load_encoder_weights(self, encoder_ckpt):
        encoder_sd = load_file(encoder_ckpt)
        # encoder_sd = {key:value for key, value in encoder_sd.items() if key.startswith("encoder.")}
        self.vae.load_state_dict(encoder_sd, strict=False)

    def decode_image_codebook(self, latents):
        # latents.shape: 1, 4, 64, 64
        assert self.args.codebook
        latents = 1 / self.vae.config.scaling_factor * latents
        latents_after_decoder_conv_in = self.decoder_conv_in(latents)
        # print(f"latents_after_decoder_conv_in.shape: {latents_after_decoder_conv_in.shape}")
        # print(f"latents_after_decoder_conv_in.max(): {latents_after_decoder_conv_in.max()}, latents_after_decoder_conv_in.min(): {latents_after_decoder_conv_in.min()}")
        
        hs = self.vqvae.encoder2(latents_after_decoder_conv_in)
        h = self.vqvae.quant_conv(hs["out"])
        batchsize = h.shape[0]
        assert h.shape[1] == 512 and h.shape[2] == 64 and h.shape[3] == 64
        # print(f"h.shape: {h.shape}")
        # print(f"h.max(): {h.max()}, h.min(): {h.min()}")
        h = h.flatten(2).permute(2, 0, 1) # (hw)bn, 4096,1,512
        # print(f"h.shape: {h.shape}")
        # print(f"h.max(): {h.max()}, h.min(): {h.min()}")
        logits = self.vqvae.idx_pred_layer(h) # (hw, b, codebook_size)
        logits = logits.permute(1, 0, 2)  # (hw, b, codebook_size) -> (b, hw, codebook_size)
        # print(f"logits.shape: {logits.shape}, {logits.max()=}, {logits.min()=}")
        logits.reshape(-1, logits.size(-1)) # (1, b * hw, 2048)
        # print(f"logits.shape: {logits.shape}, {logits.max()=}, {logits.min()=}")
        soft_one_hot = F.softmax(logits, dim=2)
        # print(f"soft_one_hot.shape: {soft_one_hot.shape}")
        # print(f"soft_one_hot.max(): {soft_one_hot.max()}, soft_one_hot.min(): {soft_one_hot.min()}")
        _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        # print(f"top_idx.shape: {top_idx.shape}")
        # print(f"top_idx.max(): {top_idx.max()}, top_idx.min(): {top_idx.min()}")
        # print(f'h.shape[0]: {h.shape[0]}')
        quant_feat = self.vqvae.quantize.get_codebook_entry(top_idx.reshape(-1), shape=[batchsize, 64, 64, 512])
        quant_feat = latents_after_decoder_conv_in + (quant_feat - latents_after_decoder_conv_in).detach()

        quant = self.vqvae.post_quant_conv(quant_feat)
        latents = self.vqvae.decoder2(quant)
        x = self.decoder(latents).float().clamp(-1, 1)
        return x
    
    def decode_image(self, latents):
        assert not self.args.codebook
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample.float().clamp(-1, 1)
        return image

    def decode_image_vae(self, latents):
        assert not self.args.codebook
        latents = 1 / self.vae.module.config.scaling_factor * latents
        image = self.vae.module.decode(latents).sample.float().clamp(-1, 1)
        return image

    def build_condition_input(self, resolution, accelerator):
        original_size = (resolution, resolution)
        target_size = (resolution, resolution)
        crop_top_left = (0, 0)

        add_time_ids = list(original_size + crop_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids], device=accelerator.device, dtype=torch.float32)
        return add_time_ids
    
    def _create_dct_matrix(self, N: int):
        import math
        n = torch.arange(N, dtype=torch.float32)
        k = torch.arange(N, dtype=torch.float32).unsqueeze(1)
        C = torch.cos(math.pi * (2*n + 1) * k / (2.0 * N))
        alpha = torch.sqrt(torch.tensor(2.0) / N) * torch.ones(N)
        alpha[0] = math.sqrt(1.0 / N)
        C = alpha.unsqueeze(1) * C
        return C  # (N,N)

    def _rgb2ycbcr(self, x: torch.Tensor):
        """RGB -> YCbCr, x in [B,C,H,W]"""
        r = x[:,0:1,:,:]
        g = x[:,1:2,:,:]
        b = x[:,2:3,:,:]
        y  = 0.299*r + 0.587*g + 0.114*b
        cb = -0.168736*r - 0.331264*g + 0.5*b
        cr = 0.5*r - 0.418688*g -0.081312*b
        return torch.cat([y, cb, cr], dim=1)  # (B,3,H,W)

    @torch.compile()
    def _dct(self, x: torch.Tensor):
        """
        8x8 block DCT, 返回 (B,C,Bh,Bw,bs,bs)
        """
        bs = self.block_size
        B, C, H, W = x.shape
        pad_h = (-H) % bs
        pad_w = (-W) % bs
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0,pad_w,0,pad_h), mode="reflect")

        B, C, H2, W2 = x.shape
        Bh, Bw = H2//bs, W2//bs
        blocks = x.unfold(2, bs, bs).unfold(3, bs, bs)  # (B,C,Bh,Bw,bs,bs)
        blocks = blocks.contiguous().view(-1, bs, bs)

        Cmat = self.dct_mat.to(x.device, x.dtype)
        dct_flat = torch.matmul(Cmat.unsqueeze(0), blocks)
        dct_flat = torch.matmul(dct_flat, Cmat.t().unsqueeze(0))
        return dct_flat.view(B, C, Bh, Bw, bs, bs)

    def _build_freq_weight(self, quality=85, mode='inv_gamma', gamma=1.0):
        # JPEG luminance & chrominance base tables
        lum_q = torch.tensor([
            [16,11,10,16,24,40,51,61],
            [12,12,14,19,26,58,60,55],
            [14,13,16,24,40,57,69,56],
            [14,17,22,29,51,87,80,62],
            [18,22,37,56,68,109,103,77],
            [24,35,55,64,81,104,113,92],
            [49,64,78,87,103,121,120,101],
            [72,92,95,98,112,100,103,99]], dtype=torch.float32)

        chr_q = torch.tensor([
            [17,18,24,47,99,99,99,99],
            [18,21,26,66,99,99,99,99],
            [24,26,56,99,99,99,99,99],
            [47,66,99,99,99,99,99,99],
            [99,99,99,99,99,99,99,99],
            [99,99,99,99,99,99,99,99],
            [99,99,99,99,99,99,99,99],
            [99,99,99,99,99,99,99,99]], dtype=torch.float32)

        def scale_q(base_q, quality):
            q = max(1, min(100, int(quality)))
            if q < 50:
                scale = 5000 / q
            else:
                scale = 200 - 2*q
            return torch.floor((base_q*scale + 50)/100).clamp(1,255)

        Q_y = scale_q(lum_q, quality)
        Q_cbcr = scale_q(chr_q, quality)

        def q_to_weight(Q):
            if mode == 'inv':
                w = 1.0 / Q
            elif mode == 'inv_gamma':
                w = (Q.mean()/Q) ** gamma
            else:
                raise ValueError("mode must be 'inv' or 'inv_gamma'")
            return w / w.mean()

        # Y -> luminance, Cb/Cr -> chrominance
        w_y = q_to_weight(Q_y)
        w_cb = q_to_weight(Q_cbcr)
        w_cr = q_to_weight(Q_cbcr)

        # stack成 (1,C,1,1,8,8)
        w = torch.stack([w_y, w_cb, w_cr], dim=0)
        return w.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # (1,3,1,1,8,8)

    def forward(self, lq, text_embedding, uncond_embedding, 
        visual=False, 
        real_train_dict=None,
        compute_generator_gradient=True,
        generator_turn=False,
        guidance_turn=False,
        guidance_data_dict=None,
        guidance_factor=1
    ):
        assert (generator_turn and not guidance_turn) or (guidance_turn and not generator_turn) 
        # print(f"lq.shape: {lq.shape}")
        if self.sdxl:
            text_embedding, pooled_text_embedding = text_embedding 

        if generator_turn:
            timesteps = torch.ones(lq.shape[0], device=lq.device, dtype=torch.long) * self.conditioning_timestep # 固定timestep为399

            if self.sdxl:
                add_time_ids = self.add_time_ids.repeat(lq.shape[0], 1)
                unet_added_conditions = {
                    "time_ids": add_time_ids,
                    "text_embeds": pooled_text_embedding
                }

                uncond_unet_added_conditions = {
                    "time_ids": add_time_ids,
                    "text_embeds": torch.zeros_like(pooled_text_embedding)
                }
                uncond_embedding = torch.zeros_like(text_embedding)
            else:
                unet_added_conditions = None
                uncond_unet_added_conditions = None

            if compute_generator_gradient:
                with self.network_context_manager:
                    lq_latent = self.vae.encode(lq*2-1).latent_dist.sample() * self.vae.config.scaling_factor
                    generated_noise = self.feedforward_model( 
                        lq_latent, timesteps.long(), 
                        text_embedding, added_cond_kwargs=unet_added_conditions
                    ).sample
            else:
                if self.gradient_checkpointing:
                    self.accelerator.unwrap_model(self.feedforward_model).disable_gradient_checkpointing()

                with torch.no_grad():
                    with self.network_context_manager:
                        lq_latent = self.vae.encode(lq*2-1).latent_dist.sample() * self.vae.config.scaling_factor
                        generated_noise = self.feedforward_model(
                            lq_latent, timesteps.long(), 
                            text_embedding, added_cond_kwargs=unet_added_conditions
                        ).sample

                if self.gradient_checkpointing:
                    self.accelerator.unwrap_model(self.feedforward_model).enable_gradient_checkpointing()
            # print(f"lq_latent.shape: {lq_latent.shape}")
            if self.args.use_x0:
                pred_latent = get_x0_from_noise(
                    lq_latent.double(), 
                    generated_noise.double(), self.alphas_cumprod.double(), timesteps
                ).float()
            else:
                pred_latent = get_prev_sample_from_noise(
                    lq_latent.double(), 
                    generated_noise.double(), 
                    self.alphas.double(), self.betas.double(), 
                    timesteps
                ).float()

            with torch.no_grad():
                with self.network_context_manager:
                    gt_latent = self.vae.encode(real_train_dict["gt_image"]*2-1).latent_dist.sample() * self.vae.config.scaling_factor
                real_train_dict["gt_latent"] = gt_latent.float()

            if self.use_fp16:
                if self.args.codebook:  
                    pred_image = self.decode_image_codebook(pred_latent.half()) * 0.5 + 0.5
                else:
                    pred_image = self.decode_image(pred_latent.half()) * 0.5 + 0.5
            else:
                if self.args.codebook:
                    pred_image = self.decode_image_codebook(pred_latent) * 0.5 + 0.5
                else:
                    pred_image = self.decode_image(pred_latent) * 0.5 + 0.5

            if compute_generator_gradient:
                if self.args.sdxl_guidance:
                    guidance_text_embedding, guidance_pooled_text_embedding = real_train_dict["guidance_text_embedding"]
                    add_time_ids = self.add_time_ids.repeat(lq.shape[0], 1)
                    unet_added_conditions = {
                        "time_ids": add_time_ids,
                        "text_embeds": guidance_pooled_text_embedding
                    }

                    uncond_unet_added_conditions = {
                        "time_ids": add_time_ids,
                        "text_embeds": torch.zeros_like(guidance_pooled_text_embedding)
                    }
                    uncond_embedding = torch.zeros_like(guidance_text_embedding)

                    generator_data_dict = {
                        "lq_image": lq,
                        "pred_latent": pred_latent,
                        "pred_image": pred_image,
                        "text_embedding": guidance_text_embedding,
                        "uncond_embedding": uncond_embedding,
                        "real_train_dict": real_train_dict,
                        "unet_added_conditions": unet_added_conditions,
                        "uncond_unet_added_conditions": uncond_unet_added_conditions
                    } 
                else:
                    generator_data_dict = {
                        "lq_image": lq,
                        "pred_latent": pred_latent,
                        "pred_image": pred_image,
                        "text_embedding": text_embedding,
                        "uncond_embedding": uncond_embedding,
                        "real_train_dict": real_train_dict,
                        "unet_added_conditions": unet_added_conditions,
                        "uncond_unet_added_conditions": uncond_unet_added_conditions
                    } 

                # avoid any side effects of gradient accumulation
                self.guidance_model.requires_grad_(False)
                loss_dict, log_dict = self.guidance_model( 
                    generator_turn=True,
                    guidance_turn=False,
                    generator_data_dict=generator_data_dict,
                    guidance_factor=guidance_factor
                )
                self.guidance_model.requires_grad_(True)

                if self.args.spatial_loss:                    
                    with self.network_context_manager:
                        spatial_loss = 0
                        mse_loss = self.args.mse_weight * F.mse_loss(pred_image, real_train_dict["gt_image"])
                        spatial_loss += mse_loss
                        loss_dict["loss_mse"] = mse_loss
                        if self.args.terminal_print_loss:
                            print(f"{loss_dict['loss_mse']=}")
                        if self.args.percep_weight > 0:
                            percep_loss = self.args.percep_weight * self.lpips_loss(pred_image, real_train_dict["gt_image"])
                            spatial_loss += percep_loss
                            loss_dict["loss_lpips"] = percep_loss
                            if self.args.edge_weight > 0:
                                edge_loss = self.lpips_loss(
                                    self.edge_detection_model(pred_image), 
                                    self.edge_detection_model(real_train_dict["gt_image"])
                                )
                                loss_dict["loss_edge"] = edge_loss
                                spatial_loss += self.args.edge_weight * edge_loss
                            if self.args.terminal_print_loss:
                                print(f'{loss_dict["loss_lpips"]=}, {self.args.edge_weight * edge_loss=}, ')
                        if self.args.dists_weight > 0:
                            dists_loss = self.args.dists_weight * self.dists_loss(pred_image, real_train_dict["gt_image"])
                            spatial_loss += dists_loss
                            loss_dict["loss_dists"] = dists_loss
                            if self.args.edge_weight > 0:
                                edge_loss = self.dists_loss(
                                    self.edge_detection_model(pred_image), 
                                    self.edge_detection_model(real_train_dict["gt_image"])
                                )
                                loss_dict["loss_edge"] = edge_loss
                                spatial_loss += self.args.edge_weight * edge_loss
                            if self.args.terminal_print_loss:
                                print(f'{loss_dict["loss_dists"]=}, {self.args.edge_weight * edge_loss=}, ')
                        if self.args.high_freq_weight > 0:
                            dists_loss = self.dists_loss(pred_image, real_train_dict["gt_image"])  # weight ???
                            loss_dict["loss_dists"] = dists_loss
                            edge_loss = F.mse_loss(
                                self.edge_detection_model(pred_image), 
                                self.edge_detection_model(real_train_dict["gt_image"])
                            )
                            loss_dict["loss_edge"] = edge_loss
                            xfm = DWTForward(J=3, mode='zero', wave='db3').to(self.accelerator.device)
                            Yl, Yh = xfm(real_train_dict["gt_image"])
                            Yl_hat, Yh_hat = xfm(pred_image)
                            dwt_loss = sum([F.mse_loss(rearrange(h, 'b c n h w -> (b n) c h w'), 
                                                    rearrange(h_hat, 'b c n h w -> (b n) c h w'), 
                                                    ) 
                                                for h, h_hat in zip (Yh, Yh_hat)])
                            
                            loss_dict["loss_dwt"] = dwt_loss
                            if self.args.terminal_print_loss:
                                print(f'{loss_dict["loss_dists"]=}, {loss_dict["loss_edge"]=}, {loss_dict["loss_dwt"]=}')
                            spatial_loss = spatial_loss + dists_loss + edge_loss + dwt_loss
                        if self.args.high_freq_weight_dists > 0:
                            dists_loss = self.dists_loss(pred_image, real_train_dict["gt_image"])
                            loss_dict["loss_dists"] = dists_loss
                            edge_loss = self.dists_loss(
                                self.edge_detection_model(pred_image), 
                                self.edge_detection_model(real_train_dict["gt_image"])
                            )
                            loss_dict["loss_edge"] = edge_loss
                            if self.args.terminal_print_loss:
                                print(f'{loss_dict["loss_dists"]=}, {loss_dict["loss_edge"]=}')
                            spatial_loss = spatial_loss + dists_loss + edge_loss
                        
                        if self.args.arcface_spatial_loss_weight > 0:
                            self.arcface_embedding.eval()
                            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                            # print(f"{pred_image.shape=}, {pred_image.max()=}, {pred_image.min()=}")
                            # print(f"{real_train_dict['gt_image'].shape=}, {real_train_dict['gt_image'].max()=}, {real_train_dict['gt_image'].min()=}")
                            # F.interpolate: input mini-batch x channels x [optional depth] x [optional height] x width.
                            emd_pred = self.arcface_embedding(F.interpolate(pred_image, (112,112), mode='bilinear', antialias=True))
                            emd_gt = self.arcface_embedding(F.interpolate(real_train_dict["gt_image"], (112,112), mode='bilinear', antialias=True))
                            arcface_loss = 1 - cos(emd_gt, emd_pred)* self.args.arcface_spatial_loss_weight
                            arcface_loss = arcface_loss.mean()
                            loss_dict["loss_arcface"] = arcface_loss * self.args.arcface_spatial_loss_weight
                            spatial_loss += loss_dict["loss_arcface"]
                            if self.args.terminal_print_loss:
                                print(f'{loss_dict["loss_arcface"]=}')
                        
                        if self.args.adaface_spatial_loss_weight > 0:
                            self.adaface_model.eval()

                            # print(f"{pred_image.shape=}, {pred_image.max()=}, {pred_image.min()=}")  # B C H W max=1, min=0
                            # print(f"{real_train_dict['gt_image'].shape=}, {real_train_dict['gt_image'].max()=}, {real_train_dict['gt_image'].min()=}")

                            pred_identify = F.interpolate(pred_image, (112,112), mode='bilinear', antialias=True) # 1, 3, 112, 112  max=1, min=0
                            gt_identify = F.interpolate(real_train_dict["gt_image"], (112,112), mode='bilinear', antialias=True)
                            # print(f"{pred_identify.shape=}, {pred_identify.max()=}, {pred_identify.min()=}")
                            # print(f"{gt_identify.shape=}, {gt_identify.max()=}, {gt_identify.min()=}")
                                
                            adaface_loss = 0
                            for pic_num in range(pred_image.shape[0]):
                                # pred_pil = transforms.ToPILImage()(pred_identify[pic_num].cpu())
                                # gt_pil = transforms.ToPILImage()(gt_identify[pic_num].cpu())

                                pred_tensor = pred_identify[pic_num].permute(1,2,0)
                                gt_tensor = gt_identify[pic_num].permute(1,2,0)

                                # import os
                                # if not os.path.exists('results/test/gt0.png') and not os.path.exists('results/test/pred0.png'):
                                #     gt_pil.save('results/test/gt0.png')
                                #     pred_pil.save('results/test/pred0.png')

                                # gt_bgr_pil_tensor = self.to_input_pil(gt_pil)
                                # pred_bgr_pil_tensor = self.to_input_pil(pred_pil)

                                gt_bgr_tensor_tensor = self.to_input_tensor(gt_tensor).to(self.accelerator.device)
                                pred_bgr_tensor_tensor = self.to_input_tensor(pred_tensor).to(self.accelerator.device)

                                # feature_gt_pil, _ = adaface_model(gt_bgr_pil_tensor)
                                # feature_pred_pil, _ = adaface_model(pred_bgr_pil_tensor)

                                feature_gt_tensor, _ = self.adaface_model(gt_bgr_tensor_tensor)
                                feature_pred_tensor, _ = self.adaface_model(pred_bgr_tensor_tensor)

                                # similarity_score_0 = feature_gt_pil @ feature_pred_pil.T
                                similarity_score_1 = feature_gt_tensor @ feature_pred_tensor.T
                                # print(similarity_score_0, similarity_score_1)
                                
                                adaface_loss += 1 - similarity_score_1[0,0] # 1e4
                            adaface_loss = adaface_loss / pred_image.shape[0]

                            loss_dict["loss_adaface"] = adaface_loss * self.args.adaface_spatial_loss_weight
                            spatial_loss += loss_dict["loss_adaface"]
                            if self.args.terminal_print_loss:
                                print(f'{loss_dict["loss_adaface"]=}')

                        if self.args.triplet_adaface_spatial_loss_weight > 0:
                            self.adaface_model.eval()
                            pred_identify = F.interpolate(pred_image, (112,112), mode='bilinear', antialias=True) # 1, 3, 112, 112  max=1, min=0
                            gt_identify = F.interpolate(real_train_dict["gt_image"], (112,112), mode='bilinear', antialias=True)
                            other_identify = F.interpolate(real_train_dict["gt_image2"], (112,112), mode='bilinear', antialias=True)
                                
                            adaface_loss = 0
                            for pic_num in range(pred_image.shape[0]):
                                pred_tensor = pred_identify[pic_num].permute(1,2,0)
                                gt_tensor = gt_identify[pic_num].permute(1,2,0)
                                other_tensor = other_identify[pic_num].permute(1,2,0)

                                gt_bgr_tensor_tensor = self.to_input_tensor(gt_tensor).to(self.accelerator.device)
                                pred_bgr_tensor_tensor = self.to_input_tensor(pred_tensor).to(self.accelerator.device)
                                other_bgr_tensor_tensor = self.to_input_tensor(other_tensor).to(self.accelerator.device)

                                feature_gt_tensor, _ = self.adaface_model(gt_bgr_tensor_tensor)
                                feature_pred_tensor, _ = self.adaface_model(pred_bgr_tensor_tensor)
                                feature_other_tensor, _ = self.adaface_model(other_bgr_tensor_tensor)

                                if self.args.triplet_adaface_spaital_minus:
                                    triplet_loss = (
                                        nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))
                                    )
                                    loss = triplet_loss(feature_gt_tensor, feature_pred_tensor, feature_other_tensor)
                                else:
                                    similarity_score_up = feature_gt_tensor @ feature_pred_tensor.T
                                    similarity_score_down = feature_gt_tensor @ feature_other_tensor.T
                                    
                                    similarity_score_up = 1 - similarity_score_up[0,0]
                                    similarity_score_down = 1 - similarity_score_down[0,0]
                                    loss = similarity_score_up / similarity_score_down
                                
                                adaface_loss += loss
                            adaface_loss = adaface_loss / pred_image.shape[0]

                            loss_dict["loss_triplet_adaface"] = adaface_loss * self.args.triplet_adaface_spatial_loss_weight
                            spatial_loss += loss_dict["loss_triplet_adaface"]
                            if self.args.terminal_print_loss:
                                print(f'{loss_dict["loss_triplet_adaface"]=}')
                        if self.args.component_loss and (self.args.comp_mse_loss_weight>0 or self.args.comp_lpips_loss_weight):
                            combined_pred = extract_and_combine_parts(pred_image, real_train_dict)
                            combined_real = extract_and_combine_parts(real_train_dict["gt_image"], real_train_dict)
                            if self.args.comp_mse_loss_weight > 0:
                                mse_loss = F.mse_loss(combined_pred, combined_real)
                                loss_dict["loss_comp_mse"] = mse_loss * self.args.comp_mse_loss_weight
                                spatial_loss += loss_dict["loss_comp_mse"]
                                if self.args.terminal_print_loss:
                                    print(f'{loss_dict["loss_comp_mse"]=}')
                            if self.args.comp_lpips_loss_weight > 0:
                                lpips_loss = self.lpips_loss(combined_pred, combined_real)
                                loss_dict["loss_comp_lpips"] = lpips_loss * self.args.comp_lpips_loss_weight
                                spatial_loss += loss_dict["loss_comp_lpips"]
                                if self.args.terminal_print_loss:
                                    print(f'{loss_dict["loss_comp_lpips"]=}')
                        
                        # if self.args.fm_freq_loss_weight > 0:
                        #     pred_image_freq = self._dct(self._rgb2ycbcr(pred_image))
                        #     gt_image_freq = self._dct(self._rgb2ycbcr(real_train_dict["gt_image"]))
                        #     fm_freq_loss = (self.freq_w.to(pred_image_freq.device)*((pred_image_freq - gt_image_freq)**2)).mean()
                        #     loss_dict["fm_freq_loss"] = fm_freq_loss * self.args.fm_freq_loss_weight
                        #     spatial_loss += loss_dict["fm_freq_loss"]
                        #     if self.args.terminal_print_loss:
                        #         print(f'{loss_dict["fm_freq_loss"]=}')

                        loss_dict["loss_spatial"] = spatial_loss

            else:
                loss_dict = {}
                log_dict = {} 

            # print(f'real_train_dict["gt_latent"].shape: {real_train_dict["gt_latent"].shape}')
            if visual:
                decode_key = [
                    "dmtrain_pred_real_image", "dmtrain_pred_fake_image"
                ]

                with torch.no_grad():
                    if compute_generator_gradient and not self.args.gan_alone:
                        for key in decode_key:
                            if self.use_fp16:
                                if self.args.codebook:
                                    log_dict[key+"_decoded"] = self.decode_image_codebook(real_train_dict[key].half()) * 0.5 + 0.5
                                else:
                                    log_dict[key+"_decoded"] = self.decode_image(log_dict[key].detach().half()) * 0.5 + 0.5
                            else:
                                if self.args.codebook:
                                    log_dict[key+"_decoded"] = self.decode_image_codebook(real_train_dict[key]) * 0.5 + 0.5
                                else:
                                    log_dict[key+"_decoded"] = self.decode_image(log_dict[key].detach()) * 0.5 + 0.5

                    if self.use_fp16:
                        if self.args.codebook:
                            log_dict["pred_image"] = self.decode_image_codebook(pred_latent.half()) * 0.5 + 0.5
                            log_dict["decoded_gt_image"] = self.decode_image_codebook(real_train_dict["gt_latent"].half()) * 0.5 + 0.5    
                        else:
                            log_dict["pred_image"] = self.decode_image(pred_latent.half()) * 0.5 + 0.5
                            log_dict["decoded_gt_image"] = self.decode_image(real_train_dict["gt_latent"].half()) * 0.5 + 0.5
                    else:
                        if self.args.codebook:
                            log_dict["pred_image"] = self.decode_image_codebook(pred_latent) * 0.5 + 0.5
                            log_dict["decoded_gt_image"] = self.decode_image_codebook(real_train_dict["gt_latent"]) * 0.5 + 0.5
                        else:
                            log_dict["pred_image"] = self.decode_image(pred_latent) * 0.5 + 0.5
                            log_dict["decoded_gt_image"] = self.decode_image(real_train_dict["gt_latent"]) * 0.5 + 0.5

            if self.args.sdxl_guidance:
                guidance_text_embedding, guidance_pooled_text_embedding = real_train_dict["guidance_text_embedding"]
                add_time_ids = self.add_time_ids.repeat(lq.shape[0], 1)
                unet_added_conditions = {
                    "time_ids": add_time_ids,
                    "text_embeds": guidance_pooled_text_embedding
                }

                uncond_unet_added_conditions = {
                    "time_ids": add_time_ids,
                    "text_embeds": torch.zeros_like(guidance_pooled_text_embedding)
                }
                uncond_embedding = torch.zeros_like(guidance_text_embedding)
                log_dict["guidance_data_dict"] = {  # to guidance as guidance_data_dict
                    "lq_image": lq.detach(),
                    "pred_latent": pred_latent.detach(),
                    "pred_image": pred_image.detach(),
                    "text_embedding": guidance_text_embedding.detach(),
                    "uncond_embedding": uncond_embedding.detach(),
                    "real_train_dict": real_train_dict,
                    "unet_added_conditions": unet_added_conditions,
                    "uncond_unet_added_conditions": uncond_unet_added_conditions
                }
            else:
                log_dict["guidance_data_dict"] = {  # to guidance as guidance_data_dict
                    "lq_image": lq.detach(),
                    "pred_latent": pred_latent.detach(),
                    "pred_image": pred_image.detach(),
                    "text_embedding": text_embedding.detach(),
                    "uncond_embedding": uncond_embedding.detach(),
                    "real_train_dict": real_train_dict,
                    "unet_added_conditions": unet_added_conditions,
                    "uncond_unet_added_conditions": uncond_unet_added_conditions
                }

            log_dict['denoising_timestep'] = timesteps
                
        elif guidance_turn:
            assert guidance_data_dict is not None 
            loss_dict, log_dict = self.guidance_model(
                generator_turn=False,
                guidance_turn=True,
                guidance_data_dict=guidance_data_dict,
                guidance_factor=guidance_factor
            )    
        return loss_dict, log_dict

    def forward_vae(self, input, real_train_dict):
        assert self.args.train_vae,"Need to train vae only"

        with self.network_context_manager:
            input_latent = self.vae.module.encode(input * 2 - 1).latent_dist.sample() * self.vae.module.config.scaling_factor
        # print(f"lq_latent.shape: {lq_latent.shape}")

        if self.use_fp16:
            if self.args.codebook:
                pred_image = self.decode_image_codebook(input_latent.half()) * 0.5 + 0.5
            else:
                pred_image = self.decode_image_vae(input_latent.half()) * 0.5 + 0.5
        else:
            if self.args.codebook:
                pred_image = self.decode_image_codebook(input_latent) * 0.5 + 0.5
            else:
                pred_image = self.decode_image_vae(input_latent) * 0.5 + 0.5

        if self.gradient_checkpointing:
            self.accelerator.unwrap_model(self.vae).enable_gradient_checkpointing()

            loss_dict, log_dict = {},{}

            if self.args.spatial_loss:
                with self.network_context_manager:
                    spatial_loss = 0
                    mse_loss = self.args.mse_weight * F.mse_loss(pred_image, input)
                    spatial_loss += mse_loss
                    loss_dict["loss_mse"] = mse_loss
                    if self.args.terminal_print_loss:
                        print(f"{loss_dict['loss_mse']=}")
                    if self.args.percep_weight > 0:
                        percep_loss = self.args.percep_weight * self.lpips_loss(pred_image,
                                                                                input)
                        spatial_loss += percep_loss
                        loss_dict["loss_lpips"] = percep_loss
                        if self.args.edge_weight > 0:
                            edge_loss = self.lpips_loss(
                                self.edge_detection_model(pred_image),
                                self.edge_detection_model(input)
                            )
                            loss_dict["loss_edge"] = edge_loss
                            spatial_loss += self.args.edge_weight * edge_loss
                        if self.args.terminal_print_loss:
                            print(f'{loss_dict["loss_lpips"]=}, {self.args.edge_weight * edge_loss=}, ')
                    if self.args.dists_weight > 0:
                        dists_loss = self.args.dists_weight * self.dists_loss(pred_image,
                                                                              input)
                        spatial_loss += dists_loss
                        loss_dict["loss_dists"] = dists_loss
                        if self.args.edge_weight > 0:
                            edge_loss = self.dists_loss(
                                self.edge_detection_model(pred_image),
                                self.edge_detection_model(input)
                            )
                            loss_dict["loss_edge"] = edge_loss
                            spatial_loss += self.args.edge_weight * edge_loss
                        if self.args.terminal_print_loss:
                            print(f'{loss_dict["loss_dists"]=}, {self.args.edge_weight * edge_loss=}, ')
                    if self.args.high_freq_weight > 0:
                        dists_loss = self.dists_loss(pred_image, input)  # weight ???
                        loss_dict["loss_dists"] = dists_loss
                        edge_loss = F.mse_loss(
                            self.edge_detection_model(pred_image),
                            self.edge_detection_model(input)
                        )
                        loss_dict["loss_edge"] = edge_loss
                        xfm = DWTForward(J=3, mode='zero', wave='db3').to(self.accelerator.device)
                        Yl, Yh = xfm(input)
                        Yl_hat, Yh_hat = xfm(pred_image)
                        dwt_loss = sum([F.mse_loss(rearrange(h, 'b c n h w -> (b n) c h w'),
                                                   rearrange(h_hat, 'b c n h w -> (b n) c h w'),
                                                   )
                                        for h, h_hat in zip(Yh, Yh_hat)])

                        loss_dict["loss_dwt"] = dwt_loss
                        if self.args.terminal_print_loss:
                            print(
                                f'{loss_dict["loss_dists"]=}, {loss_dict["loss_edge"]=}, {loss_dict["loss_dwt"]=}')
                        spatial_loss = spatial_loss + dists_loss + edge_loss + dwt_loss
                    if self.args.high_freq_weight_dists > 0:
                        dists_loss = self.dists_loss(pred_image, input)
                        loss_dict["loss_dists"] = dists_loss
                        edge_loss = self.dists_loss(
                            self.edge_detection_model(pred_image),
                            self.edge_detection_model(input)
                        )
                        loss_dict["loss_edge"] = edge_loss
                        if self.args.terminal_print_loss:
                            print(f'{loss_dict["loss_dists"]=}, {loss_dict["loss_edge"]=}')
                        spatial_loss = spatial_loss + dists_loss + edge_loss

                    if self.args.arcface_spatial_loss_weight > 0:
                        self.arcface_embedding.eval()
                        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
                        # print(f"{pred_image.shape=}, {pred_image.max()=}, {pred_image.min()=}")
                        # print(f"{real_train_dict['gt_image'].shape=}, {real_train_dict['gt_image'].max()=}, {real_train_dict['gt_image'].min()=}")
                        # F.interpolate: input mini-batch x channels x [optional depth] x [optional height] x width.
                        emd_pred = self.arcface_embedding(
                            F.interpolate(pred_image, (112, 112), mode='bilinear', antialias=True))
                        emd_gt = self.arcface_embedding(
                            F.interpolate(input, (112, 112), mode='bilinear', antialias=True))
                        arcface_loss = 1 - cos(emd_gt, emd_pred) * self.args.arcface_spatial_loss_weight
                        arcface_loss = arcface_loss.mean()
                        loss_dict["loss_arcface"] = arcface_loss * self.args.arcface_spatial_loss_weight
                        spatial_loss += loss_dict["loss_arcface"]
                        if self.args.terminal_print_loss:
                            print(f'{loss_dict["loss_arcface"]=}')

                    if self.args.adaface_spatial_loss_weight > 0:
                        self.adaface_model.eval()

                        # print(f"{pred_image.shape=}, {pred_image.max()=}, {pred_image.min()=}")  # B C H W max=1, min=0
                        # print(f"{real_train_dict['gt_image'].shape=}, {real_train_dict['gt_image'].max()=}, {real_train_dict['gt_image'].min()=}")

                        pred_identify = F.interpolate(pred_image, (112, 112), mode='bilinear',
                                                      antialias=True)  # 1, 3, 112, 112  max=1, min=0
                        gt_identify = F.interpolate(input, (112, 112), mode='bilinear',
                                                    antialias=True)
                        # print(f"{pred_identify.shape=}, {pred_identify.max()=}, {pred_identify.min()=}")
                        # print(f"{gt_identify.shape=}, {gt_identify.max()=}, {gt_identify.min()=}")

                        adaface_loss = 0
                        for pic_num in range(pred_image.shape[0]):
                            # pred_pil = transforms.ToPILImage()(pred_identify[pic_num].cpu())
                            # gt_pil = transforms.ToPILImage()(gt_identify[pic_num].cpu())

                            pred_tensor = pred_identify[pic_num].permute(1, 2, 0)
                            gt_tensor = gt_identify[pic_num].permute(1, 2, 0)

                            # import os
                            # if not os.path.exists('results/test/gt0.png') and not os.path.exists('results/test/pred0.png'):
                            #     gt_pil.save('results/test/gt0.png')
                            #     pred_pil.save('results/test/pred0.png')

                            # gt_bgr_pil_tensor = self.to_input_pil(gt_pil)
                            # pred_bgr_pil_tensor = self.to_input_pil(pred_pil)

                            gt_bgr_tensor_tensor = self.to_input_tensor(gt_tensor).to(self.accelerator.device)
                            pred_bgr_tensor_tensor = self.to_input_tensor(pred_tensor).to(self.accelerator.device)

                            # feature_gt_pil, _ = adaface_model(gt_bgr_pil_tensor)
                            # feature_pred_pil, _ = adaface_model(pred_bgr_pil_tensor)

                            feature_gt_tensor, _ = self.adaface_model(gt_bgr_tensor_tensor)
                            feature_pred_tensor, _ = self.adaface_model(pred_bgr_tensor_tensor)

                            # similarity_score_0 = feature_gt_pil @ feature_pred_pil.T
                            similarity_score_1 = feature_gt_tensor @ feature_pred_tensor.T
                            # print(similarity_score_0, similarity_score_1)

                            adaface_loss += 1 - similarity_score_1[0, 0]  # 1e4
                        adaface_loss = adaface_loss / pred_image.shape[0]

                        loss_dict["loss_adaface"] = adaface_loss * self.args.adaface_spatial_loss_weight
                        spatial_loss += loss_dict["loss_adaface"]
                        if self.args.terminal_print_loss:
                            print(f'{loss_dict["loss_adaface"]=}')

                    if self.args.triplet_adaface_spatial_loss_weight > 0:
                        self.adaface_model.eval()
                        pred_identify = F.interpolate(pred_image, (112, 112), mode='bilinear',
                                                      antialias=True)  # 1, 3, 112, 112  max=1, min=0
                        gt_identify = F.interpolate(input, (112, 112), mode='bilinear',
                                                    antialias=True)
                        other_identify = F.interpolate(real_train_dict["gt_image2"], (112, 112), mode='bilinear',
                                                       antialias=True)

                        adaface_loss = 0
                        for pic_num in range(pred_image.shape[0]):
                            pred_tensor = pred_identify[pic_num].permute(1, 2, 0)
                            gt_tensor = gt_identify[pic_num].permute(1, 2, 0)
                            other_tensor = other_identify[pic_num].permute(1, 2, 0)

                            gt_bgr_tensor_tensor = self.to_input_tensor(gt_tensor).to(self.accelerator.device)
                            pred_bgr_tensor_tensor = self.to_input_tensor(pred_tensor).to(self.accelerator.device)
                            other_bgr_tensor_tensor = self.to_input_tensor(other_tensor).to(self.accelerator.device)

                            feature_gt_tensor, _ = self.adaface_model(gt_bgr_tensor_tensor)
                            feature_pred_tensor, _ = self.adaface_model(pred_bgr_tensor_tensor)
                            feature_other_tensor, _ = self.adaface_model(other_bgr_tensor_tensor)

                            if self.args.triplet_adaface_spaital_minus:
                                triplet_loss = (
                                    nn.TripletMarginWithDistanceLoss(
                                        distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))
                                )
                                loss = triplet_loss(feature_gt_tensor, feature_pred_tensor, feature_other_tensor)
                            else:
                                similarity_score_up = feature_gt_tensor @ feature_pred_tensor.T
                                similarity_score_down = feature_gt_tensor @ feature_other_tensor.T

                                similarity_score_up = 1 - similarity_score_up[0, 0]
                                similarity_score_down = 1 - similarity_score_down[0, 0]
                                loss = similarity_score_up / similarity_score_down

                            adaface_loss += loss
                        adaface_loss = adaface_loss / pred_image.shape[0]

                        loss_dict[
                            "loss_triplet_adaface"] = adaface_loss * self.args.triplet_adaface_spatial_loss_weight
                        spatial_loss += loss_dict["loss_triplet_adaface"]
                        if self.args.terminal_print_loss:
                            print(f'{loss_dict["loss_triplet_adaface"]=}')
                    if self.args.component_loss and (
                            self.args.comp_mse_loss_weight > 0 or self.args.comp_lpips_loss_weight):
                        combined_pred = extract_and_combine_parts(pred_image, real_train_dict)
                        combined_real = extract_and_combine_parts(input, real_train_dict)
                        if self.args.comp_mse_loss_weight > 0:
                            mse_loss = F.mse_loss(combined_pred, combined_real)
                            loss_dict["loss_comp_mse"] = mse_loss * self.args.comp_mse_loss_weight
                            spatial_loss += loss_dict["loss_comp_mse"]
                            if self.args.terminal_print_loss:
                                print(f'{loss_dict["loss_comp_mse"]=}')
                        if self.args.comp_lpips_loss_weight > 0:
                            lpips_loss = self.lpips_loss(combined_pred, combined_real)
                            loss_dict["loss_comp_lpips"] = lpips_loss * self.args.comp_lpips_loss_weight
                            spatial_loss += loss_dict["loss_comp_lpips"]
                            if self.args.terminal_print_loss:
                                print(f'{loss_dict["loss_comp_lpips"]=}')

                    loss_dict["loss_spatial"] = spatial_loss

        else:
            loss_dict = {}
            log_dict = {}

            # print(f'real_train_dict["gt_latent"].shape: {real_train_dict["gt_latent"].shape}')

        log_dict["pred_image"] = pred_image
        log_dict["decoded_gt_image"] = input

        return loss_dict, log_dict


            