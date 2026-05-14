from utils.others import get_x0_from_noise, DummyNetwork, NoOpContext
from diffusers import UNet2DConditionModel, DDIMScheduler
from models.sd_unet_forward import classify_forward
import torch.nn.functional as F
import torch.nn as nn
from peft import LoraConfig
import torch
import types 
from diffusers.utils.import_utils import is_xformers_available
import AdaFace.net as adaface_net
from models.DCGAN import DCGAN_D, CategoricalLoss

def predict_noise(unet, noisy_latents, text_embeddings, uncond_embedding, timesteps, 
    guidance_scale=1.0, unet_added_conditions=None, uncond_unet_added_conditions=None
):
    CFG_GUIDANCE = guidance_scale > 1

    if CFG_GUIDANCE: # using cfg_guidance
        model_input = torch.cat([noisy_latents] * 2) 
        embeddings = torch.cat([uncond_embedding, text_embeddings]) 
        timesteps = torch.cat([timesteps] * 2) 

        if unet_added_conditions is not None:
            assert uncond_unet_added_conditions is not None 
            condition_input = {}
            for key in unet_added_conditions.keys():
                condition_input[key] = torch.cat(
                    [uncond_unet_added_conditions[key], unet_added_conditions[key]] # should be uncond, cond, check the order  
                )
        else:
            condition_input = None 

        noise_pred = unet(model_input, timesteps, embeddings, added_cond_kwargs=condition_input).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond) 
    else:
        model_input = noisy_latents 
        embeddings = text_embeddings
        timesteps = timesteps    
        noise_pred = unet(model_input, timesteps, embeddings, added_cond_kwargs=unet_added_conditions).sample

    return noise_pred  

class SDGuidance(nn.Module):
    def __init__(self, args, accelerator):
        super().__init__()
        self.args = args 

        if args.sdxl_path is None:
            args.sdxl_path = args.model_id

        self.real_unet = UNet2DConditionModel.from_pretrained(
            args.sdxl_path,
            subfolder="unet"
        ).float()

        self.real_unet.requires_grad_(False)
        self.gan_alone = args.gan_alone 

        self.fake_unet = UNet2DConditionModel.from_pretrained(
            args.sdxl_path,
            subfolder="unet"
        ).float()

        self.fake_unet.requires_grad_(False)
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
        self.fake_unet.add_adapter(lora_config)

        # somehow FSDP requires at least one network with dense parameters (models from diffuser are lazy initialized so their parameters are empty in fsdp mode)
        self.dummy_network = DummyNetwork() 
        self.dummy_network.requires_grad_(False)

        if args.use_fp16:
            self.real_unet = self.real_unet.to(torch.bfloat16)

        if self.gan_alone:
            del self.real_unet
            del self.fake_unet.up_blocks
            del self.fake_unet.conv_out
            # self.fake_unet.up_blocks.requires_grad_(False)
            # self.fake_unet.conv_out.requires_grad_(False)

        if args.enable_xformers:
            if is_xformers_available():
                import xformers

                self.real_unet.enable_xformers_memory_efficient_attention()
                self.fake_unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        self.scheduler = DDIMScheduler.from_pretrained(
            args.model_id,
            subfolder="scheduler"
        )

        alphas_cumprod = self.scheduler.alphas_cumprod
        self.register_buffer(
            "alphas_cumprod",
            alphas_cumprod
        )

        self.num_train_timesteps = args.num_train_timesteps 
        self.min_step = int(args.min_step_percent * self.scheduler.num_train_timesteps)
        self.max_step = int(args.max_step_percent * self.scheduler.num_train_timesteps)

        self.real_guidance_scale = args.real_guidance_scale 
        self.fake_guidance_scale = args.fake_guidance_scale

        assert self.fake_guidance_scale == 1, "no guidance for fake"

        self.use_fp16 = args.use_fp16

        self.accelerator = accelerator

        self.fake_unet.forward = types.MethodType(
            classify_forward, self.fake_unet
        )

        if accelerator.is_local_main_process:
            print("Note that we randomly initialized a bunch of parameters. FSDP mode 4 hybrid_shard will have non-synced parameters across nodes which would lead to training problems. The current solution is to save the checkpoint 0 and resume")

        if args.sdxl:
            self.cls_pred_branch = nn.Sequential(
                nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=2, padding=1), # 32x32 -> 16x16 
                nn.GroupNorm(num_groups=32, num_channels=1280),
                nn.SiLU(),
                nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=2, padding=1), # 16x16 -> 8x8 
                nn.GroupNorm(num_groups=32, num_channels=1280),
                nn.SiLU(),
                nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=2, padding=1), # 8x8 -> 4x4
                nn.GroupNorm(num_groups=32, num_channels=1280),
                nn.SiLU(),
                nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=4, padding=0), # 4x4 -> 1x1
                nn.GroupNorm(num_groups=32, num_channels=1280),
                nn.SiLU(),
                nn.Conv2d(kernel_size=1, in_channels=1280, out_channels=1, stride=1, padding=0), # 1x1 -> 1x1
            )
        elif args.patch_gan:
            if not args.sdxl_guidance:
                self.cls_pred_branch = nn.Sequential(
                    nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=2560, stride=1, padding=1), # 8x8 -> 8x8 
                    nn.GroupNorm(num_groups=32, num_channels=2560),
                    nn.SiLU(),
                    nn.Conv2d(kernel_size=4, in_channels=2560, out_channels=1, stride=1, padding=1), # 8x8 -> 8x8
                )
            else:
                self.cls_pred_branch = nn.Sequential(
                    nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=2, padding=1), # 32x32 -> 16x16 
                    nn.GroupNorm(num_groups=32, num_channels=1280),
                    nn.SiLU(),
                    nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=2, padding=1), # 16x16 -> 8x8 
                )
        else:
            self.cls_pred_branch = nn.Sequential(
                nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=2, padding=1), # 8x8 -> 4x4 
                nn.GroupNorm(num_groups=32, num_channels=1280),
                nn.SiLU(),
                nn.Conv2d(kernel_size=4, in_channels=1280, out_channels=1280, stride=4, padding=0), # 4x4 -> 1x1
                nn.GroupNorm(num_groups=32, num_channels=1280),
                nn.SiLU(),
                nn.Conv2d(kernel_size=1, in_channels=1280, out_channels=1, stride=1, padding=0), # 1x1 -> 1x1
            )

        self.cls_pred_branch.requires_grad_(True)

        if self.args.triplet_adaface_guidance:
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
                return model
            self.adaface_branch = load_pretrained_model('ir_50').to(accelerator.device)
            self.adaface_branch.requires_grad_(True)

        D = DCGAN_D()
        Triplet_Loss = CategoricalLoss(atoms=D.param['num_outcomes'], v_max=1.0, v_min=-1.0)

        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('BatchNorm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

        D.apply(weights_init)
        D = D.to(accelerator.device)
        Triplet_Loss.to(accelerator.device)

        self.sdxl = args.sdxl 
        self.gradient_checkpointing = args.gradient_checkpointing 

        self.diffusion_gan = args.diffusion_gan 
        self.diffusion_gan_max_timestep = args.diffusion_gan_max_timestep

        self.network_context_manager = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if self.use_fp16 else NoOpContext()

    def compute_cls_logits(self, image, text_embedding, unet_added_conditions=None):
        assert self.args.cls_on_clean_image
        # we are operating on the VAE latent space, no further normalization needed for now 
        if self.diffusion_gan:
            timesteps = torch.randint(
                0, self.diffusion_gan_max_timestep, [image.shape[0]], device=image.device, dtype=torch.long
            )
            image = self.scheduler.add_noise(image, torch.randn_like(image), timesteps)
        else:
            timesteps = torch.zeros([image.shape[0]], dtype=torch.long, device=image.device)

        with self.network_context_manager:
            rep = self.fake_unet.forward(
                image, timesteps, text_embedding,
                added_cond_kwargs=unet_added_conditions,
                classify_mode=True
            )

        # we only use the bottleneck layer 
        rep = rep[-1].float()
        if self.args.patch_gan:
            logits = self.cls_pred_branch(rep)
        else:
            logits = self.cls_pred_branch(rep).squeeze(dim=[2, 3])
        return logits
    
    def to_input_tensor(self, rgb_tensor):
        assert rgb_tensor.shape[3] == 3  # B H W C, batch size, height, width, channels
        bgr_tensor = rgb_tensor.flip(-1)
        brg_img = (bgr_tensor - 0.5) / 0.5
        tensor = brg_img.permute(0, 3, 1, 2)  # Change shape to (batch_size, channels, height, width)
        return tensor

    def compute_adaface_logits(self, image):
        assert self.args.triplet_adaface_guidance
        assert image.dim() == 4
        print(f"image shape: {image.shape}")
        
        # Resize images
        identify = F.interpolate(image, (112, 112), mode='bilinear', align_corners=False)  # 1, 3, 112, 112  max=1, min=0
        
        # Initialize features tensor
        features = torch.zeros((image.shape[0], 512), device=self.accelerator.device)
        
        with torch.no_grad():
            # Permute and convert to input tensor
            identify = identify.permute(0, 2, 3, 1)  # Change shape to (batch_size, 112, 112, 3)
            bgr_tensor_tensor = self.to_input_tensor(identify).to(self.accelerator.device)
            
            # Compute features for all images in the batch
            feature_tensors, _ = self.adaface_branch(bgr_tensor_tensor)
            features = feature_tensors
        
        return features

    def compute_distribution_matching_loss(
        self, 
        latents,
        text_embedding,
        uncond_embedding,
        unet_added_conditions,
        uncond_unet_added_conditions
    ):
        assert not self.args.gan_alone
        original_latents = latents 
        batch_size = latents.shape[0]
        with torch.no_grad():
            timesteps = torch.randint(
                self.min_step, 
                min(self.max_step+1, self.num_train_timesteps),
                [batch_size], 
                device=latents.device,
                dtype=torch.long
            )

            noise = torch.randn_like(latents)

            noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

            # run at full precision as autocast and no_grad doesn't work well together 
            pred_fake_noise = predict_noise(
                self.fake_unet, noisy_latents, text_embedding, uncond_embedding, 
                timesteps, guidance_scale=self.fake_guidance_scale,
                unet_added_conditions=unet_added_conditions,
                uncond_unet_added_conditions=uncond_unet_added_conditions
            )  

            pred_fake_image = get_x0_from_noise(
                noisy_latents.double(), pred_fake_noise.double(), self.alphas_cumprod.double(), timesteps
            )

            if self.use_fp16:
                if self.sdxl:
                    bf16_unet_added_conditions = {} 
                    bf16_uncond_unet_added_conditions = {} 

                    for k,v in unet_added_conditions.items():
                        bf16_unet_added_conditions[k] = v.to(torch.bfloat16)
                    for k,v in uncond_unet_added_conditions.items():
                        bf16_uncond_unet_added_conditions[k] = v.to(torch.bfloat16)
                else:
                    bf16_unet_added_conditions = unet_added_conditions 
                    bf16_uncond_unet_added_conditions = uncond_unet_added_conditions

                pred_real_noise = predict_noise(
                    self.real_unet, noisy_latents.to(torch.bfloat16), text_embedding.to(torch.bfloat16), 
                    uncond_embedding.to(torch.bfloat16), 
                    timesteps, guidance_scale=self.real_guidance_scale,
                    unet_added_conditions=bf16_unet_added_conditions,
                    uncond_unet_added_conditions=bf16_uncond_unet_added_conditions
                ) 
            else:
                pred_real_noise = predict_noise(
                    self.real_unet, noisy_latents, text_embedding, uncond_embedding, 
                    timesteps, guidance_scale=self.real_guidance_scale,
                    unet_added_conditions=unet_added_conditions,
                    uncond_unet_added_conditions=uncond_unet_added_conditions
                )

            pred_real_image = get_x0_from_noise(
                noisy_latents.double(), pred_real_noise.double(), self.alphas_cumprod.double(), timesteps
            )     

            p_real = (latents - pred_real_image)
            p_fake = (latents - pred_fake_image)

            grad = (p_real - p_fake) / torch.abs(p_real).mean(dim=[1, 2, 3], keepdim=True) 
            grad = torch.nan_to_num(grad)

        loss = 0.5 * F.mse_loss(original_latents.float(), (original_latents-grad).detach().float(), reduction="mean")         

        loss_dict = {
            "loss_dm": loss 
        }

        dm_log_dict = {
            "dmtrain_noisy_latents": noisy_latents.detach().float(),
            "dmtrain_pred_real_image": pred_real_image.detach().float(),
            "dmtrain_pred_fake_image": pred_fake_image.detach().float(),
            "dmtrain_grad": grad.detach().float(),
            "dmtrain_gradient_norm": torch.norm(grad).item()
        }

        return loss_dict, dm_log_dict

    def compute_generator_clean_cls_loss(self, 
        fake_image, text_embedding, 
        unet_added_conditions=None
    ):
        assert self.args.cls_on_clean_image
        loss_dict = {} 

        pred_realism_on_fake_with_grad = self.compute_cls_logits(
            fake_image, 
            text_embedding=text_embedding, 
            unet_added_conditions=unet_added_conditions
        )
        loss_dict["gen_cls_loss"] = F.softplus(-pred_realism_on_fake_with_grad).mean()
        return loss_dict 
    
    def compute_generator_triplet_adaface_loss(self, pred_image, gt_image, lq_image, other_image):
        ####################### detach??? 
        assert self.args.triplet_adaface_guidance
        assert pred_image.shape.size() == 4
        assert gt_image.shape.size() == 4
        assert lq_image.shape.size() == 4
        assert other_image.shape.size() == 4
        feat_fake = self.compute_adaface_logits(pred_image.detach())  ## detach? 
        feat_real = self.compute_adaface_logits(gt_image)
        if self.args.triplet_adaface_guidance_lq:
            anchor_fake = self.compute_adaface_logits(lq_image)
        else:
            anchor_fake = self.compute_adaface_logits(other_image)

        print(f"feat_fake shape: {feat_fake.shape}")
        print(f"feat_real shape: {feat_real.shape}")
        print(f"anchor_fake shape: {anchor_fake.shape}")

        # loss1 = torch.matmul(anchor_fake, feat_fake).diag().mean()
        # loss2 = torch.matmul(feat_real, feat_fake).diag().mean()

        loss_dict = {} 
        loss1 = F.kl_div(feat_fake.log(), anchor_fake, reduction='batchsum')
        loss2 = F.kl_div(feat_fake.log(), feat_real, reduction='batchsum')
        loss_dict["G_triplet_adaface_loss"] = -loss1 + loss2
        return loss_dict

    def generator_forward(
        self,
        lq_image,
        pred_latent,
        pred_image,
        text_embedding,
        uncond_embedding,
        real_train_dict=None,
        unet_added_conditions=None,
        uncond_unet_added_conditions=None
    ):
        loss_dict = {}
        log_dict = {}

        # image.requires_grad_(True)
        if not self.args.gan_alone:
            dm_dict, dm_log_dict = self.compute_distribution_matching_loss(
                pred_latent, text_embedding, uncond_embedding, 
                unet_added_conditions, uncond_unet_added_conditions
            )

            loss_dict.update(dm_dict)
            log_dict.update(dm_log_dict)

        if self.args.cls_on_clean_image:
            clean_cls_loss_dict = self.compute_generator_clean_cls_loss(
                pred_latent, text_embedding, unet_added_conditions
            )
            loss_dict.update(clean_cls_loss_dict)

        if self.args.triplet_adaface_guidance:
            triplet_adaface_loss_dict = self.compute_generator_triplet_adaface_loss(
                gt_image=real_train_dict['gt_image'], 
                pred_image=pred_image, 
                lq_image=lq_image, 
                other_image=real_train_dict['gt_image2']
            )
            loss_dict.update(triplet_adaface_loss_dict)

        return loss_dict, log_dict

    def compute_loss_fake(
        self,
        latents,
        text_embedding,
        uncond_embedding,
        unet_added_conditions=None,
        uncond_unet_added_conditions=None
    ):
        assert not self.args.gan_alone
        if self.gradient_checkpointing:
            self.fake_unet.enable_gradient_checkpointing()
        latents = latents.detach()
        batch_size = latents.shape[0]
        noise = torch.randn_like(latents)

        timesteps = torch.randint(
            0,
            self.num_train_timesteps,
            [batch_size], 
            device=latents.device,
            dtype=torch.long
        )
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        with self.network_context_manager:
            fake_noise_pred = predict_noise(
                self.fake_unet, noisy_latents, text_embedding, uncond_embedding,
                timesteps, guidance_scale=1, # no guidance for training dfake 
                unet_added_conditions=unet_added_conditions,
                uncond_unet_added_conditions=uncond_unet_added_conditions
            )

        fake_noise_pred = fake_noise_pred.float()

        fake_x0_pred = get_x0_from_noise(
            noisy_latents.double(), fake_noise_pred.double(), self.alphas_cumprod.double(), timesteps
        )

        # epsilon prediction loss 
        loss_fake = torch.mean(
            (fake_noise_pred.float() - noise.float())**2
        )

        loss_dict = {
            "loss_fake_mean": loss_fake,
        }

        fake_log_dict = {
            "faketrain_latents": latents.detach().float(),
            "faketrain_noisy_latents": noisy_latents.detach().float(),
            "faketrain_x0_pred": fake_x0_pred.detach().float()
        }
        if self.gradient_checkpointing:
            self.fake_unet.disable_gradient_checkpointing()
        return loss_dict, fake_log_dict

    def compute_guidance_clean_cls_loss(
            self, real_image, fake_image, 
            real_text_embedding, fake_text_embedding,
            real_unet_added_conditions=None, 
            fake_unet_added_conditions=None
        ):
        assert self.args.cls_on_clean_image
        pred_realism_on_real = self.compute_cls_logits(
            real_image.detach(), 
            text_embedding=real_text_embedding,
            unet_added_conditions=real_unet_added_conditions
        )
        pred_realism_on_fake = self.compute_cls_logits(
            fake_image.detach(), 
            text_embedding=fake_text_embedding,
            unet_added_conditions=fake_unet_added_conditions
        )

        log_dict = {
            "pred_realism_on_real": torch.sigmoid(pred_realism_on_real).squeeze(dim=1).detach(),
            "pred_realism_on_fake": torch.sigmoid(pred_realism_on_fake).squeeze(dim=1).detach()
        }

        if self.args.patch_gan:
            criterion = torch.nn.BCEWithLogitsLoss()
            real_loss = criterion(pred_realism_on_real, torch.ones_like(pred_realism_on_real))
            fake_loss = criterion(pred_realism_on_fake, torch.zeros_like(pred_realism_on_fake))
            classification_loss = (real_loss + fake_loss) / 2
        else:
            classification_loss = F.softplus(pred_realism_on_fake).mean() + F.softplus(-pred_realism_on_real).mean()
        loss_dict = {
            "guidance_cls_loss": classification_loss
        }
        return loss_dict, log_dict 

    def compute_guidance_triplet_adaface_loss(
            self, pred_image, gt_image, lq_image, other_image
        ):
        assert self.args.triplet_adaface_guidance
        pred_realism_on_real = self.compute_cls_logits(
            real_image.detach(), 
            text_embedding=real_text_embedding,
            unet_added_conditions=real_unet_added_conditions
        )
        pred_realism_on_fake = self.compute_cls_logits(
            fake_image.detach(), 
            text_embedding=fake_text_embedding,
            unet_added_conditions=fake_unet_added_conditions
        )

        log_dict = {
            "pred_realism_on_real": torch.sigmoid(pred_realism_on_real).squeeze(dim=1).detach(),
            "pred_realism_on_fake": torch.sigmoid(pred_realism_on_fake).squeeze(dim=1).detach()
        }

        if self.args.patch_gan:
            criterion = torch.nn.BCEWithLogitsLoss()
            real_loss = criterion(pred_realism_on_real, torch.ones_like(pred_realism_on_real))
            fake_loss = criterion(pred_realism_on_fake, torch.zeros_like(pred_realism_on_fake))
            classification_loss = (real_loss + fake_loss) / 2
        else:
            classification_loss = F.softplus(pred_realism_on_fake).mean() + F.softplus(-pred_realism_on_real).mean()
        loss_dict = {
            "guidance_cls_loss": classification_loss
        }
        return loss_dict, log_dict 

    def guidance_forward(
        self,
        lq_image,
        pred_latent,
        pred_image,
        text_embedding,
        uncond_embedding,
        real_train_dict=None,
        unet_added_conditions=None,
        uncond_unet_added_conditions=None
    ):
        loss_dict = {}
        log_dict = {}

        if not self.args.gan_alone:
            fake_dict, fake_log_dict = self.compute_loss_fake(
                pred_latent, text_embedding, uncond_embedding,
                unet_added_conditions=unet_added_conditions,
                uncond_unet_added_conditions=uncond_unet_added_conditions
            )

            loss_dict = fake_dict 
            log_dict = fake_log_dict

        if self.args.cls_on_clean_image:
            clean_cls_loss_dict, clean_cls_log_dict = self.compute_guidance_clean_cls_loss(
                real_image=real_train_dict['gt_latent'], 
                fake_image=pred_latent,
                real_text_embedding=text_embedding,
                fake_text_embedding=text_embedding, 
                real_unet_added_conditions=unet_added_conditions,
                fake_unet_added_conditions=unet_added_conditions
            )
            loss_dict.update(clean_cls_loss_dict)
            log_dict.update(clean_cls_log_dict)

        if self.args.triplet_adaface_guidance:
            triplet_adaface_loss_dict, triplet_adaface_log_dict = self.compute_guidance_triplet_adaface_loss(
                gt_image=real_train_dict['gt_latent'], 
                image=pred_latent,
                lq_image=real_train_dict['lq_latent'],
                other_image=real_train_dict['other_latent']
            )
            loss_dict.update(triplet_adaface_loss_dict)
            log_dict.update(triplet_adaface_log_dict)

        return loss_dict, log_dict 

    def forward(
        self,
        generator_turn=False,
        guidance_turn=False,
        generator_data_dict=None,    # in G training mode, the data dict
        guidance_data_dict=None,     # in D training mode, the data dict
        guidance_factor=1
    ):    
        if generator_turn:
            loss_dict, log_dict = self.generator_forward(
                lq_image=generator_data_dict["lq_image"],
                pred_latent=generator_data_dict["pred_latent"],
                pred_image=generator_data_dict["pred_image"],
                text_embedding=generator_data_dict["text_embedding"],
                uncond_embedding=generator_data_dict["uncond_embedding"],
                real_train_dict=generator_data_dict["real_train_dict"],
                unet_added_conditions=generator_data_dict["unet_added_conditions"],
                uncond_unet_added_conditions=generator_data_dict["uncond_unet_added_conditions"]
            )   
        elif guidance_turn:
            loss_dict, log_dict = self.guidance_forward(
                lq_image=guidance_data_dict["lq_image"],
                pred_latent=guidance_data_dict["pred_latent"],
                pred_image=guidance_data_dict["pred_image"],
                text_embedding=guidance_data_dict["text_embedding"],
                uncond_embedding=guidance_data_dict["uncond_embedding"],
                real_train_dict=guidance_data_dict["real_train_dict"],
                unet_added_conditions=guidance_data_dict["unet_added_conditions"],
                uncond_unet_added_conditions=guidance_data_dict["uncond_unet_added_conditions"]
            ) 
        else:
            raise NotImplementedError
        loss_dict = {key: value * guidance_factor for key, value in loss_dict.items()}
        return loss_dict, log_dict 
