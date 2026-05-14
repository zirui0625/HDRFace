import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from Codebook.vqvae_v3 import SDVQVAE
from models.sd_unifined_model import SDUniModel
from accelerate import Accelerator
import argparse 

def add_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="preset/models/stable-diffusion-2-1-base")
    parser.add_argument("--guidance_model_id", type=str, default="preset/models/stable-diffusion-xl-base-1.0")
    parser.add_argument('--ram_path', type=str, default='preset/models/ram_swin_large_14m.pth')
    parser.add_argument('--ram_ft_path', type=str, default='preset/models/DAPE.pth')
    parser.add_argument("--output_path", type=str, default="exp")
    parser.add_argument("--dataset_cfg", type=str, default="options/restoreformer_usm.yaml")
    parser.add_argument("--log_path", type=str, default="tb-log")
    parser.add_argument("--train_iters", type=int, default=1000000)
    parser.add_argument("--log_iters", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=114)
    # parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--visual_iters", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=10.0, help="max grad norm for network")
    parser.add_argument("--warmup_step", type=int, default=500, help="warmup step for network")
    parser.add_argument("--min_step_percent", type=float, default=0.02, help="minimum step percent for training")
    parser.add_argument("--max_step_percent", type=float, default=0.98, help="maximum step percent for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--enable_xformers", action="store_true")
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--ckpt_only_path", type=str, default=None, help="checkpoint (no optimizer state) only path")
    # parser.add_argument("--train_prompt_path", type=str)
    # parser.add_argument("--latent_resolution", type=int, default=64)
    parser.add_argument("--real_guidance_scale", type=float, default=6.0)
    parser.add_argument("--fake_guidance_scale", type=float, default=1.0)
    # parser.add_argument("--grid_size", type=int, default=2)
    parser.add_argument("--no_save", action="store_true", help="don't save ckpt for debugging only")
    parser.add_argument("--cache_dir", type=str, default="/mnt/localssd/cache")
    parser.add_argument("--log_loss", action="store_true", help="log loss at every iteration")
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--latent_channel", type=int, default=4)
    parser.add_argument("--max_checkpoint", type=int, default=5)
    parser.add_argument("--dfake_gen_update_ratio", type=int, default=1)
    parser.add_argument("--generator_lr", type=float)
    parser.add_argument("--guidance_lr", type=float)
    parser.add_argument("--spatial_loss", action="store_true")
    parser.add_argument("--cls_on_clean_image", action="store_true")
    parser.add_argument("--gen_cls_loss", action="store_true")
    parser.add_argument("--percep_weight", type=float, default=0)
    parser.add_argument("--dists_weight", type=float, default=0)
    parser.add_argument("--edge_as_weight", action="store_true")
    parser.add_argument("--edge_weight", type=float, default=0)
    parser.add_argument("--high_freq_weight", type=float, default=0)
    parser.add_argument("--high_freq_weight_dists", type=float, default=0)
    parser.add_argument("--gen_cls_loss_weight", type=float, default=1)
    parser.add_argument("--guidance_cls_loss_weight", type=float, default=1)
    parser.add_argument("--sdxl", action="store_true")
    parser.add_argument("--sdxl_guidance", action="store_true")
    parser.add_argument("--load_sdxl_tokenizer", action="store_true")
    parser.add_argument("--sdxl_path", type=str, default="preset/models/stable-diffusion-xl-base-1.0")
    parser.add_argument("--learnable_text_embedding", action="store_true")
    parser.add_argument("--random_embedding", action="store_true")
    parser.add_argument("--fsdp", action="store_true")
    parser.add_argument("--generator_ckpt_path", type=str)
    parser.add_argument("--conditioning_timestep", type=int, default=999)
    parser.add_argument("--gradient_checkpointing", action="store_true", help="apply gradient checkpointing for dfake and generator. this might be a better option than FSDP")
    parser.add_argument("--dm_loss_weight", type=float, default=1.0)

    # parser.add_argument("--denoising", action="store_true", help="train the generator for denoising")
    parser.add_argument("--use_x0", action="store_true")
    parser.add_argument("--denoising_timestep", type=int, default=1000)
    parser.add_argument("--num_denoising_step", type=int, default=1)
    parser.add_argument("--denoising_loss_weight", type=float, default=1.0)

    parser.add_argument("--diffusion_gan", action="store_true")
    parser.add_argument("--patch_gan", action="store_true")
    parser.add_argument("--diffusion_gan_max_timestep", type=int, default=0)
    parser.add_argument("--revision", type=str)

    parser.add_argument("--real_image_path", type=str)
    parser.add_argument("--gan_alone", action="store_true", help="only use the gan loss without dmd")
    parser.add_argument("--backward_simulation", action="store_true")

    parser.add_argument("--generator_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=float, default=8)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_ckpt", type=str, default=None)

    parser.add_argument("--arcface_guidance", action="store_true")
    parser.add_argument("--arcface_guidance_loss_weight", type=float, default=0)
    parser.add_argument("--arcface_guidance_cosine", action="store_true")

    parser.add_argument("--arcface_spatial", action="store_true")
    parser.add_argument("--arcface_spatial_loss_weight", type=float, default=0)

    parser.add_argument("--adaface_spatial", action="store_true")
    parser.add_argument("--adaface_spatial_loss_weight", type=float, default=0)
    parser.add_argument("--codebook", action="store_true")
    parser.add_argument("--freeze_codebook", action="store_true")
    parser.add_argument("--terminal_print_loss", action="store_true")
    return parser

if __name__ == '__main__':
    print(f"torch.__version__: {torch.__version__}")
    
    accelerator = Accelerator()
    device = accelerator.device

    resume = '/home/jkwang/data/DAEFR/experiments/logs/2024-10-23T01-45-13_SDvae_v3_edge/checkpoints/last.ckpt'
    
    # 检查文件是否存在
    if not os.path.exists(resume):
        raise FileNotFoundError(f"Checkpoint file not found: {resume}")
    
    state_dict = torch.load(resume, map_location='cpu', weights_only=False)["state_dict"]
    new_state_dict = {}

    with open('outcodebook.txt', 'w') as file:
        # 遍历原始的 state_dict
        for key in state_dict:
            # 检查键是否以 'vqvae.' 开头
            print(f"Key: {key}", file=file)
            if key.startswith('vqvae.'):
                print(f" Original key: {key}", file=file)
                # 删除 'vqvae.' 部分，只保留之后的名称
                new_key = key.replace('vqvae.', '')
                print(f" New key: {new_key}", file=file)
                # 将新的键名和对应的值存储到新的字典中
                new_state_dict[new_key] = state_dict[key]

    vqvae = SDVQVAE().to(device=device)
    
    # 打印模型加载状态
    try:
        vqvae.load_state_dict(new_state_dict, strict=False)
        print("Model state_dict loaded successfully.")
    except Exception as e:
        print(f"Error loading state_dict: {e}")
    
    latent = torch.randn(1, 4, 64, 64).to(device=device)  # 创建一个随机输入张量
    
    args = add_arg().parse_args()
    args.codebook = True
    
    sdunimodel = SDUniModel(args, accelerator)
    sdunimodel.vqvae.to(accelerator.device)
    matching = sdunimodel.decode_image_codebook(latent)  # 使用模型解码输入张量
    
    latent_after_vae_decoder = sdunimodel.vae.decoder(sdunimodel.decoder_conv_in(latent))
    print(f"{latent_after_vae_decoder.shape=}, {matching.shape=}")  # 打印输入张量和对应的输出张量
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    similarity = cos(latent_after_vae_decoder.flatten(1), matching.flatten(1))
    print(f"Similarity: {similarity}")  # 打印输入张量和输出张量的余弦相似度

