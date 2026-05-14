import torch
import os
from iresnet import create_arcface_embedding
from PIL import Image, ImageOps
from torchvision import transforms
import torchvision.transforms.functional as F
from torch.nn.functional import interpolate
device = torch.device("cuda")

# from diffusers import UNet2DConditionModel
# from models.autoencoder_kl import AutoencoderKL
# vae = AutoencoderKL.from_pretrained(
#     '/data/user/jkwang/DFOSD/preset/models/stable-diffusion-2-1-base', 
#     subfolder="vae"
# ).float().to(device)
# vae.eval()

# fake_unet = UNet2DConditionModel.from_pretrained(
#     '/data/user/jkwang/DFOSD/preset/models/stable-diffusion-2-1-base',
#     subfolder="unet"
# ).float()

# timesteps = torch.zeros([3], dtype=torch.long, device=device)
# rep = fake_unet.forward(
#     image, timesteps, text_embedding,
#     added_cond_kwargs=unet_added_conditions,
#     classify_mode=True
# )
# # use model_id to do a diffusion forward pass (default sd 2.1)

# # we only use the bottleneck layer 
# rep = rep[-1].float()

# 创建并加载 ArcFace 嵌入模型
embedding = create_arcface_embedding()
embedding.load_state_dict(torch.load("/data/user/jkwang/DFOSD/preset/models/ms1mv3_arcface_r50_fp16.pth"))
embedding.to(device)
embedding.eval()

# 输入图像路径
input_images = {
    'LQ': '/data/user/jkwang/DFOSD/data/CelebA-Test/LQ/self_celeba_512_v2/00000000.png', 
    'HQ': '/data/user/jkwang/DFOSD/data/CelebA-Test/HQ/celeba_512_validation/00000000.png',
    'CodeFormer': '/data/user/jkwang/DFOSD/results_zhanglin/CodeFormer/celebA_LQ/restored_faces/00000000.png',
    'DiffBIR': '/data/user/jkwang/DFOSD/results_zhanglin/DiffBIR/celebA_LQ/00000000.png', 
    'RestoreFormer++': '/data/user/jkwang/DFOSD/results_zhanglin/RestoreFormer++/celebA_LQ/aligned/restored_faces/00000000_00.png', 
    'PGDiff': '/data/user/jkwang/DFOSD/results_zhanglin/PGDiff/celebTest_A/s0.05-seed1234/00000000.png', 
    'DAEFR': '/data/user/jkwang/DFOSD/results_zhanglin/DAEFR/self_celeba_512_v2/restored_faces/00000000_00.png', 
}

print(input_images)

# 存储嵌入向量
embeddings = {}
embeddings_2 = {}

# 处理每个输入图像并计算嵌入向量
for bname, path in input_images.items():
    print(f"{bname=}, {path=}")
    input_image = Image.open(path).convert('RGB')
    print(f"{input_image.size=}")
    ori_width, ori_height = input_image.size
    
    with torch.no_grad():
        lq = F.to_tensor(input_image).unsqueeze(0).cuda()*2-1         # torch.Size([1, 3, 512, 512]), lq.max()=1.0, lq.min()=-1.0
        print(f"{lq.size()=}, lq.max()={lq.max()}, lq.min()={lq.min()}")
        # lq_latent = vae.encode(lq).latent_dist.sample() * vae.config.scaling_factor
        # print(f"{lq_latent.size()=}, lq_latent.max()={lq_latent.max()}, lq_latent.min()={lq_latent.min()}")
        inter = interpolate(lq, (112, 112), mode='bilinear', antialias=True)
        print(f"{inter.shape=}, {inter.size()=}, inter.max()={inter.max()}, inter.min()={inter.min()}")
        embeddings[bname] = embedding(inter) # embedding outputs'shape are (1, 512) tensor. 
        # embeddings_2[bname] = embedding(interpolate(lq_latent, (112, 112), mode='bicubic', antialias=True))
        
# 计算余弦相似度矩阵
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
keys = list(embeddings.keys())
num_embeddings = len(keys)
similarity_matrix = torch.zeros((num_embeddings, num_embeddings))

for i in range(num_embeddings):
    for j in range(num_embeddings):
        if i != j:
            similarity_matrix[i, j] = cos(embeddings[keys[i]], embeddings[keys[j]])

# 输出余弦相似度矩阵
print("Cosine Similarity Matrix:")
print(similarity_matrix)

# tensor([[0.0000, 0.1523, 0.1843, 0.3245, 0.5085],
#         [0.1523, 0.0000, 0.4484, 0.5099, 0.3640],
#         [0.1843, 0.4484, 0.0000, 0.5481, 0.4391],
#         [0.3245, 0.5099, 0.5481, 0.0000, 0.4867],
#         [0.5085, 0.3640, 0.4391, 0.4867, 0.0000]])

