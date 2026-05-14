import copy

# from models.autoencoder_kl import AutoencoderKL
from diffusers import AutoencoderKL
import time

import torch
import torch.nn as nn
import random
import math
import torch.nn.functional as F
import numpy as np
# from basicsr.utils.registry import ARCH_REGISTRY

class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        Inputs the output of the encoder network z and maps it to a discrete
        one-hot vector that is the index of the closest embedding vector e_j
        z (continuous) -> z_q (discrete)
        z.shape = (batch, channel, height, width)
        quantization pipeline:
            1. get encoder input (B,C,H,W)
            2. flatten input to (B*H*W,C)
        """
        # import pdb
        # pdb.set_trace()

        # reshape z -> (batch, height, width, channel) and flatten
        z = z.permute(0, 2, 3, 1).contiguous()
        # z_flattened -> ( batch*height*width, e_dim = 256)
        z_flattened = z.view(-1, self.e_dim)

        # distances d from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        # d shape -> ( batch*height*width, n_e = 1024)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        ## could possible replace this here
        # #\start...
        # find closest encodings

        # min_value shape -> (batch*height*width)
        # min_encoding_indices -> (batch*height*width)
        # "min_encoding_indices" indicate the corresponding code items
        min_value, min_encoding_indices = torch.min(d, dim=1)

        # min_encoding_indices -> (batch*height*width, 1)
        min_encoding_indices = min_encoding_indices.unsqueeze(1)

        # min_encodings shape -> ( batch*height*width, n_e = 1024)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # dtype min encodings: torch.float32
        # min_encodings shape: torch.Size([2048, 512])
        # min_encoding_indices.shape: torch.Size([2048, 1])

        # get quantized latent vectors
        # torch.matmul(min_encodings, self.embedding.weight)
        # shape -> ( batch*height*width, e_dim = 256)
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)
        # .........\end

        # with:
        # .........\start
        # min_encoding_indices = torch.argmin(d, dim=1)
        # z_q = self.embedding(min_encoding_indices)
        # ......\end......... (TODO)

        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z) ** 2) + self.beta * \
               torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity

        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices, d), self.embedding.weight

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # TODO: check for more easy handling with nn.Embedding
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:, None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

class SDVQVAE(nn.Module):
    def __init__(self, n_embed=2048, embed_dim=512, ch=128, out_ch=3, ch_mult=(1, 2, 4, 8),
                 num_res_blocks=2, attn_resolutions=16, dropout=0.0, in_channels=3,
                 resolution=512, z_channels=512, double_z=False, enable_mid=True,
                 fix_decoder=False, fix_codebook=False, fix_encoder=False, head_size=1,
                 model_id="/data1/pretrained/hf-models/sd21", **ignore_kwargs):
        super(SDVQVAE, self).__init__()
        self.vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae"
        ).float()
        self.vae.requires_grad_(False)
        self.decoder = self.vae.decoder
        self.decoder.conv_out.requires_grad_(True)
        self.conv_in = copy.deepcopy(self.decoder.conv_in)

        del self.decoder.conv_in

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25)
        self.z_channels = z_channels
        self.quant_conv = torch.nn.Conv2d(z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, z_channels, 1)


    def forward(self, x):
        # quant, diff, info, hs = self.encode(input)

        latents = self.vae.encode(x).latent_dist.sample() * self.vae.config.scaling_factor
        latents = 1 / self.vae.config.scaling_factor * latents
        latents = self.conv_in(latents)

        h = self.quant_conv(latents)
        quant, emb_loss, info, dictionary = self.quantize(h)
        quant = self.post_quant_conv(quant)

        x = self.decoder(quant).float().clamp(-1, 1)

        return x, emb_loss, info, latents, h, quant, dictionary
    
    def find_corresponding_code(self, latents):
        assert latents.shape[1] == self.z_channels
        h = self.quant_conv(latents)
        quant, emb_loss, info, dictionary = self.quantize(h)
        quant = self.post_quant_conv(quant)
        return quant