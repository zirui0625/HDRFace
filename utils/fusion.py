import torch
import torch.nn as nn
import torch.nn.functional as F

class SDFM(nn.Module):
    def __init__(
        self,
        token_dim: int = 1024,
        t_emb_dim: int = 0,
        hidden_dim: int = 512,
        use_layernorm: bool = True,
    ):
        super().__init__()

        self.token_dim = token_dim
        self.t_emb_dim = t_emb_dim

        self.norm = nn.LayerNorm(token_dim) if use_layernorm else nn.Identity()

        # Channel Gate
        ch_in_dim = token_dim * 3 + (t_emb_dim if t_emb_dim > 0 else 0)
        self.channel_mlp = nn.Sequential(
            nn.Linear(ch_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, token_dim),
        )

        # Token Gate
        tk_in_dim = token_dim * 3 + (t_emb_dim if t_emb_dim > 0 else 0)
        self.token_mlp = nn.Sequential(
            nn.Linear(tk_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.eps = 1e-6
    def forward(
        self,
        f_sr: torch.Tensor,
        f_lr: torch.Tensor,
        t_emb: torch.Tensor = None,
    ):
        assert f_sr.shape == f_lr.shape, "f_sr and f_lr must be same shape"
        B, N, C = f_sr.shape
        assert C == self.token_dim, f"token_dim unmatching : {C} vs {self.token_dim}"

        if self.t_emb_dim > 0:
            assert t_emb is not None, "when t_emb_dim > 0, there must have t_emb"
            assert t_emb.shape[0] == B

        f_sr = self.norm(f_sr)
        f_lr = self.norm(f_lr)

        diff = (f_sr - f_lr).abs()  # (B, N, C)

        # Channel Gate
        f_sr_pool = f_sr.mean(dim=1)    # (B, C)
        f_lr_pool = f_lr.mean(dim=1)    # (B, C)
        diff_pool = diff.mean(dim=1)    # (B, C)

        ch_feat = torch.cat([f_sr_pool, f_lr_pool, diff_pool], dim=-1)  # (B, 3C)

        if self.t_emb_dim > 0:
            ch_feat = torch.cat([ch_feat, t_emb], dim=-1)

        alpha_channel = torch.sigmoid(self.channel_mlp(ch_feat))  # (B, C)
        alpha_channel = alpha_channel.unsqueeze(1)  # (B, 1, C)

        # Token Gate
        tk_feat = torch.cat([f_sr, f_lr, diff], dim=-1)  # (B, N, 3C)

        if self.t_emb_dim > 0:
            t_map = t_emb.unsqueeze(1).expand(B, N, self.t_emb_dim)
            tk_feat = torch.cat([tk_feat, t_map], dim=-1)

        alpha_token = torch.sigmoid(self.token_mlp(tk_feat))  # (B, N, 1)

        # Final Gate：token gate × channel gate
        alpha = alpha_token * alpha_channel  # (B, N, C)
        alpha = alpha.clamp(self.eps, 1.0 - self.eps)

        f_fused = alpha * f_sr + (1.0 - alpha) * f_lr

        return f_fused, alpha
