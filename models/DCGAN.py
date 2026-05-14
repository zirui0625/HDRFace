import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class DCGAN_D(nn.Module):
    default_param = {
        'n_channels': 3,
        'D_updates' : 1,
        'D_h_size': 32,
        'image_size': 512,
        'num_outcomes': 51,
        'use_adaptive_reparam': True
    }

    def __init__(self, param=None):
        super(DCGAN_D, self).__init__()
        if param is None:
            param = self.default_param
        self.param = param
        model = []
        # start block
        model.append(spectral_norm(nn.Conv2d(self.param['n_channels'], self.param['D_h_size'], kernel_size=4, stride=2, padding=1, bias=False)))
        model.append(nn.LeakyReLU(0.2, inplace=True))

        image_size_new = self.param['image_size'] // 2

        # middle block
        mult = 1
        while image_size_new > 4:
            model.append(spectral_norm(nn.Conv2d(self.param['D_h_size'] * mult, self.param['D_h_size'] * (2*mult), kernel_size=4, stride=2, padding=1, bias=False)))
            model.append(nn.LeakyReLU(0.2, inplace=True))

            image_size_new = image_size_new // 2
            mult *= 2

        self.model = nn.Sequential(*model)
        self.mult = mult

        # end block
        in_size  = int(param['D_h_size'] * mult * 4 * 4)
        out_size = self.param['num_outcomes']
        self.fc = spectral_norm(nn.Linear(in_size, out_size, bias=False))

        # resampling trick
        self.reparam = spectral_norm(nn.Linear(in_size, out_size * 2, bias=False))

    def forward(self, input):
        y = self.model(input)

        y = y.view(-1, self.param['D_h_size'] * self.mult * 4 * 4)
        output = self.fc(y).view(-1, self.param['num_outcomes'])

        # re-parameterization trick
        if self.param['use_adaptive_reparam']:
            stat_tuple = self.reparam(y).unsqueeze(2).unsqueeze(3)
            mu, logvar = stat_tuple.chunk(2, 1)
            std = logvar.mul(0.5).exp_()
            epsilon = torch.randn(input.shape[0], self.param['num_outcomes'], 1, 1).to(stat_tuple)
            output = epsilon.mul(std).add_(mu).view(-1, self.param['num_outcomes'])

        return output
    
class CategoricalLoss(nn.Module):
    def __init__(self, atoms=51, v_max=10, v_min=-10):
        super(CategoricalLoss, self).__init__()

        self.atoms = atoms
        self.v_max = v_max
        self.v_min = v_min
        self.supports = torch.linspace(v_min, v_max, atoms).view(1, 1, atoms) # RL: [bs, #action, #quantiles]
        self.delta = (v_max - v_min) / (atoms - 1)

    def to(self, device):
        self.device = device
        self.supports = self.supports.to(device)

    def forward(self, anchor, feature, skewness=0.0):
        batch_size = feature.shape[0]
        skew = torch.zeros((batch_size, self.atoms)).to(self.device).fill_(skewness)

        # experiment to adjust KL divergence between positive/negative anchors
        Tz = skew + self.supports.view(1, -1) * torch.ones((batch_size, 1)).to(torch.float).view(-1, 1).to(self.device)
        Tz = Tz.clamp(self.v_min, self.v_max)
        b = (Tz - self.v_min) / self.delta
        l = b.floor().to(torch.int64)
        u = b.ceil().to(torch.int64)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.atoms - 1)) * (l == u)] += 1
        offset = torch.linspace(0, (batch_size - 1) * self.atoms, batch_size).to(torch.int64).unsqueeze(dim=1).expand(batch_size, self.atoms).to(self.device)
        skewed_anchor = torch.zeros(batch_size, self.atoms).to(self.device)
        skewed_anchor.view(-1).index_add_(0, (l + offset).view(-1), (anchor * (u.float() - b)).view(-1))  
        skewed_anchor.view(-1).index_add_(0, (u + offset).view(-1), (anchor * (b - l.float())).view(-1))  

        loss = -(skewed_anchor * (feature + 1e-16).log()).sum(-1).mean()

        return loss

