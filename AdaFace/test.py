import AdaFace.net as adaface_net
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
import os
from PIL import Image
import numpy as np
import csv
from tqdm import tqdm

class DCGAN_D(nn.Module):
    def __init__(self, param):
        super(DCGAN_D, self).__init__()
        self.param = param
        model = []

        # start block
        model.append(spectral_norm(nn.Conv2d(self.param.n_channels, self.param.D_h_size, kernel_size=4, stride=2, padding=1, bias=False)))
        model.append(nn.LeakyReLU(0.2, inplace=True))

        image_size_new = self.param.image_size // 2

        # middle block
        mult = 1
        while image_size_new > 4:
            model.append(spectral_norm(nn.Conv2d(self.param.D_h_size * mult, self.param.D_h_size * (2*mult), kernel_size=4, stride=2, padding=1, bias=False)))
            model.append(nn.LeakyReLU(0.2, inplace=True))

            image_size_new = image_size_new // 2
            mult *= 2

        self.model = nn.Sequential(*model)
        self.mult = mult

        # end block
        in_size  = int(param.D_h_size * mult * 4 * 4)
        out_size = self.param.num_outcomes 
        self.fc = spectral_norm(nn.Linear(in_size, out_size, bias=False))

        # resampling trick
        self.reparam = spectral_norm(nn.Linear(in_size, out_size * 2, bias=False))

    def forward(self, input):
        y = self.model(input)

        y = y.view(-1, self.param.D_h_size * self.mult * 4 * 4)
        output = self.fc(y).view(-1, self.param.num_outcomes)

        # re-parameterization trick
        if self.param.use_adaptive_reparam:
            stat_tuple = self.reparam(y).unsqueeze(2).unsqueeze(3)
            mu, logvar = stat_tuple.chunk(2, 1)
            std = logvar.mul(0.5).exp_()
            epsilon = torch.randn(input.shape[0], self.param.num_outcomes, 1, 1).to(stat_tuple)
            output = epsilon.mul(std).add_(mu).view(-1, self.param.num_outcomes)

        return output


adaface_models = {
    "ir_50": "/data/user/jkwang/DFOSD/AdaFace/pretrained/adaface_ir50_ms1mv2.ckpt",
}

def load_pretrained_model(architecture="ir_50"):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = adaface_net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])["state_dict"]
    model_statedict = {
        key[6:]: val for key, val in statedict.items() if key.startswith("model.")
    }
    model.load_state_dict(model_statedict)
    # model.output_layer = torch.nn.Identity()
    model.eval()
    # model.train()
    # torch.Size([1, 512]) torch.Size([1, 512]) torch.Size([1, 512])
    # torch.Size([1, 512, 7, 7]) torch.Size([1, 512, 7, 7]) torch.Size([1, 512, 7, 7])
    return model

def to_input_tensor(rgb_tensor):
    assert rgb_tensor.shape[2] == 3  # H W C, 112 112 3
    bgr_tensor = rgb_tensor.flip(-1)
    brg_img = (bgr_tensor - 0.5) / 0.5
    tensor = brg_img.permute(2, 0, 1)
    return tensor

def load_images(image_paths):
    images = []
    for path in tqdm(image_paths, desc="Loading images"):
        image = Image.open(path).convert("RGB")
        image = F.interpolate(
            torch.tensor(np.array(image)).permute(2, 0, 1).unsqueeze(0), (112, 112), mode="bilinear", antialias=True
        ).squeeze(0).permute(1, 2, 0)
        images.append(to_input_tensor(image))
    return torch.stack(images)

if __name__ == "__main__":
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

    adaface_branch = load_pretrained_model("ir_50").to(device)

    input_images = {
        "LQ": "/data/user/jkwang/DFOSD/data/CelebA-Test/LQ/self_celeba_512_v2",
        "HQ": "/data/user/jkwang/DFOSD/data/CelebA-Test/HQ/celeba_512_validation",
        "Pred": "/data/user/gj/DFOSD/results/baseline/bs1_vqEnc_2id_v2_99000_ACT_test",
        "Other": "/data/user/jkwang/DFOSD/data/CelebA-Test/HQ/celeba_512_validation"
    }

    num_images = 64
    results = []

    pred_paths = [os.path.join(input_images["Pred"], f"{i:08d}.png") for i in range(num_images)]
    hq_paths = [os.path.join(input_images["HQ"], f"{i:08d}.png") for i in range(num_images)]
    lq_paths = [os.path.join(input_images["LQ"], f"{i:08d}.png") for i in range(num_images)]
    other_paths = [os.path.join(input_images["Other"], f"{(i+1)%num_images:08d}.png") for i in range(num_images)]

    # print(f"{hq_paths=}")
    # print(f"{other_paths=}")

    pred_images = load_images(pred_paths).to(device)
    hq_images = load_images(hq_paths).to(device)
    lq_images = load_images(lq_paths).to(device)
    other_images = load_images(other_paths).to(device)

    print(f"{pred_images.shape=}, {hq_images.shape=}, {lq_images.shape=}, {other_images.shape=}") # torch.Size([3000, 3, 112, 112])

    feature_pred_tensor, _ = adaface_branch(pred_images)
    feature_gt_tensor, _ = adaface_branch(hq_images)
    feature_lq_tensor, _ = adaface_branch(lq_images)
    feature_other_tensor, _ = adaface_branch(other_images)

    print(f"{feature_pred_tensor.shape=}, {feature_gt_tensor.shape=}, {feature_lq_tensor.shape=}, {feature_other_tensor.shape=}")
    # torch.Size([3000, 512])

    # 计算相似度分数
    similarity_scores_1 = torch.matmul(feature_gt_tensor, feature_pred_tensor.T).diag()
    similarity_scores_2 = torch.matmul(feature_gt_tensor, feature_lq_tensor.T).diag()
    similarity_scores_3 = torch.matmul(feature_gt_tensor, feature_other_tensor.T).diag()

    # 转换相似度分数
    similarity_scores_1 = 1 - similarity_scores_1
    similarity_scores_2 = 1 - similarity_scores_2
    similarity_scores_3 = 1 - similarity_scores_3

    print(f"{similarity_scores_1.shape=}, {similarity_scores_2.shape=}, {similarity_scores_3.shape=}")

    with open('similarity_scores_train.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Filename', 'HQ/Pred Similarity', 'HQ/LQ Similarity', 'HQ/Other Similarity', 'LQ-Pred', 'Other-Pred', 'Other-LQ'])
        csvwriter.writerows(zip(range(num_images), 
                                [round(score, 4) for score in similarity_scores_1.tolist()], 
                                [round(score, 4) for score in similarity_scores_2.tolist()], 
                                [round(score, 4) for score in similarity_scores_3.tolist()], 
                                [round(score, 4) for score in (similarity_scores_2 - similarity_scores_1).tolist()],
                                [round(score, 4) for score in (similarity_scores_3 - similarity_scores_1).tolist()],
                                [round(score, 4) for score in (similarity_scores_3 - similarity_scores_2).tolist()]
                            ))

    print(f'Similarity scores saved to similarity_scores_train.csv')