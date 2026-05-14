from typing import List, Dict
import random
import math

import numpy as np
from PIL import Image
import cv2


def load_file_list(file_list_path: str) -> List[Dict[str, str]]:
    files = []
    with open(file_list_path, "r") as fin:
        for line in fin:
            path = line.strip()
            if path:
                files.append({"image_path": path, "prompt": ""})
    return files


# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py
def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/image_datasets.py
def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]

class USMSharp:
    def __init__(self, radius=50, sigma=0):
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        # 获取高斯核
        kernel = cv2.getGaussianKernel(radius, sigma)
        # 转换为 2D 卷积核
        self.kernel = np.dot(kernel, kernel.T)

    def filter2D(self, img, kernel):
        # 使用 OpenCV 进行卷积操作
        return cv2.filter2D(img, -1, kernel)

    def forward(self, img, weight=0.5, threshold=10):
        # 计算模糊图像
        blur = self.filter2D(img, self.kernel)
        # 计算图像残差
        residual = img - blur

        # 计算掩码，只有残差超过阈值的部分才进行增强
        mask = np.abs(residual) * 255 > threshold
        soft_mask = self.filter2D(mask.astype(np.float32), self.kernel)

        # 进行加权锐化
        sharp = img + weight * residual
        # 将结果裁剪到 [0, 1] 范围
        sharp = np.clip(sharp, 0, 1)

        # 最终输出
        return soft_mask * sharp + (1 - soft_mask) * img